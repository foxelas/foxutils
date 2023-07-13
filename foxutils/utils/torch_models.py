# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from .core_utils import read_image_to_tensor

#############################################
# Autoencoder

IM_WIDTH = 640
IM_HEIGHT = 368


def crop_square(img, im_width):
    img = transforms.ToPILImage()(img)
    img = img.crop((0, 0, im_width, im_width))
    return transforms.PILToTensor()(img)


#############################################

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 im_width: int = IM_WIDTH,
                 im_height: int = IM_HEIGHT,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),

        )
        self.linear = nn.Sequential(
            nn.Flatten(),  # Image grid to single feature vector
            # nn.Linear(2 * 16 * c_hid, latent_dim)
            nn.Linear(2 * c_hid * int(im_width / c_hid * 2) * int(im_height / c_hid * 2), latent_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.linear(x)
        # print(x.shape)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 im_width: int = IM_WIDTH,
                 im_height: int = IM_HEIGHT,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            # nn.Linear(latent_dim, 2 * c_hid),
            nn.Linear(latent_dim, 2 * c_hid * int(im_width / c_hid * 2) * int(im_height / c_hid * 2)),
            act_fn()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(2 * c_hid,
                                                               int(im_width / c_hid * 2),
                                                               int(im_height / c_hid * 2)))

        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        # x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.unflatten(x)
        x = self.net(x)
        return x


#############################################
def get_reconstruction_loss(x, x_hat):
    loss = F.mse_loss(x, x_hat, reduction="none")
    loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss


def reconstruct_images(model, input_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()
    reconstruction_errors = [get_reconstruction_loss(torch.unsqueeze(x, 1), torch.unsqueeze(y, 1)) for (x, y) in
                             zip(input_imgs, reconst_imgs)]

    return reconst_imgs, reconstruction_errors


def embed_imgs(model, input_imgs):
    model.eval()
    with torch.no_grad():
        img_embeds = model.encoder(input_imgs.to(model.device))

    return img_embeds


def embed_imgs_with_dataloader(model, data_loader, transform_function=None):
    img_list, embed_list, names_list = [], [], []
    model.eval()
    for imgs, names in tqdm(data_loader, desc="Encoding images", leave=False):
        with torch.no_grad():
            z = model.encoder(imgs.to(model.device))

        if transform_function is not None:
            imgs = [transform_function(x) for x in imgs]

        img_list.append(torch.stack(imgs))
        embed_list.append(z)
        names_list.append(names)

    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0), torch.cat(names_list, dim=0))


#############################################

def display_reconstruction_results_and_errors(target_files, dataset_dir, model_dict, im_height=None, im_width=None):
    # In each row displays the original image and a reconstruction from each model
    for target_filename in target_files:
        img = read_image_to_tensor(target_filename, dataset_dir, im_height, im_width)
        input_imgs = torch.stack([img])

        reconst_imgs = {}
        reconstruction_errors = {}
        for key in model_dict:
            reconst_imgs[key], reconstruction_errors[key] = reconstruct_images(model_dict[key]["model"], input_imgs)

        reconst_imgs = torch.stack([reconst_imgs[x][0] for x in reconst_imgs])
        imgs = torch.cat([input_imgs, reconst_imgs], dim=0)  # .flatten(0,1)

        labels = ["Dim " + str(key) + "(MSE: " + "%.2f" % reconstruction_errors[key][0].item() + ")" for
                  key in model_dict]
        labels.insert(0, "Original")
        fig_title = "[" + ', '.join(labels) + "]"

        grid = torchvision.utils.make_grid(imgs, normalize=True, range=(-1, 1))
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(15, 5))
        plt.title(fig_title)
        plt.imshow(grid)
        plt.axis('off')
        plt.show()


def visualize_reconstructions(model, input_imgs):
    # Shows all image reconstructions in a row
    reconst_imgs, reconstruction_errors = reconstruct_images(model, input_imgs)

    # Plotting
    imgs = torch.stack([reconst_imgs], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(15, 5))
    plt.title(f"Reconstructed from {model.hparams.latent_dim} latents")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

    return reconst_imgs, reconstruction_errors
