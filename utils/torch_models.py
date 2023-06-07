# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm

#############################################
# Autoencoder

IM_WIDTH = 640
IM_HEIGHT = 368


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
    reconstruction_errors = [get_reconstruction_loss(torch.unsqueeze(x,1), torch.unsqueeze(y,1)) for (x, y) in zip(input_imgs, reconst_imgs)]

    return reconst_imgs, reconstruction_errors


def embed_imgs(model, data_loader, transform_function=None):
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
