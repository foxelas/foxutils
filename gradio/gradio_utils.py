import gradio as gr
from PIL import Image
from tkinter import Tk, filedialog

DEFAULT_INPUT_DIRECTORY = 'input'

def get_target_directory():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    root.update()
    filename = filedialog.askdirectory()
    root.destroy()
    return str(filename)


def view_uploaded_image(file_obj):
    filepath = file_obj.name
    img = Image.open(filepath)
    return img


GET_DIRECTORY_TEXTBOX_DEFAULT_LABEL = "Target Directory"
GET_DIRECTORY_TEXTBOX_DEFAULT_INFO = "Select directory from where the data will be fetched."
GET_DIRECTORY_TEXTBOX_DEFAULT_BUTTON_TEXT = "Select directory"


def get_directory_textbox(label=GET_DIRECTORY_TEXTBOX_DEFAULT_LABEL, info=GET_DIRECTORY_TEXTBOX_DEFAULT_INFO,
                          button_text=GET_DIRECTORY_TEXTBOX_DEFAULT_BUTTON_TEXT, default_dir=DEFAULT_INPUT_DIRECTORY):
    file_dir = gr.Textbox(label=label, value=default_dir, info=info, scale=0)
    with gr.Row():
        directory_button = gr.Button(button_text, size="sm", variant="primary", scale=0)

    directory_button.click(fn=get_target_directory, inputs=[], outputs=file_dir)

    return file_dir, directory_button
