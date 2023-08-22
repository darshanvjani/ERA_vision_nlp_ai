import gradio as gr
import torch

from Utilities.model import Net
from Utilities import config
from Utilities.utils import generate_confidences, generate_gradcam, generate_missclassified_imgs

inputs = [

    gr.Image(shape=(32, 32), label="Input Image"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Number of Top Prediction to Display"),
    gr.Checkbox(default=False, label="Show GradCAM"),
    gr.Slider(minimum=-2, maximum=-1, step=1, value=-1, label="GradCAM Layer (from the end)"),
    gr.Slider(minimum=0, maximum=1, value=0.5, label="GradCAM Heatmap Opacity"),
    gr.Checkbox(label="Show Incorrect Predictions"),
    gr.Slider(minimum=5, maximum=50, step=5, label="Number of Incorrect Predictions to Display"),

]

model = Net(
    num_classes=config.NUM_CLASSES,
    dropout_percentage = config.DROPOUT_PERCENTAGE,
    norm = config.LAYER_NORM,
    criterion = config.CRITERION,
    learning_rate = config.LEARNING_RATE,
    weight_decay = config.WEIGHT_DECAY
)

model.load_state_dict(
    torch.load(
        config.MODEL_PATH,
        map_location=torch.device(config.ACCELERATOR)
    )
)

model.pred_store = torch.load(config.PRED_STORE_PATH, map_location=torch.device(config.ACCELERATOR))

def generate_gradio_output(
        input_img,
        num_top_preds,
        show_gradcam,
        gradcam_layer,
        gradcam_opacity,
        show_misclassified,
        num_misclassified,
):
    processed_img, confidences = generate_confidences(
        model=model,
        input_img=input_img,
        num_top_preds=num_top_preds 
    )

    visulization = generate_gradcam(
        model=model,
        org_img=input_img,
        input_img=processed_img,
        show_gradcam=show_gradcam,
        gradcam_layer=gradcam_layer,
        gradcam_opacity=gradcam_opacity,
    )

    plot = generate_missclassified_imgs(
        model=model,
        show_misclassified=show_misclassified,
        num_misclassified=num_misclassified,
    )

    return confidences, visulization, plot

outputs = [
    gr.Label(visible=True, scale=0.5, label="Classification Confidences"),
    gr.Image(shape=(32, 32), label="GradCAM Visualization").style(
        width=256, height=256, visible=True
    ),
    gr.Plot(visible=True, label="Misclassified Images")
]

examples = [
    [config.EXAMPLE_IMG_PATH + "cat.jpeg", 3, True, -2, 0.68, True, 40],
    [config.EXAMPLE_IMG_PATH + "horse.jpg", 3, True, -2, 0.59, True, 25],
    [config.EXAMPLE_IMG_PATH + "bird.webp", 10, True, -1, 0.55, True, 20],
    [config.EXAMPLE_IMG_PATH + "dog1.jpg", 10, True, -1, 0.33, True, 45],
    [config.EXAMPLE_IMG_PATH + "frog1.webp", 5, True, -1, 0.64, True, 40],
    [config.EXAMPLE_IMG_PATH + "deer.webp", 1, True, -2, 0.45, True, 20],
    [config.EXAMPLE_IMG_PATH + "airplane.png", 3, True, -2, 0.43, True, 40],
    [config.EXAMPLE_IMG_PATH + "shipp.jpg", 7, True, -1, 0.6, True, 30],
    [config.EXAMPLE_IMG_PATH + "car.jpg", 2, True, -1, 0.68, True, 30],
    [config.EXAMPLE_IMG_PATH + "truck1.jpg", 5, True, -2, 0.51, True, 35],
]

title = "Image Classification (CIFAR10 - 10 Classes) with GradCAM"
description = """A simple Gradio interface to visualize the output of a CNN trained on CIFAR10 dataset with GradCAM and Misclassified images. 
The architecture is inspired from David Page's (myrtle.ai) DAWNBench winning model archiecture.
Please input the image and select the number of top predictions to display - you will see the top predictions and their corresponding confidence scores.
You can also select whether to show GradCAM for the particular image (utilizes the gradients of the classification score with respect to the final convolutional feature map, to identify the parts of an input image that most impact the classification score).
You need to select the model layer where the gradients need to be plugged from - this affects how much of the image is used to compute the GradCAM.
You can also select whether to show misclassified images - these are the images that the model misclassified.
Some examples are provided in the examples tab.
"""

gr.Interface(
    fn=generate_gradio_output,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=examples
).launch()
