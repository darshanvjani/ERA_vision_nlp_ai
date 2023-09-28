import gradio as gr
import torch

from Utilities.model import YOLOv3
from Utilities import config
from Utilities.transforms import resize_transforms
from Utilities.runtime_utils import generate_gradcam_output, plot_bboxes

model = YOLOv3.load_from_checkpoint(
    config.MODEL_CHECKPOINT_PATH,
    map_location=torch.device('cpu')
)

examples = [
    [config.EXAMPLE_IMAGE_PATH + "cat.jpeg", 1],
    [config.EXAMPLE_IMAGE_PATH  + "horse.jpg", 1],
    [config.EXAMPLE_IMAGE_PATH  + "000018.jpg", 2],
    [config.EXAMPLE_IMAGE_PATH  + "bird.webp", 2],
    [config.EXAMPLE_IMAGE_PATH  + "000022.jpg", 2],
    [config.EXAMPLE_IMAGE_PATH  + "airplane.png", 0],
    [config.EXAMPLE_IMAGE_PATH  + "shipp.jpg", 0],
    [config.EXAMPLE_IMAGE_PATH  + "car.jpg", 1],
    [config.EXAMPLE_IMAGE_PATH  + "000007.jpg", 1],
    [config.EXAMPLE_IMAGE_PATH  + "000013.jpg", 2],
    [config.EXAMPLE_IMAGE_PATH  + "000012.jpg", 2],
    [config.EXAMPLE_IMAGE_PATH  + "000006.jpg", 1],
    [config.EXAMPLE_IMAGE_PATH  + "000004.jpg", 1],
    [config.EXAMPLE_IMAGE_PATH  + "000014.jpg", 0],
]

title = "Building YOLOv3 from Scratch using PyTorch Lightning"
description = """Unveiling the intricacies of YOLOv3 through PyTorch Lightning ⚡️🕵️‍♂️
---
In the rapidly evolving landscape of machine learning, expertise in building sophisticated models from scratch is invaluable. Presenting the YOLOv3 Object Detection System crafted meticulously using the cutting-edge PyTorch Lightning framework.

🎉 Key Highlights:
---
1. **Deep Dive into YOLOv3**: Ground-up development of the YOLOv3 model, showcasing proficiency in intricate model architectures and in-depth understanding of computer vision principles.

2. **PyTorch Lightning Advantage**: Leverage the robustness and efficiency of PyTorch Lightning, reflecting modern best practices and optimizing training workflows. This demonstrates strong proficiency in state-of-the-art deep learning frameworks.

3. **High Precision with GradCAM**: Integrated GradCAM (Gradient-weighted Class Activation Mapping), offering insights into model's decision-making layers, indicative of a holistic approach to model transparency and interpretability.

4. **Flexibility in Object Detection**: Multi-scale outputs (13x13, 26x26, 52x52) for versatile object detection, displaying an understanding of varying image resolutions and their impact on detection tasks.

📸 Workflow:
---
- Upload an image for object detection.
- Choose an appropriate output stream size.
- Experience real-time object identification, enriched with GradCAM visualizations, highlighting the model's decision-making areas.

✅ Recognizable Pascal VOC Classes:
---
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

🌟 Dive Deeper:
---
Explore the "Examples" section for comprehensive visual insights. Understand the YOLOv3's capabilities and analyze GradCAM results for varied output streams. This emphasizes a keen interest in not just creating, but also in understanding and optimizing machine learning models.

Venture into a hands-on demonstration of skills, innovation, and expertise in computer vision and deep learning. Dive into this YOLOv3 Object Detection System, exemplifying the forefront of machine learning prowess.
"""

def generate_gradio_output(input_img, gradcam_output_stream=0):
    input_img = resize_transforms(image=input_img)["image"]

    fig, processed_img = plot_bboxes(
        input_img=input_img,
        model=model,
        thresh=0.6,
        iou_thresh=0.5,
        anchors=model.scaled_anchors,
    )

    visualization = generate_gradcam_output(
        org_img=input_img,
        model=model,
        input_img=processed_img,
        gradcam_output_stream=gradcam_output_stream,
    )
    return fig, visualization

gr.Interface(
    fn=generate_gradio_output,
    inputs=[
        gr.Image(label="Input Image"),
        gr.Slider(0, 2, step=1, label="GradCAM Output Stream (13, 26, 52)")
    ],
    outputs=[
        gr.Plot(
            visible=True,
            label="Bounding Box Predictions",
        ),
        gr.Image(label="GradCAM Visualization").style(width=416, height=416)
    ],
    examples=examples,
    title=title,
    description=description,
).launch()

