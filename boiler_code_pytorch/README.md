<h1 align="center">Boiler Plate Code for Pytorch</h1>

<p align="center">
  <img src="https://github.com/darshanvjani/ERA_vision_nlp_ai/blob/main/boiler_code_pytorch/images/logo.png?raw=true" alt="Logo" width="200" height="200">
</p>

<p align="center">
  <strong>Boilerplate code for building and prototyping custom models with PyTorch.</strong>
</p>

---

## 📚 Overview

Building machine learning models from scratch can be a daunting task. It involves writing a lot of repetitive code for data loading, model architecture, training loops, testing loops, and more. This process can be time-consuming and error-prone, especially for beginners.

This repository aims to solve this problem by providing a boilerplate code for building and prototyping custom models with PyTorch. It provides a structured and modular approach to machine learning model development, allowing for easy customization and reuse.

By using this repository, you can focus more on the unique aspects of your model and less on the boilerplate code. This can significantly speed up your model development process and reduce the chances of errors.

### 📦 Dataloader

The `dataloader` module is responsible for loading and preprocessing the data that will be used to train and test the models. It ensures that the data is in the correct format and is ready to be fed into the models. The main script in this module is `data_loader.py`.

### 🧠 Models

The `models` module contains the implementation of various machine learning models. Currently, it includes the implementation of a ResNet model (`resnet.py`). You can add your custom models in this module.

### 🔧 Utils

The `utils` module contains several utility scripts that assist in various tasks throughout the module. These include:

- **gradcam.py**: Implementation of Grad-CAM, a technique for making Convolutional Neural Network (CNN)-based models more interpretable.
- **helper.py**: Contains helper functions that assist in various tasks.
- **plot_metrics.py**: Used for visualizing various metrics during or after the training process.
- **test.py**: Contains the testing loop for the models.
- **train.py**: Contains the training loop for the models.

## 🚀 Usage

To use this repository as a boilerplate for your custom models, follow these steps:

1. Clone the repository to your local machine.
2. Add your custom models to the `models` module.
3. If your data loading and preprocessing steps are different, modify the `dataloader` module accordingly.
4. Update the `train.py` and `test.py` scripts in the `utils` module to work with your custom models and data.
5. Run the `main.py` script to start the training and testing process.

## 📊 Interdependencies

The following diagram represents the interdependencies of the scripts in the repository:

![ERA_vision_nlp_ai](https://showme.redstarplugin.com/d/JnnMTKKe)


[You can edit this diagram online if you want to make any changes.](https://showme.redstarplugin.com/s/TmTnkCyj)

## 🤝 Contributing

Contributions are welcome! Feel free to submit a pull request.

## 📄 License

This project is licensed under the terms of the MIT license.

---

Stay updated with the latest features and improvements by following our team on [Twitter](https://twitter.com/togethergpt).

## Authors

- [@darshanvjani](https://github.com/darshanvjani)
