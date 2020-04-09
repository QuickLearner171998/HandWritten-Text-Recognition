# Handwritten text recognition on IAM dataset

## Major part of the source code belongs to [awslabs/handwritten-text-recognition-for-apache-mxnet repo](https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet).
Modification - This code takes lesser RAM by taking images in a batch of 1.


## Usage

1) install all the requirements.
    ```pip install -r requirements.txt```
2) run the python script and provide the path of directory containing the Images.
    ```python mxnet.py "Path/to/ImageDirectory"```

The text will be saved in output folder as a .txt file with the same name in as that of the Image.

## Note
1) The model is trained on IAM dataset so it may not give satisfactory results on other datasets.
2) 8Gb RAM is recommended for inferencing.
3) A Colab Notebook is also provided. Just change the paths in the notebook and run it.
