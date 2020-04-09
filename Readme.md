# Handwritten text recognition on IAM dataset

## Usage

1) install all the requirements.
    ```pip install -r requirements.txt```
2) run the python script and provide the path of directory containing the Images.
    ```python mxnet.py "Path/to/ImageDirectory"```

The text will be saved in a .txt file with the same name as that of the Image.

## Note
1) The model is trained on IAM dataset so it may not give satisfactory results on other datasets.
2) Inferencing requires at least 8Gb RAM.
