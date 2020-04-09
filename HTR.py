import argparse
import difflib
import importlib
import math
import cv2 as cv2
import numpy as np
import mxnet as mx
import random
import matplotlib.pyplot as plt
import gluonnlp as nlp
import leven
import matplotlib.patches as patches
from skimage import transform as skimage_tf, exposure
from tqdm import tqdm
import os
import nltk

from ocr.utils.expand_bounding_box import expand_bounding_box
from ocr.utils.sclite_helper import ScliteHelper
from ocr.utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from ocr.utils.iam_dataset import IAMDataset, resize_image, crop_image, crop_handwriting_page
from ocr.utils.encoder_decoder import Denoiser, ALPHABET, encode_char, decode_char, EOS, BOS
from ocr.utils.beam_search import ctcBeamSearch

import ocr.utils.denoiser_utils
import ocr.utils.beam_search

importlib.reload(ocr.utils.denoiser_utils)
from ocr.utils.denoiser_utils import SequenceGenerator

importlib.reload(ocr.utils.beam_search)
from ocr.utils.beam_search import ctcBeamSearch


from ocr.paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from ocr.word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from ocr.handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from ocr.handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding
ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

# helper functions

MAX_IMAGE_SIZE_FORM = (1120, 800)
MAX_IMAGE_SIZE_LINE = (60, 800)
MAX_IMAGE_SIZE_WORD = (30, 140)


def resize_image(image, desired_size):
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size

    bounding box: (int, int, int, int)
        (x, y, w, h) in percentages of the resized image of the original
    '''
    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = image[0][0]
    if color < 230:
        color = 230
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))
    crop_bb = (left / image.shape[1], top / image.shape[0], (image.shape[1] - right - left) / image.shape[1],
               (image.shape[0] - bottom - top) / image.shape[0])
    image[image > 230] = 255
    return image, crop_bb


# this function takes in the img file path
def _pre_process_image(img_in, _parse_method):
    im = cv2.imread(img_in, cv2.IMREAD_GRAYSCALE)
    if np.size(im) == 1:  # skip if the image data is corrupt.
        return None
    # reduce the size of form images so that it can fit in memory.
    if _parse_method in ["form", "form_bb"]:
        im, _ = resize_image(im, MAX_IMAGE_SIZE_FORM)
    if _parse_method == "line":
        im, _ = resize_image(im, MAX_IMAGE_SIZE_LINE)
    if _parse_method == "word":
        im, _ = resize_image(im, MAX_IMAGE_SIZE_WORD)
    img_arr = np.asarray(im)
    return img_arr


def get_arg_max(prob):
    '''
    The greedy algorithm convert the output of the handwriting recognition network
    into strings.
    '''
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


def get_beam_search(prob, width=5):
    possibilities = ctcBeamSearch(prob.softmax()[0].asnumpy(), alphabet_encoding, None, width)
    return possibilities[0]


def get_denoised(prob, ctc_bs=False):
    if ctc_bs:  # Using ctc beam search before denoising yields only limited improvements a is very slow
        text = get_beam_search(prob)
    else:
        text = get_arg_max(prob)
    src_seq, src_valid_length = encode_char(text)
    src_seq = mx.nd.array([src_seq], ctx=ctx)
    src_valid_length = mx.nd.array(src_valid_length, ctx=ctx)
    encoder_outputs, _ = denoiser.encode(src_seq, valid_length=src_valid_length)
    states = denoiser.decoder.init_state_from_encoder(encoder_outputs,
                                                      encoder_valid_length=src_valid_length)
    inputs = mx.nd.full(shape=(1,), ctx=src_seq.context, dtype=np.float32, val=BOS)
    output = generator.generate_sequences(inputs, states, text)
    return output.strip()



# load
paragraph_segmentation_net = SegmentationNetwork(ctx=ctx)
paragraph_segmentation_net.cnn.load_parameters("models/paragraph_segmentation2.params", ctx=ctx)
paragraph_segmentation_net.hybridize()

word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
word_segmentation_net.load_parameters("models/word_segmentation2.params")
word_segmentation_net.hybridize()


handwriting_line_recognition_net = HandwritingRecognitionNet(rnn_hidden_states=512,
                                                             rnn_layers=2, ctx=ctx, max_seq_len=160)
handwriting_line_recognition_net.load_parameters("models/handwriting_line8.params", ctx=ctx)
handwriting_line_recognition_net.hybridize()

FEATURE_LEN = 150
denoiser = Denoiser(alphabet_size=len(ALPHABET), max_src_length=FEATURE_LEN, max_tgt_length=FEATURE_LEN, num_heads=16, embed_size=256, num_layers=2)
denoiser.load_parameters('models/denoiser2.params', ctx=ctx)

denoiser.hybridize(static_alloc=True)

ctx_nlp = mx.cpu(0)
language_model, vocab = nlp.model.big_rnn_lm_2048_512(dataset_name='gbw', pretrained=True, ctx=ctx_nlp)
moses_tokenizer = nlp.data.SacreMosesTokenizer()
moses_detokenizer = nlp.data.SacreMosesDetokenizer()

beam_sampler = nlp.model.BeamSearchSampler(beam_size=20,
                                           decoder=denoiser.decode_logprob,
                                           eos_id=EOS,
                                           scorer=nlp.model.BeamSearchScorer(),
                                           max_length=150)


generator = SequenceGenerator(beam_sampler, language_model, vocab, ctx_nlp, moses_tokenizer, moses_detokenizer)


def generate_op(img_n, img_dir, folder_path):
    image_name = img_n.split('.')[0]
    img_path = os.path.join(img_dir, img_n)
    image = _pre_process_image(img_path, 'form')

    form_size = (1120, 800)

    predicted_bbs = []

    resized_image = paragraph_segmentation_transform(image, form_size)
    bb_predicted = paragraph_segmentation_net(resized_image.as_in_context(ctx))
    bb_predicted = bb_predicted[0].asnumpy()
    bb_predicted = expand_bounding_box(bb_predicted, expand_bb_scale_x=0.03,
                                       expand_bb_scale_y=0.03)
    predicted_bbs.append(bb_predicted)

    (x, y, w, h) = bb_predicted
    image_h, image_w = image.shape[-2:]
    (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)

    segmented_paragraph_size = (700, 700)
    paragraph_segmented_images = []

    bb = predicted_bbs[0]
    image = crop_handwriting_page(image, bb, image_size=segmented_paragraph_size)
    paragraph_segmented_images.append(image)

    min_c = 0.1
    overlap_thres = 0.1
    topk = 600
    predicted_words_bbs_array = []

    for i, paragraph_segmented_image in enumerate(paragraph_segmented_images):
        predicted_bb = predict_bounding_boxes(
            word_segmentation_net, paragraph_segmented_image, min_c, overlap_thres, topk, ctx)

        predicted_words_bbs_array.append(predicted_bb)
        for j in range(predicted_bb.shape[0]):
            (x, y, w, h) = predicted_bb[j]
            image_h, image_w = paragraph_segmented_image.shape[-2:]
            (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)

    line_images_array = []

    for i, paragraph_segmented_image in enumerate(paragraph_segmented_images):
        predicted_bbs = predicted_words_bbs_array[i]
        line_bbs = sort_bbs_line_by_line(predicted_bbs, y_overlap=0.4)
        line_images = crop_line_images(paragraph_segmented_image, line_bbs)
        line_images_array.append(line_images)

        for line_bb in line_bbs:
            (x, y, w, h) = line_bb
            image_h, image_w = paragraph_segmented_image.shape[-2:]
            (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)

    line_image_size = (60, 800)
    character_probs = []
    for line_images in line_images_array:
        form_character_prob = []
        for i, line_image in enumerate(line_images):
            line_image = handwriting_recognition_transform(line_image, line_image_size)
            line_character_prob = handwriting_line_recognition_net(line_image.as_in_context(ctx))
            form_character_prob.append(line_character_prob)
        character_probs.append(form_character_prob)

    FEATURE_LEN = 150
    save_path = os.path.join(folder_path, image_name + '.txt')
    file = open(save_path, 'w')

    for i, form_character_probs in enumerate(character_probs):
        for j, line_character_probs in enumerate(form_character_probs):
            decoded_line_bs = get_beam_search(line_character_probs)
            print(decoded_line_bs)
            file.write(decoded_line_bs + ' ')
    file.close()


# main code
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', help='Path to images')
    args = parser.parse_args()

    # img_dir = "dataset/small"
    img_names = os.listdir(args.img_dir)
    folder_path = os.path.abspath('output')
    if(not os.path.exists(folder_path)):
        print("creating dir")
        os.mkdir(folder_path)

    for img_n in img_names:
        print("\n New image starting\n")
        generate_op(img_n, args.img_dir, folder_path)


if __name__ == '__main__':
    main()
