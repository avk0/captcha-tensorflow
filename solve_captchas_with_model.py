import os
from keras.models import load_model
#from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
from PIL import Image
import pickle


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"


def cut_to_5(in_path):
    with Image.open(in_path) as img:
        wid = img.size[0]
        hei = img.size[1]
        # print(img.size)
        dw = int(wid / 5)
        char_imgs = []
        char_vals = []
        for piece in range(5):
            temp_img = img.copy().crop((1 + piece * dw, 0, (piece + 1) * dw - 1, hei))
            name = os.path.basename(in_path)[piece]
            char_imgs.append(temp_img)
            char_vals.append(name)
        return char_imgs, char_vals


def img_to_black_and_white(in_path, out_path, w=190):
    with Image.open(in_path) as img:
        for x in range(img.size[1]):
            for y in range(img.size[0]):
                pix = np.array(img.getpixel((y, x)))
                if sum(pix) > w * 3:
                    img.putpixel((y, x), (255, 255, 255))
                else:
                    img.putpixel((y, x), (0, 0, 0))
        img.save(out_path)


def recognize_captcha(image_file):
    # Load up the model labels (so we can translate model predictions to actual letters)
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    # Load the trained neural network
    model = load_model(MODEL_FILENAME)

    # Cut captcha at 5 pieces.
    # If it is marked, load marks in char_vals array
    char_imgs, char_vals = cut_to_5(image_file)

    letter_boxes = []

    # Now we can loop through each of the five letter boxes and extract the letter
    # inside of each one
    for char in char_imgs:
        image = np.array(char)

        image = cv2.threshold(image, 195, 255, cv2.THRESH_BINARY)[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        letter_boxes.append(image)

    predictions = []

    # loop over the letters
    for letter_box in letter_boxes:
        print(np.array(letter_box).shape)

        # Turn the single image into a 4d list of images to make Keras happy
        letter_image = np.expand_dims(letter_box, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(letter_image)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))
    return captcha_text


if __name__ == '__main__':
    # Grab some random CAPTCHA images to test against.
    captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
    #captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

    # loop over the image paths
    for image_file in captcha_image_files:
        # Load the image and convert it to grayscale
        output = cv2.imread(image_file)

        recognize_captcha(image_file)

        # Show the annotated image
        cv2.imshow("Output", output)
        cv2.waitKey()
