# fourier_synthesis.py
import random
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import cv2
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))

def normalize_data(data, width, height, negative=True):
    """
    Normalizes images in interval [0 1] or [-1 1] and returns batches.
    Args:
        data (numpy.array): Array of images to process.
        width (int): Width dimension of image in pixels.
        height (int): Height dimension of image in pixels.
        negative (bool, optional): Flag that determines interval (True: [-1 1], False: [0 1]). Defaults to True.
    Returns:
        data (tf.data.Dataset): Kvasir-SEG dataset sliced w/ batch_size and normalized.
    """
    normalized_data = []
    for image in data:
        resized_image = cv2.resize(image, (width, height))
        # if negative:
        #     image = (resized_image / 127.5) - 1
        # else:
        #     image = (resized_image / 255.0)
        normalized_data.append(image)
    return normalized_data


def load_images(cls):
    path = "../BigCats"
    folder = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
    image_path = []
    # for i in range(5):
    print( cls)
    dir = os.path.join(path, cls)
    for file in os.listdir(dir):
        image_path.append(os.path.join(dir, file))
    return image_path

def generate_modulus_dataset(X, k):
    return np.abs(np.fft.fft(X)[:, :k])

def generate_complex_dataset(X, k):
    fourier = np.fft.fft(X)[:, :k]
    return np.column_stack((fourier.real, fourier.imag))

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )

def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1]))

def display_plots(individual_grating, reconstruction, idx, image_filename,  cls):
    plt.subplot(121)
    plt.imshow(individual_grating)
    plt.axis("off")
    plt.subplot(122)
    print(reconstruction.shape)
    plt.imshow(reconstruction)
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(0.01)
    cv2.imwrite(os.path.join(dir_path, 'imgs',  cls, "resized"+image_filename), reconstruction)
    # img.show()
    # plt.figure(reconstruction)
    # plt.savefig(os.path.join(dir_path, cls, "resized"+image_filename))

folder = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
features = {}
for cls in folder:
    imgs = load_images(cls)
    cnt_img = 0

    gratings = []
    cnt = 0
    for imagine in imgs:
        cnt +=1
        image_filename = imagine
        img_name = os.path.split(image_filename)[-1]
        print(img_name)

        # Read and process image
        # image = plt.imread(image_filename)
        image = Image.open(image_filename)

        image = image.resize((333, 333))

        # image.save(image_filename+'resized')
        # image = plt.imread(image_filename+'resized')

        image = np.asarray(image)
        # print("image shape" , image.shape)
        image = image[:, :, :3].mean(axis=2)  # Convert to grayscale

        # Array dimensions (array is square) and centre pixel
        # Use smallest of the dimensions and ensure it's odd
        array_size = min(image.shape) - 1 + min(image.shape) % 2

        # Crop image so it's a square image
        image = image[:array_size, :array_size]
        centre = int((array_size - 1) / 2)

        # Get all coordinate pairs in the left half of the array,
        # including the column at the centre of the array (which
        # includes the centre pixel)
        coords_left_half = (
            (x, y) for x in range(array_size) for y in range(centre+1)
        )
        # Sort points based on distance from centre
        coords_left_half = sorted(
            coords_left_half,
            key=lambda x: calculate_distance_from_centre(x, centre)
        )

        plt.set_cmap("gray")
        ft = calculate_2dft(image)
        # Show grayscale image and its Fourier transform
        # modulus_100 = generate_modulus_dataset(image, )
        plt.subplot(121)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(np.log(abs(ft)))
        plt.axis("off")
        # plt.pause(2)
        # plt.savefig(os.path.join(dir_path, "./imgs/", str(cls)) + "Frequency-Map-"+str(cnt_img)+".png")
        # Reconstruct image
        fig = plt.figure()
        # Step 1
        # Set up empty arrays for final image and
        # individual gratings
        rec_image = np.zeros(image.shape)
        individual_grating = np.zeros(
            image.shape, dtype="complex"
        )
        idx = 0
        # All steps are displayed until display_all_until value
        display_all_until = 1001
        # After this, skip which steps to display using the
        # display_step value
        display_step = 10


        # Work out index of next step to display
        next_display = display_all_until + display_step
        # Step 2
        threshold = 0
        for coords in coords_left_half:
            # Central column: only include if points in top half of
            # the central column
            threshold += 1

            if not (coords[1] == centre and coords[0] > centre):
                idx += 1
                symm_coords = find_symmetric_coordinates(
                    coords, centre
                )
                # Step 3
                # Copy values from Fourier transform into
                # individual_grating for the pair of points in
                # current iteration
                individual_grating[coords] = ft[coords]
                individual_grating[symm_coords] = ft[symm_coords]
                # Step 4
                # Calculate inverse Fourier transform to give the
                # reconstructed grating. Add this reconstructed
                # grating to the reconstructed image
                rec_grating = calculate_2dift(individual_grating)
                rec_image += rec_grating

                # save individual gratings in the features array
                # print(features)
                gratings.append(individual_grating)
                features[cls] =  gratings
                # print(features[cls])
                # print(gratings)
                # TODO: now the grating is just overwritten for each image, it should be appended
                # print("Shape is ", len(gratings))
                # print("gratings ", individual_grating.shape)

                # Clear individual_grating array, ready for
                # next iteration
                individual_grating[coords] = 0
                individual_grating[symm_coords] = 0
                # Don't display every step
                if idx < display_all_until or idx == next_display:
                    if idx > display_all_until:
                        next_display += display_step
                        # Accelerate animation the further the
                        # iteration runs by increasing
                        # display_step
                        display_step += 10
                    # display_plots(rec_grating, rec_image, idx)
                if threshold == 1000:
                    print("threshold reached ")
                    display_plots(rec_grating, rec_image, idx, img_name, cls)
                    break

        plt.show()
        cnt_img += 1

# to return a group of the key-value
# pairs in the dictionary
result = features.items()

# Convert object to a list
data = list(result)

# Convert list to an array
numpyArray = np.array(data, dtype=object)
# print(numpyArray)
# print("finished numpt array")
np.save('features.npy', numpyArray)
#
# print(features.values())
# print(features.keys())
# df = pd.DataFrame(features.values(), columns=features.keys())
# df.to_pickle('features.pkl')
