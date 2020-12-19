import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


def main():

    # initialization of constants
    beta = 0.5
    c = 0.08

    # load and prepare image
    image_rgb = cv2.imread('../data/coronaries.jpg')
    image = convert2gray(image_rgb)
    print(image.shape)

    scales = [1.0, 1.5, 2.0, 3.0]
    images_vesselness = []
    for s in scales:

        images_vesselness.append(calculate_vesselness_2d(image, s, beta, c))

    result = compute_scale_maximum(images_vesselness)
    show_four_scales(image, result, images_vesselness, scales)


# calculate the vesselness filter image (Frangi 1998)
def calculate_vesselness_2d(image, scale, beta, c):

    # create empty result image
    vesselness = np.zeros(image.shape)

    # compute the Hessian for each pixel
    H = compute_hessian(image, scale)

    # get the eigenvalues for the Hessians
    eigenvalues = compute_eigenvalues(H)

    print('Computing vesselness...')

    # compute the vesselness measure for each pixel
    # TODO: loop over the pixels to compute the vesselness image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lam1 = eigenvalues[i, j, 0]
            lam2 = eigenvalues[i, j, 1]
            vesselness[i, j] = vesselness_measure(lam1, lam2, beta, c)
    # Hint: use the function vesselness_measure (implement it first below)

    print('...done.')
    return vesselness


def compute_hessian(image, sigma):

    # gauss filter the input with given sigma
    # TODO: filter image using sigma and zero padding (filter mode 'constant')
    image_gauss = gaussian_filter(image, sigma, mode='constant') # replace None by your result

    print('Computing Hessian...')

    # gradient calculation
    # TODO: compute first order gradient
    dx = np.gradient(image_gauss, axis=0)
    dy = np.gradient(image_gauss, axis=1)

    # Create components of the Hessian Matrix [dx2 dxy][dyx dy2]
    # TODO: compute all partial second derivatives
    dx2 = np.gradient(dx, axis=0)
    dxy = np.gradient(dx, axis=1)
    dyx = np.gradient(dy, axis=0)
    dy2 = np.gradient(dy, axis=1)
    # scale normalization -> multiply the hessian components with sigma^2
    # TODO: normalize as stated

    # save values in a single array
    H = np.empty((np.shape(image_gauss)[0], np.shape(image_gauss)[1], 2, 2))

    # TODO: fill the Hessian with the proper values from above
    H[:, :, 0, 0] = dx2 * sigma * sigma
    H[:, :, 0, 1] = dxy * sigma * sigma
    H[:, :, 1, 0] = dyx * sigma * sigma
    H[:, :, 1, 1] = dy2 * sigma * sigma

    # print(H)
    print('...done.')
    return H


# create array for the eigenvalues and compute them
def compute_eigenvalues(hessian):

    evs = np.empty((np.shape(hessian)[0], np.shape(hessian)[1], 2))
    print('Computing eigenvalues, this may take a while...')

    # TODO: implement the computation of the eigenvalues
    # TODO (Hint): make use of np.linalg.eig(...)
    evs, _ = np.linalg.eig(hessian)
    print(evs.shape)

    print('...done.')
    return evs


# calculate the 2-D vesselness measure (see Frangi paper or course slides)
def vesselness_measure(lambda1, lambda2, beta, c):

    # ensure lambda1 >= lambda2
    lambda1, lambda2 = sort_descending(lambda1, lambda2)

    # the vesselness measure is zero if lambda1 is positive (inverted/dark vessel)
    # if both eigenvalues are zero, set RB and S to zero, otherwise compute them as shown in the course
    # TODO: implement the vesselness measure and take care of lambda1 being zero
    if lambda1 == lambda2 == 0:
        RB = 0
        S = 0
    else:
        RB = lambda2/lambda1
        S = np.sqrt(lambda1*lambda1 + lambda2*lambda2)
    if lambda1 > 0:
        v = 0
    else:
        v = np.exp(-RB * RB / (2 * beta * beta)) * (1 - np.exp(-S * S / (2 * c * c)))


    # dummy result
    return v


# takes a list of vesselness images and returns the pixel-wise maximum as a result
def compute_scale_maximum(image_list):

    result = image_list[0]
    print('Computing maximum...')

    # TODO: compute the image that takes the PIXELWISE maximum from all images in image_list
    # image_list = np.array(image_list)
    # print(image_list.shape)
    result = np.max(image_list,axis=0)

    print('...done.')
    return result


# convert to gray scale and normalize for float
# (OpenCV treats color pixels as BGR)
def convert2gray(image_rgb):

    temp = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image_gray = temp.astype(np.float32) / 255.0

    return image_gray


# rearrange pair of values in descending order
def sort_descending(value1, value2):

    if np.abs(value1) < np.abs(value2):
        buf = value2
        value2 = value1
        value1 = buf

    return value1, value2


# special function to show the images from this exercise
def show_four_scales(original, result, image_list, scales):

    plt.figure('vesselness')

    prepare_subplot_image(original, 'original', 1)
    prepare_subplot_image(image_list[0], 'sigma = '+str(scales[0]), 2)
    prepare_subplot_image(image_list[1], 'sigma = '+str(scales[1]), 3)
    prepare_subplot_image(result, 'result', 4)
    prepare_subplot_image(image_list[2], 'sigma = '+str(scales[2]), 5)
    prepare_subplot_image(image_list[3], 'sigma = '+str(scales[3]), 6)

    plt.show()


# helper function
def prepare_subplot_image(image, title='', idx=1):

    if idx > 6:
        return

    plt.gcf()
    plt.subplot(2, 3, idx)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap='gray', vmin=0, vmax=np.max(image))


# function for displaying an image and waiting for user input
def show_image(i, t, destroy_windows=True):

    cv2.imshow(t, i)

    print('Press a key to continue...')
    cv2.waitKey(0)

    if destroy_windows:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
