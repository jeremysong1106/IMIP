import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d


def main():
    # load and prepare image
    image_rgb = cv2.imread('../data/rectangles.jpeg')
    show_image(image_rgb, 'original image', destroy_windows=False)

    image_gray = convert2gray(image_rgb)
    show_image(image_gray, 'gray-scale image')

    J = compute_structure_tensor(image_gray, sigma=0.5, rho=0.5, show=True)

    eigenvalues = compute_eigenvalues(J)
    c, e, f = generate_feature_masks(eigenvalues, thresh=0.003)
    show_feature_masks(c, e, f)


# convert to gray scale and normalize for float
# (OpenCV treats color pixels as BGR)
def convert2gray(image_rgb):
    temp = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image_gray = temp.astype(np.float32) / 255.0

    return image_gray


# compute smooth version of image with parameter sigma (zero-padding)
def filter_gauss(image, sigma):
    # TODO: filter image using sigma and zero padding (filter mode 'constant')
    img_filtered = gaussian_filter(image, sigma, mode='constant')  # TODO: replace None by your result
    return img_filtered


# compute the structure tensor for an image
def compute_structure_tensor(image, sigma=0.5, rho=0.5, show=False):
    # perform gaussian filtering before computing the gradient (parameter is sigma)
    image = filter_gauss(image, sigma)

    # compute the gradient image
    img_gradient = compute_gradient(image)
    if show:
        show_gradient(img_gradient)

    # create and compute the structure tensor (dimY x dimX x 2 x 2)
    # local 2x2-tensor J = [[f_x ^ 2 f_x * f_y], [f_x * f_y f_y ^ 2]]
    J = np.empty((np.shape(image)[0], np.shape(image)[1], 2, 2))
    # print(J.shape)
    # f_x squared
    # TODO
    J[:, :, 0, 0] = img_gradient[:, :, 0] * img_gradient[:, :, 0]
    # f_y squared
    # TODO
    J[:, :, 1, 1] = img_gradient[:, :, 1] * img_gradient[:, :, 1]
    # f_x * f_y
    J[:, :, 0, 1] = img_gradient[:, :, 0] * img_gradient[:, :, 1]
    J[:, :, 1, 0] = J[:, :, 0, 1]

    # TODO
    # relaxation step (parameter is rho)
    # TODO: use filter_gauss to filter the tensor components
    # print(J.shape)
    J = filter_gauss(J, rho)
    return J


# compute gradients
def compute_gradient(image):
    # create two channel image for the gradients (dimY x dimX x 2)
    img_gradient = np.empty((np.shape(image)[0], np.shape(image)[1], 2))
    # print(image.shape)

    # the filter kernel and convolution for forward differences in x direction
    x_kernel = np.asarray([[1, -1]], dtype=np.float32)  # TODO: fix the kernel for forward differences
    # print(x_kernel)
    img_gradient[:, :, 0] = convolve2d(image, x_kernel, mode='same')  # TODO: apply x_kernel
    # print(img_gradient[:,:,0])

    # the filter kernel and convolution for forward differences in y direction
    y_kernel = np.asarray([[1], [-1]], dtype=np.float32)  # TODO: fix the kernel for forward differences
    img_gradient[:, :, 1] = convolve2d(image, y_kernel, mode='same')  # TODO: apply y_kernel

    return img_gradient


# create array for the eigenvalues and compute them
def compute_eigenvalues(tensor):
    evs = np.empty((np.shape(tensor)[0], np.shape(tensor)[1], 2))
    print('Computing eigenvalues, this may take a while...')

    # TODO: implement the computation of the eigenvalues
    # TODO (Hint): make use of np.linalg.eig(...)
    evs, _ = np.linalg.eig(tensor)

    return evs


# generate masks for corners, straight edges, and flat areas
def generate_feature_masks(evs, thresh=0.005):
    corners = np.zeros(np.shape(evs[:, :, 0]))
    straight_edges = np.zeros(np.shape(evs[:, :, 0]))
    flat_areas = np.zeros(np.shape(evs[:, :, 0]))

    for i in range(0, np.shape(evs)[0]):
        for j in range(0, np.shape(evs)[1]):

            # TODO: analyze the eigenvalues evs, and assign corners, edges, and flat areas accordingly
            lam1 = evs[i, j, 0]
            lam2 = evs[i, j, 1]
            if lam1 < thresh:
                if lam2 < thresh:
                    flat_areas[i, j] = 1
            if lam1 > thresh:
                if lam2 < thresh:
                    straight_edges[i, j] = 1
            if lam2 > thresh:
                if lam1 < thresh:
                    straight_edges[i, j] = 1
            if lam1 > thresh:
                if lam2 > thresh:
                    corners[i, j] = 1

    return corners, straight_edges, flat_areas


# show feature masks
def show_feature_masks(c, e, f, destroy_windows=True):
    print('[| Features are indicated by white. |]')

    show_image(c, 'corners', False)
    show_image(e, 'straight edges', False)
    show_image(f, 'flat areas', destroy_windows)


# visualize the gradient components and the gradient norm
def show_gradient(img_gradient, destroy_windows=True):
    # compute the norm for display purposes
    img_gradient_norm = np.sqrt(
        img_gradient[:, :, 0] * img_gradient[:, :, 0] + img_gradient[:, :, 1] * img_gradient[:, :, 1])

    # the gradients are in the range of [-1, +1], so we rescale for display purposes
    show_image(img_gradient[:, :, 0] / 2.0 + 0.5, 'x gradients', False)
    show_image(img_gradient[:, :, 1] / 2.0 + 0.5, 'y gradients', False)
    show_image(img_gradient_norm, 'gradient L2-norm', destroy_windows)


# function for displaying an image and waiting for user input
def show_image(i, t, destroy_windows=True):
    cv2.imshow(t, i)

    print('Press a key to continue...')
    cv2.waitKey(0)

    if destroy_windows:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
