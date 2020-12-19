import numpy as np
import cv2


# TASK: Implementation of the guided filter as described in the original paper by He et al. (2013)
#
# Step 1:   Implement a box filter function to perform the mean filter with box/window parameter r in 2D.
#           In the original paper a 1D box filter via moving sum is given (cf. Algorithm 2).
#           Here we expand this for the 2D case by successively applying it for each dimension.
# Step 2:   Implement the guided filter as shown in the paper (Algorithm 1).

def main():
    # initialization of parameters (try also other value combinations!)
    r = 2
    epsilon = 0.01

    # load and prepare images
    image_rgb = cv2.imread('../data/coronaries_noisy.jpg')
    guidance_rgb = cv2.imread('../data/coronaries.jpg')
    #
    image = convert2gray(image_rgb)
    guidance = convert2gray(guidance_rgb)

    # example 1: guidance is original image without noise
    filtered_image_1 = guided_filter(image, guidance, r, epsilon)

    # example 2: guidance is original image without noise
    filtered_image_2 = guided_filter(image, image, r, epsilon)

    show_image(image, 'input', False)
    show_image(guidance, 'guidance', False)
    show_image(filtered_image_1, 'example 1', False)
    show_image(filtered_image_2, 'example 2')


# compute the (unnormalized) box-filtered image for a 2D input image and a given "radius" r (equal for both dimensions)
def box_filter(img, r):
    rows, columns = img.shape

    # check for valid input
    if rows < 2 * r or columns < 2 * r:
        raise ValueError('Error computing box filter, the value of r was selected too large.')

    # compute the 2D filtered image by successively (!) applying 1D filtering (once for each dimension):
    # - before you start, consider this approach and make yourself clear how it works
    # - use your implementations of box_filter_rows(...) and box_filter_columns(...) below
    result = box_filter_rows(img, r)
    result = box_filter_columns(result, r)
    # TODO: replace the right side with the correct implementation

    return result


# perform mean filtering for each column
def box_filter_columns(img, r):
    rows, _ = img.shape
    result = np.zeros(img.shape)

    # use the given variables and inputs, no others should be necessary
    # IMPORTANT: ensure the O(n) complexity!
    # (do not use any loop and make use of clever Python indexing)

    # (a) compute the cumulative sum along each column ("partial integral image")
    # (hint: search NumPy for a function that does that for you)
    integral_image_cols = np.cumsum(img, axis=0)  # TODO: replace the right side with the correct implementation

    # (b) implement the column-wise 1D box filter result with proper boundary handling:
    # - neglect values the box would include outside of the image
    # - use numpy.tile(...) to repeat an array/matrix
    #
    # image boundary (top)
    result[0: r + 1, :] = integral_image_cols[r : 2*r+1, :]  # TODO: implement the correct steps here
    # regular areas (box fully overlapping with image)
    result[r + 1: rows - r, :] = integral_image_cols[2*r+1 : rows, :] - integral_image_cols[0 : rows-2*r-1, :]  # TODO: implement the correct steps here
    # image boundary (bottom)
    result[rows - r: rows, :] = np.tile(integral_image_cols[rows-1, :], [r, 1]) - integral_image_cols[rows-2*r-1 : rows-r-1, :] # TODO: implement the correct steps here

    return result


# perform mean filtering for each row
def box_filter_rows(img, r):
    _, columns = img.shape
    result = np.zeros(img.shape)

    # use the given variables and inputs, no others should be necessary
    # IMPORTANT: ensure the O(n) complexity!
    # (do not use any loop and make use of clever Python indexing)

    # (a) compute the cumulative sum along each row ("partial integral image")
    # (hint: search NumPy for a function that does that for you)
    integral_image_rows = np.cumsum(img, axis=1)  # TODO: replace the right side with the correct implementation

    # (b) implement the row-wise 1D box filter result with proper boundary handling:
    # - neglect values the box would include outside of the image
    # - use numpy.tile(...) to repeat an array/matrix
    #
    # image boundary (left)
    result[:, 0: r + 1] = integral_image_rows[:, r : 2*r+1] # TODO: implement the correct steps here
    # regular areas (box fully overlapping with image)
    result[:, r + 1: columns - r] = integral_image_rows[:, 2*r + 1: columns] - integral_image_rows[:, 0:columns-2*r-1] # TODO: implement the correct steps here
    # image boundary (right)
    result[:, columns - r: columns] = np.tile(integral_image_rows[:,columns-1],[r,1]).T - integral_image_rows[:, columns-2*r-1 : columns-r-1] # TODO: implement the correct steps here

    return result


# compute the array of normalization constants for the 2D box filter to create a mean filter
# (depends on r and the pixel position, the number of pixels effectively included in a box is lower at the boundaries)
def get_box_norm(img, r):
    result = box_filter(np.ones(img.shape), r)  # TODO: replace the right side with the correct implementation
    return result


# compute the mean-filtered image
def mean_filter(img, r, n=None):
    if n is None:
        result = box_filter(img, r) / get_box_norm(img, r)
    else:
        result = box_filter(img, r) / n

    return result


# compute a guided filter image from an input image g with a guidance image i
# (g and i are used as in the course, r is the parameter for the mean filter, epsilon is the regularization parameter)
def guided_filter(g, i, r, epsilon):
    # IMPORTANT: before you start here, finish implementing all required functions for the mean_filter(...) routine

    # your implementation requires nothing else than the input arguments, provided variables below, and mean_filter(...)
    # (hint: check the original paper)

    # normalization term
    n = get_box_norm(g, r)

    # (1a) compute mean-filtered images of the guidance and input image
    mean_i = mean_filter(i, r,n)  # TODO: implement the correct steps here
    mean_g = mean_filter(g, r,n)  # TODO: implement the correct steps here
    # (1b) compute the auto-correlation of the guidance image and cross-correlation with the input image g
    corr_i = mean_filter(i * i, r,n)  # TODO: implement the correct steps here
    corr_ig = mean_filter(i * g, r,n)  # TODO: implement the correct steps here

    # (2) calculate variance and covariance
    var_i = corr_i - mean_i * mean_i  # TODO: implement the correct steps here
    cov_ig = corr_ig - mean_i * mean_g  # TODO: implement the correct steps here

    # (3) calculate a and b
    a = cov_ig / (var_i + epsilon)  # TODO: implement the correct steps here
    b = mean_g - a * mean_i  # TODO: implement the correct steps here

    # (4) mean filter a and b
    mean_a = mean_filter(a,r,n)  # TODO: implement the correct steps here
    mean_b = mean_filter(b,r,n)  # TODO: implement the correct steps here

    # (5) compute the output image
    result = mean_a * i + mean_b  # TODO: replace the right side with the correct implementation
    print(result)

    return result


# convert to gray scale and normalize for float
# (OpenCV treats color pixels as BGR)
def convert2gray(image_rgb):
    temp = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image_gray = temp.astype(np.float32) / 255.0

    return image_gray


# function for displaying an image and waiting for user input
def show_image(i, t, destroy_windows=True):
    cv2.imshow(t, i)

    print('Press a key to continue...')
    cv2.waitKey(0)

    if destroy_windows:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
