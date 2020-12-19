import numpy as np
import cv2

from skimage.transform import radon, rescale


# TASK: Implementation of the epipolar consistency measure as described
# in "Efficient Epipolar Consistency" by Aichert et al. (2016)


def main():
    shape = [1920, 2480]
    scale_computation = 0.05  # keep that at 0.05 for the tests
    scales_show = (0.4, 4)

    # first, load all data from the disk
    projections_original = load_projections('../data/ProjectionImage', shape)  # TODO implement this method
    projection_matrices = load_geometry('../data/ProjectionMatrix')  # TODO implement this method

    # downscale the projection images
    projections = scale_images(projections_original, scale_computation)

    # compute the derivatives on the sinograms (Radon transform for each projection)
    radon_derivatives = compute_radon_derivatives(projections)

    K0, K1 = compute_mapping_circle_to_epipolar_lines(projection_matrices[0],
                                                      projection_matrices[1], shape)  # TODO implement this method
    print(K0)
    print(K1)
    consistency = compute_consistency(radon_derivatives, [K0, K1])

    # show everything
    show_projections(projections_original, scales_show[0])
    show_radon_derivatives(radon_derivatives, scales_show[1])
    print(consistency - 14.61544)
    if np.abs(consistency - 14.61544) < 1e-2:
        print('Your mapping is correct!')
    else:
        print('The mapping still doesn\'t work!')


# load the raw image data from disk
def load_projections(projection_filename, shape):
    projections = []  # each list entry is one NumPy array

    # use projection_filename and load all (two) enumerated files from the data subdirectory
    # TODO
    projections.append(np.load(projection_filename + '0.npy'))
    # make sure that the projections are returned with correct dimensions
    # TODO
    projections.append(np.load(projection_filename + '1.npy'))
    projections = np.reshape(projections, [2, shape[0], shape[1]])
    return projections


# load the raw data for the projection matrices from disk
def load_geometry(matrices_filename):
    projection_matrices = []  # each list entry is one NumPy array

    # use matrices_filename and load all (two) enumerated files from the data subdirectory
    # TODO
    projection_matrices.append(np.load(matrices_filename + '0.npy'))
    # make sure that the projection matrices are returned in the correct shape
    # TODO
    projection_matrices.append(np.load(matrices_filename + '1.npy'))
    projection_matrices = np.reshape(projection_matrices, [2, 3, 4])
    return projection_matrices


def compute_mapping_circle_to_epipolar_lines(p0, p1, shape):
    C0 = compute_projection_center(p0)  # TODO implement this method
    C1 = compute_projection_center(p1)

    # Plücker coordinates of the baseline
    B = get_pluecker_coordinates(C0, C1)  # TODO implement this method

    # mapping from [cos(kappa) sin(kappa)]' to epipolar plane
    K = compute_mapping_circle_to_plane(B)  # TODO implement this method
    # for each projection
    K0 = compute_mapping_per_projection(p0, K, shape)  # TODO implement this method
    K1 = compute_mapping_per_projection(p1, K, shape)

    return K0, K1


# calculate the center of projection from the projection matrix
def compute_projection_center(pm):
    # consider P = [M, p4] where p4 is simply the fourth column vector
    # and solve for ker(P) by -M^{-1} * p
    # pm = pm.reshape((4, 3)).T
    M = -1 * np.linalg.inv(pm[:, :3])
    print(M)
    p4 = pm[:, 3:]
    print(p4)
    C = np.squeeze(np.dot(M, p4).T)  # TODO
    print(C)
    return np.concatenate([C, [1]])  # return the center in homogeneous coordinates


# compute the Pluecker coordinates
def get_pluecker_coordinates(C0, C1):
    # TODO
    B01 = C0[0] * C1[1] - C1[0] * C0[1]
    B02 = C0[0] * C1[2] - C1[0] * C0[2]
    B03 = C0[0] * C1[3] - C1[0] * C0[3]
    B12 = C0[1] * C1[2] - C1[1] * C0[2]
    B13 = C0[1] * C1[3] - C1[1] * C0[3]
    B23 = C0[2] * C1[3] - C1[2] * C0[3]
    B = (B01, B02, B03, B12, B13, B23)  # 6-tuple, contains all six different entries of the Plücker matrix
    print(B)
    return B


# compute the mapping as described in the article section III.C
def compute_mapping_circle_to_plane(B):
    K = np.zeros([4, 2])

    a2, s2 = compute_pluecker_base_moment(B)
    a3, s3 = compute_pluecker_base_direction(B)
    a1 = np.cross(a3, a2)
    # TODO
    K[0, 0] = (a2 * s3)[0]
    K[0, 1] = a1[0]
    K[1, 0] = (a2 * s3)[1]
    K[1, 1] = a1[1]
    K[2, 0] = (a2 * s3)[2]
    K[2, 1] = a1[2]
    K[3, 0] = 0
    K[3, 1] = -np.power(s2, 2)
    print(K)
    return K


# compute the individual mappings for a specific projection as described in the article section II.D
def compute_mapping_per_projection(p, K, shape):  # TODO
    ny, nx = shape
    H_inv_T = np.asarray([[1, 0, 0], [0, 1, 0], [nx/2, ny/2, 1]])
    p_pseudo_T = np.linalg.pinv(p).T
    # hint: use the NumPy function numpy.linalg.pinv for the pseudoinverse
    Kp = np.dot(np.dot(H_inv_T, p_pseudo_T), K)  # TODO

    # Lines transform contra-variant by their inverse transpose
    # TODO

    return Kp


def compute_radon_derivatives(projections):
    theta = np.linspace(0., 180., np.ceil(max(projections[0].shape) * np.sqrt(2)), endpoint=False)

    radon_derivatives = []
    for projection in projections:
        sinogram = radon(projection, theta=theta, circle=False)

        radon_derivative = np.gradient(sinogram, axis=1)
        radon_derivatives.append(radon_derivative)

    return radon_derivatives


# Plücker line moment
def compute_pluecker_base_moment(B):
    a2 = np.array([B[3], -B[1], B[0]])
    s2 = np.linalg.norm(a2)

    return a2, s2


# Plücker direction of line
def compute_pluecker_base_direction(B):
    a3 = np.array([-B[2], -B[4], -B[5]])
    s3 = np.linalg.norm(a3)

    return a3, s3


def compute_consistency(radon_derivatives, K):
    consistency = 0
    for i in range(-180, 180):

        kappa = i * 0.0035
        ls = []
        x_kappa = np.array([np.cos(kappa), np.sin(kappa)])
        ls.append(np.dot(K[0], x_kappa))
        ls.append(np.dot(K[1], x_kappa))

        values = []
        list_radon_domain = []
        dtrs = []
        for l, dtr in zip(ls, radon_derivatives):

            length = np.linalg.norm(l[0:2])
            alpha = np.arctan2(l[1], l[0])
            t = l[2] / length
            list_radon_domain.append((alpha, t))

            # scale to 0 - 2 range
            dtr_x = alpha / np.pi + 1
            # scale to 0 - 1
            dtr_y = t / 3164 + 0.5

            dtrs.append((dtr_x, dtr_y))

            # compute normalized coordinates in Radon derivative
            # also accounts for symmetry rho(alpha,t)=-rho(alpha+pi,-t)
            if dtr_x > 1:
                weight = -1
                dtr_x = dtr_x - 1.0
                dtr_y = 1.0 - dtr_y
            else:
                weight = 1

            dtr_x = (1 - dtr_x) * dtr.shape[0]
            dtr_y = (1 - dtr_y) * dtr.shape[1]

            # nearest neighbor interpolation
            values.append(weight * dtr[np.int(np.round(dtr_x)), np.int(np.round(dtr_y))])
        consistency += np.square(values[0] - values[1])

    return consistency


# function to display all of the projection images on screen
def show_projections(projections, scale=1.):
    projections_scaled = scale_images(projections, scale)

    for i, projection in enumerate(projections_scaled):
        p_min = np.min(projection)
        projection_range_scaled = ((projection - p_min) / (np.max(projection) - p_min) * 255).astype(np.uint8)

        projection_colored = cv2.applyColorMap(projection_range_scaled, cv2.COLORMAP_BONE)

        show_image(projection_colored, 'projection {}'.format(i), False)


# function to display all of the derived Radon images on screen
def show_radon_derivatives(radon_derivatives, scale=1.):
    radon_derivatives_range_scaled = []
    for i, radon_derivative in enumerate(radon_derivatives):
        rd_min = np.min(radon_derivative)
        radon_derivative_range_scaled = ((radon_derivative - rd_min) / (np.max(radon_derivative) - rd_min) * 255)
        radon_derivatives_range_scaled.append(radon_derivative_range_scaled.astype(np.uint8))

    radon_derivatives_scaled = scale_images(radon_derivatives_range_scaled, scale)

    for i, radon_derivative in enumerate(radon_derivatives_scaled):
        show_image(radon_derivative, 'd rho {}/ dt'.format(i), False)


# function to scale each entry of a list of images
def scale_images(images, scale=1.):
    images_scaled = []
    for image in images:
        images_scaled.append(rescale(image, scale=scale, mode='reflect', multichannel=False, anti_aliasing=False))

    return images_scaled


# function for displaying an image and waiting for user input
def show_image(i, t, destroy_windows=True):
    cv2.imshow(t, i)

    print('Press a key to continue...')
    cv2.waitKey(0)

    if destroy_windows:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
