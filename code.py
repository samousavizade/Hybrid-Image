import cv2 as cv
import matplotlib.pyplot as pp
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt


def is_even(number):
    return number % 2 == 0


def gaussian_kernel(std):
    return cv.getGaussianKernel(6 * std + 1, std) if is_even(6 * std) else cv.getGaussianKernel(int(6 * std), std)


def low_pass_filter(image, std, cutoff):
    # 2d gaussian kernel
    kernel = gaussian_kernel(std)
    kernel_2d = kernel * kernel.T

    # pp.imsave('08_lowpass_{}.jpg'.format(std), kernel_2d, cmap='gray')

    # gaussian kernel shifted fft
    kernel_shifted_fft = (fft.fftshift(np.abs(fft.fft2(kernel_2d, image.shape))))

    # pp.imsave('08_lowpass_{}_frequency_domain.jpg'.format(std), kernel_shifted_fft, cmap='gray')

    # apply cutoff
    h = kernel_shifted_fft.shape[0]
    w = kernel_shifted_fft.shape[1]
    center_h = h // 2 if is_even(h) else (h + 1) // 2
    center_w = w // 2 if is_even(w) else (w + 1) // 2
    for row in range(h):
        for col in range(w):
            if (row - center_h) ** 2 + (col - center_w) ** 2 > cutoff ** 2:
                kernel_shifted_fft[row, col] = 0

    # pp.imsave('10_lowpass_cutoff.jpg', kernel_shifted_fft, cmap='gray')

    # image shifted fft
    image_shifted_fft = fft.fftshift((fft.fft2((image))))

    # filtered image shifted fft
    filtered_image_frequency_domain = image_shifted_fft * kernel_shifted_fft

    return filtered_image_frequency_domain


def high_pass_filter(image, std, cutoff):
    # 2d gaussian kernel
    kernel = gaussian_kernel(std)
    kernel_2d = kernel * kernel.T

    # pp.imsave('q4_07_highpass_{}.jpg'.format(std), 1 - kernel_2d, cmap='gray')

    # gaussian kernel shifted fft
    kernel_shifted_fft = 1 - (fft.fftshift(np.abs(fft.fft2(kernel_2d, image.shape))))

    # pp.imsave('07_highpass_{}_frequency_domain.jpg'.format(std), kernel_shifted_fft, cmap='gray')

    # apply cutoff
    h = kernel_shifted_fft.shape[0]
    w = kernel_shifted_fft.shape[1]
    center_h = h // 2 if is_even(h) else (h + 1) // 2
    center_w = w // 2 if is_even(w) else (w + 1) // 2
    for row in range(h):
        for col in range(w):
            if (row - center_h) ** 2 + (col - center_w) ** 2 < cutoff ** 2:
                kernel_shifted_fft[row, col] = 0

    # pp.imsave('09_highpass_cutoff.jpg', kernel_shifted_fft, cmap='gray')

    # image shifted fft
    image_shifted_fft = (fft.fftshift((fft.fft2((image)))))

    # filtered image shifted fft
    filtered_image_frequency_domain = image_shifted_fft * kernel_shifted_fft

    return filtered_image_frequency_domain


def hybrid(low, high, factor=.8, cutoff_low=25, cutoff_high=20):
    low_pass_input_image, low_std = low
    high_pass_input_image, high_std = high

    # get low fft
    low_fft = low_pass_filter(low_pass_input_image, low_std, cutoff_low)

    # save low image frequency domain
    # low_fft_image = 8 * np.log(np.abs(low_fft))
    # cv.imwrite('q4_12_lowpassed.jpg', low_fft_image, )

    # calculate low image in spatial domain
    low = np.abs(fft.ifft2(fft.ifftshift(low_fft)))

    ##############

    # get high fft
    high_fft = high_pass_filter(high_pass_input_image, high_std, cutoff_high)

    # save high image frequency domain
    # high_fft_image = 8 * np.log(np.abs(high_fft))
    # cv.imwrite('q4_11_highpassed.jpg', high_fft_image)

    # calculate low image in spatial domain
    high = np.abs(fft.ifft2(fft.ifftshift(high_fft)))

    # calculate hybrid fft
    hybrid_fft = factor * low_fft + (1 - factor) * high_fft

    # save hybrid image frequency domain
    # hybrid_fft_image = 8 * np.log(np.abs(hybrid_fft))
    # cv.imwrite('q4_13_hybrid_frequency.jpg', hybrid_fft_image)

    hybrid_result = np.abs(fft.ifft2(hybrid_fft))

    return hybrid_result, low, high


def get_image_log_frequency_domain(image):
    # change color space to gray
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # change to frequency domain
    image_gray_fft = np.abs(fft.fftshift((fft.fft2(image_gray,
                                                   image_gray.shape))))
    # return 8 * log()
    log_image_gray_fft = 8 * np.log2(image_gray_fft)
    return log_image_gray_fft


def main():
    # initial values
    image_high_file_name = 'input/motorcycle.bmp'
    image_low_file_name = 'input/bicycle.bmp'

    zoom_in_file_name = 'output/zoom_in.jpg'
    zoom_out_file_name = 'output/zoom_out.jpg'

    low_std = 5
    high_std = 5
    cutoff_low = 1000000000
    cutoff_high = 0

    # read images
    image_high = cv.imread(image_high_file_name, cv.IMREAD_COLOR)  # Low Pass
    image_low = cv.imread(image_low_file_name, cv.IMREAD_COLOR)  # High Pass

    # save image_low fourier transform and return 8 * log()
    # image_low_log_of_dft = get_image_log_frequency_domain(image_low)
    # cv.imwrite('06_dft_far.jpg', image_low_log_of_dft)

    # save image_high fourier transform and return 8 * log()
    # image_high_log_of_dft = get_image_log_frequency_domain(image_high)
    # cv.imwrite('05_dft_near.jpg', image_high_log_of_dft)

    # create empty array for hybrid image and low result and high result
    hybrid_image = np.zeros(image_low.shape, dtype='uint8')
    low_result = np.zeros(image_low.shape, dtype='uint8')
    high_result = np.zeros(image_low.shape, dtype='uint8')

    # calculate hybrid channel for all RGB channel
    print(image_low.shape)
    print(high_result.shape)

    for i in range(3):
        hybrid_image[:, :, i], low_result[:, :, i], high_result[:, :, i] = hybrid(
            (image_low[:, :, i], low_std),
            (image_high[:, :, i], high_std),
            .65, cutoff_low, cutoff_high
        )

    hybrid_image = cv.cvtColor(hybrid_image, cv.COLOR_BGR2RGB)

    # create near and far of hybrid image
    zoom_in = cv.resize(hybrid_image, (0, 0), fx=5, fy=5)
    zoom_out = cv.resize(hybrid_image, (0, 0), fx=.5, fy=.5)

    pp.imsave(zoom_in_file_name, zoom_in)
    pp.imsave(zoom_out_file_name, zoom_out)
    pp.imsave('output/result.jpg', hybrid_image)


if __name__ == '__main__':
    main()
