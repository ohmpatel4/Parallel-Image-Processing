# Parallel-Image-Processing
Parallel implementation of applying a convolution kernel to an image and essentially blurring it.

This project consists of 2 portions. 

The first portion is a simple serial program that applies a convolution kernel to an image. We chose the kernel in a way that we would achieve blurring. In this case the kernel was a 5x5 matrix consisting of all 0.04. The code is attatched. The results of the program is shown below.

**Origial Image
**
![PurplePlanets](https://github.com/user-attachments/assets/b8a0cc18-5c3c-4185-a0eb-07a1669b4590)

**Filtered Image
**
![part_1_processed_image](https://github.com/user-attachments/assets/b153a982-b1da-4f1f-acf8-27dd072bf571)

The second portion aims for a parallel implementation to portion one. The image is first split into equal sizes for each processor. Open MPI is used to accomplish these tasks. A halo is added to each sub image before being sent to each processors. Each processor is then responsible for applying the convolution kernel to the sub image. All filtered sub images are then gathered onto rank 0 and combined into one final image. The kernel used was the same as portion 1 and the output images were identical.

To compare if both implementations gave the same results, we used the same 5x5 uniform kernel to both images with nproc=1. Once we had recieved the final images, we compared them by calculating the laplacian variance for blur detection. We basically turn the image into greyscale, then use the Laplacian kernel and convolve it with a channel of the image. We then take calculate the variance of that response. This provides us with a quantitative value to the blur level of the image. We used this method on both outputs. We got an almost identical blur level indicating a successful implementation.
