import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


def jpeg_compression_demo(img_path):
    # Step 1: Read the image and convert to grayscale
    original_image = cv2.imread(img_path)
    if original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Step 1b: Select an 8x8 block (example: rows 141-148, cols 61-68)
    block = original_image[141:149, 61:69].astype(float)

    # Shift pixel values from [0,255] to [-128,127]
    shifted_block = block - 128

    # Step 2: Apply 2D DCT
    dct_block = dct(dct(shifted_block.T, norm='ortho').T, norm='ortho')

    # Step 3: Quantization (standard JPEG luminance table)
    quantization_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Quantize the DCT coefficients
    quantized_block = np.round(dct_block / quantization_table)

    # Step 4: De-quantization
    dequantized_block = quantized_block * quantization_table

    # Step 5: Inverse DCT
    reconstructed_shifted_block = idct(idct(dequantized_block.T, norm='ortho').T, norm='ortho')

    # Step 6: Rescale reconstructed block to [0,255]
    reconstructed_block = np.round(reconstructed_shifted_block + 128).astype(np.uint8)

    # Replace the block in the original image
    compressed_image = original_image.copy()
    compressed_image[141:149, 61:69] = reconstructed_block

    # Step 7: Calculate sizes
    original_size_bits = block.size * 8
    non_zero_coeffs = np.count_nonzero(quantized_block)
    estimated_compressed_bits = non_zero_coeffs * 16
    if non_zero_coeffs < 64:
        estimated_compressed_bits += 4
    compression_ratio = original_size_bits / estimated_compressed_bits

    # Step 8: Display results
    plt.figure(figsize=(12, 6))
    plt.suptitle('JPEG DCT Logic on an 8x8 Block')

    # Original Block
    plt.subplot(2, 3, 1)
    plt.imshow(block, cmap='gray')
    plt.title('Original Block')
    plt.xlabel(f'Size: {original_size_bits} bits ({original_size_bits / 8:.2f} bytes)')

    # Quantized DCT
    plt.subplot(2, 3, 2)
    plt.imshow(quantized_block, cmap='viridis')
    plt.colorbar()
    plt.title('Quantized DCT Coefficients')
    plt.xlabel(f'{non_zero_coeffs} non-zero values')

    # Reconstructed Block
    plt.subplot(2, 3, 3)
    plt.imshow(reconstructed_block, cmap='gray')
    plt.title('Reconstructed Block')

    # Original Image
    plt.subplot(2, 3, 4)
    plt.imshow(original_image, cmap='gray')
    plt.title(f'Original Image\n{original_size_bits / 8:.2f} bytes')

    # Compressed Image
    plt.subplot(2, 3, 6)
    plt.imshow(compressed_image, cmap='gray')
    plt.title(f'Compressed Image\n{estimated_compressed_bits / 8:.2f} bytes')

    plt.show()

    # Print size info
    print("\n--- Size Comparison for the 8x8 Block ---")
    print(f"Original Size      : {original_size_bits} bits ({original_size_bits / 8:.1f} bytes)")
    print(
        f"Compressed Estimate: {non_zero_coeffs} non-zero coeffs * ~16 bits = {estimated_compressed_bits} bits ({estimated_compressed_bits / 8:.1f} bytes)")
    print(f"Compression Ratio  : {compression_ratio:.1f} : 1")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    jpeg_compression_demo("lena.png")
