import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# -------------------------------
# Shannon-Fano Encoding Functions
# -------------------------------
def shannon_fano(symbols_freq):
    if len(symbols_freq) <= 1:
        return {symbols_freq[0][0]: "0"} if len(symbols_freq) == 1 else {}

    # Split into two parts with nearly equal total frequencies
    total = sum(freq for _, freq in symbols_freq)
    cumulative = 0
    split_index = 0
    for i, (_, freq) in enumerate(symbols_freq):
        cumulative += freq
        if cumulative >= total / 2:
            split_index = i + 1
            break

    left = symbols_freq[:split_index]
    right = symbols_freq[split_index:]

    left_codes = shannon_fano(left)
    right_codes = shannon_fano(right)

    # Prefix 0 for left, 1 for right
    for k in left_codes:
        left_codes[k] = "0" + left_codes[k]
    for k in right_codes:
        right_codes[k] = "1" + right_codes[k]

    return {**left_codes, **right_codes}


# -------------------------------
# Image Compression with Shannon-Fano
# -------------------------------
def shannon_fano_image(image_path):
    # Step 1: Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    flat_pixels = img.flatten()

    # Step 2: Calculate frequency of each pixel value
    freq_counter = Counter(flat_pixels)
    symbols_freq = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)

    # Step 3: Get Shannon-Fano codes
    codes = shannon_fano(symbols_freq)

    # Step 4: Encode the image
    encoded_bits = "".join(codes[p] for p in flat_pixels)

    # Step 5: Decode the image
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_pixels = []
    current_bits = ""
    for bit in encoded_bits:
        current_bits += bit
        if current_bits in reverse_codes:
            decoded_pixels.append(reverse_codes[current_bits])
            current_bits = ""

    decoded_img = np.array(decoded_pixels, dtype=np.uint8).reshape(img.shape)

    # -------------------------------
    # Display Original & Reconstructed
    # -------------------------------
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(decoded_img, cmap='gray')
    plt.title("Decoded Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Print compression stats
    original_bits = flat_pixels.size * 8
    compressed_bits = len(encoded_bits)
    print(f"Original Size: {original_bits} bits")
    print(f"Compressed Size: {compressed_bits} bits")
    print(f"Compression Ratio: {original_bits / compressed_bits:.2f}")


# -------------------------------
# Run the function
# -------------------------------
shannon_fano_image("lena.png")
