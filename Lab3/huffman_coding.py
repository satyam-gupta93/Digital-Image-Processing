import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict


# ---------------- Huffman Coding ----------------
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    # For priority queue
    def __lt__(self, other):
        return self.freq < other.freq


def huffman_tree(frequencies):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    return heap[0]


def huffman_codes(node, code="", mapping={}):
    if node is None:
        return
    if node.symbol is not None:
        mapping[node.symbol] = code
    huffman_codes(node.left, code + "0", mapping)
    huffman_codes(node.right, code + "1", mapping)
    return mapping


def huffman_encode(image):
    flat = image.flatten()
    frequencies = defaultdict(int)
    for pixel in flat:
        frequencies[pixel] += 1

    root = huffman_tree(frequencies)
    codes = huffman_codes(root)

    encoded_data = ''.join(codes[p] for p in flat)
    return encoded_data, codes


def huffman_decode(encoded_data, codes, shape):
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_pixels = []
    code = ""
    for bit in encoded_data:
        code += bit
        if code in reverse_codes:
            decoded_pixels.append(reverse_codes[code])
            code = ""
    return np.array(decoded_pixels, dtype=np.uint8).reshape(shape)


# ---------------- Main Script ----------------
def huffman_image_demo(image_path):
    # Step 1: Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Encode using Huffman
    encoded_data, codes = huffman_encode(img)

    # Step 3: Decode back
    decoded_img = huffman_decode(encoded_data, codes, img.shape)

    # Step 4: Calculate compression ratio
    original_bits = img.size * 8
    compressed_bits = len(encoded_data)
    compression_ratio = original_bits / compressed_bits

    # Step 5: Plot results
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.text(0.1, 0.6,
             f"Original bits: {original_bits}\nCompressed bits: {compressed_bits}\nCompression ratio: {compression_ratio:.2f}x",
             fontsize=12, va='center')
    plt.axis('off')
    plt.title("Compression Info")

    plt.subplot(1, 3, 3)
    plt.imshow(decoded_img, cmap='gray')
    plt.title("Decoded Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
huffman_image_demo("lena.png")
