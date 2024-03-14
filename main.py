import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import skimage


USE_CIELAB = False

def palettize_image_rgb(image, palette):
    # Flatten the image array to shape (w*h, 3) for easier processing
    flattened_image = image.reshape(-1, 3)
    # Initialize an array to hold the indices of the closest palette colors
    indices = np.zeros((flattened_image.shape[0]), dtype=int)

    # Calculate the Euclidean distance from each pixel to each palette color
    for i, color in enumerate(flattened_image):
        distances = np.sqrt(np.sum((palette - color) ** 2, axis=1))
        indices[i] = np.argmin(distances)

    # Replace each pixel with the closest palette color
    palettized_image = palette[indices].reshape(image.shape)

    return palettized_image


def rgb_to_lab(rgb_image):
    # skimage expects float32 in the range [0, 1] for RGB images
    normalized_rgb_image = rgb_image.astype(np.float32) / 255
    # Convert RGB to LAB
    lab_image = skimage.color.rgb2lab(normalized_rgb_image)

    return lab_image


def palettize_image_cielab(image, palette):
    # Flatten the image array to shape (w*h, 3) for easier processing
    flattened_image = image.reshape(-1, 3)
    flattened_image_lab = rgb_to_lab(flattened_image)
    palette_lab = rgb_to_lab(palette)
    # Initialize an array to hold the indices of the closest palette colors
    indices = np.zeros((flattened_image.shape[0]), dtype=int)

    # Calculate the Euclidean distance from each pixel to each palette color
    for i, color_lab in enumerate(flattened_image_lab):
        distances = np.sqrt(np.sum((palette_lab - color_lab) ** 2, axis=1))
        indices[i] = np.argmin(distances)

    # Replace each pixel with the closest palette color
    palettized_image = palette[indices].reshape(image.shape)

    return palettized_image


def palettize_image(image, palette, use_cielab=False):
    if use_cielab:
        return palettize_image_cielab(image, palette)
    else:
        return palettize_image_rgb(image, palette)


def read_palette(palette_filename):
    palette = []
    for l in open(PALETTE_FILENAME, "r"):
        if l.startswith("#"):
            continue
        l = l.split("#")[0]
        rgb = list(map(int, l.split()))
        palette.append(rgb)

    palette = np.array(palette)
    return palette.astype(np.float32) / 255


def counts_for_palette(image, palette):
    counts = []
    for color in palette:
        cnt = np.all(image == color[None, None, :], axis=-1).sum()
        counts.append(cnt)
    return np.array(counts)


def show_palettized(palettized_image):
    im = plt.imshow(palettized_image, interpolation='none', vmin=0, vmax=1, aspect='equal')

    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, w, 1))
    ax.set_yticks(np.arange(0, h, 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, w+1, 1))
    ax.set_yticklabels(np.arange(1, h+1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.title("Palettized Image")
    return im


def show_bars(palette, color_counts):
    assert len(color_counts) == len(palette)
    indices = np.arange(len(color_counts))

    # Creating the bar chart
    display_bars = plt.bar(indices, color_counts, color=palette)

    plt.ylim(0, color_counts.max() * 2) # Adjust ylim to leave space for text and slider adjustment

    # Adding the counts above the bars
    display_bar_texts = []
    for i, count in enumerate(color_counts):
        txt = plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
        display_bar_texts.append(txt)

    # Optional: if you want to have a legend or labels for the colors, additional steps are needed

    plt.xticks(indices) # Set x-ticks if necessary
    plt.ylabel('Counts')
    return display_bars, display_bar_texts


def show_3d_scatter(image):
    colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    palettized_colors = palettize_image(colors, palette, use_cielab=USE_CIELAB)

    # Setup 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot palette colors as balls
    for color in palette:
        ax.scatter(color[0], color[1], color[2], color=color, s=100) # Adjust 's' for ball size

    # Draw lines between original and palettized colors
    # Assuming 'original_and_palettized_pairs' contains (original_color, palettized_color) tuples
    for original_color, palettized_color in zip(colors, palettized_colors):
        ax.plot([original_color[0], palettized_color[0]], [original_color[1], palettized_color[1]], [original_color[2], palettized_color[2]], color=original_color)

    # Set labels and show plot
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')


IMAGE_FILENAME, PALETTE_FILENAME = sys.argv[1:3]
image = Image.open(IMAGE_FILENAME).convert('RGB')

# optional reshape
if len(sys.argv) >= 4:
    assert len(sys.argv) == 4
    SHAPE = sys.argv[3]
    w, h = map(int, SHAPE.split("x"))
else:
    w, h = image.width, image.height

image = np.asarray(image.resize((w, h)))
image = image.astype(np.float32) / 255.0

palette = read_palette(PALETTE_FILENAME)

palettized_image = palettize_image(image, palette, use_cielab=USE_CIELAB)
color_counts = counts_for_palette(palettized_image, palette)

# Plotting

# Display original image
plt.subplot(1, 3, 1)
plt.subplots_adjust(bottom=0.3)

img_display = plt.imshow(image)
plt.title('Original Image')
plt.axis('off')  # Hide axes ticks

# Display palettized image
plt.subplot(1, 3, 2)
palettized_img_display = show_palettized(palettized_image)
plt.title("Palettized Image")

plt.subplot(1, 3, 3)
display_bars, display_bar_texts = show_bars(palette, color_counts)

# Adjust the axes for the red, green, and blue gamma sliders
axcolor = 'lightgoldenrodyellow'
ax_red_gamma = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_green_gamma = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_blue_gamma = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)

# Create the sliders
red_gamma_slider = Slider(ax=ax_red_gamma, label='Red Gamma', valmin=0.1, valmax=2.0, valinit=1.0)
green_gamma_slider = Slider(ax=ax_green_gamma, label='Green Gamma', valmin=0.1, valmax=2.0, valinit=1.0)
blue_gamma_slider = Slider(ax=ax_blue_gamma, label='Blue Gamma', valmin=0.1, valmax=2.0, valinit=1.0)


# Update function for the sliders
def update(val):
    r_gamma = red_gamma_slider.val
    g_gamma = green_gamma_slider.val
    b_gamma = blue_gamma_slider.val
    
    # Apply gamma correction separately
    r = image[:,:,0] ** r_gamma
    g = image[:,:,1] ** g_gamma
    b = image[:,:,2] ** b_gamma
    
    # Combine the corrected channels back into an image
    img_corrected = np.stack([r, g, b], axis=2)
    img_display.set_data(img_corrected)

    palettized_image = palettize_image(img_corrected, palette, use_cielab=USE_CIELAB)
    palettized_img_display.set_data(palettized_image)

    color_counts = counts_for_palette(palettized_image, palette)

    for bar, new_count in zip(display_bars, color_counts):
        bar.set_height(new_count)

    for i, (cnt, text) in enumerate(zip(color_counts, display_bar_texts)):
        text.set_position((i, cnt + 0.5))
        text.set_text(str(cnt))

    # fig.canvas.draw_idle()

# Connect the update function to the sliders
red_gamma_slider.on_changed(update)
green_gamma_slider.on_changed(update)
blue_gamma_slider.on_changed(update)

plt.show()


show_3d_scatter(image)
plt.show()