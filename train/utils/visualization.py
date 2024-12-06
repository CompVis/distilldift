import torch
import matplotlib.pyplot as plt

def display_image_pair(image_pair, show_bbox=False, labels=None, ground_truth=None):
    """
    Display an image pair with keypoints and bounding boxes (optional).

    Args:
        image_pair (dict): dictionary containing the source and target images, keypoints and bounding boxes
        show_bbox (bool, optional): whether to show the bounding boxes
    """

    # Get the images, keypoints and bounding boxes from the image pair
    source_image = image_pair['source_image']
    target_image = image_pair['target_image']
    source_points = image_pair['source_points']
    target_points = image_pair['target_points']
    source_bbox = image_pair['source_bbox']
    target_bbox = image_pair['target_bbox']

    fig, ax = plt.subplots()

    # Calculate the offset of the target image (based on the width of the source image)
    offset = source_image.size[0]

    # Draw the images in specific location (left, right, bottom, top)
    ax.imshow(source_image, extent=[0, source_image.size[0], 0, source_image.size[1]])
    ax.imshow(target_image, extent=[offset, offset + target_image.size[0], 0, target_image.size[1]])

    # Get a list of colors from the 'tab20' colormap which has 20 distinct colors
    colors = plt.cm.tab20.colors

    def print_line(i, sp, tp, color=None):
        ax.plot([sp[0], tp[0] + offset], # x-coordinates
                [source_image.size[1] - sp[1], target_image.size[1] - tp[1]], # y-coordinates (inverted)
                color=colors[i % len(colors)] if color is None else color)

    def is_correct(tp, gp):
        y, x, h, w = target_bbox
        distances = torch.linalg.norm(tp - gp, axis=-1)
        return distances <= 0.1 * max(h, w)

    # Draw lines between the keypoints, ensuring the target points are offset correctly
    if isinstance(target_points, list):
        for j in range(len(target_points)):
            for i, (sp, tp) in enumerate(zip(source_points, target_points[j])):
                if ground_truth is not None:
                    print_line(i, sp, tp, color='lime' if is_correct(tp, ground_truth[i][j]) else 'red')
                else:
                    print_line(j, sp, tp)
    else:
        for i, (sp, tp) in enumerate(zip(source_points, target_points)):
            if ground_truth is not None:
                print_line(i, sp, tp, color='lime' if is_correct(tp, ground_truth[i]) else 'red')
            else:
                print_line(i, sp, tp)

    # Add labels to legend
    if labels is not None:
        for i, label in enumerate(labels):
            ax.plot([], [], color=colors[i], label=label)

    # Draw the bounding boxes if required
    if show_bbox:
        # Extract the coordinates and dimensions of the bounding boxes
        source_x, source_y, source_w, source_h = source_bbox
        target_x, target_y, target_w, target_h = target_bbox

        # Draw the bounding box for the source image
        source_rect = plt.Rectangle((source_x, source_image.size[1] - source_y - source_h),
                                    source_w, source_h,
                                    linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(source_rect)

        # Draw the bounding box for the target image
        target_rect = plt.Rectangle((target_x + offset, target_image.size[1] - target_y - target_h),
                                    target_w, target_h,
                                    linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(target_rect)

    # Disable the axis labels
    ax.axis('off')

    # Plot legend if labels are provided
    if labels is not None:
        ax.legend()

    # Show the plot
    plt.show()

def plot_results(pck_img, pck_bbox, layers, save_path):
    """
    Plot the PCK values for each layer.

    Args:
        pck_img (torch.Tensor): PCK values for each image
        pck_bbox (torch.Tensor): PCK values for each bounding box
        layers (list, optional): List of layers to plot
        save_path (str, optional): Path to save the plot
    """

    plt.figure(figsize=(10, 6))

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    r1 = range(len(layers))
    r2 = [x + bar_width for x in r1]

    # Plot the bars for PCK_img and PCK_bbox next to each other
    plt.bar(r1, pck_img, width=bar_width, color='blue', edgecolor='grey', label='PCK_img')
    plt.bar(r2, pck_bbox, width=bar_width, color='orange', edgecolor='grey', label='PCK_bbox')

    # Add xticks on the middle of the group bars
    plt.xlabel('Layer', fontweight='bold')
    plt.xticks([r + bar_width / 2 for r in range(len(layers))], layers)

    # Set the y-axis label
    plt.ylabel('PCK', fontweight='bold')

    # Create legend & Show graphic
    plt.legend()

    # Save the plot to a file
    plt.savefig(save_path)
    plt.show()