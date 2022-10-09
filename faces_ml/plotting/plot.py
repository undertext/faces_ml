import math
from typing import List

import matplotlib.pyplot as plt
from tensorflow import Tensor
import matplotlib.patches as patches


def plot_image_data(image_list: List[Tensor], bb_boxes_list: List[Tensor] = None,
                    bb_boxes_real_list: List[Tensor] = None, sizes=(400, 400)):
    plt.figure(dpi=600)
    for i, image in enumerate(image_list):
        size = round(math.sqrt(len(image_list))) + 1
        print(size)
        ax = plt.subplot(size, size, i + 1)
        plt.imshow(image)
        bb_boxes = bb_boxes_list[i]
        bb_boxes_real = bb_boxes_real_list[i]
        for bb_box in [bb_boxes[0:4]]:
            rect = patches.Rectangle((bb_box[0] * sizes[0], bb_box[1] * sizes[1]),
                                     (bb_box[2]) * sizes[0],
                                     (bb_box[3]) * sizes[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

        for bb_box in [bb_boxes_real[0:4]]:
            print(bb_box)
            rect = patches.Rectangle((bb_box[0] * sizes[0], bb_box[1] * sizes[1]),
                                     (bb_box[2]) * sizes[0],
                                     (bb_box[3]) * sizes[1], linewidth=1, edgecolor='green',
                                     facecolor='none')
            ax.add_patch(rect)
        plt.axis("off")
    fig = plt.gcf()
    plt.show()
    return fig
