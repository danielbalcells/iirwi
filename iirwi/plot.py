import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches

LINEWIDTH = 2
EDGECOLOR = 'r'

class MemePlotter:
    def __init__(self):
        pass

    def plot(self, input_img, similar_img, input_img_crop_coords, show=False, save_path=None):
        similar_img_crop_coords = similar_img.bbox
        similar_img_parent = similar_img.load_parent()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.subplots_adjust(top=0.8, bottom=0.1)
        ax1.imshow(similar_img_parent)
        ax2.imshow(input_img)
        radius1 = (similar_img_crop_coords[2] - similar_img_crop_coords[0]) / 2
        rect1 = patches.Circle(((similar_img_crop_coords[0] + similar_img_crop_coords[2]) / 2, 
                            (similar_img_crop_coords[1] + similar_img_crop_coords[3]) / 2), 
                            radius1, linewidth=LINEWIDTH, edgecolor=EDGECOLOR, facecolor='none')
        ax1.add_patch(rect1)
        
        radius2 = (input_img_crop_coords[2] - input_img_crop_coords[0]) / 2
        rect2 = patches.Circle(((input_img_crop_coords[0] + input_img_crop_coords[2]) / 2, 
                            (input_img_crop_coords[1] + input_img_crop_coords[3]) / 2),
                            radius2, linewidth=LINEWIDTH, edgecolor=EDGECOLOR, facecolor='none')
        ax2.add_patch(rect2)

        ax1.axis('off')
        ax2.axis('off')
        fig.suptitle('Is It Really Worth It?', fontsize=20, weight='bold')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        if show:
            plt.show()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (3,))
        return img