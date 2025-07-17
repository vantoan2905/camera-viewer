import matplotlib.pyplot as plt
import numpy as np
# ------ this class for visualizing the dataset ------
# function viz_dataset,viz_one_image, viz_sample_5_image_form_df




class DataVisualizer:
    def __init__(self, data=None):
        self.data = data
        
    def viz_one_image(self, img):
        """
        Visualize one image from the dataset.

        Parameters
        ----------
        img : array, shape (H, W, C)
            The image to visualize.

        Returns
        -------
        None
        """
        
        plt.imshow(img)
        plt.show()

    def viz_dataset(self):
        """
        Visualize all images in the dataset.

        If self.data is not None, iterate through all images in the dataset and
        call viz_one_image on each one. Otherwise, print a message to the user
        indicating that there is no data to visualize.
        """
        if self.data is not None:
            for i in range(len(self.data)):
                self.viz_one_image(self.data[i]) 
        else:
            print("No data to visualize.")
            
    def viz_sample_5_images(self):
        """
        Visualize 5 sample images from the dataset.

        This function will visualize up to 5 images from the dataset.
        If there are less than 5 images in the dataset, it will only
        visualize the images that are available.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.data is not None and len(self.data) >= 5:
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                axes[i].imshow(self.data[i])
                axes[i].axis('off')
            plt.show()
        else:
            print("Not enough data to visualize 5 images.")