# from ..defect_preprocess.noise import Noise  ❌
from defect_preprocess.noise import Noise  
import matplotlib.pyplot as plt
import cv2

path_img = r"D:\object_detect_tracking\camera-viewer\data\WindTurbines_CableTower_Copy\train\images\cabletower5_jpg.rf.cfedbcd99c645625f01b2e0f780936a2.jpg"
noise = Noise(path_img)

median_filtered = noise.MedianFilter()
tv_filtered = noise.TotalVariationFilter()
gaussian_noisy = noise.GaussianNoise()
adaptive_thresholded = noise.AdaptiveThreshold()
contraharmonic_filtered = noise.ContraharmonicMeanFilter()
decision_based_filtered = noise.DecisionBasedSwitchingFilter()




# Display the results
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
plt.title('Median Filter')
plt.axis('off')
plt.subplot(232)
plt.imshow(cv2.cvtColor(tv_filtered, cv2.COLOR_BGR2RGB))
plt.title('Total Variation Filter')
plt.axis('off')
plt.subplot(233)
plt.imshow(cv2.cvtColor(gaussian_noisy, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Noise')
plt.axis('off')
plt.subplot(234)
plt.imshow(cv2.cvtColor(adaptive_thresholded, cv2.COLOR_BGR2RGB))
plt.title('Adaptive Thresholding')
plt.axis('off')
plt.subplot(235)
plt.imshow(cv2.cvtColor(contraharmonic_filtered, cv2.COLOR_BGR2RGB))
plt.title('Contraharmonic Mean Filter')
plt.axis('off')
plt.subplot(236)
plt.imshow(cv2.cvtColor(decision_based_filtered, cv2.COLOR_BGR2RGB))
plt.title('Decision Based Switching Filter')
plt.axis('off')
plt.tight_layout()
plt.show()



# save plot 
plt.savefig('noise_filters_comparison.png', bbox_inches='tight', dpi=300)
