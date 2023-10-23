import cv2
import numpy as np

# Load the image
image = cv2.imread('Grou.jpg')

# Define the transformation parameters
rows, cols, _ = image.shape
center = (cols // 2, rows // 2)

# Perform scaling transformation
scaling_matrix = cv2.getRotationMatrix2D(center, 0.5, 2)
scaled_image = cv2.warpAffine(image, scaling_matrix, (cols, rows))

# Perform rotation transformation
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

# Perform translation transformation
translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])
translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

# Perform shearing transformation
shearing_matrix = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
sheared_image = cv2.warpAffine(image, shearing_matrix, (cols, rows))

# Save the transformed images
cv2.imwrite('scaled_image.jpg', scaled_image)
cv2.imwrite('rotated_image.jpg', rotated_image)
cv2.imwrite('translated_image.jpg', translated_image)
cv2.imwrite('sheared_image.jpg', sheared_image)
