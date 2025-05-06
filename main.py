import cv2
import matplotlib.pyplot as plt

from clustering import assign_clusters
from pickle_save_load import load_pickle
from prep_processing import make_alter_spots

pickle_file = "data/mspots1.pkl"
image_path = "data/test1.JPG"

spots = load_pickle(pickle_file)
image = cv2.imread(image_path)

spots = [spot for spot in spots if spot.area > 8000]

overlay = image.copy()
alpha = 0.7

alter_spots = make_alter_spots(spots, image)

assign_clusters(alter_spots, tolerance_coef=0.4)

unique_clusters = set(sp.ps.cluster for sp in alter_spots if sp.ps.cluster != -1)
print(f"Quantity unique clusters: {len(unique_clusters)}")

for i, spot in enumerate(spots):
    spot.visualize_cluster(image, overlay)

cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

plt.figure(figsize=(16, 16))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("on")
plt.show()


