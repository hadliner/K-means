from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    start_x = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        end_x = start_x + (percent * 300)
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50),
                      color.astype("uint8").tolist(), -1)
        start_x = end_x

    # return the bar chart
    return bar

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-c", "--clusters", required=True, type=int,
                help="# of clusters")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# check if the image is loaded successfully
if image is None:
    print("Error: Unable to load the image. Please check the path provided.")
    exit()

# convert the image from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show the original image
plt.figure()
plt.axis("off")
plt.imshow(image)
plt.show()

# reshape the image to be a list of pixels
image_pixels = image.reshape((-1, 3))

# cluster the pixel intensities
clt = KMeans(n_clusters=args["clusters"])
clt.fit(image_pixels)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)

# show the color bar
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
