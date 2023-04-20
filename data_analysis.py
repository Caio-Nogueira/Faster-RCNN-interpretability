import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class_count = {x:0 for x in utils.idx_to_label.values()}
object_areas = []
num_detections = []

def object_area(object):
    s1 = int(object["bndbox"]["xmax"]) - int(object["bndbox"]["xmin"])
    s2 = int(object["bndbox"]["ymax"]) - int(object["bndbox"]["ymin"])
    return s1 * s2

for img, annotation in tqdm(utils.train_dataset):
    objects = annotation["annotation"]["object"]
    num_detections.append(len(objects))
    for obj in objects:
        class_count[obj["name"]] += 1
        object_areas.append(object_area(obj))

# class_count =  {'background': 0, 'Car': 28742, 'Van': 2914, 'Truck': 1094, 'Pedestrian': 4487, 'Person_sitting': 222, 'Cyclist': 1627, 'Tram': 511, 'Misc': 973, 'DontCare': 11295}

# print(len(utils.train_dataset))
plt.bar(class_count.keys(), class_count.values())
plt.xticks(rotation=90)
plt.savefig("class_count_kitti.pdf")
plt.show()


print("Average number of detections per image: ", np.mean(num_detections))
print("Average object area: ", np.mean(object_areas))
print("Number of objects per class: ", class_count)
