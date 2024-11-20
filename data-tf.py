import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras_cv import bbox as keras_cv
import matplotlib.pyplot as plt
import cv2
import pickle

def load_images_boxes_labels(filename, sub_df, img_width, img_height, mouse_name, df):
    image = cv2.imread(os.path.join(mouse_name, 'cropped', filename))
    image = cv2.resize(image, (img_height, img_width), interpolation=cv2.INTER_AREA)
    bbox_list = []
    label_list = []
    for i in sub_df.index:
        x_modify = img_width / df.iloc[i]['width'].astype(float).astype(int)
        y_modify = img_height / df.iloc[i]['height'].astype(float).astype(int)
        x_min = df.iloc[i]['xmin'].astype(float).astype(int) * x_modify
        y_min = df.iloc[i]['ymin'].astype(float).astype(int) * y_modify
        x_max = df.iloc[i]['xmax'].astype(float).astype(int) * x_modify
        y_max = df.iloc[i]['ymax'].astype(float).astype(int) * y_modify
        label = df.iloc[i]['label']
        boxes = keras_cv.bounding_box.convert_format(
            tf.cast([x_min, y_min, x_max, y_max], dtype=np.float32),
            images=image,
            source="xyxy",
            target="xywh")
        bbox_list.append(boxes)
        label_list.append(label)

    return {'images': tf.cast(image, dtype=np.float32),
            'bounding_boxes': {
                'boxes': tf.cast(bbox_list, dtype=np.float32),
                'classes': tf.cast(label_list, dtype=np.float32)
            }}

def padding(data):
    maximum_bb = max(d['bounding_boxes']['boxes'].shape[0] for d in data)
    maximum_cls = max(d['bounding_boxes']['classes'].shape[0] for d in data)

    for d in data:
        if d['bounding_boxes']['boxes'].shape[0] < maximum_bb:
            paddings = [[0, maximum_bb - d['bounding_boxes']['boxes'].shape[0]], [0, 0]]
            d['bounding_boxes']['boxes'] = tf.pad(d['bounding_boxes']['boxes'], paddings, 'CONSTANT',
                                                  constant_values=-1)
        if d['bounding_boxes']['classes'].shape[0] < maximum_cls:
            d['bounding_boxes']['classes'] = tf.concat([d['bounding_boxes']['classes'],
                                                        tf.fill([maximum_cls - d['bounding_boxes']['classes'].shape[0]],
                                                                -1)], 0)

    return data

def tf_dataset(data):
    bboxes = tf.data.Dataset.from_tensor_slices([d['bounding_boxes']['boxes'] for d in data])
    images = tf.data.Dataset.from_tensor_slices([d['images'] for d in data])
    labels = tf.data.Dataset.from_tensor_slices([d['bounding_boxes']['classes'] for d in data])

    ds = tf.data.Dataset.zip((images, bboxes, labels))
    ds = ds.map(lambda x, y, z: {"images": x, 'bounding_boxes': {"boxes": y, "classes": z}})
    return ds

def visualize_obj_freq(data):
    class_label_dic = {0: "antralfollicle", 1: "antralfolliclewn", 2: "corpusluteum"}

    freq = {label: sum(1 for box_cls in d["bounding_boxes"]["classes"] if box_cls == label) for label in
            class_label_dic}

    plt.bar(freq.keys(), freq.values())
    plt.xlabel("Classes")
    plt.ylabel("Frequencies")
    plt.title(f"Total num of images: {len(data)}")
    for x, y in freq.items():
        plt.text(x, y, str(y), ha="center", va="bottom")

    plt.show()

def main():
    img_width = 640
    img_height = 640

    mouse_name = ['mouse9147', 'mouse34727', '097B', '855B', '108B', '784B', '924B', '077B', 'mousead21', '975B',
                  '916B','567B', '925B', '948B', '899B', '934B', '940B']
    data = []

    for j in mouse_name:
        df = pd.read_csv(os.path.join(j, 'cropped-labels-{}.csv'.format(j)))
        filenames = df["filename"].unique().tolist()
        for i in filenames:
            sub_df = df[df['filename'] == i]
            s_df = sub_df[sub_df['class'] != 'negative']
            if s_df.shape[0] > 0:
                data.append(load_images_boxes_labels(i, s_df, img_width, img_height, j, df))

    data = padding(data)
    full_ds = tf_dataset(data)

    tf.data.Dataset.save(full_ds, 'tf_data', compression='GZIP')
    with open('tf_data' + '/element_spec', 'wb') as out_:
        pickle.dump(full_ds.element_spec, out_)

    visualize_obj_freq(data)

if __name__ == "__main__":
    main()
