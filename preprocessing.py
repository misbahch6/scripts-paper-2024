import os
import pandas as pd
import cv2

def convert_xml_to_csv(mouse_name):
    """
    Convert XML annotations to CSV format.
    
    Args:
        mouse_name (str): Name of the mouse.
        
    Returns:
        DataFrame: DataFrame with image names, labels, classes, width, height, and coordinates of objects.
    """
    xml_file = os.path.join(mouse_name, "annotations.xml")
    try:
        xml_data = pd.read_xml(xml_file)
    except FileNotFoundError:
        print(f"XML file not found for {mouse_name}.")
        return None
    
    cols = ["filename", "width", "height", "class", "label", "xmin", "ymin", "xmax", "ymax"]
    rows = []
    
    class_label_dic = {"antralfollicle": 0, "antralfolliclewn": 1, "corpusluteum": 2, "negative": 3}

    for image in xml_data.findall(".//image"):
        filename = image.get("name")
        width = image.get("width")
        height = image.get("height")

        for box in image.findall(".//box"):
            classs = box.get("label").lower()
            label = class_label_dic.get(classs, -1)

            xmin = box.get("xtl")
            ymin = box.get("ytl")
            xmax = box.get("xbr")
            ymax = box.get("ybr")

            rows.append({"filename": filename, "width": width, "height": height, "class": classs, 
                         "label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    
    df = pd.DataFrame(rows, columns=cols)
    csv_file = f"{mouse_name}/labels-{mouse_name}.csv"
    df.to_csv(csv_file, index=False)

    return df

def crop_image(mouse_name, image_name, img_width, img_height, df):
    """
    Crop objects from the image based on bounding box coordinates.
    
    Args:
        mouse_name (str): Name of the mouse.
        image_name (str): Name of the image.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        df (DataFrame): DataFrame containing bounding box coordinates.
        
    Returns:
        numpy.ndarray: Cropped image.
        list: List of bounding box coordinates.
        list: List of labels.
    """
    color = (255, 220, 81)
    thickness = 1
    font_size = 0.4
    bbox_list = []
    labels = []

    image_path = os.path.join(mouse_name, "images", image_name)
    im = cv2.imread(image_path)
    im = cv2.resize(im, (img_width, img_height), interpolation=cv2.INTER_AREA)

    for _, row in df[df['filename'] == image_name].iterrows():
        x_modify = img_width / row['width']
        y_modify = img_height / row['height']

        x_min = int(float(row['xmin']) * x_modify)
        x_max = int(float(row['xmax']) * x_modify)
        y_min = int(float(row['ymin']) * y_modify)
        y_max = int(float(row['ymax']) * y_modify)

        bbox_list.append([x_min, y_min, x_max, y_max])
        labels.append(row['class'])

        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.putText(im, row['class'], (x_min, y_min - 5), 0, font_size, color, thickness)

    cv2.imshow("Cropped Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return im, bbox_list, labels

def main():
    mouse_names = ['mouse9147', 'mouse34727', '097B', '855B', '108B', '784B', '924B', '077B', 'mousead21', '975B',
                  '916B','567B', '925B', '948B', '899B', '934B', '940B']

    for mouse_name in mouse_names:
        df = convert_xml_to_csv(mouse_name)
        if df is not None:
            for filename in df['filename'].unique():
                crop_image(mouse_name, filename, 640, 480, df)

if __name__ == "__main__":
    main()
