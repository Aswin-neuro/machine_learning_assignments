import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Install VOC2012 dataset on your laptop (You may copy the dataset from me or Siddharth on
    your pendrive). Use YOLOv8 to train and identify objects in this dataset. Describe the results
    of your evaluation.
    """)
    return


@app.cell
def _():
    import os
    import xml.etree.ElementTree as ET

    classes = [
        "aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
        "chair","cow","diningtable","dog","horse","motorbike","person",
        "pottedplant","sheep","sofa","train","tvmonitor"
    ]

    annotations_path = "VOC2012_test/Annotations"
    labels_path = "/home/aswin/ml_proj/a10/dset/Annotations_txt/"

    os.makedirs(labels_path, exist_ok=True)

    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        return (x*dw, y*dh, w*dw, h*dh)

    for file in os.listdir(annotations_path):
        if not file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(annotations_path, file))
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        out_file = open(os.path.join(labels_path, file.replace(".xml", ".txt")), "w")

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue

            cls_id = classes.index(cls)

            xmlbox = obj.find('bndbox')
            b = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text)
            )

            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")

        out_file.close()
    return


if __name__ == "__main__":
    app.run()
