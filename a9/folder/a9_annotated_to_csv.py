import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    a) Model predicts bounding box(location), confidence score(probability of presence of object), class label(what kind of object it is).

    Annotation provides the location(bounding box) and class label of each object, which serve as ground truth for supervised learning. This enables the model to learn how to localize and classify objects in unseen data.

    If the annotation is not accurate the bounding box and classification can become inaccurate.

    b,c) used labelme for annotation
    The output file is a .json file with these labels:

    label,points,imagePath, frame_00024.jpg", imageData, imageHeight, imageWidth.

    It shows the bounding box x,y position, label=mice, diagonal coordinates of anchorbox.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    d) Do annotation of some of the objects on the extracted frames from the video that you have taken (in Problem 2) and create the csv ﬁle or a text ﬁle of annota5on as the output ﬁle.
    """)
    return


@app.cell
def _():
    import json
    import csv
    from pathlib import Path

    ann_dir2 = Path("a9/frames/training/")   # update path
    csv_out2 = "annotations_my.csv"
    rows_out = []
    for jfile in ann_dir2.glob("*.json"):
        with open(jfile, "r") as f:
            jd = json.load(f)
        img_name2 = jd.get("imagePath")
    
        # skip if no corresponding image (safety)
        if not (ann_dir2 / img_name2).exists():
            continue
        
        for shp2 in jd.get("shapes", []):
            lbl2 = shp2.get("label", "unknown")
            pts2 = shp2.get("points", [])
            if len(pts2) == 2:
                (x1_, y1_), (x2_, y2_) = pts2
                rows_out.append([
                    img_name2,
                    lbl2,
                    x1_, y1_, x2_, y2_
                ])

    # write CSV
    with open(csv_out2, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "xmin", "ymin", "xmax", "ymax"])
        writer.writerows(rows_out)

    print("Saved:", csv_out2)
    return


if __name__ == "__main__":
    app.run()
