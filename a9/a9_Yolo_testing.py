import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import random
    import cv2
    from pathlib import Path
    from ultralytics import YOLO

    return Path, YOLO, cv2, random


@app.cell
def _(Path, YOLO, cv2, random):

    model=YOLO("yolov8s.pt")
    img_dir=Path("a9/data/val2017")

    random.seed(42)
    images=random.sample(sorted(img_dir.glob("*.jpg")), 100)

    for img_path in images:
        if cv2.imread(str(img_path)) is None:
            print(f"[{img_path.name}] skipped (unreadable)")
            continue

        result=model.predict(str(img_path), conf=0.25, verbose=False)[0]

        print(f"\n[{img_path.name}]")
        for box in result.boxes:
            x1,y1,x2,y2=[round(v, 1) for v in box.xyxy[0].tolist()]
            cls=result.names[int(box.cls)]
            conf=round(float(box.conf), 2)
            print(f"{cls:<20} conf={conf} bbox=[{x1},{y1},{x2},{y2}]")
    return img_dir, model


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The pretrained model shows good object performance on the COCO validation subset.
    High confidence predictions for common classes like: person, vehicle, and animals

    Low confidence threshold(0.25) introduces multiple low-confidence detections, duplicate bounding boxes and occasional false positives

    In scene with lots of sheeps and for small ambiguous objects the low confidence intervals can show false positives. Performance degrades for small-scale or partially occluded objects, where confidence drops and misclassifications increase.
    """)
    return


@app.cell
def _(cv2, img_dir, model, random):
    #3x3 grid visualization block
    import matplotlib.pyplot as plt

    imgs_9=random.sample(sorted(img_dir.glob("*.jpg")), 9)

    grid_imgs=[]

    for p in imgs_9:
        im=cv2.imread(str(p))
        if im is None:
            continue

        res=model.predict(str(p), conf=0.25, verbose=False)[0]

        for b in res.boxes:
            xa,ya,xb,yb=map(int, b.xyxy[0])
            c_name=res.names[int(b.cls)]
            c_score=float(b.conf)

            lbl=f"{c_name} {c_score:.2f}"
            cv2.rectangle(im, (xa, ya), (xb, yb), (0, 255, 0), 2)
            cv2.putText(im, lbl, (xa, ya - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        grid_imgs.append(im)

    plt.figure(figsize=(10, 10))
    for i in range(len(grid_imgs)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(grid_imgs[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
