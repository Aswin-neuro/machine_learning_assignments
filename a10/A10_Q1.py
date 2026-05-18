import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    # Cell 1 — Imports
    import subprocess, sys, os
    from pathlib import Path
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # # Cell 3 #training dont run
    # from ultralytics import YOLO

    # model = YOLO("yolov8n.pt")  #wts dwld
    # results = model.train(
    #     data="VOC.yaml",
    #     epochs=20,
    #     imgsz=640,
    #     batch=16,
    #     device=0,
    #     project="runs/detect",
    #     name="voc_train"
    # )
    return


@app.cell
def _():
    # Cell 4 — Evaluate best checkpoint
    from ultralytics import YOLO

    model = YOLO("/home/aswin/runs/detect/runs/detect/voc_train2/weights/best.pt")

    metrics = model.val(data="VOC.yaml", device=0)
    print(f"Precision:    {metrics.box.mp:.3f}")
    print(f"Recall:       {metrics.box.mr:.3f}")
    print(f"mAP@0.5:      {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The settings for which I ran this for.
    epochs=20
    imgsz=640
    batch=16

    - I got precision of 0.823 which is really good
    - Recall of 0.748, meaning some missed detection is there.
    - mAP@0.5 got 0.835 meaning high detection capability.
    - mAP@0.5:0.95 got 0.629 meaning moderate localization precision

    - So these says our bounding boxes are coarse and the precision is not that good. This can be due to the low number of epochs as well as the smaller yolo model being used to train without much computational expense.

    - Strong classes obtained are - car(0.93), horse(0.93), bus(0.93) and person(0.90). This is because higher representation in our VOC dataset, relatively large objects and easy distinguishability.
    """)
    return


if __name__ == "__main__":
    app.run()
