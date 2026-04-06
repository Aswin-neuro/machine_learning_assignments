import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q2

    Take a short video of about 30 seconds. Write a program to extract each frame of this
    video and save each frame as jpg ﬁle with naming conven5on frame_nnnnn.jpg. nnnnn
    indicate the frame number in the video. Describe proper5es of the video and the
    extracted jpg ﬁles.
    """)
    return


@app.cell
def _():
    import cv2
    vid="video.mp4"
    cap=cv2.VideoCapture(vid)
    if not cap.isOpened():
        print("video can't be opened")
        exit()
    return cap, cv2


@app.cell
def _(cap, cv2):

    fps=cap.get(cv2.CAP_PROP_FPS)
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length=tot_frames/fps
    print(f"fps:{fps}\nresolution: {w}x{h}\ntotal frames: {tot_frames}\nduration (s): {length}")
    return (tot_frames,)


@app.cell
def _(cap, cv2, tot_frames):
    count=0
    for count in range(tot_frames):
        ret, frame=cap.read()
        if not ret:
            break
        cv2.imwrite(f"frame_{count:05d}.jpg", frame)
    cap.release()

    print("frames saved:", count+1)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The video taken is that of a mice in a maze. You can observe the black mice moving around in a static maze puzzle. The mouse comes in, explores and finds a way out. Then it goes in again. It is a suitable footage for annotation
    """)
    return


if __name__ == "__main__":
    app.run()
