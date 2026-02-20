import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
    import os
    return os, random


@app.cell
def _(os, random):
    import cv2

    img1 = cv2.imread("fence.jpg", cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape

    def bw_convert1(img,T=30):
        bw = img.copy()
        h, w = img.shape
        for i in range(h):
            for j in range(w):
                if img[i,j]>=T:
                    bw[i,j] = 255
                else:
                    bw[i,j] = 0
        return bw
    radius = 5


    def random_circle_generator(img):
        os.makedirs('with_circle', exist_ok=True)
        for i in range(100):
            img_circle = img.copy()
            x = random.randint(radius, w - radius - 1)
            y = random.randint(radius, h - radius - 1)
            cv2.circle(img_circle, (x, y), radius, 0, -1)
            cv2.imwrite(f"with_circle/img_{i+1}.jpg", img_circle)
        print('done')

    img_fence = bw_convert1(img1)
    random_circle_generator(img_fence)
    return (cv2,)


@app.cell
def _(cv2, os):
    import shutil

    roll_no = "ms22276"
    source_folder = "with_circle/"

    # destination folders
    in_folder = f"inside_{roll_no}"
    out_folder = f"outside_{roll_no}"
    os.makedirs(in_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    # counters
    in_count = 0
    out_count = 0
    target = 25

    print("sort manually by yourelf")
    print(f"Press 'i' for INSIDE.")
    print(f"Press 'o' for OUTSIDE.")
    print(f"Press 's' to SKIP (if circle is on the line).")
    print(f"Press 'q' to QUIT.\n")

    #list of generated images
    files = sorted(os.listdir(source_folder))

    for filename in files:
        if in_count >= target and out_count >= target:
            print("\nSuccess! You have collected 25 images for both categories.")
            break

        if not filename.endswith((".jpg", ".png")):
            continue
        # read image
        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        # show image
        cv2.imshow("Sort Images: 'i'=Inside, 'o'=Outside", img)
        key = cv2.waitKey(0) & 0xFF  #wait indefinitely for key

        if key == ord('i'):  #inside
            if in_count < target:
                in_count += 1
                # Rename format: in_rollno_1.jpg
                new_name = f"in_{roll_no}_{in_count}.jpg"
                shutil.copy(img_path, os.path.join(in_folder, new_name))
                print(f"Saved to inside ({in_count}/25)")
            else:
                print("Inside folder is full! Skipping...")

        elif key == ord('o'):  # outside
            if out_count < target:
                out_count += 1
                # format: out_rollno_1.jpg
                new_name = f"out_{roll_no}_{out_count}.jpg"
                shutil.copy(img_path, os.path.join(out_folder, new_name))
                print(f"Saved to outside({out_count}/25)")
            else:
                print("Outside folder is full")
        elif key == ord('s'):  #skip
            print("Skipped.")
        elif key == ord('q'):  #quit
            break

    cv2.destroyAllWindows()
    print("\nSorting complete.")
    return


@app.cell
def _(cv2):
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    app.run()
