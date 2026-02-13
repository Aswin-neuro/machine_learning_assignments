import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cv2
    import matplotlib.pyplot as plt
    import math
    return cv2, math, mo, plt


@app.cell
def _(cv2):
    img = cv2.imread('a4/img.png', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    print(h,w)
    return h, img


@app.cell
def _(h, img):
    ## output frame
    ## resusable
    def output_img_frame(img,output_dim=h-1,f_size=2):
        stride = 1
        output_img_frame = []
        for i in range(output_dim):
            row = []
            for j in range(output_dim):
                row.append(0)
            output_img_frame.append(row)
        return output_img_frame

    output_img_frame_frame=output_img_frame(img)
    return (output_img_frame,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Robert Detection
    """)
    return


@app.cell
def _(math):
    ##Robert
    def robert(reg):
        p_00 = reg[0][0]
        p_01 = reg[0][1]
        p_10 = reg[1][0]
        p_11 = reg[1][1]
        g_x = p_00 - p_11
        g_y = p_01 - p_10
        mag = math.sqrt(g_x**2 + g_y**2)
        return (int(mag))

    ## Robert detect
    def robert_detect(img, output_img_frame):
        pixels = img.tolist()
        stride = 1
        f_size = 2  # Filter size
        output_dim = 306
        for r in range(output_dim):
            for c in range(output_dim):
         # Determine where the window starts on the big image
                reg = []
                start_y = r * stride
                start_x = c * stride
                # Loop through the filter window
                for ky in range(f_size):
                    row = []
                    for kx in range(f_size):
                        # Get the pixel value
                        i=(start_y + ky)
                        j=(start_x + kx)
                        val = pixels[i][j]
                        row.append(val)
                    reg.append(row)
                ## edge calc
                edge_value = robert(reg)
                output_img_frame[r][c] = edge_value
        return output_img_frame
    return (robert_detect,)


@app.cell
def _(img, output_img_frame, plt, robert_detect):
    empty_out_robert = output_img_frame(img)
    robert_detect_out = robert_detect(img, empty_out_robert)

    plt.imshow(robert_detect_out, cmap='gray')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prewitt Detection
    """)
    return


@app.cell
def _(math):
    def prewitt(reg):
        g_x = ((reg[0][2] + reg[1][2] + reg[2][2])-
        (reg[0][0] + reg[1][0] +reg[2][0]))

        g_y = ((reg[2][0] + reg[2][1] + reg[2][2])-
        (reg[0][0] + reg[0][1] + reg[0][2]))

        mag = math.sqrt(g_x**2 + g_y**2)
        return(int(mag))
    return


@app.function
## Robert detect
def main_detect(img, output_img_frame,filter):
    if hasattr(img, 'tolist'):
        pixels = img.tolist()
    else:
        pixels = img
    stride = 1
    f_size = 3  # Filter size
    output_dim = 305 # 307-2
    for r in range(output_dim):
        for c in range(output_dim):
     # Determine where the window starts on the big image
            reg = []
            start_y = r * stride
            start_x = c * stride
            # Loop through the filter window
            for ky in range(f_size):
                row = []
                for kx in range(f_size):
                    # Get the pixel value
                    i=(start_y + ky)
                    j=(start_x + kx)
                    val = pixels[i][j]
                    row.append(val)
                reg.append(row)
            ## edge calc
            edge_value = filter(reg)
            output_img_frame[r][c] = edge_value
    return output_img_frame


@app.cell
def _(img, output_img_frame, plt, sobel):
    empty_out_prewitt = output_img_frame(img, output_dim=305, f_size=3)

    prewitt_detect_out = main_detect(img, empty_out_prewitt,sobel)

    plt.imshow(prewitt_detect_out, cmap='gray')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sobel detect
    """)
    return


@app.cell
def _(math):
    ## Sobel 
    def sobel(reg):
        g_x = ((reg[0][2] + (2 * reg[1][2]) + reg[2][2])-
        (reg[0][0] + (2 * reg[1][0]) + reg[2][0]))
        g_y = ((reg[2][0] + (2 * reg[2][1]) + reg[2][2])-
        (reg[0][0] + (2 * reg[0][1]) + reg[0][2]))

        mag = math.sqrt(g_x**2+g_y**2)
        return int(mag)
    return (sobel,)


@app.cell
def _(img, output_img_frame, plt, sobel):
    empty_out_sobel = output_img_frame(img, output_dim=305, f_size=3)

    ## we can reuse prewitt_detect due to same logic
    sobel_detect_out = main_detect(img, empty_out_sobel,sobel)

    plt.imshow(sobel_detect_out, cmap='gray')
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Laplacian Detection
    """)
    return


@app.function
def laplacian(reg):
    # Map the kernel directly to the pixels in the 3x3 'reg' patch
    center = -4 * reg[1][1]
    top    = reg[0][1]
    bottom = reg[2][1]
    left   = reg[1][0]
    right  = reg[1][2]

    # total value
    val = center + top + bottom + left + right
    # The Laplacian detects the rate of change, which can result in negative numbers.
    mag = abs(val)
    return int(mag)


@app.cell
def _(img, output_img_frame, plt):
    empty_out_laplacian = output_img_frame(img, output_dim=305, f_size=3)

    ## we can reuse prewitt_detect due to same logic
    laplacian_detect_out = main_detect(img, empty_out_laplacian,laplacian)

    plt.imshow(laplacian_detect_out, cmap='gray')
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gaussian -> Laplacian
    """)
    return


@app.cell
def _(img, output_img_frame, plt):
    # convert the input into gaussian output
    def gaussian_matrix(size,sigma):
        import math
        radius =size//2
        filter = []
        total=0

        ## matrix, inc zeroth pos at centre
        for x in range(-radius, radius+1): 
            row = []
            for y in range(-radius, radius+1):
                exp = -(x**2 + y**2)/(2*sigma**2) # power
                val = math.exp(exp)
                row.append(val)
                total += val
            filter.append(row)
        # normalize
        for i in range(size):
            for j in range(size):
                filter[i][j] = filter[i][j] / total
        return filter



    ## GAUSSIAN FILTER
    def gaussian_filter(img, output_img_frame, filter):
        pixels = img.tolist()
        stride = 1
        f_size = len(filter)
        output_dim = len(output_img_frame)
        for r in range(output_dim):
            for c in range(output_dim):
                # image window on the big image
                start_y = r * stride
                start_x = c * stride
                weighted_sum = 0
                # each pixels in filter
                for ky in range(f_size):
                    for kx in range(f_size):
                        # Get the pixel value
                        i=(start_y + ky)
                        j=(start_x + kx)
                        val = pixels[i][j]
                        # apply filter, gives output weight depending on the ith and j th pos of pixel inside filter
                        wt = filter[ky][kx]
                        weighted_sum +=(val*wt)
                output_img_frame[r][c] = int(weighted_sum)
        return output_img_frame

    ## Running the filter
    # Image_store_dimension
    empty_out_gaussian = output_img_frame(img, output_dim=303,f_size=5)
    ## create the gaussian matrix for out specific filter size and sigma
    gaussian_matrix = gaussian_matrix(5,2)

    gaussian_filtered_img = gaussian_filter(img, empty_out_gaussian, gaussian_matrix)

    ###
    plt.imshow(gaussian_filtered_img, cmap='gray')
    plt.gca()

    return (gaussian_filtered_img,)


@app.cell
def _(gaussian_filtered_img, img, output_img_frame, plt):

    ## Apply Laplacian on this
    # output dimension should be nrow-frow/stride +1 = 307-5+1 =303
    #Laplacian filter is 3x3
    #so 303 - 3+1 = 301
    def combined_detect(img, output_img, func):
        # Safe list conversion for the transition from cv2 -> gaussian -> laplacian  
        pixels = img    
        stride = 1
        f_size = 3 
        output_dim = 301
    
        for r in range(output_dim):
            for c in range(output_dim):
                start_y = r * stride
                start_x = c * stride
    # filter region
                reg = []
                for ky in range(f_size):
                    row = []
                    for kx in range(f_size):
                        i = (start_y + ky)
                        j = (start_x + kx)
                        val = pixels[i][j]  # Your existing syntax
                        row.append(val)
                    reg.append(row)
                output_img[r][c] = func(reg)
        return output_img

    empty_out_com = output_img_frame(img, output_dim=301, f_size=3)
    combined_detect_out = combined_detect(gaussian_filtered_img, empty_out_com, laplacian)

    plt.imshow(combined_detect_out, cmap='gray')
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
