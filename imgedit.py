import cv2 as cv
import numpy as np

# Read the given image
img = cv.imread('cv_tutorial/data/mandril_color.tif')

if img is not None:
    rotateList = [cv.ROTATE_90_CLOCKWISE,
                  cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
    rotateState = [90, 180, 270]
    rotateCode = 0

    gamma = 1

    while True:

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_tran = cv.equalizeHist(img_gray)
        cv.intensity_transform.gammaCorrection(img_gray, img_tran, gamma)
        # Rotate the given image
        img_rotate = cv.rotate(img, rotateList[rotateCode])

        # change channel to merge
        img_gray = np.repeat(img_gray[:, :, np.newaxis], 3, -1)
        img_tran = np.repeat(img_tran[:, :, np.newaxis], 3, -1)

        # Show all images
        info = f'Gamma : {gamma}, Rotation: {rotateState[rotateCode]}'

        # merge all images and show
        merge = np.hstack((img, img_rotate, img_gray, img_tran))

        cv.putText(merge, info, (10, 25),
                   cv.FONT_HERSHEY_DUPLEX, 0.6, 255, thickness=2)
        cv.putText(merge, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, 0)

        cv.imshow(
            'Original | Image Rotation | Gray Scale | Contrast', merge)

        # Process the key event
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == ord('+') or key == ord('='):
            gamma -= 0.1
        elif key == ord('-') or key == ord('_'):
            gamma += 0.1
        elif key == ord('r'):
            rotateCode += 1
            if rotateCode > 2:
                rotateCode = 0
        elif key == ord('d'):
            key = cv.waitKey(1000)
            if key == ord('1'):
                cv.imwrite('img.jpg', img)
            elif key == ord('2'):
                cv.imwrite('img_rotate.jpg', img_rotate)
            elif key == ord('3'):
                cv.imwrite('img_gray.jpg', img_gray)
            elif key == ord('4'):
                cv.imwrite('img_tran.jpg', img_tran)
