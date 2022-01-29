import cv2
import numpy as np

def main(img_path, next_img, output_name, epochs = 20):
    # read image & template
    img = cv2.imread(img_path)
    next_i = cv2.imread(next_img)

    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Click on the screen and press any key for end process")
    p0 = get_points(img)
    for i in range(len(p0)):
        cv2.circle(img, (int(p0[i][0,0]), int(p0[i][0,1])), 3, (255,0,0), -1)
    save_image(img, f"{output_name}_bluepoint", "data/")
    
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(img)

    # calculate optical flow
    frame_gray = cv2.cvtColor(next_i, cv2.COLOR_BGR2GRAY)

    for epoch in range(1,epochs, 2):
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, epoch, 0.03))
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if epoch==1:
            good_new = p1[st==1]
            good_old = p0[st==1]
        else:
            good_new = p1[st==1]

        # draw the tracks   
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()       
            cv2.line(next_i, (int(a), int(b)), (int(c), int(d)), (255,0,255), 2)
            if epoch==1:
                cv2.circle(next_i, (int(c), int(d)), 3, (255,0,0), -1) #old
            elif epoch==epochs-1:
                cv2.circle(next_i, (int(a), int(b)), 3, (0,255,0), -1) # new
            else:
                cv2.circle(next_i, (int(a), int(b)), 3, (0,0,255), -1) # new
            good_old = good_new

    # save
    save_image(next_i, f"{output_name}_track", "data/")


def save_image(fig, figname, report_path):
    cv2.imwrite(f'{report_path}/{figname}.jpg', fig)

def get_points(img):
    data = dict()
    data['img'] = img.copy()
    data['points'] = list()

    # create a window
    cv2.namedWindow("Image", 0)

    # resize the Window
    h, w, dim = img.shape
    print(f"Img height, width:({h}, {w})")
    cv2.resizeWindow('Image', w, h)

    # show image of window
    cv2.imshow("Image", img)

    # Use the mouse to return the value, the data is all stored in the dict.
    cv2.setMouseCallback("Image", mouse_handler, data)

    # release opencv resource
    cv2.waitKey()
    cv2.destroyAllWindows()

    return np.array(data['points'])

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        # marked point position
        cv2.circle(data['img'], (x,y), 1, (255,0,0), 5, 16)

        # title
        cv2.imshow("Image", data['img'])

        # show (x,y) and store it to list
        print(f"get point: [x,y] = [{x}, {y}]")
        data["points"].append(np.array([[x,y]], dtype='float32'))


if __name__=="__main__":
    main("data/Cup0.jpg", "data/Cup1.jpg", "Cup")
    main("data/Pillow0.jpg", "data/Pillow1.jpg", "Pillow")