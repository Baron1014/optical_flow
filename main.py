import cv2
import numpy as np

def main(img_path, next_img, output_name):
     # read image & template
    img = cv2.imread(img_path)
    next_i = cv2.imread(next_img)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Click on the screen and press any key for end process")
    p0 = get_points(img)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(img)

    # calculate optical flow
    frame_gray = cv2.cvtColor(next_i, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks   
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()       
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255,0,255), 2)
        old = cv2.circle(next_i, (int(c), int(d)), 3, (255,0,0), -1)
        next = cv2.circle(next_i, (int(a), int(b)), 3, (0,255,0), -1)
    output = cv2.add(mask, next, old)

    # save
    save_image(output, output_name, "data/")


def save_image(fig, figname, report_path):
    cv2.imwrite(f'{report_path}/{figname}.jpg', fig)

def get_points(img):
    data = dict()
    data['img'] = img.copy()
    data['points'] = list()

    # 建立一個window
    cv2.namedWindow("Image", 0)

    # 改變Window為適當大小
    h, w, dim = img.shape
    print(f"Img height, width:({h}, {w})")
    cv2.resizeWindow('Image', w, h)

    # 顯示圖片至window
    cv2.imshow("Image", img)

    # 利用滑鼠回傳值，資料皆保存於data dict中
    cv2.setMouseCallback("Image", mouse_handler, data)

    # 按下任意鍵釋放opencv資源
    cv2.waitKey()
    cv2.destroyAllWindows()

    return np.array(data['points'])

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 標記點位置
        cv2.circle(data['img'], (x,y), 1, (255,0,0), 5, 16)

        #改變顯示內容
        cv2.imshow("Image", data['img'])

        # 顯示(x,y)並儲存到list中
        print(f"get point: [x,y] = [{x}, {y}]")
        data["points"].append(np.array([[x,y]], dtype='float32'))


if __name__=="__main__":
    main("data/Cup0.jpg", "data/Cup1.jpg", "Cup_track")
    main("data/Pillow0.jpg", "data/Pillow1.jpg", "Pillow_track")