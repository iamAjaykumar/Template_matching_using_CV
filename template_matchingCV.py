import cv2 as cv
import numpy as np
rubik=cv.imread("dog.jpg")
gray_rubik=cv.cvtColor(rubik,cv.COLOR_BGR2GRAY)
template=cv.imread("dog copy.jpg",0)
originalTemp=cv.imread("dog copy.jpg",1)
result=cv.matchTemplate(gray_rubik,template,cv.TM_CCOEFF_NORMED)
w,h=template.shape[: : -1]
threshold=0.8
loc=np.where(result>=threshold)
for pt in  zip(*loc[::-1]):
    cv.rectangle(rubik,pt,(pt[0]+w,pt[1]+h),(0,0,255),1)

#cv.imshow("Gray",gray_rubik)
#cv.imshow("template",template)
#cv.imshow("result",result)
cv.imshow("original template",originalTemp)
cv.imshow("detection",rubik)
cv.waitKey(0)
cv.destroyAllWindows()