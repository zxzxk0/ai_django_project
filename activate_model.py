from face_predict import face_prediction

import matplotlib.colors as colors


answer , red, green, blue = face_prediction('D:/AI/notebook/Face_Expression/best_model.pkl',r'C:\Users\user\Downloads\opencv-4.x\opencv-4.x\data\haarcascades\haarcascade_frontalface_default.xml')

r = (red/(red+green+blue))*255
g = (green/(red+green+blue))*255
b = (blue/(red+green+blue))*255
print(answer,red,green,blue)


r, g, b = r/255, g/255, b/255
rgb_color = (r, g, b)
hex_color = colors.to_hex(rgb_color)

print(hex_color)  # 출력 결과: #7cd40b 