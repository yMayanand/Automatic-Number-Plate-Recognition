# object-detection
🕵️ Implementation of yolo like network from scatch on ANPR(Automatic Number Plate Detection) Dataset
Dataset used for this project is obtained from kaggle 
## Dataset 
- [Indian vehicle license plate dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset)
Model is built in **YOLO** style final layer output 49 predictions as it receives final feature map of size 7x7.
To tackle Automatic Number Plate Recognition we first train an Objec Detection Network to find Numbers plates and then after detecting plates we train another network to recognize the characters on Number plate.
