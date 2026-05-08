from ultralytics import YOLO











if __name__ == "__main__":



    model = YOLO("weights/best.pt")



    metrics = model.val(



        data="datasets/ship.yaml",



        ch=6,



        imgsz=1024,



    )



    print(metrics.box.map75)



