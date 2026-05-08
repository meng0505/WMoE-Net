from ultralytics import YOLO











if __name__ == "__main__":



    model = YOLO("weights/best.pt")



    model.predict(



        source="datasets/images/val",



        save=True,



        imgsz=640,



        conf=0.25,



        iou=0.45,



        show=False,



        project="runs/predict",



        name="WMoE-Net",



        save_txt=False,



        save_conf=True,



        save_crop=False,



        show_labels=True,



        show_conf=True,



        vid_stride=1,



        line_width=2,



        visualize=False,



        augment=False,



        agnostic_nms=False,



        retina_masks=False,



    )



