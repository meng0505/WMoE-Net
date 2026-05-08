from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/fuse/wmoenet_wdca3_moe_top2.yaml")
    model.train(
        data="datasets/ship.yaml",
        ch=6,
        imgsz=1024,
        epochs=500,
        batch=4,
        close_mosaic=10,
        workers=8,
        device="0",
        optimizer="SGD",
        patience=50,
        amp=True,
        cache=False,
        project="runs/train",
        name="WMoE-Net",
        resume=False,
        fraction=1,
        seed=2,
    )
