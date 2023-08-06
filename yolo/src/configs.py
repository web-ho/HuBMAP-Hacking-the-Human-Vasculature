class config():

    seed = 42
    imgsz=512
    epochs=100
    patience=50
    batch=4
    workers=2
    device='0'
    optimizer='Adam'
    lr0=0.0001
    pretrained=True
    iou=0.1
    max_det=1000
    augment=True
    name='yolov8n-seg-100epoch'
    model = "yolov8x-seg.pt"
    #rest of hyperparameteres were default

    DATA_PATH = '/data/'
    TRAIN_PATH = DATA_PATH + 'train/'
    POLYGON_PATH = DATA_PATH + 'polygons.jsonl'
    TEST_PATH = DATA_PATH + "test/"
    
    DEST_DIR = DATA_PATH + 'yolo/'

    data="/yolo/dataset.yaml"
