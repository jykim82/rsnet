# Water Meter OCR with OpenCV and CNN

This folder contains a minimal example showing how to build a
mechanical water meter digit recognizer using OpenCV for image
preprocessing and a small CNN for classification.

The example assumes that you have a dataset of cropped digit images.
You can train the CNN using this dataset and then apply it to new
photos of the meter.

## Quick start

1. Prepare digit images sorted into directories by label.
2. Run `python water_meter_ocr.py --train /path/to/dataset` to train
the model.
3. Run `python water_meter_ocr.py --predict /path/to/image.jpg` to
read digits from a photo.

See the comments in `water_meter_ocr.py` for more details.
