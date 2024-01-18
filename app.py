from pathlib import Path
import gradio as gr
import fastai.vision.all as fv
from iirwi import IIRWI

EXTRACTOR_NAME = Path('extractor.pt')
STORAGE_NAME = Path('storage.pkl')

iirwi = IIRWI.from_filenames(EXTRACTOR_NAME, STORAGE_NAME)

def predict(input_image):
     img = fv.PILImage.create(input_image)
     return iirwi.process(img)
    
iface = gr.Interface(
    fn=predict,
    inputs='image',
    outputs='image',
)

iface.launch(debug=True)