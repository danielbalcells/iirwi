from .feature_extractor import FeatureExtractor
from .storage import ImgFeatureStorage
from .retrieval import CroppingImgRetriever
from .plot import MemePlotter

class IIRWI:
    def __init__(self, retriever):
        self.retriever = retriever
        self.plotter = MemePlotter()

    @classmethod
    def from_filenames(cls, extractor_filename, storage_filename):
        retriever = CroppingImgRetriever.from_filenames(extractor_filename, storage_filename)
        return cls(retriever)
    
    def process(self, input_img):
        similar_img, crop_coords = self.retriever.process(input_img)
        meme = self.plotter.plot(input_img, similar_img, crop_coords)
        return meme