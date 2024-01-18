from feature_extractor import FeatureExtractor
from storage import ImgFeatureStorage
from retrieval import CroppingImgRetriever
from plot import MemePlotter

class IIRWI:
    def __init__(self, extractor, storage):
        self.extractor = extractor
        self.storage = storage
        self.retriever = CroppingImgRetriever(extractor, storage)
        self.plotter = MemePlotter()

    @classmethod
    def from_filenames(cls, extractor_filename, storage_filename):
        extractor = FeatureExtractor.load(extractor_filename)
        storage = ImgFeatureStorage.load(storage_filename)
        return cls(extractor, storage)
    
    def process(self, input_img):
        similar_img, crop_coords = self.retriever.process(input_img)
        meme = self.plotter.plot(input_img, similar_img, crop_coords)
        return meme