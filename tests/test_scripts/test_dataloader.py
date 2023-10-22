import os
import sys
import tensorflow as tf
import numpy as np

from bird_classifier.data_loader import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.test_scripts.test_config import Config


url_dict = {
    0:'https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg',
    1:'https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg',
    2:'https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg',
    3:'https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg',
    4:'https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg'
}


class TestClass:
    """Test class for dataloader."""

    def test_dataloader_initialisation(self):
        self.data_loader = DataLoader(Config, url_dict)
        assert len(self.data_loader) == 5


    def test_blank_img(self):
        self.data_loader = DataLoader(Config, url_dict)
        blank_img = self.data_loader.blank_image()
        assert blank_img.shape == (*Config.img_size, 3)


    def test_preprocess_image(self):
        self.data_loader = DataLoader(Config, url_dict)
        blank_img = np.ones((500, 500, 3), dtype=np.float32) * 255
        preprocessed_img = self.data_loader.preprocess_image(blank_img)
        assert preprocessed_img.shape == (*Config.img_size, 3)
        if Config.img_devide_by_255:
            assert np.max(preprocessed_img) <= 1.0
            assert np.min(preprocessed_img) >= 0.0
        else:
            assert np.max(preprocessed_img) <= 255
            assert np.min(preprocessed_img) >= 0.0


    def test_get_image(self):
        self.data_loader = DataLoader(Config, url_dict)
        image, load_status = self.data_loader.get_image(self.data_loader.img_url_dict[0])
        assert isinstance(image, tf.Tensor)
        assert isinstance(load_status, tf.Tensor)
        image = image.numpy()
        load_status = load_status.numpy()
        assert image.shape == (*Config.img_size, 3)
        if Config.img_devide_by_255:
            assert np.max(image) <= 1.0
            assert np.min(image) >= 0.0
        else:
            assert np.max(image) <= 255
            assert np.min(image) >= 0.0
        assert load_status == 1


    def test_load_sample(self):
        self.data_loader = DataLoader(Config, url_dict)
        (idx, image, load_status) = self.data_loader.load_sample(tf.constant(0))
        assert isinstance(idx, tf.Tensor)
        assert isinstance(image, tf.Tensor)
        assert isinstance(load_status, tf.Tensor)
        idx = idx.numpy()
        image = image.numpy()
        load_status = load_status.numpy()
        assert idx == 0
        assert image.shape == (*Config.img_size, 3)
        if Config.img_devide_by_255:
            assert np.max(image) <= 1.0
            assert np.min(image) >= 0.0
        else:
            assert np.max(image) <= 255
            assert np.min(image) >= 0.0
        assert load_status == 1


    def test_iteration(self):
        self.data_loader = DataLoader(Config, url_dict)
        (sample_ids, images, load_status) = next(self.data_loader)
        assert isinstance(sample_ids, tf.Tensor)
        assert isinstance(images, tf.Tensor)
        assert isinstance(load_status, tf.Tensor)
        idx = sample_ids.numpy()[0]
        image = images.numpy()[0]
        load_status = load_status.numpy()[0]
        assert idx == 0
        assert image.shape == (*Config.img_size, 3)
        if Config.img_devide_by_255:
            assert np.max(image) <= 1.0
            assert np.min(image) >= 0.0
        else:
            assert np.max(image) <= 255
            assert np.min(image) >= 0.0
        assert load_status == 1
