import torch
import unittest
import os
from models.rope_vit import RoPEViT
from models.vit import ViT
from lightning_modules.lightning_liere_image_classification import LiereImageClassification
import ipdb

# Test hyperparameters
LEARNING_RATE = 1E-5
MODEL_SIZE = "tiny"
PATH_SIZE = 4
IMSIZE = 32

# Directory to save outputs
OUTPUT_DIR = "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_number_of_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total number of parameters: ", total_params)

def forward_resolutions(model, input_dimensionality):
    assert input_dimensionality in [2, 3]

    image = torch.randn(2, 3, IMSIZE, IMSIZE) if input_dimensionality == 2 else torch.randn(2, 3, IMSIZE, IMSIZE, IMSIZE)
    out = model(image)

    print_number_of_parameters(model)
    return out

def save_tensor(tensor, filename):
    torch.save(tensor, filename)
    print(f"Tensor saved to {filename}")

def load_tensor(filename):
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        return None

class TestForward(unittest.TestCase):

    def compare_with_saved(self, tensor, filename):
        saved_tensor = load_tensor(filename)
        if saved_tensor is None:
            print("No saved tensor found. Saving current tensor for future comparison.")
            save_tensor(tensor, filename)
        else:
            self.assertTrue(torch.equal(tensor, saved_tensor), f"Tensors do not match. Saved tensor: {filename}")

    def test_absolute_2d(self):
        input_dimensionality = 2
        model = LiereImageClassification(learning_rate=LEARNING_RATE, imsize=IMSIZE, model_architecture="absolute", 
                                         model_size=MODEL_SIZE, patch_size=PATH_SIZE, input_dimensionality=input_dimensionality)
        out = forward_resolutions(model, input_dimensionality)
        self.compare_with_saved(out, os.path.join(OUTPUT_DIR, "absolute_2d.pt"))

    def test_absolute_3d(self):
        input_dimensionality = 3
        model = LiereImageClassification(learning_rate=LEARNING_RATE, imsize=IMSIZE, model_architecture="absolute", 
                                         model_size=MODEL_SIZE, patch_size=PATH_SIZE, input_dimensionality=input_dimensionality)
        out = forward_resolutions(model, input_dimensionality)
        self.compare_with_saved(out, os.path.join(OUTPUT_DIR, "absolute_3d.pt"))

    def test_liere_2d(self):
        input_dimensionality = 2
        model = LiereImageClassification(learning_rate=LEARNING_RATE, imsize=IMSIZE, model_architecture="liere", 
                                         model_size=MODEL_SIZE, patch_size=PATH_SIZE, input_dimensionality=input_dimensionality)
        out = forward_resolutions(model, input_dimensionality)
        self.compare_with_saved(out, os.path.join(OUTPUT_DIR, "liere_2d.pt"))

    def test_liere_3d(self):
        input_dimensionality = 3
        model = LiereImageClassification(learning_rate=LEARNING_RATE, imsize=IMSIZE, model_architecture="liere", 
                                         model_size=MODEL_SIZE, patch_size=PATH_SIZE, input_dimensionality=input_dimensionality)
        out = forward_resolutions(model, input_dimensionality)
        self.compare_with_saved(out, os.path.join(OUTPUT_DIR, "liere_3d.pt"))

    def test_naver_2d(self):
        input_dimensionality = 2
        model = LiereImageClassification(learning_rate=LEARNING_RATE, imsize=IMSIZE, model_architecture="rope_mixed", 
                                         model_size=MODEL_SIZE, patch_size=PATH_SIZE, input_dimensionality=input_dimensionality)
        out = forward_resolutions(model, input_dimensionality)
        self.compare_with_saved(out, os.path.join(OUTPUT_DIR, "naver_2d.pt"))

    def test_naver_3d(self):
        input_dimensionality = 3
        model = LiereImageClassification(learning_rate=LEARNING_RATE, imsize=IMSIZE, model_architecture="rope_mixed", 
                                         model_size=MODEL_SIZE, patch_size=PATH_SIZE, input_dimensionality=input_dimensionality)
        out = forward_resolutions(model, input_dimensionality)
        self.compare_with_saved(out, os.path.join(OUTPUT_DIR, "naver_3d.pt"))

    def test_visionllama_2d(self):
        input_dimensionality = 2
        model = LiereImageClassification(learning_rate=LEARNING_RATE, imsize=IMSIZE, model_architecture="visionllama", 
                                         model_size=MODEL_SIZE, patch_size=PATH_SIZE, input_dimensionality=input_dimensionality)
        out = forward_resolutions(model, input_dimensionality)
        self.compare_with_saved(out, os.path.join(OUTPUT_DIR, "visionllama_2d.pt"))

    def test_visionllama_3d(self):
        input_dimensionality = 3
        model = LiereImageClassification(learning_rate=LEARNING_RATE, imsize=IMSIZE, model_architecture="visionllama", 
                                         model_size=MODEL_SIZE, patch_size=PATH_SIZE, input_dimensionality=input_dimensionality)
        out = forward_resolutions(model, input_dimensionality)
        self.compare_with_saved(out, os.path.join(OUTPUT_DIR, "visionllama_3d.pt"))
