import os
from dataclasses import dataclass
from typing import Union

import numpy as np
import supervision as sv
import torch
from autodistill.classification import ClassificationBaseModel
from autodistill.core.embedding_model import EmbeddingModel
from autodistill.core.embedding_ontology import (  # noqa: E501
    EmbeddingOntology,
    compare_embeddings,
)
from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image
from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EvaCLIP(ClassificationBaseModel, EmbeddingModel):
    ontology: Union[EmbeddingOntology, CaptionOntology]

    def __init__(self, ontology: Union[EmbeddingOntology, CaptionOntology]):
        self.ontology = ontology
        model_name_or_path = "BAAI/EVA-CLIP-8B"
        preprocess = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", device=DEVICE
        )

        self.clip_model = (
            AutoModel.from_pretrained(
                model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
            )
            .to("cuda")
            .eval()
        )
        self.clip_preprocess = preprocess
        self.tokenize = CLIPTokenizer.from_pretrained(model_name_or_path)

        # if Ontology is EmbeddingOntologyImage, then run process
        if isinstance(self.ontology, EmbeddingOntology):
            self.ontology.process(self)

        # get ontology class name
        self.ontology_type = self.ontology.__class__.__name__

    def embed_image(self, input: str) -> np.ndarray:
        image = load_image(input, return_format="PIL")
        image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)

        return image_features.cpu().numpy()

    def embed_text(self, input: str) -> np.ndarray:
        return (
            self.clip_model.encode_text(self.tokenize([input]).to(DEVICE)).cpu().numpy()
        )

    def predict(self, input: str) -> sv.Classifications:
        image = load_image(input, return_format="PIL")
        image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)

        if isinstance(self.ontology, EmbeddingOntology):
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)

                return compare_embeddings(
                    image_features.cpu().numpy(), self.ontology.embeddingMap.values()
                )
        else:
            labels = self.ontology.prompts()

            text = self.tokenize(labels).to(DEVICE)

            with torch.no_grad():
                logits_per_image, _ = self.clip_model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            return sv.Classifications(
                class_id=np.array([i for i in range(len(labels))]),
                confidence=np.array(probs).flatten(),
            )
