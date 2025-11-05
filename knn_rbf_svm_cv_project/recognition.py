import os
from collections import Counter

import numpy as np
from joblib import load

from features import extract_features

LABELS = ['0','1','2','3','4','5','6','7','8','9','plus','minus','mul','div','equals','x','y','z']


class Recognizer:
    def __init__(self, models_dir: str = 'models', prefer: str = 'svm'):
        self.models = {}
        for name in ['svm', 'knn', 'svm_rbf']:
            fp = os.path.join(models_dir, f'{name}.joblib')
            if os.path.exists(fp):
                self.models[name] = load(fp)
        if not self.models:
            raise FileNotFoundError(f"No models found in {models_dir}. Train with train_models.py")
        prefer_key = 'svm_rbf' if prefer == 'rbf' else prefer
        self.preferred_model = prefer_key if prefer_key in self.models else next(iter(self.models))
        self.model_name = self.preferred_model

    def predict_one(self, roi_gray: np.ndarray) -> str:
        feat = extract_features(roi_gray)
        feat = feat.reshape(1, -1)
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = str(model.predict(feat)[0])

        if not predictions:
            raise RuntimeError("No models available for prediction.")

        counter = Counter(predictions.values())
        top_label, top_count = counter.most_common(1)[0]
        tied_labels = [label for label, count in counter.items() if count == top_count]

        if len(tied_labels) > 1 and self.preferred_model in predictions:
            preferred_label = predictions[self.preferred_model]
            if preferred_label in tied_labels:
                top_label = preferred_label
            else:
                top_label = tied_labels[0]

        mapping = {
            'plus': '+',
            'minus': '-',
            'mul': '*',
            'div': '/',
            'equals': '=',
        }
        out = mapping.get(top_label, top_label)
        return out