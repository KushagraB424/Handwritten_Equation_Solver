import os
import glob
import cv2
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import classification_report
from joblib import dump
from features import extract_features
import time

LABELS = ['0','1','2','3','4','5','6','7','8','9','plus','minus','mul','div','equals','x','y','z']


def augment_image(img, enable_full=False):
    """Apply augmentation to increase training data diversity.
    
    Args:
        img: Input grayscale image
        enable_full: If True, applies all augmentations (slow). 
                     If False, only applies essential ones (fast).
    """
    augmented = [img]  # Always include original
    
    if not enable_full:
        # Fast mode: Only apply small rotation (most important for handwriting)
        for angle in [-3, 3]:
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
            augmented.append(rotated)
        return augmented  # 3 images total (1 original + 2 rotations)
    
    # Full augmentation mode (slower but better accuracy)
    # Rotations
    for angle in [-5, -2, 2, 5]:
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)
        augmented.append(rotated)
    
    # Translations
    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)
        augmented.append(shifted)
    
    # Light noise (helps with generalization)
    noisy = img.copy()
    noise = np.random.randn(*img.shape) * 8
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)
    
    return augmented  # 14 images total


def load_dataset(root_dir: str, augment: bool = False, full_augment: bool = False):
    """Load dataset with optional augmentation.
    
    Args:
        root_dir: Path to data directory
        augment: Whether to apply augmentation
        full_augment: If True, applies all augmentations (slower)
    """
    X, y = [], []
    total_images = 0
    
    print(f"Loading dataset from {root_dir}")
    for lbl in LABELS:
        cls_dir = os.path.join(root_dir, lbl)
        if not os.path.isdir(cls_dir):
            print(f"Warning: No directory found for label '{lbl}'")
            continue
        
        label_count = 0
        for fp in glob.glob(os.path.join(cls_dir, '*.png')) + glob.glob(os.path.join(cls_dir, '*.jpg')):
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Apply augmentation if requested
            if augment:
                images = augment_image(img, enable_full=full_augment)
            else:
                images = [img]
            
            for aug_img in images:
                feat = extract_features(aug_img)
                X.append(feat)
                y.append(lbl)
            
            label_count += len(images)
        
        total_images += label_count
        print(f"  Loaded {label_count} samples for '{lbl}'")
    
    if not X:
        raise RuntimeError(f"No training data found in {root_dir}. Expected folders: {LABELS}")
    
    print(f"\nTotal samples: {total_images}")
    return np.array(X), np.array(y)


def train_and_save(
    models_dir: str,
    data_dir: str,
    quick: bool = False,
    model_filter: str = 'all',
    search: str = 'random',
    train_fraction: float = 1.0,
    n_iter: int = 15,
    verbose: int = 1,
    augment: bool = False,
    full_augment: bool = False,
):
    """Train models with optimized parameters.
    
    Args:
        models_dir: Output directory for models
        data_dir: Input data directory
        quick: Use minimal search (fastest)
        model_filter: Which models to train ('all', 'svm', 'knn', 'rbf')
        search: Search strategy ('grid', 'random')
        train_fraction: Fraction of data to use (for testing)
        n_iter: Number of iterations for RandomizedSearchCV
        verbose: Verbosity level
        augment: Apply data augmentation
        full_augment: Apply full augmentation (slower but better)
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Load dataset
    aug_msg = " with " + ("full" if full_augment else "fast") + " augmentation" if augment else ""
    print(f"\n{'='*60}")
    print(f"Loading dataset from {data_dir}{aug_msg}")
    print(f"{'='*60}")
    
    X, y = load_dataset(data_dir, augment=augment, full_augment=full_augment)
    
    # Optional subsampling
    if train_fraction < 1.0:
        print(f"\nSubsampling to {train_fraction*100}% of data...")
        _, X, _, y = train_test_split(X, y, train_size=train_fraction, random_state=42, stratify=y)
    
    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain size: {len(Xtr)}, Test size: {len(Xte)}")
    
    # Cross-validation setup
    n_splits = 2 if quick else 3
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # -----------------  Linear SVM (FASTEST) -----------------
    if model_filter in ('all', 'svm'):
        print(f"\n{'='*60}")
        print("Training LinearSVC (fastest, good for linearly separable data)")
        print(f"{'='*60}")
        start = time.time()
        
        svm_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(random_state=42, max_iter=5000))
        ])
        
        if quick:
            param_grid = {
                'clf__C': [0.5, 1.0],
                'clf__loss': ['squared_hinge']
            }
        else:
            param_grid = {
                'clf__C': [0.1, 0.5, 1, 2, 5],
                'clf__loss': ['hinge', 'squared_hinge'],
                'clf__class_weight': [None, 'balanced']
            }
        
        if search == 'random' and not quick:
            searcher = RandomizedSearchCV(
                svm_pipe, param_grid, n_iter=min(n_iter, 10), 
                cv=cv, scoring='f1_weighted', n_jobs=2, 
                random_state=42, verbose=verbose
            )
        else:
            searcher = GridSearchCV(
                svm_pipe, param_grid, cv=cv, 
                scoring='f1_weighted', n_jobs=2, verbose=verbose
            )
        
        searcher.fit(Xtr, ytr)
        best_svm = searcher.best_estimator_
        yp = best_svm.predict(Xte)
        
        print(f"\nBest params: {searcher.best_params_}")
        print(f"CV score: {searcher.best_score_:.4f}")
        print(f"\nTest set performance:")
        print(classification_report(yte, yp, zero_division=0))
        
        dump(best_svm, os.path.join(models_dir, 'svm.joblib'))
        print(f"✓ Saved to {models_dir}/svm.joblib")
        print(f"Time: {time.time()-start:.1f}s")
    
    # -----------------  KNN -----------------
    if model_filter in ('all', 'knn'):
        print(f"\n{'='*60}")
        print("Training KNN (fast, no training phase)")
        print(f"{'='*60}")
        start = time.time()
        
        knn_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier())
        ])
        
        if quick:
            param_grid = {
                'clf__n_neighbors': [3, 5],
                'clf__weights': ['distance']
            }
        else:
            param_grid = {
                'clf__n_neighbors': [1, 3, 5, 7],
                'clf__weights': ['uniform', 'distance'],
                'clf__p': [1, 2]
            }
        
        if search == 'random' and not quick:
            searcher = RandomizedSearchCV(
                knn_pipe, param_grid, n_iter=min(n_iter, 12), 
                cv=cv, scoring='f1_weighted', n_jobs=2, 
                random_state=42, verbose=verbose
            )
        else:
            searcher = GridSearchCV(
                knn_pipe, param_grid, cv=cv, 
                scoring='f1_weighted', n_jobs=2, verbose=verbose
            )
        
        searcher.fit(Xtr, ytr)
        best_knn = searcher.best_estimator_
        yp = best_knn.predict(Xte)
        
        print(f"\nBest params: {searcher.best_params_}")
        print(f"CV score: {searcher.best_score_:.4f}")
        print(f"\nTest set performance:")
        print(classification_report(yte, yp, zero_division=0))
        
        dump(best_knn, os.path.join(models_dir, 'knn.joblib'))
        print(f"✓ Saved to {models_dir}/knn.joblib")
        print(f"Time: {time.time()-start:.1f}s")
    
    # -----------------  RBF SVM (SLOWEST, BEST ACCURACY) -----------------
    if model_filter in ('all', 'rbf'):
        print(f"\n{'='*60}")
        print("Training RBF SVM (slow but often most accurate)")
        print(f"{'='*60}")
        start = time.time()
        
        rbf_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', random_state=42))
        ])
        
        if quick:
            param_grid = {
                'clf__C': [1, 5],
                'clf__gamma': ['scale', 0.01]
            }
        else:
            param_grid = {
                'clf__C': [0.5, 1, 2, 5, 10],
                'clf__gamma': ['scale', 'auto', 0.01, 0.1],
                'clf__class_weight': [None, 'balanced']
            }
        
        # Always use RandomizedSearchCV for RBF (much faster)
        searcher = RandomizedSearchCV(
            rbf_pipe, param_grid, n_iter=min(n_iter, 15), 
            cv=cv, scoring='f1_weighted', n_jobs=2, 
            random_state=42, verbose=verbose
        )
        
        searcher.fit(Xtr, ytr)
        best_rbf = searcher.best_estimator_
        yp = best_rbf.predict(Xte)
        
        print(f"\nBest params: {searcher.best_params_}")
        print(f"CV score: {searcher.best_score_:.4f}")
        print(f"\nTest set performance:")
        print(classification_report(yte, yp, zero_division=0))
        
        dump(best_rbf, os.path.join(models_dir, 'svm_rbf.joblib'))
        print(f"✓ Saved to {models_dir}/svm_rbf.joblib")
        print(f"Time: {time.time()-start:.1f}s")
    
    print(f"\n{'='*60}")
    print(f"✓ Training complete! Models saved to {models_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Train handwriting recognition models')
    ap.add_argument('--data', required=True, help='Path to dataset root (folders per class label)')
    ap.add_argument('--out', default='models', help='Output models directory')
    ap.add_argument('--quick', action='store_true', help='Fast training with minimal search (2-3 min)')
    ap.add_argument('--model', default='all', choices=['all', 'svm', 'knn', 'rbf'], 
                    help='Which model to train')
    ap.add_argument('--search', default='random', choices=['grid', 'random'], 
                    help='Search strategy (random is much faster)')
    ap.add_argument('--train-fraction', type=float, default=1.0, 
                    help='Use fraction of data (0<f<=1) for faster testing')
    ap.add_argument('--n-iter', type=int, default=15, 
                    help='RandomizedSearchCV iterations (default: 15)')
    ap.add_argument('--verbose', type=int, default=1, help='Verbosity level (0-2)')
    ap.add_argument('--augment', action='store_true', 
                    help='Apply data augmentation (recommended!)')
    ap.add_argument('--full-augment', action='store_true',
                    help='Apply full augmentation (slower, best accuracy)')
    
    args = ap.parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Mode: {'QUICK (fastest)' if args.quick else 'THOROUGH'}")
    print(f"Augmentation: {'FULL (slow, best)' if args.full_augment else 'FAST' if args.augment else 'OFF'}")
    print(f"Search: {args.search.upper()}")
    print(f"Models: {args.model.upper()}")
    print("="*60 + "\n")
    
    train_and_save(
        args.out, args.data, 
        quick=args.quick,
        model_filter=args.model,
        search=args.search,
        train_fraction=args.train_fraction,
        n_iter=args.n_iter,
        verbose=args.verbose,
        augment=args.augment,
        full_augment=args.full_augment,
    )