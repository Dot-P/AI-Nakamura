import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
from scipy import linalg
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from mylib.data_io import CSVBasedDataset


# Inception Score を計算するクラス
class InceptionScore:

    # 認識処理
    def __get_pred(self, dataloader):
        pred = None
        self.classifier.to(self.device)
        for X in tqdm(dataloader):
            with torch.inference_mode():
                if self.preprocess is not None:
                    X = self.preprocess(X)
                X = X.to(self.device)
                Y = F.softmax(self.classifier(X), dim=1)
            if pred is None:
                pred = Y.to('cpu').detach().numpy()
            else:
                pred = np.concatenate([pred, Y.to('cpu').detach().numpy()], axis=0)
        self.classifier.to('cpu')
        return pred

    # コンストラクタ
    def __init__(
            self,
            device, # 使用するデバイス
            img_range = [0, 1], # 画像の画素値の範囲（feature_extractor_model=None の場合はデフォルト値を使用すること）
            img_transform = None, # 認識器に画像を入力する際の前処理（feature_extractor_model=None の場合は無視される）
            classifier_model = None, # 認識器として使用するモデル（None の場合は inception-v3 を使用）
            batch_size = 128, # バッチサイズ（一度に処理する画像の枚数）
    ):
        self.device = device
        self.img_range = img_range
        self.batch_size = batch_size

        # 認識器の用意
        if classifier_model is None:
            self.inception_flag = True
            self.classifier = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
            self.preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(299, antialias=True),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.inception_flag = False
            self.classifier = classifier_model
            self.preprocess = img_transform
        self.classifier.eval()

    # Inception Score の計算を実行
    def __call__(
            self,
            image_list, # 画像ファイルのリスト
            image_item, # 画像リストにおける読み出し対象列の項目名
            image_dir,  # 画像ファイルの保存先ディレクトリ
            n_splits=1  # 画像集合の分割数
    ):
        # 画像リストのデータセット化
        dataset = CSVBasedDataset(
            filename = image_list,
            items = [image_item],
            dtypes = ['image'],
            dirname = image_dir,
            img_range = self.img_range,
        )
        dataset_size = len(dataset)

        # 画像リストのデータローダー化
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False)

        # 画像を認識
        pred = self.__get_pred(dataloader)

        # inception score を計算する
        M = dataset_size // n_splits
        split_scores = []
        for k in range(n_splits):
            scores = []
            part = pred[k*M:(k+1)*M,:]
            py = np.mean(part, axis=0)
            for i in range(part.shape[0]):
                scores.append(entropy(part[i,:], py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)


# Frechet Inception Distance を計算するクラス
class FID:

    # 特徴量抽出
    def __get_features(self, dataloader):
        features = None
        self.extractor.to(self.device)
        for X in tqdm(dataloader):
            with torch.inference_mode():
                if self.preprocess is not None:
                    X = self.preprocess(X)
                X = X.to(self.device)
                Y = torch.squeeze(self.extractor(X)['feature']) if self.inception_flag else self.extractor(X)
            if features is None:
                features = Y.to('cpu').detach().numpy()
            else:
                features = np.concatenate([features, Y.to('cpu').detach().numpy()], axis=0)
        self.extractor.to('cpu')
        return features

    # コンストラクタ
    def __init__(self,
                 device, # 使用するデバイス
                 img_range = [0, 1], # 画像の画素値の範囲（feature_extractor_model=None の場合はデフォルト値を使用すること）
                 img_transform = None, # 特徴抽出器に画像を入力する際の前処理（feature_extractor_model=None の場合は無視される）
                 feature_extractor_model = None, # 特徴抽出器として使用するモデル（None の場合は inception-v3 を使用）
                 batch_size = 128, # バッチサイズ（一度に処理する画像の枚数）
    ):
        self.device = device
        self.img_range = img_range
        self.batch_size = batch_size

        # 特徴抽出器の用意
        if feature_extractor_model is None:
            self.inception_flag = True
            inception = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
            self.extractor = create_feature_extractor(inception, {'avgpool': 'feature'})
            self.preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(299, antialias=True),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.inception_flag = False
            self.extractor = feature_extractor_model
            self.preprocess = img_transform
        self.extractor.eval()

    # FID計算を実行
    def __call__(self,
                 real_image_list, # Real画像ファイルのリスト
                 real_image_item, # Real画像リストにおける読み出し対象列の項目名
                 real_image_dir,  # Real画像ファイルの保存先ディレクトリ
                 fake_image_list, # Fake画像ファイルのリスト
                 fake_image_item, # Fake画像リストにおける読み出し対象列の項目名
                 fake_image_dir,  # Fake画像ファイルの保存先ディレクトリ
    ):
        # Real画像リストのデータセット化
        real_dataset = CSVBasedDataset(
            filename = real_image_list,
            items = [real_image_item],
            dtypes = ['image'],
            dirname = real_image_dir,
            img_range = self.img_range,
        )

        # Fake画像リストのデータセット化
        fake_dataset = CSVBasedDataset(
            filename = fake_image_list,
            items = [fake_image_item],
            dtypes = ['image'],
            dirname = fake_image_dir,
            img_range = self.img_range,
        )

        # 画像リストのデータローダー化
        real_dataloader = DataLoader(real_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False)
        fake_dataloader = DataLoader(fake_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False)

        # Real画像から特徴量を抽出
        real_features = self.__get_features(real_dataloader)
        fake_features = self.__get_features(fake_dataloader)

        # 画像の特徴量の平均と分散を求める
        real_m = np.mean(real_features, axis=0)
        fake_m = np.mean(fake_features, axis=0)
        real_v = np.cov(real_features, rowvar=False)
        fake_v = np.cov(fake_features, rowvar=False)

        # FIDを求める
        diff = real_m - fake_m
        covmean, _ = linalg.sqrtm(real_v.dot(fake_v), disp=False)
        tr_covmean = np.trace(covmean)
        score = diff.dot(diff) + np.trace(real_v) + np.trace(fake_v) - 2 * tr_covmean

        return score
