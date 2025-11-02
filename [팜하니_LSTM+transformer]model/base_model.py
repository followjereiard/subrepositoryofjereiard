#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base Model Class for Vancomycin TDM Prediction
공통 기능을 포함한 기본 모델 클래스
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BaseVancomycinModel:
    """
    반코마이신 TDM 예측을 위한 기본 모델 클래스
    """
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = None
        self.target_column = None
        
    def prepare_features(self, data, target):
        """
        모델 학습을 위한 특성 준비
        """
        # 기본 특성
        base_features = [
            'Age_Numeric', 'Weight', 'Height', 'BMI',
            'Is_Pediatric', 'Is_Obese', 'Is_CKD', 'Is_Normal_Adult',
            'TDM_Runcount', 'Cumulative_DOSE', 'Total_Treatment_Days'
        ]
        
        # Target에 따른 추가 특성
        if target in ['Peak', 'Trough']:
            # 최근 용량 정보 추가
            dose_features = [col for col in data.columns if 'Dose' in col and 'Day' in col]
            recent_doses = dose_features[-3:] if len(dose_features) >= 3 else dose_features
            base_features.extend(recent_doses)
            
            # 최근 신기능 정보
            cr_features = [col for col in data.columns if 'Cr_Day' in col]
            recent_cr = cr_features[-3:] if len(cr_features) >= 3 else cr_features
            base_features.extend(recent_cr)
            
        elif target in ['AUC', 'AUC_MIC']:
            # AUC 예측을 위한 추가 특성
            clearance_features = [col for col in data.columns if 'Clearence_Day' in col]
            recent_clearance = clearance_features[-3:] if len(clearance_features) >= 3 else clearance_features
            base_features.extend(recent_clearance)
            
            # Half-life 정보
            halflife_features = [col for col in data.columns if 'Halflife_Day' in col]
            recent_halflife = halflife_features[-3:] if len(halflife_features) >= 3 else halflife_features
            base_features.extend(recent_halflife)
        
        # 실제 존재하는 특성만 필터링
        available_features = [f for f in base_features if f in data.columns]
        
        return available_features
    
    def calculate_metrics(self, y_true, y_pred):
        """
        예측 성능 지표 계산
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # 20% 오차 이내 예측 비율
        within_20_percent = np.sum(np.abs((y_true - y_pred) / (y_true + 1e-8)) <= 0.2) / len(y_true) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'within_20_percent': within_20_percent
        }
    
    def plot_predictions(self, y_true, y_pred, target, save_path=None):
        """
        예측 결과 시각화
        """
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'{self.model_name} - {target} Prediction')
        
        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(f'Predicted {target}')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        모델 학습 (각 하위 클래스에서 구현)
        """
        raise NotImplementedError("Subclass must implement train method")
    
    def predict(self, X):
        """
        예측 수행
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """
        모델 평가
        """
        y_pred = self.predict(X)
        metrics = self.calculate_metrics(y, y_pred)
        return metrics, y_pred