#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Support Vector Regression Model for Vancomycin TDM Prediction
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
import joblib
import os
from base_model import BaseVancomycinModel

class SVRVancomycin(BaseVancomycinModel):
    """
    Support Vector Regression 모델을 사용한 반코마이신 농도 예측
    """
    
    def __init__(self):
        super().__init__("Support Vector Regression")
        # SVR은 RobustScaler가 더 효과적
        self.scaler = RobustScaler()
        
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize_hyperparameters=True):
        """
        SVR 모델 학습
        """
        # 데이터 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if optimize_hyperparameters:
            print(f"Optimizing hyperparameters for {self.model_name}...")
            
            # 하이퍼파라미터 그리드
            param_grid = {
                'kernel': ['rbf', 'linear', 'poly'],
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5, 1],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
            # GridSearchCV
            svr = SVR(max_iter=10000)
            grid_search = GridSearchCV(
                svr, 
                param_grid, 
                cv=5, 
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # 기본 파라미터로 학습
            self.model = SVR(
                kernel='rbf',
                C=10,
                epsilon=0.1,
                gamma='scale',
                max_iter=10000
            )
            self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Validation 성능 확인
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_pred_val = self.model.predict(X_val_scaled)
            val_metrics = self.calculate_metrics(y_val, y_pred_val)
            print(f"Validation R²: {val_metrics['r2']:.4f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
            
        return self
    
    def get_support_vectors_info(self):
        """
        서포트 벡터 정보 반환
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        n_support = len(self.model.support_)
        support_ratio = n_support / len(self.model.support_vectors_) * 100
        
        info = {
            'n_support_vectors': n_support,
            'support_ratio': support_ratio,
            'kernel': self.model.kernel,
            'C': self.model.C,
            'epsilon': self.model.epsilon,
            'gamma': self.model.gamma
        }
        
        return info
    
    def plot_svr_analysis(self, X_test, y_test, save_path=None):
        """
        SVR 분석 시각화
        """
        import matplotlib.pyplot as plt
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 예측 vs 실제
        axes[0].scatter(y_test, y_pred, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Predictions vs Actual')
        
        # 2. 잔차 분포
        residuals = y_test - y_pred
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        # 3. 오차율 분포
        error_rate = np.abs(residuals) / (y_test + 1e-8) * 100
        axes[2].hist(error_rate, bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(x=20, color='r', linestyle='--', label='20% threshold')
        axes[2].set_xlabel('Error Rate (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Error Rate Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_model(self, path):
        """
        모델 저장
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.model_name
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path):
        """
        저장된 모델 로드
        """
        model_data = joblib.load(path)
        
        instance = cls()
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.is_trained = True
        
        return instance


# 사용 예시를 위한 독립 실행 함수
if __name__ == "__main__":
    print("SVR Vancomycin Model Test")
    print("-" * 50)
    
    # 데이터 로드 (예시)
    # data = pd.read_csv('final_tdm_data_processed.csv')
    # svr_model = SVRVancomycin()
    # svr_model.train(X_train, y_train, X_val, y_val)