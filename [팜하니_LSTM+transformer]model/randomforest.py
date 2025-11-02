#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Forest Model for Vancomycin TDM Prediction
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os
from base_model import BaseVancomycinModel

class RandomForestVancomycin(BaseVancomycinModel):
    """
    Random Forest 모델을 사용한 반코마이신 농도 예측
    """
    
    def __init__(self):
        super().__init__("Random Forest")
        self.feature_importance = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize_hyperparameters=True):
        """
        Random Forest 모델 학습
        """
        # 데이터 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if optimize_hyperparameters:
            print(f"Optimizing hyperparameters for {self.model_name}...")
            
            # 하이퍼파라미터 그리드
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            
            # GridSearchCV
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, 
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
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Feature importance 계산
        self.feature_importance = self.model.feature_importances_
        self.is_trained = True
        
        # Validation 성능 확인
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_pred_val = self.model.predict(X_val_scaled)
            val_metrics = self.calculate_metrics(y_val, y_pred_val)
            print(f"Validation R²: {val_metrics['r2']:.4f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
            
        return self
    
    def get_feature_importance(self, feature_names, top_n=15):
        """
        특성 중요도 반환
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        
        # 특성 중요도 정렬
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, feature_names, top_n=15, save_path=None):
        """
        특성 중요도 시각화
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        importance_df = self.get_feature_importance(feature_names, top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'{self.model_name} - Feature Importance (Top {top_n})')
        plt.xlabel('Importance')
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
            'feature_importance': self.feature_importance,
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
        instance.feature_importance = model_data['feature_importance']
        instance.is_trained = True
        
        return instance


# 사용 예시를 위한 독립 실행 함수
if __name__ == "__main__":
    import pandas as pd
    
    print("Random Forest Vancomycin Model Test")
    print("-" * 50)
    
    # 데이터 로드 (예시)
    # data = pd.read_csv('final_tdm_data_processed.csv')
    # rf_model = RandomForestVancomycin()
    # rf_model.train(X_train, y_train, X_val, y_val)