#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost Model for Vancomycin TDM Prediction
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import os
from base_model import BaseVancomycinModel

class XGBoostVancomycin(BaseVancomycinModel):
    """
    XGBoost 모델을 사용한 반코마이신 농도 예측
    """
    
    def __init__(self):
        super().__init__("XGBoost")
        self.feature_importance = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize_hyperparameters=True):
        """
        XGBoost 모델 학습
        """
        # 데이터 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if optimize_hyperparameters:
            print(f"Optimizing hyperparameters for {self.model_name}...")
            
            # 하이퍼파라미터 그리드
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.3, 0.5]
            }
            
            # GridSearchCV
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                xgb_model, 
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
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            # Early stopping을 위한 설정
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=eval_set,
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
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
    
    def get_feature_importance(self, feature_names, importance_type='gain', top_n=15):
        """
        특성 중요도 반환
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        
        # XGBoost의 다양한 중요도 타입
        if importance_type in ['gain', 'weight', 'cover']:
            importance = self.model.get_booster().get_score(importance_type=importance_type)
            # Feature 이름 매핑
            importance_dict = {}
            for i, fname in enumerate(feature_names):
                feat_name = f'f{i}'
                if feat_name in importance:
                    importance_dict[fname] = importance[feat_name]
                else:
                    importance_dict[fname] = 0
            
            importance_df = pd.DataFrame(
                list(importance_dict.items()), 
                columns=['feature', 'importance']
            )
        else:
            # 기본 feature importance 사용
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            })
        
        return importance_df.sort_values('importance', ascending=False).head(top_n)
    
    def plot_tree(self, tree_index=0, save_path=None):
        """
        개별 트리 시각화
        """
        import matplotlib.pyplot as plt
        from xgboost import plot_tree
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(self.model, num_trees=tree_index, ax=ax)
        plt.title(f'XGBoost Tree {tree_index}')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, feature_names, importance_type='gain', top_n=15, save_path=None):
        """
        특성 중요도 시각화
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        importance_df = self.get_feature_importance(feature_names, importance_type, top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'{self.model_name} - Feature Importance ({importance_type}) - Top {top_n}')
        plt.xlabel(f'Importance ({importance_type})')
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
    print("XGBoost Vancomycin Model Test")
    print("-" * 50)
    
    # 데이터 로드 (예시)
    # data = pd.read_csv('final_tdm_data_processed.csv')
    # xgb_model = XGBoostVancomycin()
    # xgb_model.train(X_train, y_train, X_val, y_val)