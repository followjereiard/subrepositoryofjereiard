#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Script for Vancomycin TDM Prediction Models
모든 모델을 통합하여 실행하는 메인 스크립트
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# 모델 import
from randomforest import RandomForestVancomycin
from svr import SVRVancomycin
from xgboost_model import XGBoostVancomycin
from lstm_attention import LSTMAttentionVancomycin
from transformer import TransformerVancomycin

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class VancomycinTDMManager:
    """
    반코마이신 TDM 예측 모델 통합 관리 클래스
    """
    
    def __init__(self, data_path='final_tdm_data_processed.csv'):
        """
        초기화 및 데이터 로드
        """
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        self.load_data()
        
        # 출력 디렉토리 생성
        self.create_output_directories()
    
    def create_output_directories(self):
        """
        결과 저장을 위한 디렉토리 생성
        """
        directories = [
            'results',
            'results/models',
            'results/plots',
            'results/reports'
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """
        데이터 로드 및 기본 전처리
        """
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # 기본 정보 출력
        print(f"Data shape: {self.df.shape}")
        print(f"Number of patients: {self.df['PatientID'].nunique()}")
        
        # 데이터 검증
        self.validate_data()
    
    def validate_data(self):
        """
        데이터 검증 및 전처리
        """
        # 필수 컬럼 확인
        required_columns = ['PatientID', 'Age_Numeric', 'Weight', 'Height', 'BMI']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        # 결측치 처리
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_columns] = self.df[numeric_columns].fillna(self.df[numeric_columns].median())
    
    def prepare_target_data(self, target):
        """
        타겟별 데이터 준비
        """
        print(f"\nPreparing data for {target} prediction...")
        
        # 타겟별 컬럼 매핑
        target_column_map = {
            'Peak': 'EstimatedPeak_Day_',
            'Trough': 'EstimatedTrough_Day_',
            'AUC': 'AUC_Day_',
            'AUC_MIC': 'AUC_MIC_Day_'
        }
        
        target_prefix = target_column_map.get(target)
        if not target_prefix:
            print(f"Unknown target: {target}")
            return None
        
        # 타겟 컬럼 찾기
        target_columns = [col for col in self.df.columns if col.startswith(target_prefix)]
        
        if not target_columns:
            print(f"No target columns found for {target}")
            return None
        
        # 가장 최근 값 사용 (또는 평균값)
        self.df[f'{target}_value'] = self.df[target_columns].mean(axis=1, skipna=True)
        
        # 유효한 타겟 값이 있는 행만 필터링
        valid_data = self.df[self.df[f'{target}_value'].notna()].copy()
        
        print(f"Valid samples for {target}: {len(valid_data)}")
        
        if len(valid_data) < 50:
            print(f"Insufficient data for {target} (< 50 samples)")
            return None
        
        return valid_data
    
    def get_features_for_target(self, target):
        """
        타겟별 특성 선택
        """
        # 기본 특성
        base_features = [
            'Age_Numeric', 'Weight', 'Height', 'BMI',
            'Is_Pediatric', 'Is_Obese', 'Is_CKD', 'Is_Normal_Adult',
            'TDM_Runcount', 'Cumulative_DOSE', 'Total_Treatment_Days'
        ]
        
        # 타겟별 추가 특성
        if target in ['Peak', 'Trough']:
            # 최근 용량 정보
            dose_features = [col for col in self.df.columns if 'Dose' in col and 'Day' in col][:5]
            base_features.extend(dose_features)
            
            # 최근 신기능 정보
            cr_features = [col for col in self.df.columns if 'Cr_Day' in col][:5]
            base_features.extend(cr_features)
            
        elif target in ['AUC', 'AUC_MIC']:
            # 클리어런스 정보
            clearance_features = [col for col in self.df.columns if 'Clearence_Day' in col][:5]
            base_features.extend(clearance_features)
            
            # 반감기 정보
            halflife_features = [col for col in self.df.columns if 'Halflife_Day' in col][:5]
            base_features.extend(halflife_features)
        
        # 실제 존재하는 특성만 필터링
        available_features = [f for f in base_features if f in self.df.columns]
        
        return available_features
    
    def train_models_for_target(self, target):
        """
        특정 타겟에 대해 모든 모델 학습
        """
        # 데이터 준비
        data = self.prepare_target_data(target)
        if data is None:
            return
        
        # 특성 선택
        features = self.get_features_for_target(target)
        X = data[features].values
        y = data[f'{target}_value'].values
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )
        
        print(f"\nTraining models for {target}...")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        results = {}
        
        # 1. Random Forest
        try:
            print("\n1. Training Random Forest...")
            rf_model = RandomForestVancomycin()
            rf_model.train(X_train, y_train, X_val, y_val, optimize_hyperparameters=False)
            
            # 평가
            metrics, y_pred = rf_model.evaluate(X_test, y_test)
            results['RandomForest'] = metrics
            
            # 시각화
            rf_model.plot_predictions(y_test, y_pred, target, 
                                    f'results/plots/rf_{target}_predictions.png')
            rf_model.plot_feature_importance(features, top_n=15,
                                           save_path=f'results/plots/rf_{target}_importance.png')
            
            # 모델 저장
            rf_model.save_model(f'results/models/rf_{target}_model.pkl')
            
        except Exception as e:
            print(f"Error training Random Forest: {e}")
        
        # 2. XGBoost
        try:
            print("\n2. Training XGBoost...")
            xgb_model = XGBoostVancomycin()
            xgb_model.train(X_train, y_train, X_val, y_val, optimize_hyperparameters=False)
            
            # 평가
            metrics, y_pred = xgb_model.evaluate(X_test, y_test)
            results['XGBoost'] = metrics
            
            # 시각화
            xgb_model.plot_predictions(y_test, y_pred, target,
                                     f'results/plots/xgb_{target}_predictions.png')
            xgb_model.plot_feature_importance(features, importance_type='gain', top_n=15,
                                            save_path=f'results/plots/xgb_{target}_importance.png')
            
            # 모델 저장
            xgb_model.save_model(f'results/models/xgb_{target}_model.pkl')
            
        except Exception as e:
            print(f"Error training XGBoost: {e}")
        
        # 3. SVR
        try:
            print("\n3. Training SVR...")
            svr_model = SVRVancomycin()
            svr_model.train(X_train, y_train, X_val, y_val, optimize_hyperparameters=False)
            
            # 평가
            metrics, y_pred = svr_model.evaluate(X_test, y_test)
            results['SVR'] = metrics
            
            # 시각화
            svr_model.plot_predictions(y_test, y_pred, target,
                                     f'results/plots/svr_{target}_predictions.png')
            svr_model.plot_svr_analysis(X_test, y_test,
                                       f'results/plots/svr_{target}_analysis.png')
            
            # 모델 저장
            svr_model.save_model(f'results/models/svr_{target}_model.pkl')
            
        except Exception as e:
            print(f"Error training SVR: {e}")
        
        # 4. LSTM with Attention
        try:
            print("\n4. Training LSTM with Attention...")
            lstm_model = LSTMAttentionVancomycin()
            lstm_model, history = lstm_model.train(
                X_train, y_train, X_val, y_val,
                epochs=50, batch_size=32, sequence_length=5
            )
            
            # 평가
            metrics, y_pred = lstm_model.evaluate(X_test, y_test)
            results['LSTM_Attention'] = metrics
            
            # 시각화
            lstm_model.plot_predictions(y_test, y_pred, target,
                                      f'results/plots/lstm_{target}_predictions.png')
            lstm_model.plot_training_history(history,
                                           f'results/plots/lstm_{target}_history.png')
            
            # 모델 저장
            lstm_model.save_model(f'results/models/lstm_{target}_model.pkl')
            
        except Exception as e:
            print(f"Error training LSTM: {e}")
        
        # 5. Transformer
        try:
            print("\n5. Training Transformer...")
            transformer_model = TransformerVancomycin()
            transformer_model, history = transformer_model.train(
                X_train, y_train, X_val, y_val,
                epochs=50, batch_size=32, sequence_length=10
            )
            
            # 평가
            metrics, y_pred = transformer_model.evaluate(X_test, y_test)
            results['Transformer'] = metrics
            
            # 시각화
            transformer_model.plot_predictions(y_test, y_pred, target,
                                             f'results/plots/transformer_{target}_predictions.png')
            transformer_model.plot_training_history(history,
                                                  f'results/plots/transformer_{target}_history.png')
            
            # 모델 저장
            transformer_model.save_model(f'results/models/transformer_{target}_model.pkl')
            
        except Exception as e:
            print(f"Error training Transformer: {e}")
        
        # 결과 저장
        self.results[target] = results
        
        # 결과 출력
        self.print_target_results(target, results)
    
    def print_target_results(self, target, results):
        """
        타겟별 결과 출력
        """
        print(f"\n{'='*60}")
        print(f"Results for {target} Prediction")
        print(f"{'='*60}")
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  Within 20%: {metrics['within_20_percent']:.2f}%")
    
    def create_comparison_plot(self):
        """
        모든 모델의 성능 비교 플롯 생성
        """
        if not self.results:
            print("No results to plot")
            return
        
        # 데이터 준비
        models = []
        targets = []
        r2_scores = []
        rmse_scores = []
        
        for target, target_results in self.results.items():
            for model, metrics in target_results.items():
                models.append(model)
                targets.append(target)
                r2_scores.append(metrics['r2'])
                rmse_scores.append(metrics['rmse'])
        
        comparison_df = pd.DataFrame({
            'Model': models,
            'Target': targets,
            'R2': r2_scores,
            'RMSE': rmse_scores
        })
        
        # 플롯 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R2 score comparison
        pivot_r2 = comparison_df.pivot(index='Model', columns='Target', values='R2')
        pivot_r2.plot(kind='bar', ax=ax1)
        ax1.set_title('Model Performance Comparison - R² Score')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        ax1.legend(title='Target')
        ax1.grid(True, alpha=0.3)
        
        # RMSE comparison
        pivot_rmse = comparison_df.pivot(index='Model', columns='Target', values='RMSE')
        pivot_rmse.plot(kind='bar', ax=ax2)
        ax2.set_title('Model Performance Comparison - RMSE')
        ax2.set_ylabel('RMSE')
        ax2.legend(title='Target')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/model_comparison.png', dpi=300)
        plt.close()
    
    def save_results_summary(self):
        """
        결과 요약 저장
        """
        # JSON 형식으로 저장
        with open('results/reports/results_summary.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # CSV 형식으로 저장
        rows = []
        for target, target_results in self.results.items():
            for model, metrics in target_results.items():
                row = {
                    'Target': target,
                    'Model': model,
                    **metrics
                }
                rows.append(row)
        
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv('results/reports/results_summary.csv', index=False)
        
        print("\nResults saved to results/reports/")
    
    def run(self):
        """
        전체 파이프라인 실행
        """
        print("\n" + "="*60)
        print("Vancomycin TDM Prediction Models")
        print("="*60)
        
        # 타겟별로 모델 학습
        targets = ['Peak', 'Trough', 'AUC', 'AUC_MIC']
        
        for target in targets:
            self.train_models_for_target(target)
        
        # 비교 플롯 생성
        self.create_comparison_plot()
        
        # 결과 저장
        self.save_results_summary()
        
        print("\n" + "="*60)
        print("All models trained successfully!")
        print(f"Results saved in 'results' directory")
        print("="*60)


def main():
    """
    메인 실행 함수
    """
    # 매니저 초기화 및 실행
    manager = VancomycinTDMManager('final_tdm_data_processed.csv')
    manager.run()


if __name__ == "__main__":
    main()