#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM with Attention Model for Vancomycin TDM Prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import os
from base_model import BaseVancomycinModel

class LSTMAttentionVancomycin(BaseVancomycinModel):
    """
    LSTM with Attention 메커니즘을 사용한 반코마이신 농도 예측
    """
    
    def __init__(self):
        super().__init__("LSTM with Attention")
        # 딥러닝 모델은 MinMaxScaler가 더 효과적
        self.scaler = MinMaxScaler()
        self.attention_weights = None
        
    def build_model(self, input_shape, output_dim=1):
        """
        LSTM + Attention 모델 구조 생성
        """
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm_out = layers.LSTM(64, return_sequences=True, dropout=0.2)(lstm_out)
        
        # Attention mechanism
        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=64,
            dropout=0.2
        )(lstm_out, lstm_out)
        
        # Add & Norm
        attention_output = layers.LayerNormalization()(layers.Add()([lstm_out, attention]))
        
        # Global pooling
        pooled = layers.GlobalAveragePooling1D()(attention_output)
        
        # Dense layers
        dense = layers.Dense(128, activation='relu')(pooled)
        dense = layers.Dropout(0.3)(dense)
        dense = layers.Dense(64, activation='relu')(dense)
        dense = layers.Dropout(0.3)(dense)
        
        # Output layer
        outputs = layers.Dense(output_dim)(dense)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def prepare_sequence_data(self, X, sequence_length=5):
        """
        시계열 데이터 준비 (LSTM input shape를 위해)
        """
        n_samples, n_features = X.shape
        
        # 시퀀스 데이터 생성
        if n_samples < sequence_length:
            # 샘플이 부족한 경우 패딩
            pad_length = sequence_length - n_samples
            X_padded = np.vstack([np.zeros((pad_length, n_features)), X])
            X_seq = X_padded.reshape(1, sequence_length, n_features)
            return np.repeat(X_seq, n_samples, axis=0)
        else:
            # 슬라이딩 윈도우 방식으로 시퀀스 생성
            X_seq = []
            for i in range(n_samples - sequence_length + 1):
                X_seq.append(X[i:i+sequence_length])
            return np.array(X_seq)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, sequence_length=5):
        """
        LSTM + Attention 모델 학습
        """
        # 데이터 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 시퀀스 데이터 준비
        X_train_seq = self.prepare_sequence_data(X_train_scaled, sequence_length)
        
        # y_train 조정 (sequence로 인해 길이가 달라질 수 있음)
        if len(X_train_seq) < len(y_train):
            y_train_adj = y_train[sequence_length-1:]
        else:
            y_train_adj = y_train[:len(X_train_seq)]
        
        # 모델 생성
        input_shape = (sequence_length, X_train.shape[1])
        self.model = self.build_model(input_shape)
        
        # 모델 컴파일
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Validation 데이터 준비
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq = self.prepare_sequence_data(X_val_scaled, sequence_length)
            if len(X_val_seq) < len(y_val):
                y_val_adj = y_val[sequence_length-1:]
            else:
                y_val_adj = y_val[:len(X_val_seq)]
            validation_data = (X_val_seq, y_val_adj)
        
        # 모델 학습
        print(f"Training {self.model_name}...")
        history = self.model.fit(
            X_train_seq, y_train_adj,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Validation 성능 확인
        if validation_data:
            val_loss = self.model.evaluate(validation_data[0], validation_data[1], verbose=0)
            print(f"Validation Loss: {val_loss[0]:.4f}")
            print(f"Validation MAE: {val_loss[1]:.4f}")
        
        return self, history
    
    def predict(self, X, sequence_length=5):
        """
        예측 수행
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_seq = self.prepare_sequence_data(X_scaled, sequence_length)
        
        predictions = self.model.predict(X_seq)
        
        # 시퀀스로 인한 길이 조정
        if len(predictions) < len(X):
            # 앞부분 패딩
            pad_length = len(X) - len(predictions)
            predictions = np.concatenate([predictions[:1]] * pad_length + [predictions])
        
        return predictions.flatten()[:len(X)]
    
    def plot_training_history(self, history, save_path=None):
        """
        학습 이력 시각화
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Model Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Model MAE')
        ax2.legend()
        ax2.grid(True)
        
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
        
        # 모델 저장
        model_path = path.replace('.pkl', '_model.h5')
        self.model.save(model_path)
        
        # Scaler 저장
        import joblib
        scaler_path = path.replace('.pkl', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    @classmethod
    def load_model(cls, path):
        """
        저장된 모델 로드
        """
        import joblib
        
        instance = cls()
        
        # 모델 로드
        model_path = path.replace('.pkl', '_model.h5')
        instance.model = keras.models.load_model(model_path)
        
        # Scaler 로드
        scaler_path = path.replace('.pkl', '_scaler.pkl')
        instance.scaler = joblib.load(scaler_path)
        
        instance.is_trained = True
        
        return instance


# 사용 예시를 위한 독립 실행 함수
if __name__ == "__main__":
    print("LSTM with Attention Vancomycin Model Test")
    print("-" * 50)
    
    # 데이터 로드 (예시)
    # data = pd.read_csv('final_tdm_data_processed.csv')
    # lstm_model = LSTMAttentionVancomycin()
    # lstm_model.train(X_train, y_train, X_val, y_val)