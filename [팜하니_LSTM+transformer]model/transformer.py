#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer Model for Vancomycin TDM Prediction
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

class TransformerBlock(layers.Layer):
    """
    Transformer 블록 구현
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    """
    위치 인코딩 레이어
    """
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # 짝수 인덱스에는 sin 적용
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        # 홀수 인덱스에는 cos 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerVancomycin(BaseVancomycinModel):
    """
    Transformer 모델을 사용한 반코마이신 농도 예측
    """
    
    def __init__(self):
        super().__init__("Transformer")
        self.scaler = MinMaxScaler()
        self.attention_weights = None
        
    def build_model(self, input_shape, embed_dim=64, num_heads=4, ff_dim=128, 
                    num_transformer_blocks=2, mlp_units=[128], dropout=0.1):
        """
        Transformer 모델 구조 생성
        """
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # 입력 투영
        x = layers.Dense(embed_dim)(x)
        
        # 위치 인코딩 추가
        x = PositionalEncoding(input_shape[0], embed_dim)(x)
        
        # Transformer 블록들
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # MLP head
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
        
        # 출력층
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def prepare_sequence_data(self, X, sequence_length=10):
        """
        시계열 데이터 준비 (Transformer input shape를 위해)
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
              epochs=100, batch_size=32, sequence_length=10):
        """
        Transformer 모델 학습
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
        self.model = self.build_model(
            input_shape,
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            num_transformer_blocks=2,
            mlp_units=[128, 64],
            dropout=0.1
        )
        
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
        print(f"Model architecture:")
        print(f"- Sequence length: {sequence_length}")
        print(f"- Embedding dimension: 64")
        print(f"- Number of heads: 4")
        print(f"- Transformer blocks: 2")
        
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
    
    def predict(self, X, sequence_length=10):
        """
        예측 수행
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_seq = self.prepare_sequence_data(X_scaled, sequence_length)
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        # 시퀀스로 인한 길이 조정
        if len(predictions) < len(X):
            # 앞부분 패딩
            pad_length = len(X) - len(predictions)
            predictions = np.concatenate([predictions[:1]] * pad_length + [predictions])
        
        return predictions.flatten()[:len(X)]
    
    def get_attention_weights(self, X, sequence_length=10, layer_index=0):
        """
        Attention weights 추출
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Attention 레이어 찾기
        attention_layers = [l for l in self.model.layers if 'multi_head_attention' in l.name]
        
        if layer_index >= len(attention_layers):
            raise ValueError(f"Layer index {layer_index} out of range")
        
        # 중간 모델 생성
        attention_model = Model(
            inputs=self.model.input,
            outputs=attention_layers[layer_index].output
        )
        
        # 데이터 준비
        X_scaled = self.scaler.transform(X)
        X_seq = self.prepare_sequence_data(X_scaled, sequence_length)
        
        # Attention weights 추출
        attention_output = attention_model.predict(X_seq, verbose=0)
        
        return attention_output
    
    def plot_attention_heatmap(self, X, sample_idx=0, sequence_length=10, save_path=None):
        """
        Attention weights 히트맵 시각화
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        attention_weights = self.get_attention_weights(X, sequence_length)
        
        # 단일 샘플의 attention weights
        sample_attention = attention_weights[sample_idx]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sample_attention, cmap='Blues', cbar=True)
        plt.title(f'Transformer Attention Weights - Sample {sample_idx}')
        plt.xlabel('Sequence Position')
        plt.ylabel('Sequence Position')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
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
    print("Transformer Vancomycin Model Test")
    print("-" * 50)
    
    # 데이터 로드 (예시)
    # data = pd.read_csv('final_tdm_data_processed.csv')
    # transformer_model = TransformerVancomycin()
    # transformer_model.train(X_train, y_train, X_val, y_val)