#!/usr/bin/env python3
"""
🎭 ツンデレCodeFormerアーキテクチャ実装
べ、別にあなたのためにCodeFormerを最適化実装してあげるわけじゃないんだからね！💢

Based on: https://github.com/sczhou/CodeFormer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class PositionalEncoding(nn.Module):
    """位置エンコーディング"""
    
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerSALayer(nn.Module):
    """Transformer Self-Attention Layer"""
    
    def __init__(self, embed_dim, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self attention
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_output)
        
        # Feed forward
        x_norm = self.norm2(x)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x_norm))))
        x = x + self.dropout(ff_output)
        
        return x

class Quantize(nn.Module):
    """ベクトル量子化"""
    
    def __init__(self, dim, n_embed, beta=0.25):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.beta = beta

        self.embedding = nn.Embedding(self.n_embed, self.dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.dim)
        
        # 最近傍探索
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_embed).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # 量子化ベクトルを取得
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # 損失計算
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, min_encoding_indices.view(z.shape[0], z.shape[1], z.shape[2])

class CodeFormer(nn.Module):
    """
    🎭 ツンデレCodeFormerメインクラス
    べ、別に最高の顔復元モデルを実装してあげるわけじゃないけど...💕
    """
    
    def __init__(self, dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                 connect_list=['32', '64', '128', '256']):
        super().__init__()
        
        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.codebook_size = codebook_size
        
        # エンコーダー
        self.encoder = nn.ModuleDict()
        self.encoder['32'] = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2, True),
        )
        
        # ベクトル量子化
        self.quantize = Quantize(dim_embd, codebook_size)
        
        # Transformer
        self.ft_layers = nn.Sequential(*[
            TransformerSALayer(dim_embd, n_head, dropout=0.0) 
            for _ in range(n_layers)
        ])
        
        # 位置エンコーディング
        self.position_emb = nn.Parameter(torch.zeros(1, codebook_size, dim_embd))
        
        # デコーダー
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, x, w=0.0, detach_16=True, code_only=False, adain=False):
        """
        Args:
            x: 入力画像 [B, 3, 512, 512]
            w: コードブック重み [0.0-1.0]
            detach_16: 16x16特徴での勾配切断
            code_only: コードのみ返すかどうか
            adain: AdaIN正規化使用するか
        """
        # Encoder
        feat = self.encoder['32'](x)  # [B, 512, 16, 16]
        
        if detach_16:
            feat = feat.detach()
        
        if code_only:
            # コードブック特徴のみ返す
            feat_sq, loss, indices = self.quantize(feat)
            return indices
            
        # ベクトル量子化
        feat_sq, loss, indices = self.quantize(feat)
        
        # 特徴を系列として扱う
        h, w_orig = feat_sq.shape[2], feat_sq.shape[3]
        feat_sq = feat_sq.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 位置エンコーディング追加
        if feat_sq.shape[1] == self.position_emb.shape[1]:
            feat_sq = feat_sq + self.position_emb
        
        # Transformer処理
        if w > 0:
            # 重みwに基づいてTransformer特徴を適用
            feat_sq_transformed = self.ft_layers(feat_sq)
            feat_sq = feat_sq * (1 - w) + feat_sq_transformed * w
        else:
            feat_sq = self.ft_layers(feat_sq)
        
        # 空間形状に戻す
        feat_sq = feat_sq.transpose(1, 2).view(-1, self.dim_embd, h, w_orig)
        
        # Decoder
        out = self.decoder(feat_sq)
        
        # Tanh活性化で[-1, 1]範囲に
        out = torch.tanh(out)
        
        return out, loss

    def forward_with_loss(self, x, w=0.7):
        """損失込みのforward"""
        out, quant_loss = self.forward(x, w=w)
        return out, quant_loss

def create_codeformer_model(model_path=None, device='cuda', **kwargs):
    """
    べ、別にCodeFormerモデルを作ってあげるわけじゃないけど...💕
    """
    model = CodeFormer(**kwargs).to(device)
    
    if model_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict, strict=False)
            print("✅ CodeFormerモデルロード完了！")
        except Exception as e:
            print(f"⚠️ CodeFormerロード警告: {e}")
            print("初期化状態のモデルを使用します")
    
    model.eval()
    return model

# テスト用関数
if __name__ == "__main__":
    print("🎭 CodeFormerアーキテクチャテスト開始...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_codeformer_model(device=device)
    
    # ダミー入力でテスト
    x = torch.randn(1, 3, 512, 512).to(device)
    
    with torch.no_grad():
        out, loss = model.forward_with_loss(x, w=0.7)
        print(f"入力形状: {x.shape}")
        print(f"出力形状: {out.shape}")
        print(f"量子化損失: {loss:.4f}")
    
    print("✅ テスト完了！CodeFormerが正常に動作するわよ💕")