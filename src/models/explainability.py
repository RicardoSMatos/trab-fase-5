"""
Sistema de explicabilidade para modelos de ML usando SHAP.

Este módulo fornece ferramentas para entender quais features
influenciam as predições do modelo LSTM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP não está instalado. Feature importance não estará disponível.")

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Classe para explicar predições do modelo usando SHAP.
    
    SHAP (SHapley Additive exPlanations) calcula a contribuição
    de cada feature para a predição final.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Inicializa o explainer.
        
        Args:
            model: Modelo treinado (Keras model)
            feature_names: Lista com nomes das features
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP não está instalado. Instale com: pip install shap"
            )
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        logger.info(f"ModelExplainer inicializado com {len(feature_names)} features")
    
    def create_explainer(
        self, 
        background_data: np.ndarray, 
        max_samples: int = 100
    ):
        """
        Cria o explainer SHAP.
        
        Args:
            background_data: Dados de background para SHAP (amostra representativa)
            max_samples: Número máximo de amostras de background
        """
        logger.info("Criando SHAP explainer...")
        
        # Limitar amostras de background para performance
        if len(background_data) > max_samples:
            indices = np.random.choice(
                len(background_data), 
                max_samples, 
                replace=False
            )
            background_data = background_data[indices]
        
        # Criar explainer para redes neurais
        try:
            self.explainer = shap.DeepExplainer(
                self.model,
                background_data
            )
            logger.info("SHAP DeepExplainer criado com sucesso")
        except Exception as e:
            logger.warning(f"Erro ao criar DeepExplainer: {e}")
            logger.info("Tentando criar KernelExplainer...")
            
            # Fallback para KernelExplainer (mais lento mas mais robusto)
            def model_predict(data):
                return self.model.predict(data, verbose=0)
            
            self.explainer = shap.KernelExplainer(
                model_predict,
                background_data[:50]  # Usar menos amostras para KernelExplainer
            )
            logger.info("SHAP KernelExplainer criado")
    
    def explain(self, data: np.ndarray, max_samples: int = 50) -> np.ndarray:
        """
        Calcula SHAP values para os dados fornecidos.
        
        Args:
            data: Dados para explicar (shape: [n_samples, timesteps, features])
            max_samples: Número máximo de amostras a explicar
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("Crie o explainer primeiro com create_explainer()")
        
        logger.info(f"Calculando SHAP values para {len(data)} amostras...")
        
        # Limitar amostras para performance
        if len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            data = data[indices]
        
        # Calcular SHAP values
        self.shap_values = self.explainer.shap_values(data)
        
        # Se retornar lista (múltiplas saídas), pegar primeira
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]
        
        logger.info(f"SHAP values calculados. Shape: {self.shap_values.shape}")
        
        return self.shap_values
    
    def get_feature_importance(
        self,
        aggregation: str = 'mean_abs'
    ) -> pd.DataFrame:
        """
        Calcula importância global das features.
        
        Args:
            aggregation: Método de agregação ('mean_abs', 'mean', 'std')
            
        Returns:
            DataFrame com importância das features
        """
        if self.shap_values is None:
            raise ValueError("Execute explain() primeiro")
        
        # Agregar sobre timesteps e amostras
        # Shape: [n_samples, timesteps, features] -> [features]
        
        if aggregation == 'mean_abs':
            # Média dos valores absolutos
            importance = np.mean(np.abs(self.shap_values), axis=(0, 1))
        elif aggregation == 'mean':
            # Média simples
            importance = np.mean(self.shap_values, axis=(0, 1))
        elif aggregation == 'std':
            # Desvio padrão (variabilidade)
            importance = np.std(self.shap_values, axis=(0, 1))
        else:
            raise ValueError(f"Agregação '{aggregation}' não suportada")
        
        # Criar DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Ordenar por importância
        df = df.sort_values('importance', ascending=False)
        df['importance_pct'] = (df['importance'] / df['importance'].sum()) * 100
        
        return df
    
    def plot_feature_importance(
        self,
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plota importância das features.
        
        Args:
            top_n: Número de features mais importantes para mostrar
            save_path: Caminho para salvar figura (opcional)
            
        Returns:
            Figure matplotlib
        """
        importance_df = self.get_feature_importance()
        
        # Top N features
        top_features = importance_df.head(top_n)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gráfico de barras
        bars = ax.barh(
            range(len(top_features)),
            top_features['importance'],
            color='skyblue',
            edgecolor='navy'
        )
        
        # Configurar eixos
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('SHAP Value (Importância)')
        ax.set_title(f'Top {top_n} Features Mais Importantes')
        ax.invert_yaxis()  # Features mais importantes no topo
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            pct = top_features.iloc[i]['importance_pct']
            ax.text(
                width, bar.get_y() + bar.get_height()/2,
                f' {pct:.1f}%',
                ha='left', va='center'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em {save_path}")
        
        return fig
    
    def plot_summary(
        self,
        data: np.ndarray,
        max_display: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Cria summary plot do SHAP (mostra distribuição de valores).
        
        Args:
            data: Dados originais usados para calcular SHAP values
            max_display: Número máximo de features para mostrar
            save_path: Caminho para salvar figura (opcional)
        """
        if self.shap_values is None:
            raise ValueError("Execute explain() primeiro")
        
        # Agregar sobre timesteps (média)
        shap_values_2d = np.mean(self.shap_values, axis=1)
        data_2d = np.mean(data, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        shap.summary_plot(
            shap_values_2d,
            features=data_2d,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot salvo em {save_path}")
        
        plt.tight_layout()
    
    def explain_single_prediction(
        self,
        sample_data: np.ndarray,
        sample_index: int = 0,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Explica uma predição específica.
        
        Args:
            sample_data: Dados de entrada
            sample_index: Índice da amostra a explicar
            save_path: Caminho para salvar visualização (opcional)
            
        Returns:
            Dict com explicação detalhada
        """
        if self.shap_values is None:
            raise ValueError("Execute explain() primeiro")
        
        # SHAP values da amostra específica
        sample_shap = self.shap_values[sample_index]  # Shape: [timesteps, features]
        
        # Agregar sobre timesteps
        feature_contributions = np.mean(sample_shap, axis=0)
        
        # Criar DataFrame
        contributions_df = pd.DataFrame({
            'feature': self.feature_names,
            'contribution': feature_contributions
        })
        contributions_df = contributions_df.sort_values(
            'contribution', 
            key=abs, 
            ascending=False
        )
        
        # Criar visualização waterfall
        plt.figure(figsize=(10, 6))
        
        # Agregar SHAP values para waterfall plot
        shap_values_sample = np.mean(sample_shap, axis=0)
        
        try:
            # Criar Explanation object para waterfall plot
            base_value = 0  # Base value (pode ser média das predições)
            
            # Waterfall plot mostra como cada feature contribui
            colors = ['red' if x < 0 else 'green' for x in shap_values_sample]
            
            plt.barh(
                range(len(self.feature_names)),
                shap_values_sample,
                color=colors,
                alpha=0.7,
                edgecolor='black'
            )
            
            plt.yticks(range(len(self.feature_names)), self.feature_names)
            plt.xlabel('SHAP Value (Contribuição)')
            plt.title(f'Contribuição das Features para Amostra #{sample_index}')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Waterfall plot salvo em {save_path}")
        
        except Exception as e:
            logger.error(f"Erro ao criar waterfall plot: {e}")
        
        return {
            'sample_index': sample_index,
            'feature_contributions': contributions_df.to_dict('records'),
            'top_positive': contributions_df[contributions_df['contribution'] > 0].head(3).to_dict('records'),
            'top_negative': contributions_df[contributions_df['contribution'] < 0].head(3).to_dict('records')
        }
    
    def save_explanation(
        self,
        filepath: str = "logs/feature_importance.json"
    ):
        """
        Salva explicação em arquivo JSON.
        
        Args:
            filepath: Caminho do arquivo
        """
        import json
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        importance_df = self.get_feature_importance()
        
        explanation = {
            'feature_importance': importance_df.to_dict('records'),
            'top_5_features': importance_df.head(5)['feature'].tolist(),
            'summary': {
                'most_important': importance_df.iloc[0]['feature'],
                'least_important': importance_df.iloc[-1]['feature'],
                'top_3_importance': float(importance_df.head(3)['importance_pct'].sum())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(explanation, f, indent=2)
        
        logger.info(f"Explicação salva em {filepath}")


def create_feature_names() -> List[str]:
    """
    Cria lista de nomes das features usadas no modelo.
    
    Returns:
        Lista de nomes das features
    """
    return [
        'Close',          # Preço de fechamento
        'Volume',         # Volume de negociação
        'High',           # Preço máximo
        'Low',            # Preço mínimo
        'MA_7',           # Média móvel 7 dias
        'MA_21',          # Média móvel 21 dias
        'MA_50',          # Média móvel 50 dias
        'RSI',            # Relative Strength Index
        'MACD',           # Moving Average Convergence Divergence
        'Volatility'      # Volatilidade
    ]


if __name__ == "__main__":
    """
    Exemplo de uso do ModelExplainer.
    """
    import sys
    from pathlib import Path
    
    # Adicionar src ao path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models.lstm_model import LSTMStockPredictor
    from data.preprocessing import TimeSeriesPreprocessor
    import pandas as pd
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not SHAP_AVAILABLE:
        print("❌ SHAP não está instalado!")
        print("Instale com: pip install shap")
        sys.exit(1)
    
    print("Exemplo de Feature Importance com SHAP")
    print("=" * 60)
    
    # Carregar dados
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "stock_data_features.csv"
    
    if not data_path.exists():
        print(f"Arquivo não encontrado: {data_path}")
        sys.exit(1)
    
    data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    print(f"Dados carregados: {len(data)} registros")
    
    # Preparar dados
    preprocessor = TimeSeriesPreprocessor()
    normalized = preprocessor.normalize_data(data)
    X, y = preprocessor.create_sequences(normalized, lookback=30)
    
    # Carregar modelo
    model_path = Path(__file__).parent.parent.parent / "models" / "lstm_model.keras"
    
    if not model_path.exists():
        print(f"Modelo não encontrado: {model_path}")
        print("Treine o modelo primeiro!")
        sys.exit(1)
    
    model = LSTMStockPredictor()
    model.load_model(model_path)
    print("Modelo carregado")
    
    # Criar explainer
    feature_names = create_feature_names()
    explainer = ModelExplainer(model.model, feature_names)
    
    # Criar explainer com dados de background
    print("\nCriando SHAP explainer...")
    explainer.create_explainer(X[:100])
    
    # Explicar predições
    print("Calculando SHAP values...")
    explainer.explain(X[100:150])
    
    # Obter importância das features
    print("\n" + "=" * 60)
    print("IMPORTÂNCIA DAS FEATURES")
    print("=" * 60)
    importance_df = explainer.get_feature_importance()
    print(importance_df.to_string(index=False))
    
    # Criar visualizações
    print("\nGerando visualizações...")
    explainer.plot_feature_importance(save_path='logs/feature_importance.png')
    explainer.plot_summary(X[100:150], save_path='logs/shap_summary.png')
    
    # Explicar predição específica
    print("\nExplicando predição específica...")
    explanation = explainer.explain_single_prediction(
        X[100:150],
        sample_index=0,
        save_path='logs/single_prediction_explanation.png'
    )
    
    print(f"\nTop 3 features positivas:")
    for feat in explanation['top_positive']:
        print(f"  {feat['feature']}: {feat['contribution']:.4f}")
    
    # Salvar resultados
    explainer.save_explanation()
    
    print("\n✅ Análise concluída!")
    print("Resultados salvos em logs/")
