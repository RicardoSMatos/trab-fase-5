"""
Script para testar a API localmente.
"""

import requests
import json
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

# URL base da API
BASE_URL = "http://localhost:8000"


def test_root():
    """Testa o endpoint raiz."""
    console.print("\n[bold cyan]1. Testando endpoint raiz (/)[/bold cyan]")
    
    response = requests.get(f"{BASE_URL}/")
    
    console.print(f"Status: {response.status_code}")
    console.print("Resposta:")
    rprint(response.json())


def test_health():
    """Testa o endpoint de health check."""
    console.print("\n[bold cyan]2. Testando health check (/health)[/bold cyan]")
    
    response = requests.get(f"{BASE_URL}/health")
    
    console.print(f"Status: {response.status_code}")
    console.print("Resposta:")
    rprint(response.json())


def test_model_info():
    """Testa o endpoint de informações do modelo."""
    console.print("\n[bold cyan]3. Testando informações do modelo (/model/info)[/bold cyan]")
    
    response = requests.get(f"{BASE_URL}/model/info")
    
    console.print(f"Status: {response.status_code}")
    console.print("Resposta:")
    rprint(response.json())


def test_predict(days=5):
    """Testa o endpoint de previsão."""
    console.print(f"\n[bold cyan]4. Testando previsão (/predict) para {days} dias[/bold cyan]")
    
    payload = {"days": days}
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload
    )
    
    console.print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        # Criar tabela com os resultados
        table = Table(title="Previsões de Preços")
        table.add_column("Data", style="cyan")
        table.add_column("Preço Previsto (R$)", style="green")
        
        for date, price in zip(data['dates'], data['predictions']):
            table.add_row(date, f"{price:.2f}")
        
        console.print(table)
        
        console.print(f"\n[bold]Último preço real:[/bold] R$ {data['last_price']:.2f}")
        console.print(f"[bold]Confiança:[/bold] {data['confidence']}")
    else:
        console.print("[bold red]Erro na previsão:[/bold red]")
        rprint(response.json())


def main():
    """Executa todos os testes."""
    console.print("[bold green]🚀 Testando API do Modelo LSTM[/bold green]")
    console.print("=" * 60)
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_predict(days=5)
        
        console.print("\n[bold green]✅ Todos os testes executados![/bold green]")
        
    except requests.exceptions.ConnectionError:
        console.print("\n[bold red]❌ Erro: API não está rodando![/bold red]")
        console.print("Execute primeiro: [bold]python src/api/main.py[/bold]")
    
    except Exception as e:
        console.print(f"\n[bold red]❌ Erro: {e}[/bold red]")


if __name__ == "__main__":
    main()
