# Projeto Orthos
---

## Pré-requisitos
- Python 3.10+
- Frameworks de cliente para LLM (ex.: `openai`, `anthropic`, `transformers*`, `deepseek`, `groq`)
- Chave de API válida para o provedor de LLM

---

## Como executar
1. Clone o repositório ou faça um forke:
   ```bash
   git clone https://github.com/carlosmandele/Orthos.git
   cd Orthos

2. Na raiz do projeto, crie e ative um ambiente virtual:
   ```bash
   py -3.13 -m venv env    # Cria o ambiente
   .\env\Scripts\Activate.ps1  # Ativar o ambiente

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt

4. Configure sua chave de API
   ```bash
   export API_KEY="sua_apikey_aqui"

5. Execute o script principal de interação com o LLM (agente):
   ```bash
   streamlit run orthus.py

----
