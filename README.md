# Projeto Orthos

## Configuração do ambiente

### 1. Pré-requisitos

- Python 3.10+
- Git 
- Ter um chave de API do OpenRouter

### 2. Criar uma chave de API no OpenRouter

1. Acesse o [OpenRouter](https://openrouter.ai/).
2. Crie uma conta ou faça login.
3. Acesse a seção **Models** e consulte os modelos disponíveis.
4. Filtre os modelos gratuitos (*Free*) e escolha um modelo compatível com o projeto **Google Gemma 4 31B**.
5. Acesse a seção **API Keys** e clique em **Create API Key**.
6. Copie a chave gerada e guarde-a em um local seguro.

> **Importante:** não compartilhe sua chave de API nem a envie para o GitHub. Ela é privada e pode permitir o uso da sua conta.

### 3. Criar e ativar o ambiente virtual

No terminal, acesse a pasta do projeto e execute:

```powershell
py -m venv env
```

Ative o ambiente virtual no Windows:

```powershell
.\env\Scripts\Activate.ps1
```

Com o ambiente ativo, o terminal deverá exibir `(env)` antes do caminho atual.

### 4. Configurar as variáveis de ambiente

Na raiz do projeto, crie um arquivo chamado `.env` e adicione:

```env
M_API_KEY=sua_chave_api_aqui
OPENROUTER_MODEL=openrouter/free
```

Substitua `sua_chave_api_aqui` pela chave gerada no OpenRouter.

### 5. Instalar as dependências

Com o ambiente virtual ativado, execute:

```powershell
python -m pip install --upgrade pip
pip install streamlit langchain-core langchain-openai langgraph python-dotenv
pip install pypdf python-docx
```

Se o projeto já possuir um arquivo `requirements.txt` atualizado com todas as dependências necessárias, use:

```powershell
pip install -r requirements.txt
```

### 6. Executar a aplicação

Na raiz do projeto, execute:

```powershell
streamlit run orthus.py
```

O Streamlit iniciará o servidor local e exibirá um endereço no terminal, geralmente:

```text
http://localhost:8501
```

Abra esse endereço no navegador para acessar a aplicação.


- **Python:** linguagem principal.
- **Streamlit:** interface da aplicação.
- **LangChain:** integração com modelos de linguagem.
- **LangGraph:** construção e organização de fluxos de agentes.
- **OpenRouter:** acesso a modelos de IA por meio de uma API.
- **python-dotenv:** carregamento de variáveis de ambiente.


