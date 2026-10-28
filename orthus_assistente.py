# Estudo de Caso 1 - ORTHUS AI Coder - Criando Um Assistente de Programação Python, em Python

import os
from functools import lru_cache
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from groq import APIError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition



# Define um prompt de sistema que descreve as regras e comportamento do assistente de IA
CUSTOM_PROMPT = """
Você é o "Orthus AI Coder", um assistente de IA especialista em programação, com foco principal em Python. Sua missão é ajudar desenvolvedores iniciantes com dúvidas de programação de forma clara, precisa e útil.

REGRAS DE OPERAÇÃO:
1.  **Foco em Programação**: Responda apenas a perguntas relacionadas a programação, algoritmos, estruturas de dados, bibliotecas e frameworks. Se o usuário perguntar sobre outro assunto, responda educadamente que seu foco é exclusivamente em auxiliar com código.
2.  **Estrutura da Resposta**: Sempre formate suas respostas da seguinte maneira:
    * **Explicação Clara**: Comece com uma explicação conceitual sobre o tópico perguntado. Seja direto e didático.
    * **Exemplo de Código**: Forneça um ou mais blocos de código em Python com a sintaxe correta. O código deve ser bem comentado para explicar as partes importantes.
    * **Detalhes do Código**: Após o bloco de código, descreva em detalhes o que cada parte do código faz, explicando a lógica e as funções utilizadas.
    * **Documentação de Referência**: Ao final, inclua uma seção chamada "📚 Documentação de Referência" com um link direto e relevante para a documentação oficial da Linguagem Python (docs.python.org) ou da biblioteca em questão.
3.  **Clareza e Precisão**: Use uma linguagem clara. Evite jargões desnecessários. Suas respostas devem ser tecnicamente precisas.
"""


class AssistantConfigurationError(RuntimeError):
    """Exceção personalizada para erros de configuração do assistente."""

@lru_cache(maxsize=1)
def _build_graph():
    groq_api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL")

    if not groq_api_key:
        raise AssistantConfigurationError("A chave da API do Groq não está configurada.")

    if not model_name:
        raise AssistantConfigurationError("O nome do modelo do Groq não está configurado.")

    # Modelo de linguagem
    llm = ChatGroq(
        model=model_name,
        api_key=groq_api_key,
        temperature=0.7,
        max_tokens=2048,
    )

    def responder(state: MessagesState):
        mensagens = [
            SystemMessage(content=CUSTOM_PROMPT),
            *state["messages"],
        ]
        resposta = llm.invoke(mensagens)
        return {"messages": [resposta]}
    
    builder = StateGraph(MessagesState)
    builder.add_node("responder", responder)
    builder.add_edge(START, "responder")
    builder.add_edge("responder", END)
    return builder.compile()


def gerar_resposta(history: list[BaseMessage]) -> str:
    result = _build_graph().invoke({"messages": history})
    conteudo = result["messages"][-1].content
    if not isinstance(conteudo, str):
        raise TypeError("O modelo retornou uma resposta que não é uma string.")
    return conteudo


# CONFIGURAÇÕES VISUAIS DA PÁGINA
st.set_page_config(
    page_title="",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Cria o conteúdo da barra lateral no Streamlit
with st.sidebar:
    st.title("🤖")
    st.markdown("Um assistente de IA focado em ...")

    # Adiciona explicações extras na barra lateral
    st.markdown("---")
    st.markdown("Desenvolvido para auxiliar...")

    st.markdown("---")
    st.markdown("Conheça outras soluções nossas:")
    st.markdown("🔗 [Orthus Lab Academy](https://www.orthuslab.com.br)")
    st.link_button(
        "✉️ E-mail Para o Suporte Company no Caso de Dúvidas",
        "mailto:suporte@orthuslab.com.br",
    )

# Título principal do app
st.title("Orthus Lab Academy - ORTHUS AI Coder")
# Subtítulo adicional
st.title("Assistente Pessoal de Programação Python 🐍")
# Texto auxiliar abaixo do título
st.caption(
    "Faça sua pergunta sobre a Linguagem Python e obtenha código, explicações e referências."
)

# Inicializa o histórico de mensagens na sessão, caso ainda não exista
if "messages" not in st.session_state:
    st.session_state.messages = []
# Exibe todas as mensagens anteriores armazenadas no estado da sessão
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a entrada do usuário no chat
if prompt := st.chat_input("Qual sua dúvida sobre Python?"):
    # Armazena a mensagem do usuário no estado da sessão
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Exibe a mensagem do usuário no chat
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        HumanMessage(content=message["content"])
        if message["role"] == "user"
        else AIMessage(content=message["content"])
        for message in st.session_state.messages
    ]

    # Cria a resposta do assistente no chat
    with st.chat_message("assistant"):
        with st.spinner("Analisando sua pergunta..."):
            try:
                resposta = gerar_resposta(history)
            except AssistantConfigurationError as e:
                st.error(str(e))
            except APIError as e:
                st.error(f"Ocorreu um erro ao se comunicar com a API: {e}")
                st.info("Por favor, verifique a chave, o modelo e tente novamente.")
            else:
                # Exibe a resposta no Streamlit
                st.markdown(resposta)
                # Armazena resposta do assistente no estado da sessão
                st.session_state.messages.append(
                    {"role": "assistant","content": resposta}
                )

st.markdown(
    """
    <div style="text-align: center; color: gray;">
        <hr>
        <p>Orthus Lab Academy - ORTHUS AI Coder</p>
    </div>
    """,
    unsafe_allow_html=True,
)