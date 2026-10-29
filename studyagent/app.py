"""StudyAgent Orthos — FastAPI ready for Vercel Python runtime."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from studyagent_core.engine import COMPETENCIAS, docs, answer, quiz_public, grade

load_dotenv()  # Carrega studyagent/.env no desenvolvimento local; não substitui variáveis reais.
app = FastAPI(title="StudyAgent Orthos MVP", version="1.0.0")
ROOT = Path(__file__).resolve().parent

class Message(BaseModel):
    role: str
    content: str = Field(max_length=1200)

class Question(BaseModel):
    pergunta: str = Field(min_length=2, max_length=2500)
    competencia: str
    nivel: str = "Iniciante"
    historico: list[Message] = Field(default_factory=list, max_length=12)

class Assessment(BaseModel):
    competencia: str
    respostas: list[int]

@app.get("/api/status")
def status():
    return {"online": True, "llm_configurado": bool(os.getenv("OPENROUTER_API_KEY")), "competencias": len(COMPETENCIAS), "conteudos":len(docs())}

@app.get("/api/competencias")
def competencies():
    return [{"id":code,"titulo":title, "assuntos":[d["assunto"] for d in docs() if d["competencia"]==code]} for code,title in COMPETENCIAS.items()]

@app.post("/api/perguntar")
def ask(payload: Question):
    try:
        return answer(payload.pergunta,payload.competencia,payload.nivel,[x.model_dump() for x in payload.historico])
    except ValueError as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=502,detail=str(exc)) from None

@app.get("/api/quiz/{competencia}")
def get_quiz(competencia: str):
    try:
        return {"competencia":competencia,"questoes":quiz_public(competencia),"criterio_aprovacao":80}
    except ValueError as exc:
        raise HTTPException(status_code=404,detail=str(exc)) from None

@app.post("/api/avaliar")
def assess(payload:Assessment):
    try:
        return grade(payload.competencia,payload.respostas)
    except ValueError as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from None

@app.get("/",include_in_schema=False)
def home():
    return FileResponse(ROOT/"public"/"index.html")
