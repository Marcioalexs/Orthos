"""StudyAgent Orthos: grafo educativo curado e revisões por conceito.

MVP: arquivos JSON locais, sem banco nem dados pessoais de estudantes.
Relações de prerequisito são orientações pedagógicas propostas para revisão.
"""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .engine import COMPETENCIAS, docs, quiz_bank

GRAFO_PATH = Path(__file__).resolve().parents[1] / "data" / "grafo_conhecimento.json"


@lru_cache(maxsize=1)
def _validated():
    raw = json.loads(GRAFO_PATH.read_text(encoding="utf-8"))
    concepts = raw.get("conceitos", [])
    edges = raw.get("relacoes", [])
    quiz_map = raw.get("quiz_conceitos", {})
    document_ids = {d["id"] for d in docs()}
    ids = [c["id"] for c in concepts]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Grafo com conceitos repetidos ou vazio")
    index = {c["id"]: c for c in concepts}
    for concept in concepts:
        if concept["competencia"] not in COMPETENCIAS:
            raise ValueError("Competência inválida no grafo")
        if not concept.get("documentos") or not set(concept["documentos"]) <= document_ids:
            raise ValueError("Referência de documento inválida no grafo")
    for edge in edges:
        if edge["origem"] not in index or edge["destino"] not in index or edge["origem"] == edge["destino"]:
            raise ValueError("Relação inválida no grafo")
        if edge["tipo"] not in {"prerequisito", "relacionado"}:
            raise ValueError("Tipo de relação não permitido")
    for competency, bank in quiz_bank().items():
        mapped = quiz_map.get(competency, [])
        if len(mapped) != len(bank) or any(
            concept_id not in index or index[concept_id]["competencia"] != competency
            for concept_id in mapped
        ):
            raise ValueError(f"Mapeamento incompleto do Quiz: {competency}")
    if set(quiz_map) != set(quiz_bank()):
        raise ValueError("Mapeamento de Quiz possui competências divergentes")
    # Rejeita ciclos nos pré-requisitos: orientações devem ter uma ordem possível.
    dependencies = {
        name: [e["origem"] for e in edges if e["tipo"] == "prerequisito" and e["destino"] == name]
        for name in ids
    }
    visited, visiting = set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError("Ciclo de pré-requisitos detectado")
        if name in visited:
            return
        visiting.add(name)
        for dependency in dependencies[name]:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in ids:
        visit(name)
    return raw, index, dependencies


def public_graph():
    raw, index, prerequisites = _validated()
    document_index = {doc["id"]: doc for doc in docs()}
    mapped = [item for values in raw["quiz_conceitos"].values() for item in values]
    nodes = []
    for concept in raw["conceitos"]:
        related = sorted({
            edge["destino"] if edge["origem"] == concept["id"] else edge["origem"]
            for edge in raw["relacoes"]
            if edge["tipo"] == "relacionado" and concept["id"] in (edge["origem"], edge["destino"])
        })
        nodes.append({
            **concept,
            "prerequisitos": prerequisites[concept["id"]],
            "relacionados": related,
            "questoes": mapped.count(concept["id"]),
            "fontes": [src for doc_id in concept["documentos"] for src in document_index[doc_id]["fontes"]],
        })
    return {"versao": raw["versao"], "estado": raw["estado"],
            "competencias": COMPETENCIAS, "conceitos": nodes, "relacoes": raw["relacoes"]}


def learning_path(concept_id: str):
    _, index, deps = _validated()
    if concept_id not in index:
        raise ValueError("Conceito não encontrado")
    ordered, done = [], set()

    def walk(current):
        if current in done:
            return
        done.add(current)
        for dependency in deps[current]:
            walk(dependency)
        ordered.append({"id": current, "titulo": index[current]["titulo"],
                        "competencia": index[current]["competencia"]})

    walk(concept_id)
    return {"conceito": concept_id, "etapas": ordered, "observacao": "Trilha sugerida, não avaliação de domínio."}


def suggest_review(competency: str, answers: list[int]):
    """Only incorrect quiz questions generate review recommendations."""
    _, index, _ = _validated()
    mapped = _validated()[0]["quiz_conceitos"].get(competency, [])
    bank = quiz_bank().get(competency, [])
    if len(mapped) != len(bank) or len(answers) != len(bank):
        raise ValueError("Questionário incompatível com o grafo")
    errors = Counter(mapped[i] for i, item in enumerate(bank) if answers[i] != item["correta"])
    result = []
    for concept_id, mistakes in sorted(errors.items(), key=lambda pair: (-pair[1], pair[0])):
        c = index[concept_id]
        result.append({"id": concept_id, "titulo": c["titulo"],
                       "documentos": c["documentos"], "erros": mistakes})
    return result
