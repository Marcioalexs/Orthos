"""Smoke tests da evolução Knowledge Graph. Execute na raiz studyagent."""
import unittest

from studyagent_core.knowledge_graph import public_graph, learning_path, suggest_review
from studyagent_core.engine import quiz_bank


class GraphTests(unittest.TestCase):
    def test_nodes_and_edges(self):
        graph = public_graph()
        self.assertEqual(len(graph["conceitos"]), 8)
        self.assertEqual(len(graph["relacoes"]), 7)
        self.assertEqual(sum(c["questoes"] for c in graph["conceitos"]), 20)

    def test_path_is_ordered(self):
        path = learning_path("ml_validacao")
        self.assertEqual([step["id"] for step in path["etapas"]],
                         ["ml_supervisionado", "ml_validacao"])

    def test_invalid_path(self):
        with self.assertRaises(ValueError):
            learning_path("inexistente")

    def test_quiz_recommendations(self):
        bank = quiz_bank()["ml"]
        correct = [q["correta"] for q in bank]
        self.assertEqual(suggest_review("ml", correct), [])
        wrong = correct.copy()
        wrong[2] = (wrong[2] + 1) % len(bank[2]["opcoes"])
        review = suggest_review("ml", wrong)
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]["id"], "ml_validacao")
        self.assertEqual(review[0]["documentos"], ["ML02"])


if __name__ == "__main__":
    unittest.main()
