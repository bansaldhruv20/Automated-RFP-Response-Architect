import importlib
import json
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import main


class WorkflowTests(unittest.TestCase):
    def test_import_does_not_load_models(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("network on import"),
        ):
            importlib.reload(main)

        importlib.reload(main)

    def test_default_input_independent_of_working_directory(self):
        self.assertTrue(main.arguments([]).input.is_absolute())

        self.assertEqual(
            main.validate_input(main.arguments([]).input).name,
            "sample_rfp.txt",
        )

    def test_rejects_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            for name, content in [
                ("empty.txt", ""),
                ("rfp.pdf", "not a PDF"),
            ]:
                path = root / name
                path.write_text(content)

                with self.assertRaises(ValueError):
                    main.validate_input(path)

            with self.assertRaises(ValueError):
                main.validate_input(root / "missing.txt")

    def test_missing_ollama_has_actionable_error(self):
        with patch.object(
            main,
            "urlopen",
            side_effect=OSError("offline"),
        ):
            with self.assertRaisesRegex(RuntimeError, "ollama serve"):
                main.check_ollama(
                    "http://localhost:11434",
                    "llama3.2:3b",
                )

    def test_ollama_model_presence_over_http(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()

                self.wfile.write(
                    json.dumps(
                        {"models": [{"name": "llama3.2:3b"}]}
                    ).encode()
                )

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            Handler,
        )

        thread = Thread(
            target=server.serve_forever,
            daemon=True,
        )
        thread.start()

        try:
            url = f"http://127.0.0.1:{server.server_port}"

            main.check_ollama(url, "llama3.2:3b")

            with self.assertRaisesRegex(RuntimeError, "ollama pull"):
                main.check_ollama(url, "missing")

        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_retrieval_outputs_are_isolated_and_auditable(self):
        evidence = {
            name: [
                {
                    "source": "sample.txt",
                    "chunk": 1,
                    "score": 0.8,
                    "text": "MFA required",
                }
            ]
            for name in main.QUERIES
        }

        with tempfile.TemporaryDirectory() as directory:
            args = main.arguments(
                ["--retrieve-only", "--output", directory]
            )

            with (
                patch.object(main, "retrieve", return_value=evidence),
                patch.object(main, "check_ollama") as check,
            ):
                first = main.run(args)
                second = main.run(args)

            check.assert_not_called()

            self.assertNotEqual(first, second)

            self.assertEqual(
                json.loads((first / "evidence.json").read_text()),
                evidence,
            )

            self.assertEqual(
                json.loads((first / "run.json").read_text())["mode"],
                "retrieval",
            )

            self.assertFalse(
                (first / "Final_RFP_Response.md").exists()
            )

    def test_real_crew_configuration_and_context(self):
        evidence = {
            name: [
                {
                    "source": "sample.txt",
                    "chunk": 1,
                    "text": "MFA required",
                }
            ]
            for name in main.QUERIES
        }

        crew = main.build_crew(
            evidence,
            "llama3.2:3b",
            "http://localhost:11434",
            300,
        )

        self.assertEqual(len(crew.agents), 3)
        self.assertEqual(crew.process.value, "sequential")
        self.assertEqual(crew.tasks[2].context, crew.tasks[:2])

        self.assertIn(
            "[sample.txt#chunk-1]",
            crew.tasks[0].description,
        )

        self.assertIn(
            "Requires vendor confirmation",
            crew.tasks[2].description,
        )

    def test_generation_saves_draft_and_metrics_without_replacing_previous_run(self):
        evidence = {
            name: [
                {
                    "source": "sample.txt",
                    "chunk": 1,
                    "text": "MFA required",
                }
            ]
            for name in main.QUERIES
        }

        with tempfile.TemporaryDirectory() as directory:
            args = main.arguments(["--output", directory])

            with (
                patch.object(main, "check_ollama"),
                patch.object(main, "retrieve", return_value=evidence),
                patch.object(main, "build_crew") as build,
            ):
                build.return_value.kickoff.return_value = (
                    "# Security & Compliance\nMFA required."
                )

                first = main.run(args)

                original = (
                    first / "Final_RFP_Response.md"
                ).read_text()

                second = main.run(args)

            self.assertNotEqual(first, second)

            self.assertEqual(
                (first / "Final_RFP_Response.md").read_text(),
                original,
            )

            self.assertIn("No vendor capabilities", original)

            self.assertEqual(
                json.loads((first / "run.json").read_text())["mode"],
                "proposal",
            )

    def test_actual_crew_runs_all_three_tasks_with_simulated_llm(self):
        # Tests orchestration, not model quality or real Ollama inference.
        from crewai import LLM

        evidence = {
            name: [
                {
                    "source": "sample.txt",
                    "chunk": 1,
                    "text": "MFA required",
                }
            ]
            for name in main.QUERIES
        }

        crew = main.build_crew(
            evidence,
            "llama3.2:3b",
            "http://localhost:11434",
            300,
        )

        with patch.object(
            LLM,
            "call",
            return_value=(
                "Final Answer: Draft requiring vendor confirmation."
            ),
        ) as call:
            result = crew.kickoff()

        self.assertEqual(call.call_count, 3)
        self.assertEqual(len(result.tasks_output), 3)
        self.assertIn("vendor confirmation", str(result))

    def test_audit_flags_invented_citations_without_remapping_them(self):
        evidence = {
            "security": [
                {
                    "source": "rfp.txt",
                    "chunk": 1,
                    "text": "MFA",
                }
            ]
        }

        text, report = main.audit_proposal(
            "MFA [rfp.txt#chunk-1]. Uptime [rfp.txt#chunk-2].",
            evidence,
        )

        self.assertEqual(
            report["unknown_citations"],
            ["rfp.txt#chunk-2"],
        )

        self.assertIn(
            "[UNVERIFIED CITATION: rfp.txt#chunk-2]",
            text,
        )

        self.assertIn("MFA [rfp.txt#chunk-1]", text)

        self.assertEqual(
            report["status"],
            "human_review_required",
        )

        self.assertTrue(report["has_known_citation"])

    def test_audit_detects_missing_citations(self):
        _, report = main.audit_proposal(
            "An uncited response",
            {},
        )

        self.assertFalse(report["has_known_citation"])


if __name__ == "__main__":
    unittest.main()
