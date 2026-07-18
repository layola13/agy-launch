import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402


class TranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        main.TARGET_MODEL = "test-model"
        main.AGY_CLI_MODEL = "gemini-3.5-flash-low"
        main.MODEL_DISPLAY_NAME = "test-model"
        main.MODEL_PROVIDER = ""

    def test_model_role_function_response_becomes_tool_message(self) -> None:
        req = {
            "contents": [
                {"role": "user", "parts": [{"text": "list files"}]},
                {
                    "role": "model",
                    "parts": [
                        {"text": "I will inspect the directory."},
                        {
                            "functionCall": {
                                "id": "call-1",
                                "name": "list_dir",
                                "args": {"DirectoryPath": "/tmp"},
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call-1",
                                "name": "list_dir",
                                "response": {"output": "ok"},
                            }
                        }
                    ],
                },
            ]
        }

        messages = main.gemini_to_openai_messages(req)

        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(messages[2]["tool_call_id"], "call-1")
        self.assertEqual(json.loads(messages[2]["content"]), {"output": "ok"})

    def test_tool_config_validated_maps_to_auto_tool_choice(self) -> None:
        payload = main.build_openai_payload(
            {
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": "read"}]}],
                    "tools": [
                        {
                            "functionDeclarations": [
                                {
                                    "name": "read_file",
                                    "description": "Read a file",
                                    "parameters": {
                                        "type": "OBJECT",
                                        "properties": {"path": {"type": "STRING"}},
                                        "required": ["path"],
                                    },
                                }
                            ]
                        }
                    ],
                    "toolConfig": {"functionCallingConfig": {"mode": "VALIDATED"}},
                }
            }
        )

        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["tools"][0]["function"]["parameters"]["type"], "object")
        self.assertEqual(
            payload["tools"][0]["function"]["parameters"]["properties"]["path"]["type"],
            "string",
        )

    def test_tool_config_any_single_allowed_tool_maps_to_specific_choice(self) -> None:
        payload = main.build_openai_payload(
            {
                "contents": [{"role": "user", "parts": [{"text": "read"}]}],
                "tools": [
                    {
                        "functionDeclarations": [
                            {"name": "read_file", "parameters": {"type": "object"}},
                            {"name": "write_file", "parameters": {"type": "object"}},
                        ]
                    }
                ],
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": ["read_file"],
                    }
                },
            }
        )

        self.assertEqual([t["function"]["name"] for t in payload["tools"]], ["read_file"])
        self.assertEqual(
            payload["tool_choice"],
            {"type": "function", "function": {"name": "read_file"}},
        )

    def test_configure_models_resp_promotes_cli_model_and_display_name(self) -> None:
        raw = {
            "models": {
                "gemini-3-flash": {
                    "displayName": "Gemini 3 Flash",
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                }
            },
            "defaultAgentModelId": "gemini-3-flash",
            "commandModelIds": ["gemini-3-flash"],
        }
        main.TARGET_MODEL = "gpt-4.1-mini"
        main.AGY_CLI_MODEL = "gemini-3.5-flash-low"
        main.MODEL_DISPLAY_NAME = "gpt-4.1-mini"
        main.MODEL_PROVIDER = "openai"

        configured = main._configure_models_resp(raw)

        self.assertEqual(configured["defaultAgentModelId"], "gemini-3.5-flash-low")
        self.assertEqual(configured["commandModelIds"][0], "gemini-3.5-flash-low")
        self.assertIn("gemini-3.5-flash-low", configured["models"])
        self.assertEqual(
            configured["models"]["gemini-3.5-flash-low"]["displayName"],
            "gpt-4.1-mini",
        )
        self.assertEqual(
            configured["models"]["gemini-3.5-flash-low"]["modelProvider"],
            "MODEL_PROVIDER_OPENAI",
        )

    def test_candidate_env_paths_prefers_repo_env_over_user_env(self) -> None:
        old_here = main._HERE
        old_env = os.environ.copy()
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as home:
                repo_dir = os.path.join(home, "repo", "agy-launch")
                os.makedirs(repo_dir, exist_ok=True)
                main._HERE = repo_dir
                os.chdir(cwd)
                os.environ.clear()
                os.environ["HOME"] = home
                paths = main._candidate_env_paths()
                self.assertLess(
                    paths.index(os.path.join(repo_dir, ".env")),
                    paths.index(os.path.join(home, ".config", "agy-launch", ".env")),
                )
        finally:
            main._HERE = old_here
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)


if __name__ == "__main__":
    unittest.main()
