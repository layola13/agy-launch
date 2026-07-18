import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main  # noqa: E402


class TranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        main.TARGET_MODEL = "test-model"

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


if __name__ == "__main__":
    unittest.main()
