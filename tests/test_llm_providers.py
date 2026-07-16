from pathlib import Path

from backend.llm_providers import generate
from backend.token_meter import TokenMeter


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.text = str(payload)

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_grok_responses_parsing_and_meter(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    def fake_post(url, **kwargs):
        assert url.endswith("/v1/responses")
        assert kwargs["json"]["max_output_tokens"] == 300
        return FakeResponse(
            {
                "output": [{"content": [{"type": "output_text", "text": "Grok briefing"}]}],
                "usage": {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60},
            }
        )

    monkeypatch.setattr("backend.llm_providers.requests.post", fake_post)
    meter = TokenMeter(tmp_path / "tokens.json")
    result = generate("grok", "grok-4.5", [{"role": "user", "content": "test"}], meter, 300)
    assert result.text == "Grok briefing"
    assert meter.snapshot()["providers"]["grok"]["total_tokens"] == 60


def test_groq_chat_parsing_and_meter(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def fake_post(url, **kwargs):
        assert "api.groq.com" in url
        assert kwargs["json"]["max_completion_tokens"] == 400
        return FakeResponse(
            {
                "choices": [{"message": {"content": "Groq briefing"}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
            }
        )

    monkeypatch.setattr("backend.llm_providers.requests.post", fake_post)
    meter = TokenMeter(tmp_path / "tokens.json")
    result = generate("groq", "openai/gpt-oss-120b", [{"role": "user", "content": "test"}], meter, 400)
    assert result.text == "Groq briefing"
    assert meter.snapshot()["providers"]["groq"]["total_tokens"] == 45


def test_gemini_parsing_and_meter(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_post(url, **kwargs):
        assert "generateContent" in url
        assert kwargs["json"]["generationConfig"]["maxOutputTokens"] == 500
        return FakeResponse(
            {
                "candidates": [{"content": {"parts": [{"text": "Gemini briefing"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 25,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 35,
                },
            }
        )

    monkeypatch.setattr("backend.llm_providers.requests.post", fake_post)
    meter = TokenMeter(tmp_path / "tokens.json")
    result = generate("gemini", "gemini-2.5-flash", [{"role": "user", "content": "test"}], meter, 500)
    assert result.text == "Gemini briefing"
    assert meter.snapshot()["providers"]["gemini"]["total_tokens"] == 35
