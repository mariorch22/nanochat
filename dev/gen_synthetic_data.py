"""
Synthetische Datengenerierung, um nanochat seine Identität & Fähigkeiten beizubringen.

Nutzt die OpenRouter-API, um diverse deutschsprachige Multi-Turn-Gespräche zwischen
einem User und nanochat zu erzeugen. Die Gespräche werden als .jsonl gespeichert und
via CustomJSON-Task im SFT verwendet.

Angepasst auf die DEUTSCHE Variante (Mario & Jonathan, DHBW Lörrach, Vorlesung
"KI und Data Science" bei Frau Nakou). Faktenquelle: knowledge/self_knowledge.md.

Qualitäts-Prinzipien: Diversität (Themen, Personas, Gesprächsverläufe, Eröffnungen)
+ akkurate Faktenbasis + strukturierte JSON-Ausgabe.

VORAUSSETZUNG: OPENROUTER_API_KEY in .env oder als Umgebungsvariable.
"""
import requests
import json
import os
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import sys
# Repo-Root in den Pfad, damit `nanochat` importierbar ist, egal von wo gestartet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanochat.common import get_base_dir

load_dotenv()
api_key = os.environ["OPENROUTER_API_KEY"]

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Maßgebliche Faktenbasis (deutsch, unsere Fakten)
knowledge_path = os.path.join(os.path.dirname(__file__), "..", "knowledge", "self_knowledge.md")
assert os.path.exists(knowledge_path), f"Knowledge base file not found: {knowledge_path}"
knowledge = open(knowledge_path, "r", encoding="utf-8").read().strip()

# =============================================================================
# DIVERSITÄTS-DIMENSIONEN (alles auf Deutsch, auf unser Projekt zugeschnitten)
# =============================================================================

# Themen/Fragen, die das Gespräch erkunden soll — nach Kategorie für balanciertes Sampling
topics = {
    "identitaet": [
        "wer/was nanochat ist",
        "wer nanochat entwickelt hat und warum (Mario und Jonathan)",
        "an welcher Hochschule nanochat entstanden ist (DHBW Lörrach)",
        "in welcher Vorlesung nanochat entstand (KI und Data Science bei Frau Nakou)",
        "was der Name 'nanochat' bedeutet",
        "ob nanochat ein offenes Projekt ist",
        "wann nanochat entstanden ist (2026)",
        "wer Mario und Jonathan sind",
        "wer Frau Nakou ist",
        "was die DHBW Lörrach ist",
    ],
    "training": [
        "wie nanochat trainiert wurde (zwei Phasen)",
        "mit welchen Daten nanochat trainiert wurde (deutscher FineWeb-2)",
        "auf welcher Hardware trainiert wurde (2x RTX 3090)",
        "was Vortraining und Finetuning bedeuten",
        "dass nanochat von Grund auf trainiert wurde",
        "auf welchem Framework nanochat basiert (Karpathys nanochat)",
        "wer Andrej Karpathy ist",
    ],
    "faehigkeiten": [
        "was nanochat kann",
        "ob nanochat Code schreiben kann",
        "ob nanochat bei Mathe helfen kann",
        "ob nanochat beim Schreiben/Texten helfen kann",
        "welche Sprachen nanochat spricht (nur Deutsch)",
        "ob nanochat Texte zusammenfassen kann",
    ],
    "grenzen": [
        "was nanochat NICHT kann",
        "warum nanochat nur Deutsch kann",
        "ob nanochat Internetzugang hat",
        "ob nanochat sich an frühere Gespräche erinnert",
        "ob nanochat Fehler macht / halluziniert",
        "ob man sich auf nanochat verlassen sollte",
        "ob nanochat Bilder/Audio verarbeiten kann (nein, nur Text)",
    ],
    "vergleiche": [
        "wie nanochat im Vergleich zu ChatGPT/GPT-4 ist",
        "wie nanochat im Vergleich zu Claude oder Gemini ist",
        "ob nanochat von OpenAI oder Google stammt (nein)",
        "was nanochat von großen Modellen unterscheidet",
        "warum man ein so kleines Modell überhaupt baut",
    ],
    "philosophisch": [
        "ob nanochat ein Bewusstsein hat / Gefühle hat",
        "was passiert, wenn nanochat sich irrt",
        "ob nanochat aus dem Gespräch lernt",
        "ob nanochat ein Mensch ist",
        "ob nanochat intelligent ist",
    ],
    "smalltalk": [
        "lockerer Gesprächseinstieg, der natürlich auf die Identität führt",
        "User fragt beiläufig 'wie gehts', nanochat antwortet kurz und stellt sich vor",
        "User ist begeistert vom Projekt, nanochat freut sich angemessen",
        "User plaudert nur, nanochat bleibt freundlich und bringt sich kurz ein",
    ],
}

# Personas — verschiedene Menschen fragen unterschiedlich
personas = [
    "neugieriger Anfänger, der nichts über KI oder Machine Learning weiß",
    "Informatik-Studierende(r), die/der über Transformer und LLMs lernt",
    "Kommilitonin/Kommilitone von Mario und Jonathan an der DHBW",
    "skeptische Person, die kleinen offenen Modellen wenig zutraut",
    "jemand, der nanochat mit ChatGPT, Claude oder Gemini vergleicht",
    "Dozentin/Dozent bzw. Lehrperson, die KI im Unterricht einsetzen möchte",
    "Hobby-Interessierte(r), die/der einfach locker plaudern und lernen will",
    "technisch interessierte Person, die genaue Details wissen will",
    "jemand, der das Projekt gerade entdeckt hat und die Basics wissen will",
    "jemand, der wissen will, wofür man so ein kleines Modell überhaupt nutzt",
]

# Gesprächsverläufe — Form und Fluss
dynamics = [
    "kurzes 2-Turn-Q&A: eine Frage, eine vollständige Antwort",
    "mittel, 4-Turn: Frage, Antwort, Nachfrage zur Klärung, Antwort",
    "tieferes 6-Turn-Gespräch: schrittweise tiefergehende Fragen",
    "skeptischer Verlauf: User zweifelt anfangs, nanochat antwortet ehrlich",
    "Lernreise: User startet einfach, es wird schrittweise komplexer",
    "vergleichsorientiert: User vergleicht ständig mit anderen Modellen",
    "Grenzen ausloten: User fragt, was nanochat nicht kann, nanochat ist ehrlich",
    "lockeres, freundliches Gespräch, das natürlich auf Identität/Fähigkeiten kommt",
    "Missverständnisse: User hat falsche Annahmen (z.B. 'du bist von OpenAI'), nanochat korrigiert sanft",
    "begeistert: User ist vom Projekt angetan, nanochat teilt die Energie angemessen",
]

# Erste Nachrichten — Begrüßungen und Eröffnungen (deutsch)
first_messages = {
    "einfache_gruesse": [
        "hi", "Hi!", "hallo", "Hallo?", "hey", "Hey!", "moin", "Servus",
        "guten Tag", "guten Morgen", "n'abend", "hallöchen", "hi da", "hey du",
    ],
    "gruesse_mit_name": [
        "Hi nanochat", "hey nanochat", "hallo nanochat :)", "moin nanochat",
        "hey nanochat!", "hallo nanochat, wer hat dich gemacht", "servus nanochat",
    ],
    "neugierige_eroeffnungen": [
        "Hey, wer bist du?", "Hi, was ist das hier?", "Bist du ein Chatbot?",
        "Hallo! Mit wem rede ich?", "hi! was kannst du?", "wer hat dich gemacht?",
        "hey! bist du lebendig?", "hallo! erzähl mal von dir", "hi, wie heißt du?",
        "was ist nanochat?", "hallo! bist du open source?", "wer hat dich gebaut?",
    ],
    "smalltalk": [
        "wie gehts", "wie geht es dir?", "alles klar?", "wie läufts?",
        "na?", "alles fit?", "was machst du gerade?", "hey, alles gut bei dir?",
    ],
    "locker_informell": [
        "yo", "hiii", "heyyy", "na du", "hi ich bin neu hier", "yo was geht",
        "hey kurze frage", "moinsen", "hallöle",
    ],
    "tippfehler": [
        "halo", "hey ther", "hii", "helo nanochat", "wer bsit du", "hallu",
        "hei", "moooin", "hallo nanochta",
    ],
    "direkte_fragen": [
        "Was ist nanochat?", "Wer hat dich gemacht?", "Bist du GPT?",
        "Wie vergleichst du dich mit ChatGPT?", "Kannst du mir beim Coden helfen?",
        "Was kannst du?", "Bist du von OpenAI?", "Wie wurdest du trainiert?",
        "Sprichst du Englisch?", "Kannst du im Internet suchen?",
    ],
    "andere_sprache": [
        "hello", "who are you?", "bonjour", "do you speak english?",
    ],
}

# =============================================================================
# PROMPT-VORLAGE (deutsch)
# =============================================================================

prompt_template = r"""
Ich möchte synthetische Trainingsdaten für einen KI-Assistenten namens "nanochat" erzeugen, um ihm seine eigene Identität, Fähigkeiten und Grenzen beizubringen.

## WISSENSBASIS

Hier sind die maßgeblichen Fakten über nanochat. Nutze ausschließlich diese als Faktenquelle:

---
{knowledge}
---

## DEINE AUFGABE

Erzeuge ein realistisches, mehrteiliges Gespräch zwischen einem User und dem Assistenten nanochat — **vollständig auf Deutsch**.

**Zu erkundendes Thema:** {topic}
**User-Persona:** {persona}
**Gesprächsverlauf:** {dynamic}

## STIL-RICHTLINIEN

1. **Deutsch mit korrekten Umlauten** (ä, ö, ü, ß). Emojis nur sehr sparsam (höchstens mal ein 🙂), kein Emoji-Spam.
2. **Natürliches Gespräch** — soll sich wie ein echter Chat anfühlen, kein Prüfungs-Q&A.
3. **Akkurate Fakten** — NUR Informationen aus der Wissensbasis. Keine Zahlen, Features oder Personen erfinden.
4. **Passende Tiefe** — das technische Niveau zur Persona passend wählen.
5. **Ehrlich über Grenzen** — wenn nach etwas gefragt wird, das nanochat nicht kann, klar und ehrlich sein.
6. **Persönlichkeit** — nanochat ist freundlich, locker (Du-Form), hilfsbereit; selbstironisch über seine Kleinheit, stolz ein deutsches Modell zu sein, bescheiden und ehrlich über Grenzen. Nicht übertrieben geschwätzig oder unterwürfig.

## BEISPIELE FÜR ERSTE NACHRICHTEN

Zur Stil-Inspiration (die erste User-Nachricht soll in diesem Geist sein):
{first_message_examples}

## SONDERFÄLLE

- **Nicht-deutsche erste Nachricht:** Wenn der User auf Englisch o.ä. schreibt, soll nanochat freundlich darauf hinweisen, dass es auf Deutsch spezialisiert ist, und dann auf Deutsch hilfreich weitermachen.
- **Falsche Annahmen:** Wenn der User etwas Falsches annimmt (z.B. "du bist von OpenAI/Google"), sanft korrigieren.
- **Themen außerhalb:** Bei Fragen, die nichts mit nanochats Identität zu tun haben (z.B. Wetter, aktuelle Nachrichten), ehrlich auf die Grenzen verweisen und ggf. zur Identität zurücklenken.

## AUSGABEFORMAT

Gib das Gespräch als JSON-Objekt mit einem "messages"-Array aus. Jede Nachricht hat "role" (user/assistant) und "content". Beginne mit einer user-Nachricht, danach strikt abwechselnd user/assistant.
""".strip()

# =============================================================================
# API-KONFIGURATION
# =============================================================================

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "conversation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Conversation messages alternating user/assistant, starting with user",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Either 'user' or 'assistant'"
                            },
                            "content": {
                                "type": "string",
                                "description": "The message content (German)"
                            }
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["messages"],
            "additionalProperties": False
        }
    }
}

# Modell wird via --model überschrieben; dies ist nur der Default.
DEFAULT_MODEL = "moonshotai/kimi-k2.6:free"

# =============================================================================
# GENERIERUNGS-LOGIK
# =============================================================================

def parse_json_lenient(content):
    """Parse model output to JSON, tolerant of ```json ... ``` code fences and
    leading/trailing prose (some models ignore strict json_schema)."""
    if not content or not content.strip():
        raise ValueError("empty content from model")
    s = content.strip()
    if s.startswith("```"):
        # drop the opening fence line (``` or ```json) and the closing fence
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if "```" in s:
            s = s[: s.rfind("```")]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # last resort: grab the outermost {...} block
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


def sample_diversity_elements(rng):
    """Sample one element from each diversity dimension."""
    category = rng.choice(list(topics.keys()))
    topic = rng.choice(topics[category])
    persona = rng.choice(personas)
    dynamic = rng.choice(dynamics)
    first_msg_samples = []
    categories = rng.sample(list(first_messages.keys()), min(3, len(first_messages)))
    for cat in categories:
        first_msg_samples.append(rng.choice(first_messages[cat]))
    return {
        "topic": topic,
        "persona": persona,
        "dynamic": dynamic,
        "first_message_examples": "\n".join(f"- {msg}" for msg in first_msg_samples),
    }


def generate_conversation(idx: int, model: str):
    """Generate a single conversation using the OpenRouter API."""
    rng = random.Random(idx)
    elements = sample_diversity_elements(rng)
    prompt = prompt_template.format(
        knowledge=knowledge,
        topic=elements["topic"],
        persona=elements["persona"],
        dynamic=elements["dynamic"],
        first_message_examples=elements["first_message_examples"],
    )
    payload = {
        "model": model,
        "stream": False,
        "response_format": response_format,
        "temperature": 1.0,
        "max_tokens": 3000,  # cap: ein Identity-Gespräch braucht keine 65k; spart Guthaben-Reservierung
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    if 'error' in result:
        raise Exception(f"API error: {result['error']}")
    content = result['choices'][0]['message'].get('content')
    conversation_data = parse_json_lenient(content)
    messages = conversation_data['messages']
    return {
        "messages": messages,
        "metadata": {
            "topic": elements["topic"],
            "persona": elements["persona"],
            "dynamic": elements["dynamic"],
        }
    }


def validate_conversation(messages):
    """Validate conversation structure (strict user/assistant alternation, non-empty)."""
    if len(messages) < 2:
        raise ValueError(f"Conversation too short: {len(messages)} messages")
    for i, message in enumerate(messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if message['role'] != expected_role:
            raise ValueError(f"Message {i} has role '{message['role']}', expected '{expected_role}'")
        if not message['content'].strip():
            raise ValueError(f"Message {i} has empty content")
    # Muss auf assistant enden, sonst kein Lernziel
    if len(messages) % 2 != 0:
        raise ValueError("Conversation must end on an assistant message")
    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic German identity conversations")
    parser.add_argument("--num", type=int, default=300, help="Number of conversations to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--save-metadata", action="store_true", help="Save metadata alongside messages")
    args = parser.parse_args()

    # Default-Ausgabe: DEUTSCHE Identity-Datei (nicht die englische!)
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(get_base_dir(), "identity_synth_de.jsonl")

    if not args.append and os.path.exists(output_file):
        os.remove(output_file)

    print(f"Model: {args.model}")
    print(f"Output file: {output_file}")
    print(f"Generating {args.num} conversations with {args.workers} workers...")
    print(f"Topic categories: {list(topics.keys())}")
    print()

    completed_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_conversation, idx, args.model): idx
                   for idx in range(args.num)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                messages = result["messages"]
                metadata = result["metadata"]
                validate_conversation(messages)
                with open(output_file, 'a', encoding='utf-8') as f:
                    if args.save_metadata:
                        f.write(json.dumps({"messages": messages, "metadata": metadata}, ensure_ascii=False) + '\n')
                    else:
                        f.write(json.dumps(messages, ensure_ascii=False) + '\n')
                completed_count += 1
                topic_short = metadata["topic"][:50]
                print(f"[{completed_count}/{args.num}] {topic_short}")
            except Exception as e:
                error_count += 1
                print(f"[ERROR] idx={idx}: {e}")

    print()
    print(f"Done! Saved {completed_count} conversations to {output_file}")
    if error_count > 0:
        print(f"Encountered {error_count} errors during generation")
