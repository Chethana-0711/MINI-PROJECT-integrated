# backend/models/llm_infer.py
import os
import json
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", None)

def generate_description(context: dict):
    """
    context: {
      species, species_prob, camo_pct, regions, bbox
    }
    Returns: (description string, [adaptation strings])
    """
    species = context.get("species") or "Unknown species"
    camo = context.get("camo_pct", 0)
    regions = context.get("regions", [])
    # If OPENAI_KEY is set, call OpenAI (user must implement)
    if OPENAI_KEY:
        try:
            import requests
            headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type":"application/json"}
            prompt = (
                f"Given these findings:\n"
                f"Species: {species}\nCamoPercent: {camo}\nRegions: {json.dumps(regions)}\n"
                "Write a 2-3 sentence descriptive analysis of the camouflage and a short list "
                "of specific adaptations (2-4 items). Return as (description, [adaptations])."
            )
            # simple call to OpenAI ChatCompletions (v1/chat/completions)
            payload = {
                "model": "gpt-4o-mini", 
                "messages": [{"role":"user","content":prompt}],
                "temperature":0.3,
                "max_tokens":400
            }
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
            r.raise_for_status()
            resp = r.json()
            content = resp["choices"][0]["message"]["content"]
            # naive split: assume content contains two parts
            # We'll try to parse JSON if model outputs JSON, else simple fallback.
            try:
                # Try to find JSON block
                import re
                m = re.search(r'(\{[\s\S]*\})', content)
                if m:
                    parsed = json.loads(m.group(1))
                    return parsed.get("description",""), parsed.get("adaptations",[])
            except Exception:
                pass
            # fallback: split into sentences
            sentences = content.strip().split("\n")
            description = sentences[0] if sentences else content
            adaptations = [s.strip("- ").strip() for s in sentences[1:4] if s.strip()]
            return description, adaptations
        except Exception as e:
            # fallback to deterministic
            pass

    # deterministic fallback
    desc = f"A camouflage level of {camo}% was detected. The object blends strongly with surrounding textures and colors."
    adapt = []
    if species and species!="Unknown species":
        adapt.append(f"Coloration similar to {species}'s typical background")
    adapt += ["Cryptic coloration", "Patterned skin/feathers to break outline"]
    return desc, adapt
