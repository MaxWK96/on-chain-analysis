import os
import json
import requests
from flask import Flask, request, jsonify
import anthropic

app = Flask(__name__)

ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def fetch_transactions(address: str) -> list:
    """Fetch up to 100 most recent transactions from Etherscan."""
    resp = requests.get(
        "https://api.etherscan.io/v2/api",
        params={
            "chainid": 1,
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": 100,
            "sort": "desc",
            "apikey": ETHERSCAN_API_KEY,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if data["status"] == "0":
        msg = data.get("message", "")
        if "No transactions" in msg:
            return []
        raise ValueError(f"Etherscan: {msg}")

    return data["result"]


def build_tx_summary(transactions: list) -> list:
    """Convert raw Etherscan transactions to a compact, token-friendly format."""
    summary = []
    for tx in transactions[:50]:  # cap at 50 to stay well within context
        summary.append({
            "hash": tx["hash"],
            "from": tx["from"],
            "to": tx["to"] or "(contract creation)",
            "value_eth": round(int(tx["value"]) / 1e18, 6),
            "gas_used": tx.get("gasUsed"),
            "timestamp_unix": tx["timeStamp"],
            "status": "error" if tx["isError"] == "1" else "ok",
            "function": tx.get("functionName", "")[:80],
        })
    return summary


@app.route("/analyze-wallet", methods=["POST"])
def analyze_wallet():
    if not ETHERSCAN_API_KEY or not ANTHROPIC_API_KEY:
        return jsonify({"error": "Server misconfiguration: API keys not set"}), 500

    body = request.get_json(silent=True)
    if not body or "address" not in body:
        return jsonify({"error": "Request body must contain 'address'"}), 400

    address = body["address"].strip()
    if not address.startswith("0x") or len(address) != 42:
        return jsonify({"error": "Invalid Ethereum address (must be 0x + 40 hex chars)"}), 400

    # 1. Fetch on-chain data
    try:
        transactions = fetch_transactions(address)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 502
    except requests.RequestException as exc:
        return jsonify({"error": f"Etherscan request failed: {exc}"}), 502

    tx_summary = build_tx_summary(transactions)
    total_tx = len(transactions)
    shown_tx = len(tx_summary)

    # 2. Build prompt (Swedish legal report)
    prompt = f"""Du är en juridisk analytiker med specialisering inom blockchain-teknologi och kryptovalutor.
Analysera nedanstående Ethereum-plånbok och generera en professionell juridisk sammanfattningsrapport på svenska.

Plånboksadress: {address}
Totalt antal hämtade transaktioner: {total_tx} (rapport baserad på de {shown_tx} senaste)

Transaktionsdata (JSON):
{json.dumps(tx_summary, indent=2, ensure_ascii=False)}

---

Rapporten ska vara strukturerad enligt följande rubriker:

1. SAMMANFATTNING
   Övergripande bild av plånbokens aktivitet, ålder och volym.

2. TRANSAKTIONSMÖNSTER OCH VOLYMER
   Totala in-/utflöden (ETH), frekvens, genomsnittliga belopp och eventuella upprepade motparter.

3. ANMÄRKNINGSVÄRDA TRANSAKTIONER
   Lyft fram transaktioner med ovanligt höga belopp, felstatus, kontraktsskapande eller okänd funktion.

4. JURIDISKA ÖVERVÄGANDEN
   - Skattemässiga implikationer (kapitalvinstskatt, inkomstskatt)
   - AML/KYC-aspekter (penningtvättsrisker, misstänkta mönster)
   - Regulatorisk exponering (MICA, FATF Travel Rule m.fl.)

5. SLUTSATSER OCH REKOMMENDATIONER
   Sammanfattande bedömning och eventuella åtgärdsrekommendationer.

Rapporten ska hålla juridisk standard och vara lämplig för dokumentation, revision eller domstolsanvändning."""

    # 3. Call Claude
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            report = stream.get_final_message()

        text = next((b.text for b in report.content if b.type == "text"), "")
    except anthropic.APIError as exc:
        return jsonify({"error": f"Claude API error: {exc}"}), 502

    return text, 200, {"Content-Type": "text/plain; charset=utf-8"}


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))