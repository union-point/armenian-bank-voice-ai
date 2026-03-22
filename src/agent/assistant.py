import json
import logging
from typing import Optional

from livekit.agents import Agent, StopResponse, llm

from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """
You are a banking query classifier for Armenian banks. 
Analyze the user's query and output ONLY a valid JSON object.

JSON fields:
- "category": Choose one of ["credit", "deposit", "branch"] if the query is specifically about one of these. Otherwise, null.
- "bank_name": The name of the bank if mentioned (e.g., "acba", "idbank", "vtb"), otherwise null.
- "in_scope": true if the query is about bank loans (credit), deposits, or branch information/locations. false if it is about any other topic (weather, general talk, other services).

Examples:
"Ի՞նչ վարկեր ունեք" -> {"category": "credit", "bank_name": null, "in_scope": true}
"Որտե՞ղ է Ակբա մասնաճյուղը" -> {"category": "branch", "bank_name": "acba", "in_scope": true}
"Ինչպե՞ս է ձեր տրամադրությունը" -> {"category": null, "bank_name": null, "in_scope": false}

Output ONLY JSON.
"""

REFUSAL_MESSAGE = "Կներեք, ես կարող եմ պատասխանել միայն բանկի վարկերի, ավանդների և մասնաճյուղերի վերաբերյալ հարցերին։"


class BankAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""Դուք բանկային օգնական եք։

Ձեր դերը՝ պատասխանել հաճախորդների հարցերին միայն հետևյալ թեմաների շրջանակում՝

* Վարկեր (Credits)
* Ավանդներ (Deposits)
* Մասնաճյուղեր (Branch locations)

ԿԱՆՈՆՆԵՐ:

1. ՕԳՏԱԳՈՐԾԵՔ ՄԻԱՅՆ ՏՐԱՄԱԴՐՎԱԾ CONTEXT-Ը

* Պատասխանեք միայն "Տեղեկատվություն" բաժնում տրված տվյալների հիման վրա
* ՉՕԳՏԱԳՈՐԾԵՔ ձեր սեփական գիտելիքները

2. ԽՍՏԻՎ ՍԱՀՄԱՆԱՓԱԿՈՒՄ (SCOPE)

* Եթե հարցը դուրս է այս թեմաներից → ՄԵՐԺԵՔ
* Եթե հարցը կապված է, բայց context-ում չկա պատասխան → ՄԵՐԺԵՔ

3. ՊԱՏԱՍԽԱՆԻ ՈՃ

* Կարճ և հստակ
* Բարեկիրթ և պրոֆեսիոնալ
""",
        )

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """
        Intervene after user turn is completed to perform intent classification and RAG.
        This hook runs before the LLM generates a response.
        """
        transcript = new_message.text_content
        if not transcript:
            return

        logger.info(f"RAG Hook: Processing user turn: {transcript!r}")

        # delegate query understanding to the LLM
        extraction_ctx = llm.ChatContext()
        extraction_ctx.add_message(role="system", content=EXTRACTION_SYSTEM_PROMPT)
        extraction_ctx.add_message(role="user", content=transcript)

        extraction_text = ""
        try:
            async with self.session.llm.chat(chat_ctx=extraction_ctx) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        extraction_text += chunk.delta.content

            # parse JSON response
            clean_json = extraction_text.strip().strip("`").replace("json\n", "")
            data = json.loads(clean_json)
            category = data.get("category")
            bank_name = data.get("bank_name")
            in_scope = data.get("in_scope", True)
        except Exception as e:
            logger.error(f"Extraction failed: {e}. Raw: {extraction_text!r}")
            # Recovery: perform broad retrieval if extraction fails
            category, bank_name, in_scope = None, None, True

        logger.info(
            f"Extracted: category={category!r}, bank_name={bank_name!r}, in_scope={in_scope}"
        )

        # Guardrails: If out of scope, refuse immediately
        if not in_scope:
            logger.info("Guardrail hit: Query out of scope. Blocking reply.")
            await self.session.say(REFUSAL_MESSAGE)
            raise StopResponse()

        results = retrieve(
            query=transcript, category=category, bank_name=bank_name, top_k=5
        )

        if results:
            context_str = "\n\n".join([f"{r.text}" for r in results])
            turn_ctx.add_message(
                role="system",
                content=f"""ՏԵՂԵԿԱՏՎՈՒԹՅՈՒՆ (Context):

{context_str}

Օգտագործեք ՄԻԱՅՆ վերևի տեղեկատվությունը պատասխանելու համար։
Եթե պատասխանը չկա → ՄԵՐԺԵՔ։
""",
            )
            logger.info(f"Injected {len(results)} chunks into chat context.")
        else:
            logger.info("No relevant chunks found in RAG.")
