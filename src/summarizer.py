# src/summarizer.py
import os
import openai
from typing import List
from math import ceil
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CHAT_CONTEXT_TOKENS = int(os.getenv("MAX_CHAT_CONTEXT_TOKENS", "3500"))

openai.api_key = OPENAI_KEY

def call_chatgpt(system: str, user: str, model: str = MODEL, max_tokens: int = 1024):
    """
    Simple wrapper for ChatCompletions (chat completion usage).
    """
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=max_tokens,
        temperature=0.0
    )
    return resp['choices'][0]['message']['content']

def iterative_summarize(chunks: List[str], target_token_budget: int = 2000) -> str:
    """
    Iteratively reduce chunks to a single summary that fits into target_token_budget.
    Strategy:
        - Group chunks into batches that, when summarized, reduce number of chunks.
        - Keep summarizing until we have a single block.
    """
    # conservative: assume ~4 chars per token => tokens ≈ chars/4
    def approx_tokens(s: str):
        return max(1, int(len(s) / 4))

    # If already small, join
    joined = "\n\n".join(chunks)
    if approx_tokens(joined) < target_token_budget:
        return joined

    summaries = chunks
    round_num = 0
    while True:
        round_num += 1
        batched = []
        batch = []
        batch_tokens = 0
        for ch in summaries:
            t = approx_tokens(ch)
            if batch_tokens + t > target_token_budget * 0.8 and batch:
                batched.append(batch)
                batch = [ch]
                batch_tokens = t
            else:
                batch.append(ch)
                batch_tokens += t
        if batch:
            batched.append(batch)

        new_summaries = []
        for i, b in enumerate(batched):
            text = "\n\n".join(b)
            system = "You are a concise summarizer. Produce a short precise summary capturing the important facts, named entities, and any numeric details."
            prompt = f"Summarize the following text into a concise paragraph (aim for <= {int(target_token_budget/4)} tokens):\n\n{text}"
            summary = call_chatgpt(system, prompt, max_tokens= int(target_token_budget/2))
            new_summaries.append(summary.strip())

        summaries = new_summaries
        # stop if single summary or small enough
        joined = "\n\n".join(summaries)
        if len(summaries) == 1 or approx_tokens(joined) < target_token_budget:
            break
        # safety cap
        if round_num > 6:
            break

    return summaries[0] if len(summaries) == 1 else joined
