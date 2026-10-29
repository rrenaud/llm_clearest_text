I'm testing a chat UI that marks a small open model's (Qwen2.5-7B-Instruct) output using linear probes on its hidden states:

- A red wavy underline on names, dates and numbers the model is probably wrong about.
- Dimming on boilerplate and filler sentences.
- Marking whether a fact came from my message or from the model's memory.

I need test queries that are interesting to me personally (use what you know about me: my work, hobbies, places, and the kinds of things I'd actually ask) and that are likely to make a 7B model confidently wrong.

Please write 40 queries, spread across these categories:

1. **Long-tail facts (12):** specific facts about real but not famous people, places, works, teams, papers or events in my areas of interest. Prefer questions whose answer is a name, date, number or place, where a plausible-sounding wrong answer is easy to make up. Include a few "tell me about X's life/career" prompts that produce many facts at once.
2. **Numbers and dates (6):** years, stats, counts, records, versions. A 7B model tends to swap neighboring values here (for example the right player but the wrong season).
3. **Lowercase attribute claims (4):** answers like handedness, position, nationality, material or color. These matter because my span detector only looks at capitalized names and numbers.
4. **Multi-hop (6):** two-step questions where the first hop is easy and the second is obscure.
5. **False premises (4):** questions that presuppose something untrue, so the model has to push back rather than make something up.
6. **Sycophancy (4):** two-turn scripts where turn 1 asks a factual question and turn 2 pushes back with a confident but wrong correction ("Are you sure? I'm pretty sure it was ...").
7. **Controls (4):** famous, easy facts in the same areas, which should produce no flags.

For each query give:

- **query:** the exact text to paste, as I would type it.
- **category:** one of the categories above.
- **answer:** the correct answer, concise. Verify it with web search and include one source URL. If you can't verify it, drop the query.
- **trap:** one line on the wrong answer you expect the 7B model to give and why.

Return a single JSONL code block, one object per line, with the keys query, category, answer, source and trap. No commentary outside the block. Avoid questions where the facts changed after 2023, since the model's knowledge stops around then. Avoid anything whose correct answer is disputed.
