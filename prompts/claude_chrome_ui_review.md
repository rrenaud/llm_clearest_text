You're reviewing a research prototype: a chat UI that marks up a language model's answers using probes on the model's internal states. Please use it like a skeptical first-time user, find problems, and write a critique. Don't change any settings, don't delete or rename anything, and don't send more than about 10 messages in total (the model runs on a rented GPU, one request at a time).

**Where:** http://127.0.0.1:8502 (Open WebUI, no login). The model is `qwen2.5-7b-instruct-probed`. The `-probed-debug` variant shows scores on every span, and `-plain` has no markup.

**What the markup is supposed to mean:**
- **Red wavy underline:** a name, date or number the probe thinks is likely wrong. Hovering shows the risk score and the model's top alternative words at its least-certain token.
- **Dimmed sentence** (opacity rises on hover): boilerplate (stock disclaimers) or filler (adds nothing). Hovering shows which one, with a score.
- **Dotted blue underline:** a number the model computed that doesn't appear in the user's message.
- **Faint tint:** a fact recalled from memory. This only appears when the user pasted a long text (over 400 characters).
- Flags are meant to be rare: only about the riskiest 5% of spans.

**Pre-made chats:** the sidebar has folders of chats I already ran and classified:
- **working:** flags land on real errors.
- **mixed:** some flags right, some missed or noisy.
- **missed errors:** confidently wrong answers with few or no flags.
- **false alarms:** flags on correct text.

These were saved before two fixes: headings like "Personal Life" and lone acronyms like "GPS" no longer get flagged, and the alternatives no longer list duplicate words with a leading space. Flag those issues in old chats only once; check whether they still happen in new chats.

**What to do:**
1. Open 2–3 chats in each folder. Hover every red underline and dimmed sentence.
2. Start 4–6 new chats of your own, mixing:
   - obscure biography or sports-stats questions likely to produce wrong details;
   - one medical or financial advice question, to see the dimming;
   - one with a pasted paragraph of 500+ characters followed by a question that needs arithmetic on numbers in it, to see the tint and dotted underline.
3. Check each answer yourself (use web search) and compare: which flags were right, which were false alarms, and which errors went unflagged.
4. Try copying an answer to the clipboard, and check that no tags or markup leak into the copied text.
5. Watch the answer stream in. Note any flicker, jumps, raw `<probe` tags, stray spaces before punctuation, or broken Markdown (lists, headings, bold, code blocks) around marked text.
6. Try the UI at a narrow (phone-width) window and in dark mode. Check that tooltips are readable, not clipped, and dismissible.

**Questions to answer in the critique:**
- **First glance:** without being told, would a user understand what the red underline and the dimming mean? Is anything misleading, such as a red underline reading as a spelling error, or dimming reading as "deleted"?
- **Tooltips:** are they useful? Are the risk numbers and the "alternatives" table understandable to a non-expert, and do the alternatives help you spot the right answer?
- **Trust:** does the markup make you trust unflagged text too much? Given how many errors go unflagged, how should the UI communicate that?
- **Dimming:** does the dimming ever hide something substantive?
- **Distraction:** does anything feel noisy, janky or distracting? What would you remove?
- **Top 3 improvements:** which three changes would make it most useful? Consider things like sentence- or paragraph-level "shaky section" tints, a legend, a per-answer summary ("3 claims flagged"), or a toggle.

**Report format:**
1. A short verdict (3–4 sentences).
2. A table of concrete bugs: where (chat title or prompt), what you did, what happened, what you expected, and severity.
3. Probe accuracy on what you checked: counts of correct flags, false alarms and missed errors, with 2–3 examples of each.
4. Usability critique, answering the questions above.
5. Prioritized recommendations.

Quote exact text and give chat titles so each issue can be found again. Screenshots are welcome.
