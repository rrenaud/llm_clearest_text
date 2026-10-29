# Qwen3.6-27B screening review (25 queries, temperature 0, 2026-10-28)

Graded from model knowledge, without web search. Flags are spans at or above risk threshold 0.986.

| # | Query | Folder | Notes |
|---|---|---|---|
| 0 | John Clare | missed errors | Helpston, 1793 and 1820 are right. "Published with Leigh Hunt's help" is wrong (it was John Taylor) and unflagged. |
| 1 | Gesualdo | degenerate | Greedy repetition loop. Parents are wrong (actually Fabrizio Gesualdo and Geronima Borromeo), unflagged. |
| 2 | Fielding 2nd wife | mixed | Says "Charlotte Crutchley"; the answer is Mary Daniel, and the first wife was Charlotte Cradock. Unflagged. Only the fabricated daughter is flagged. |
| 3 | Gooden | mixed | About 11 correct flags: Nolan Ryan as "Dr. K", 2.56 ERA, strikeout titles, Tommy John, Rangers trade, 11-11, 138-102, "Kings of New York". Missed: "lost the 1986 WS to the Red Sox", Game 6 start, Mets return in 1996, Hall of Fame 2022, No. 16 retired in 2014. False alarms: 1985 Cy Young ×2, 1990. |
| 4 | 1983 Booker | missed errors | Coetzee is right. "Only two double winners" ignores Mantel and Atwood. Unflagged. |
| 5 | Rubinstein | mixed | Flags sit near the false "won St. Petersburg 1914" and Holocaust claims. Missed: born and died in Łódź (actually Stawiski and Antwerp), 1906 Carlsbad, 1910 and 1911 wins. |
| 6 | US sub-4 mile | mixed | Says Jim Ryun 1964 3:55.3; the answer is Don Bowden 1957, unflagged. The flags on Bannister's "4:03.2" and on "1964" are right. |
| 7 | Ramanujan mentor | mixed | Seshu Aiyar was his teacher, not "Registrar" (unflagged). "Pillai Venkata" is correctly flagged, "Kumbakonam" is a false alarm, and the fabricated "Prof. R. R. Mahapatra" is unflagged. |
| 8 | Fidrych | degenerate | Loops on "traded to the Detroit Tigers". "University of Michigan" is correctly flagged. The birth date and place and "1st round" are wrong and unflagged. |
| 9 | Wainwright | correct, unflagged | |
| 10 | 1989 Pistons | working | Correct flags: Vinnie Johnson "Sixth Man of the Year", ECF vs "Cleveland" (it was Chicago), Aguirre timing. False alarm: Larry Bird. |
| 11 | Transformer | false alarms | The answer is right; "WMT 2014" is flagged. |
| 12 | Shannon thesis | correct, unflagged | Minor: Hitchcock was the formal supervisor. |
| 13 | Commodore | working | "Commodore Plastic Moulding Company" is correctly flagged (it was the Portable Typewriter Co.). "Ontario" is a false alarm. |
| 14 | Shusaku | mixed | Correct flags: Jwa-in, Oishinbo, the "award", "Meijin at 21". Missed: born in Edo (actually Innoshima), the Meijin title and its forced resignation (both fabricated), the "Sakata Hirota" game. |
| 15 | Wilt 100 | false alarms | Everything is right; "Hershey Sports Arena" is flagged. |
| 16 | Utzon | missed errors | Askin did not succeed Cahill in 1959, Pritzker was 2003 not 1999, the 2003 apology is invented, and there was no "$100M by 1966". Nothing flagged. |
| 17 | Stoner | mixed | Correct flags: Booker International 2014 (invented), Chabon, the "2000s" resurgence. Missed: the 2006 reissue was NYRB, not New Directions, and DFW. |
| 18 | Dominion | missed errors | Mayfair in Europe, the AEG 2015 rights deal, "Magic online" and the DSP 2nd place (it won) are wrong and unflagged. The Golden Geek flag is a false alarm. |
| 19 | Fields 1990 | missed errors | Lists Zelmanov and Yoccoz (1994 winners) instead of Mori and Witten. Nothing flagged. |
| 20 | Meitner | missed errors | The KWI for Physics, the 1912 start of the Hahn collaboration and the Nobel Institute director offer are wrong. Nothing flagged. |
| 21 | Bozeman | false alarms | "John Bozeman" and "2023" are flagged and both are right. The "gold in Big Hole Valley" cabin story is wrong and unflagged. |
| 22 | Headaches | false alarms | "Sleep Apnea" is flagged. The disclaimer is correctly dimmed as boilerplate. |
| 23 | Refinance | false alarms | "0.5%" is flagged. The opening sentence that carries the answer is dimmed as filler. |
| 24 | Marathon | correct, unflagged | |

Tally: about 33 correct flags and about 14 false alarms (precision about 70%). Recall is low: whole
fabricated claims went unflagged (Fields 1990, Fielding's wife, Utzon, Ryun).

# Premade questions (13 of 56, temperature 0.7, top-p 0.8; graded against prompts/premade_eval.jsonl)

| # | ID | Grade | Folder | Notes |
|---|---|---|---|---|
| 0 | ml-001 Chinchilla | correct | false alarms | 20:1, Hoffmann 2022, 70B/1.4T all right. Minor: "216 models on C4" is wrong. "Pre-Chinchilla Era" heading flagged. |
| 1 | ml-002 misspecification | correct | correct, unflagged | |
| 2 | ml-003 KV cache | wrong | missed errors | "Linear" is right, but the formula has no layer count (uses L for sequence length). Inside math, so unchecked. |
| 3 | ml-004 Llama-2-70B KV | wrong | mixed | Assumes MHA with 64 KV heads and gets 2.5 MB: the expected error. The flag lands on the pivot, "Multi-Query Attention" (alternatives include "Group" 14%), but the 2.5 MB conclusion is unflagged. |
| 4 | ml-005 Q/K/V | correct | correct, unflagged | |
| 5 | ml-006 FlashAttention | correct | correct, unflagged | Minor unflagged slip: says FA makes attention "memory-bandwidth limited rather than compute-limited" (backwards). |
| 6 | ml-007 Hopfield capacity | correct | false alarms | Ramsauer, exp vs linear. "MHN"/"MHNs" flagged twice. |
| 7 | ml-008 0.138N | correct (rambling) | mixed | Ends at 0.138N but dithers over "1/(2 ln 2) = 0.721". Invents "Kurt A. Kaneko" and "Werner & Köhler" (the 1985 flag is next to them) and says Hopfield himself gave 0.138 (he gave about 0.15). |
| 8 | ml-009 Expert Choice | partly correct | false alarms | Router and Expert Choice are right. The batch-1 argument ("all experts process the token") misses the train/inference mismatch. "Multiple" flagged twice. |
| 9 | ml-010 Switch | correct core | mixed | k=1 and Fedus/Zoph/Shazeer are right. The "TACL" venue is correctly flagged (it was JMLR). "arXiv July 2021" (actually January), the straight-through estimator and "top-k needs load balancing, top-1 doesn't" are unflagged. The flag on the year 2022 is a false alarm. |
| 10 | ml-011 MoE specialization | correct core | missed errors | The conclusion (token/syntax level) is right. The citations "Hernandez et al. 2018" and "Merrill et al. 2024" are invented and unflagged. |
| 11 | ml-012 Mamba-2 vs GDN | wrong | missed errors | The Gated DeltaNet update has no delta rule (it's an EMA). The only flag was inside a $$ block (streaming bug, now fixed in segment.py). |
| 12 | ml-013 Mamba | partly | missed errors | Gu & Dao is right. The NYU affiliation, July 2023, ICLR 2024 acceptance, the Mamba-2 title and "SSM vs convolution" duality are wrong and unflagged. "EleutherAI" is correctly flagged. |
