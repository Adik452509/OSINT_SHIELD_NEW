# Propaganda annotation guide

Used for the M8 re-annotation (D18). The same definitions are given to Llama when it screens the
corpus, so the screener and the human reviewer apply one standard.

---

## The one rule that decides most cases

> **Label the article's own voice, not the voices it reports.**

An article is propaganda only if **its own words** — the headline, the narration, the framing the
journalist or outlet chose — use one of the four techniques below.

**Reporting** that someone else said something loaded is **not** propaganda, even if what they
said is propaganda. The article becomes propaganda when it **adopts** the framing: drops the
attribution, repeats the loaded term as fact, or echoes it in its own headline.

| article text | label | why |
|---|---|---|
| *Will bring laws against 'love jihad', table UCC soon: Bengal CM Adhikari* | **No** | attributed, quoted statement — reporting |
| *Bengal finally acts against love jihad menace* | **Yes — Justification** | the outlet uses the loaded term as its own fact |
| *Six brave sons were sacrificed in Operation Sindoor* | **Yes — Glorification** | glorifying language in the outlet's own headline |
| *Six soldiers killed in Operation Sindoor; government releases names* | **No** | neutral reporting of the same event |

---

## The four techniques

### 1 · Glorification
The article's own voice exalts the military, national sacrifice or the country's role **beyond
what neutral reporting needs**: heroic superlatives, sacred language for death in service,
triumphal framing.

- **Yes:** "brave sons", "martyred heroes", "glorious victory", "India became a shield in crisis",
  "the nation bows to its warriors".
- **No:** factual casualty reports; an official's tribute that is clearly attributed; ceremonial
  events described plainly ("wreaths were laid at the memorial").

### 2 · Justification
The article's own voice uses loaded or one-sided framing to **justify a position or delegitimise
an opponent**: scapegoating, conspiracy framing, partisan slurs presented as fact, emotionally
loaded labels in place of description.

- **Yes:** "appeasement politics" or "vote-bank politics" stated as fact; "foreign sabotage" of
  national programmes asserted without evidence; headlines built on insinuation ("Is X trying to
  sabotage India…?").
- **No:** analysis that argues a position with evidence; reporting a party's accusation with
  attribution; criticism stated in neutral terms.

### 3 · Victimhood
The article's own voice frames a person, group or nation as **persecuted or wronged in order to
serve a partisan narrative**, rather than reporting a grievance.

- **Yes:** "wrongly imprisoned", "silenced", "political prisoner" asserted as fact in a one-sided
  piece; framing every development as a deliberate attack on a group.
- **No:** reporting a court verdict and both sides' reactions; quoting a rights group's statement
  with attribution.

### 4 · Recruitment
The article's own voice **calls readers to act, mobilise or join**.

- **Yes:** "join the cause", "rise up", "answer the call", "defend your faith", "volunteers needed"
  addressed to the reader.
- **No:** reporting that an organisation is recruiting; a quoted rally speech with attribution.

An article can use more than one technique — tick every one that applies.

---

## Decision procedure

For each article, in order:

1. **Find the loaded language.** Is there any glorifying, delegitimising, victimising or
   mobilising language at all? If not → **No**.
2. **Whose voice is it?** Is it inside an attributed quote or a "X said" construction? If it
   is **only** there → **No**.
3. **Does the article adopt it?** Does the headline or narration repeat it as fact, drop the
   attribution, or add its own loaded framing? If yes → **Yes**, and name the technique(s).
4. **Quote the evidence** — copy the exact words from the article's own voice that decided it.

When genuinely unsure after step 3, label **No** and mark it **unsure**. A small set of
confident positives is worth more than a large noisy one — that is the lesson of M7.

---

## What you fill in for each article

| column | values |
|---|---|
| `label` | `yes` / `no` |
| `types` | any of `glorification`, `justification`, `victimhood`, `recruitment` (comma-separated); empty if `no` |
| `sure` | `sure` / `unsure` |
| `evidence` | the exact words that decided a `yes` |
| `notes` | optional |

You will not be shown why an article is in your sheet, or what Llama or the original annotator
said. That is deliberate: knowing it would pull your judgement towards theirs.
