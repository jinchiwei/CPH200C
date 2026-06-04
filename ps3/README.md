# PS3 — Three Governing Systems for (Clinical) AI

CPH 200C Problem Set 3 (Recent Advances in Deployed Clinical Models). Deliverable: a 5–15 min class presentation. **Authors: Curtis Chambers · Jinchi Wei.**

Topic (pset #10, additional clinical applications): how AI deployment differs by **geographic region** — US, Europe, Asia — across **policy, funding, technical developments, and landscape**, with deployed clinical AI as the central, research-anchored test case.

## Deliverable

| File | What |
|---|---|
| `CPH200C_PS3_Chambers-Wei.pptx` | **The talk** (~10 min, current slot). 22 slides (14 content + 5 section dividers + cover + thanks), clinical-AI-centered, organized by the four axes → synthesis. Broad-over-deep cut: keeps the intro trio, both chip-war slides, the US frontier-standing slide (Opus 4.8 / GPT-5.5 / Gemini 3.1 on top; China a cheap, open fast-follower), the real-world deployed comparison (Aidoc / Viz.ai · Oxipit ChestLink · Infervision / Ant AQ / DeepSeek-R1), "Asia beyond China" (India / Korea / Japan / SE-Asia), and ROI. Trims the two deepest methodology slides (clinical-AI measurement; clinician-AI interaction), which remain in the 15-min variant and the report. Speaker notes embedded in the PowerPoint notes pane. |
| `CPH200C_PS3_Chambers-Wei.pdf` | Slides as PDF. |
| `CPH200C_PS3_Chambers-Wei_notes.pdf` | Presenter handout: one page per slide (slide image + full speaker notes). |
| `CPH200C_PS3_Chambers-Wei_15min.pptx` / `.pdf` / `_notes.pdf` | The longer **15-min variant** (24 slides) — same deck plus the two methodology slides. Kept in case the slot expands. |
| `CPH200C_PS3_report.pdf` / `.docx` | **The written report** (~21 pp, Jin-branded): a comprehensive prose treatment of all four scopes + the clinical-AI thread, with in-text citations and a full reference list. Built from `report.md`. Includes §4.4 (the US frontier standing) and named real-world deployed systems in §6. |
| `report.md` | Source markdown for the written report (prose, cited). |
| `research.md` | The underlying cited dossier / source notes (bullet form): every region × scope + the clinical-AI cross-comparison, with URLs and caveats. |
| `reference_deck_full.pptx/.pdf` | Supplementary 39-page comprehensive reference deck (broad US/EU/Asia comparison across all scopes). |
| `talk-10min.md` + `.layout.json` | Build source for the 10-min primary talk: slide-delimited markdown + the build-pptx layout sidecar (reproducible re-render). |
| `talk-15min.md` + `.layout.json` | Build source for the 15-min variant. |
| `figs/` | Analytical figures (matplotlib, brand-styled) + a public-domain / CC0 radiology gallery on the lens slide (chest X-ray, mammogram, retinal fundus, chest CT), each tagged with a real deployed AI system (Oxipit ChestLink, Lunit INSIGHT, IDx-DR, Infervision). |

## Thesis

Three regions run three different AI governing systems: **US = market-led + closed frontier + capital dominance; Europe = comprehensive regulation + little building; Asia (China-led) = state-driven + open-weight + population-scale deployment + owns the hardware.** The same split defines clinical AI: the US *approves* the most tools, Europe *regulates* hardest, Asia *deploys* at the most scale. The deployed-AI bottleneck is institutional (regulation, payment, coordination), not model quality.

A sharper sub-thesis (Curtis): US export controls may be backfiring — the GPU ceiling is pushing China toward genuine method innovation (MoE, distillation, low-cost training) while the US mostly scales hardware.

## Evidence base

Peer-reviewed and primary sources throughout, several from the assignment's own reading list: Han et al. (Lancet Digital Health 2024), Goh et al. (JAMA Network Open 2024), Eisemann et al. (Nature Medicine 2025), Bedi et al. (npj Digital Medicine 2025 / HealthAdminBench 2026), Mello & Rose (JAMA Health Forum 2024), Arora et al. (HealthBench 2025), the Stanford AI Index 2026, and the MIT NANDA report. Full source list with URLs in `report.md`.

Built with the build-pptx pipeline (Geist brand, `bone` theme, expressive mode). UCSF + Cal logos on the title slide.
