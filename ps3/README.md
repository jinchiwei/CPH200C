# PS3 — Three Governing Systems for (Clinical) AI

CPH 200C Problem Set 3 (Recent Advances in Deployed Clinical Models). Deliverable: a 5–15 min class presentation. **Authors: Curtis Chambers · Jinchi Wei.**

Topic (pset #10, additional clinical applications): how AI deployment differs by **geographic region** — US, Europe, Asia — across **policy, funding, technical developments, and landscape**, with deployed clinical AI as the central, research-anchored test case.

## Deliverable

| File | What |
|---|---|
| `CPH200C_PS3_Chambers-Wei.pptx` | **The talk** (~15 min). 21 slides, clinical-AI-centered, organized by the four axes → synthesis. Speaker notes embedded in the PowerPoint notes pane. |
| `CPH200C_PS3_Chambers-Wei.pdf` | Slides as PDF. |
| `CPH200C_PS3_Chambers-Wei_notes.pdf` | Presenter handout: one page per slide (slide image + full speaker notes). |
| `CPH200C_PS3_report.pdf` / `.docx` | **The written report** (~16 pp, Jin-branded): a comprehensive prose treatment of all four scopes + the clinical-AI thread, with in-text citations and a full reference list. Built from `report.md`. |
| `report.md` | Source markdown for the written report (prose, cited). |
| `research.md` | The underlying cited dossier / source notes (bullet form): every region × scope + the clinical-AI cross-comparison, with URLs and caveats. |
| `reference_deck_full.pptx/.pdf` | Supplementary 39-page comprehensive reference deck (broad US/EU/Asia comparison across all scopes). |
| `talk-15min.md` + `.layout.json` | Build source: slide-delimited markdown + the build-pptx layout sidecar (reproducible re-render). |
| `figs/` | Analytical figures (matplotlib, brand-styled) + the public-domain chest radiograph used on the lens slide. |

## Thesis

Three regions run three different AI governing systems: **US = market-led + closed frontier + capital dominance; Europe = comprehensive regulation + little building; Asia (China-led) = state-driven + open-weight + population-scale deployment + owns the hardware.** The same split defines clinical AI: the US *approves* the most tools, Europe *regulates* hardest, Asia *deploys* at the most scale. The deployed-AI bottleneck is institutional (regulation, payment, coordination), not model quality.

A sharper sub-thesis (Curtis): US export controls may be backfiring — the GPU ceiling is pushing China toward genuine method innovation (MoE, distillation, low-cost training) while the US mostly scales hardware.

## Evidence base

Peer-reviewed and primary sources throughout, several from the assignment's own reading list: Han et al. (Lancet Digital Health 2024), Goh et al. (JAMA Network Open 2024), Eisemann et al. (Nature Medicine 2025), Bedi et al. (npj Digital Medicine 2025 / HealthAdminBench 2026), Mello & Rose (JAMA Health Forum 2024), Arora et al. (HealthBench 2025), the Stanford AI Index 2026, and the MIT NANDA report. Full source list with URLs in `report.md`.

Built with the build-pptx pipeline (Geist brand, `bone` theme, expressive mode). UCSF + Cal logos on the title slide.
