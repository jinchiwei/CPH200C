# Research dossier — AI deployment patterns across the US, Europe, and China/Asia

Source-of-truth for the deck `regional-ai-deployment`. Compiled 2026-05-30 from a 7-agent parallel web sweep + the local `china-geopolitics` autofeeder. Lens: broad AI deployment with a healthcare/clinical-AI thread. Every nontrivial claim carries a source; figures flagged as estimates/press-reported are marked.

> **The one-line thesis:** three regions, three operating systems. **US = market-led + light federal + closed frontier + capital dominance.** **EU = comprehensive horizontal regulation (AI Act) + sovereignty money but little frontier building.** **China = state-driven vertical rules + industrial policy + open-weight export strategy + population-scale deployment.** They differ less in *whether* they deploy AI than in *how* — and the gap between research and realized deployment is wide everywhere.

---

## 0. The framing: Regulate / Build / Deploy

| | United States | Europe / EU | China |
|---|---|---|---|
| **Governance style** | Market-led, executive-order driven, light federal + state patchwork | Comprehensive, horizontal, risk-tiered (AI Act) | State-driven, vertical/iterative, ministry rules |
| **Frontier models** | Dominant, mostly **closed** (OpenAI, Anthropic, Google) | Thin (Mistral, DeepMind-London under Google) | Fast-following, mostly **open-weight** (DeepSeek, Qwen) |
| **Posture 2025-26** | Deregulate to "win the race" | *Softening* under competitiveness pressure | Pilots-now-statute-later; push global standards |
| **Deployment shape** | Enterprise SaaS / vendor layer | Compliance-gated, slow | Super-app + state-coordinated + hardware-integrated |
| **Capital** | ~83% of global private AI investment | Structural late-stage gap | State guidance funds replacing pulled-back VC |
| **Binding constraint** | Power / grid | Compute + capital + its own regulation | Advanced chips (export controls) |

---

## 1. United States

### Policy & politics
- **Jan 20 2025:** Trump rescinds Biden **EO 14110** (Oct 2023). **Jan 23 2025:** EO "Removing Barriers to American Leadership in AI." ([Wiley](https://www.wiley.law/alert-President-Trump-Revokes-Biden-Administrations-AI-EO-What-To-Know), [White House](https://www.whitehouse.gov/presidential-actions/2025/01/removing-barriers-to-american-leadership-in-artificial-intelligence/))
- **Jul 23 2025:** "Winning the Race: America's AI Action Plan" — 3 pillars (accelerate innovation, build infrastructure, lead diplomacy/security); deregulation, open-weight encouragement, data-center permitting. Plus **EO 14319 "Preventing Woke AI in the Federal Government"** (ideological-neutrality procurement). ([White House PDF](https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf), [Skadden](https://www.skadden.com/insights/publications/2025/07/the-white-house-releases-ai-action-plan))
- **State preemption fight:** Senate stripped the proposed state-AI-law moratorium **99–1** (summer 2025); dropped from FY2026 NDAA. **Dec 11 2025 EO** directs DOJ/Commerce/FCC/FTC to challenge "burdensome" state laws + an AI Litigation Task Force. ([StateScoop](https://statescoop.com/state-ai-law-moratorium-omitted-2026-defense-bill-trump-eo/), [Akin](https://www.akingump.com/en/insights/alerts/president-trump-unveils-ai-eo-advancing-federal-preemption-of-state-laws))
- **State patchwork:** Colorado AI Act (delayed to **Jun 30 2026**); Texas TRAIGA (effective **Jan 1 2026**); California **SB 53** Transparency in Frontier AI Act (signed Sep 29 2025, effective Jan 1 2026; successor to vetoed SB 1047); **NY RAISE Act** (signed Dec 19 2025, effective Jan 1 2027, penalties to $1M/$3M); Illinois employment + AI-therapy-ban laws. ([Mintz](https://www.mintz.com/insights-center/viewpoints/54731/2025-10-03-charting-future-ai-governance-californias-sb-53-sets), [Governor NY](https://www.governor.ny.gov/news/governor-hochul-signs-nation-leading-legislation-require-ai-frameworks-ai-frontier-models))

### Corporate
- **Anthropic** ~$380B (Series G) → reported **~$965B** valuation, run-rate revenue reportedly ~$47B (press). **OpenAI** ~$852B after ~$122B raise. **NVIDIA** market cap **~$5.2T** (#1). **Alphabet ~$4.8T**, closing (Gemini 750M users, owns full stack/TPUs). **xAI** absorbed into SpaceX (~$1.25T combined, Feb 2026). **Meta pivoted from open Llama toward closed models** after Llama 4 stumbled + DeepSeek leveraged its weights. ([CNBC](https://www.cnbc.com/2026/05/28/anthropic-open-ai-startup-value.html), [WinBuzzer](https://winbuzzer.com/2025/12/09/meta-pivots-from-llama-to-closed-ai-models-abandoning-open-source-roots-xcxwbn/))

### Compute & energy
- **Project Stargate** (OpenAI/SoftBank/Oracle/MGX), Jan 2025: up to **$500B / 10 GW** by 2029; >$450B committed by late 2025. Hyperscaler **2026 capex ~$700B** (~2x 2025's ~$388B). ([OpenAI](https://openai.com/index/five-new-stargate-sites/), [CNBC](https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html))
- **Power is the binding constraint:** Microsoft $80B Azure backlog unfulfillable on power; ~16 GW US pipeline but only ~5 GW under construction; nuclear deals (Three Mile Island/Crane 835 MW ~2028, Google-Kairos SMRs, Amazon-Susquehanna). ([tech-insider](https://tech-insider.org/us-ai-data-center-delays-cancellations-7gw-capacity-crisis-2026/))

### Health AI (thread)
- FDA cumulative AI/ML device authorizations **~1,200–1,451 by end-2025** (peer-reviewed npj snapshot 1,016 mid-2025; tracker ~1,451 end-2025); record **295 cleared in 2025**. **Radiology ~76%.** Median 2025 review 142 days. ([Nature npj](https://www.nature.com/articles/s41746-025-01800-1), [IntuitionLabs](https://intuitionlabs.ai/articles/fda-ai-medical-device-tracker))
- **PCCP** final guidance (Dec 2024) lets pre-authorized model updates ship without new submission (only ~10% of 2025 clearances used it). **Jan 6 2026:** FDA eased CDS + General Wellness guidance — but **silent on generative/LLM CDS** (a regulatory void). ([Covington](https://www.cov.com/en/news-and-insights/insights/2026/01/5-key-takeaways-from-fdas-revised-clinical-decision-support-cds-software-guidance))
- **Reimbursement is the bottleneck:** mostly CPT Cat III (temporary) codes; few Cat I; NTAP caps; Health Tech Investment Act (Apr 2025) proposes a Medicare pathway. ([Bipartisan Policy Center](https://bipartisanpolicy.org/issue-brief/paying-for-ai-in-u-s-health-care/))

---

## 2. Europe / EU

### The AI Act + the 2025-26 softening
- **Regulation 2024/1689** in force **Aug 1 2024**, phased: **prohibited practices Feb 2 2025**, **GPAI obligations Aug 2 2025**, high-risk 2026-27. Penalties up to **€35M / 7% turnover**. ([EC](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai), [Art. 99](https://artificialintelligenceact.eu/article/99/))
- **GPAI Code of Practice** (Jul 2025): signing = presumption of conformity. Google/Microsoft/OpenAI/Anthropic/Amazon signed; **xAI signed only the safety chapter.** ([TTMS](https://ttms.com/eu-ai-act-update-2025-code-of-practice-enforcement-industry-reactions/))
- **Digital Omnibus (the big story):** published **Nov 19 2025**; Council–Parliament provisional deal **May 7 2026**; Parliament approved **Mar 26 2026 (569–45)**. **Delays high-risk to Dec 2 2027 (standalone) / Aug 2 2028 (embedded, incl. medical devices)**; content-watermarking to Nov 2026; adds a **ban on non-consensual deepfake "nudifier" apps**; SME/"small mid-cap" relief. The **AI Liability Directive was scrapped in 2025.** Critics: "Big Tech's roll-back of digital rights." ([White & Case](https://www.whitecase.com/insight-alert/eu-agrees-digital-omnibus-deal-simplify-ai-rules), [Corporate Europe Observatory](https://corporateeurope.org/en/2026/01/article-article-how-big-tech-shaped-eus-roll-back-digital-rights))

### Sovereignty & the "regulates but doesn't build" critique
- **InvestAI** (Paris summit, Feb 2025): mobilize **~€200B**, **€20B for 4–5 AI gigafactories** (~100k chips each); 76 bids worth >€230B; EIB backing (Dec 2025). ([EC IP/25/467](https://ec.europa.eu/commission/presscorner/detail/en/ip_25_467))
- **EuroStack** coalition (200+ firms): **~90% of EU digital infrastructure is foreign/US-controlled.** **Gaia-X widely judged a failure.** ([TechPolicy.Press](https://www.techpolicy.press/europe-tried-to-take-control-of-its-digital-stack-in-2025-where-does-it-stand-now/))
- **Draghi report** facts: **only 11% of EU firms use AI** (75% 2030 target); **73% of foundation models US-origin vs 15% China**; 2023 AI VC **EU $8B vs US $68B**; only **4 of top-50 tech firms are European.** ([Carnegie](https://carnegieendowment.org/research/2025/05/the-eus-ai-power-play-between-deregulation-and-innovation))

### Corporate — the application layer + one chokepoint
- **Mistral (FR):** €1.7B Series C (Sep 2025), **€11.7B/~$13.8B**; **ASML led €1.3B for ~11%.** **Helsing** (defense) €600M at €12B. **Black Forest Labs** (powers Grok images) ~$3.25B. **ElevenLabs** $11B (Feb 2026). **Synthesia** ~$4B, **Wayve** $1.2B raised. European AI startups raised a record **$21.6B in 2025**. ([CNBC](https://www.cnbc.com/2025/09/09/ai-firm-mistral-valued-at-14-billion-as-asml-takes-major-stake.html), [Tech.eu](https://tech.eu/2026/05/21/ai-10-companies-that-raised-the-most-in-2025/))
- **The crown jewel is hardware: ASML (NL)** — EUV monopoly (~90% litho share, ~$250M/machine). 2025 net sales **€32.7B**, EUV +39%, net profit €9.6B. Every advanced Nvidia/TSMC chip depends on it. China ~33% of 2025 revenue, guided to ~20% in 2026. ([ASML 6-K](https://www.sec.gov/Archives/edgar/data/0000937966/000162828026003701/a2026_01x28xpresentation.htm))

### Politics
- **France leads the deregulation camp** (Macron €109B; France+Germany backed the Nov 2025 simplification). **US VP JD Vance at Paris (Feb 2025) warned Europe to ease rules.** Big Tech spent **>€35M** lobbying EU tech rules in 2025. ([Investigate Europe](https://www.investigate-europe.eu/posts/france-spearheads-member-state-campaign-dilute-european-artificial-intelligence-regulation), [Corporate Europe Observatory](https://corporateeurope.org/en/2026/01/article-article-how-big-tech-shaped-eus-roll-back-digital-rights))

### Health AI (thread)
- **Double burden:** medical AI = high-risk under AI Act **stacked on MDR/IVDR**; few notified bodies hold both designations → bottleneck. Embedded medical-AI deadline now **Aug 2028.** ([Reed Smith](https://www.reedsmith.com/our-insights/blogs/viewpoints/102kq35/the-eu-ai-act-and-medical-devices-navigating-high-risk-compliance/))
- **European Health Data Space (Reg. 2025/327)** in force Mar 26 2025, but secondary-use (AI-training) provisions apply only from **Mar 2029.** ([Arnold & Porter](https://www.arnoldporter.com/en/perspectives/advisories/2025/03/european-health-data-space-regulation-published))
- **Germany DiGA** = clearest working reimbursement path (~56–61 apps; >1M prescriptions; €234M; 2026 reforms tie ≥20% of price to outcomes). France copied it (PECAN). NHS 10-Year Plan (Jul 2025) names AI a priority but deployment stalls. ([Prova Health](https://www.provahealth.com/insights/diga-reimbursement-germany-guide))

---

## 3. China (+ Japan / South Korea / India)

### Regulation: vertical, iterative, alignment-to-state-values
- **Interim Measures for GenAI** (effective **Aug 15 2023**) — only public-facing services need security assessment + **algorithm filing**; **748 GenAI services filed by Dec 2025.** ([China-Briefing](https://www.china-briefing.com/news/how-to-interpret-chinas-first-effort-to-regulate-generative-ai-measures/))
- **AI-content labeling** ("Measures for Labeling AI-Generated Content" + GB 45438-2025): effective **Sep 1 2025**, requires explicit + implicit (metadata) labels. ([Loeb](https://www.loeb.com/en/insights/publications/2025/03/chinas-ai-labeling-measures-and-mandatory-national-standards-take-effect-september-1))
- **China removed a comprehensive AI law from its 2025 agenda** — pilots/standards/targeted rules instead. Draft verticals: anthropomorphic AI, digital humans, interactive AI services (2026). ([East Asia Forum](https://eastasiaforum.org/2025/12/25/china-resets-the-path-to-comprehensive-ai-governance/))
- **Global governance push:** at WAIC (Shanghai, Jul 2025) Premier Li Qiang proposed a **World AI Cooperation Organization (WAICO)**, HQ Shanghai, courting the Global South. ([gov.cn](https://english.www.gov.cn/news/202507/26/content_WS6884bea8c6d0868f4e8f4732.html)) Autofeeder: "China's AI Governance Offensive Threatens U.S. Tech Leadership."

### Industrial policy
- **NGAIDP (2017):** world leader by 2025, primary global leader + ~$150B industry by 2030. **"AI Plus" Initiative** (State Council, **Aug 26 2025**): >70% smart-terminal/agent penetration by 2027, >90% by 2030 (skeptics call 90% unrealistic). **Homegrown-chip "secure and reliable" procurement** list certifies **9 domestic processors**; state DCs must use domestic chips; auto-industry AI/EV/semiconductor standards blueprint. ([gov.cn](https://english.www.gov.cn/policies/latestreleases/202508/27/content_WS68ae7976c6d0868f4e8f51a0.html), [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/china-certifies-nine-domestic-ai-chips-for-government-procurement))

### Models & deployment
- **DeepSeek R1** (Jan 20 2025, MIT-licensed, claimed ~$5.6M training) → topped US App Store, triggered **~18% Nvidia drop (~$600B wiped)** — the "DeepSeek moment." ([Britannica](https://www.britannica.com/money/DeepSeek))
- **Open-weight ecosystem outpaces the West in model count:** Qwen (Alibaba), Ernie 4.5 (Baidu, Apache-2.0), GLM (Zhipu/Z.ai), Kimi (Moonshot), Hunyuan (Tencent), Doubao (ByteDance), Pangu (Huawei). **Chinese open models >45% of OpenRouter weekly traffic by Apr 2026** (another measure: ~30%); **HF downloads China 17.1% > US 15.86%** (year to Aug 2025). The Global South adopts Chinese open models for "AI sovereignty" (Singapore gov chose Qwen; Malaysia on DeepSeek). ([MIT Tech Review](https://www.technologyreview.com/2026/04/21/1135658/china-open-source-models-ai-artificial-intelligence/), [Stanford HAI](https://hai.stanford.edu/policy/beyond-deepseek-chinas-diverse-open-weight-ai-ecosystem-and-its-policy-implications))
- **Price wars + scale:** Alibaba cut Qwen prices up to 97%; Doubao **>120T tokens/day** (Apr 2026), ~155M WAU; DeepSeek ~81.6M WAU. ([HelloChinaTech](https://hellochinatech.com/p/china-token-economy-140-trillion))
- **Surveillance/state deployment:** ~**700M cameras**; **City Brain 3.0** (Mar 2025) on DeepSeek-R1; shift to active "agent" surveillance. Models bound to "core socialist values." ([China Media Project](https://chinamediaproject.org/2026/02/24/chinese-surveillance-gets-the-ai-treatment/))

### Health AI (thread) — population scale
- **NMPA: 154 AI medical devices by H1 2025, 79.9% Class III** (mandatory trials), ~49.5% CAGR; **radiology 68.8%.** First "world-first" Class III imaging AI (Diagens AutoVision) May 2026. ([JMIR](https://medinform.jmir.org/2026/1/e85538), [GlobeNewswire](https://www.globenewswire.com/news-release/2026/05/27/3301744/0/en/World-s-First-Class-III-Approval-for-a-Breakthrough-Medical-Imaging-AI-Diagens-AI-AutoVision.html))
- **Deployed at scale neither US nor EU has matched:** ~261 hospitals locally deployed **DeepSeek-R1** in Jan–Mar 2025 (84% tertiary); **Ant Group's AQ app** (Jun 2025) links **5,000+ hospitals, ~1M doctors, 100M+ users.** TCM AI (face/tongue/pulse instruments; TCM LLMs BenCao/ShizhenGPT) still mostly research-stage. ([Nature Medicine](https://www.nature.com/articles/s41591-025-03836-y), [Baidu Baike AQ](https://baike.baidu.com/en/item/AQ/1457021))

### Asia brief
- **Japan:** AI Promotion Act (effective Jun 2025) — innovation-first, **no penalties.** **South Korea:** **AI Basic Act effective Jan 22 2026** (2nd comprehensive regime after EU; 1-yr grace). **India:** light-touch + IndiaAI Mission; AI Governance Guidelines (Nov 2025). ([White & Case](https://www.whitecase.com/insight-alert/japans-first-ai-legislation-becomes-law-focus-promoting-research-and-development-no), [Cooley](https://www.cooley.com/news/insight/2026/2026-01-27-south-koreas-ai-basic-act-overview-and-key-takeaways))

---

## 4. Compute & chip-war geopolitics

### Export-control timeline
- **Oct 7 2022** first advanced-chip + SME controls (China). **Oct 17 2023** closed the A800/H800 interconnect loophole, +43 countries. **Dec 2 2024** +140 entities, first HBM controls. **AI Diffusion Rule** (Jan 15 2025) three-tier country framework → **rescinded May 13 2025** (2 days before effect). **H200/MI325X moved to case-by-case (BIS final rule Jan 15 2026)** with **25% tariff, 50% volume cap, US testing, KYC**. House passed **AI Overwatch Act** (bar Blackwell to China 2 yrs). ([CSET](https://cset.georgetown.edu/article/bis-2023-update-explainer/), [BIS](https://www.bis.gov/press-release/department-commerce-rescinds-biden-era-artificial-intelligence-diffusion-rule-strengthens-chip-related), [Introl](https://introl.com/blog/bis-h200-china-export-policy-ai-overwatch-act-2026))

### NVIDIA China saga
- **H20:** Apr 9 2025 effective ban → **~$4.5–5.5B charge**; Jul 2025 resumed under a reported **15% China-revenue cut to the US government** (NVIDIA says not formally finalized). **China discourages purchases, pushes domestic chips**; banned the **RTX 5090D V2 ~May 16 2026 during Jensen Huang's visit**; told firms to **pause H200 purchases.** ([Tom's Hardware H20](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-writes-off-usd5-5-billion-in-gpus-as-us-govt-chokes-off-supply-of-h20s-to-china), [PBS](https://www.pbs.org/newshour/politics/under-new-unusual-agreement-u-s-will-get-a-15-cut-of-nvidia-and-amd-chip-sales-to-china), [Tom's Hardware 5090D](https://www.tomshardware.com/tech-industry/china-banned-nvidia-5090d-v2-while-ceo-jensen-huang-was-in-town-report-claims-move-comes-as-beijing-pushes-its-ai-tech-companies-to-use-homegrown-chips))

### China's domestic stack
- **Huawei Ascend 910C** (SMIC 7nm) ≈ 60% of an H100 per DeepSeek; **CloudMatrix 384** system (~300 BF16 PF) beats GB200 NVL72 on aggregate but not efficiency; target ~600K 910C units 2026 (caveat: TSMC die-bank reportedly exhausted early 2026). Memory: **CXMT** ~$8B 2025 rev (+130%), HBM3 by end-2026; **YMTC**. Domestic AI-GPU share rose to **34%**; Chinese firms shipped **1.65M AI GPUs in 2025 (41% of local AI servers)**, NVIDIA China share <60%. ([SemiAnalysis](https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72), [Tom's Hardware](https://www.tomshardware.com/tech-industry/nvidia-market-share-in-china-falls-to-less-than-60-percent-chinese-chip-makers-deliver-1-65-million-ai-gpus-as-the-government-pushes-data-centers-to-use-domestic-chips))

### Smuggling, energy, ASML
- **~$2.5B Supermicro-linked smuggling case** unsealed Mar 2026; **Taiwan's first GPU-smuggling crackdown** (Japan as transshipment). ([CNBC](https://www.cnbc.com/2026/03/19/us-tech-execs-smuggled-nvidia-chips-to-china-prosecutors-say.html))
- **Energy:** US power-constrained; **China ~40 GW DC capacity by end-2026, ~400 GW spare power by 2030** (added 430 GW wind+solar in 2025) — a structural advantage. ([Al Jazeera](https://www.aljazeera.com/economy/2026/5/28/chinas-secret-weapon-in-ai-race-with-us-lots-of-cheap-energy))
- **ASML:** EU chokepoint; **EU Chips Act 2.0** proposal ~May 27 2026; cumulative commitments >€80B. ([Science|Business](https://sciencebusiness.net/news/semiconductors/eu-chips-act-20-must-better-link-rd-deployment-industry-says))

---

## 5. Adoption & deployment patterns

- **Enterprise:** McKinsey (Nov 2025): **88% of orgs use AI in ≥1 function**, but only ~⅓ scaled, 39% see EBIT impact; **Greater China 56% ≈ North America 57%** overall use. **Eurostat: only 20.0% of EU enterprises used AI in 2025** (Denmark 42% … Romania 5%). US MSMEs 62% vs EU/UK 50%. ([McKinsey](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai), [Eurostat](https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20251211-2))
- **Consumer:** ChatGPT **~900M WAU** (2026), ~2.5x Gemini; **none meaningful in China.** China: Doubao ~155M WAU, DeepSeek ~81.6M WAU. ([technologychecker.io](https://technologychecker.io/blog/chatgpt-statistics), [abhs.in](https://www.abhs.in/blog/china-ai-model-war-doubao-qwen-deepseek-kimi-2026))
- **Open vs closed** is the sharpest structural divergence (see §3). 
- **Deployment channels:** China super-app/state/hardware (WeChat agents, agentic commerce); US enterprise-SaaS/vendor (specialized-vendor buys succeed ~67% vs ~⅓ internal builds); Europe compliance-gated (~60% of EU/UK devs report AI Act launch delays). ([CNBC](https://www.cnbc.com/2026/01/21/china-tech-ai-agentic-commerce-super-apps-alibaba-taobao-qwen-tencent-wechat-doubbao-weixin.html), [Fortune](https://fortune.com/2025/08/18/mit-report-95-percent-generative-ai-pilots-at-companies-failing-cfo/))
- **The ROI gap (headline 2025 story):** MIT Project NANDA — **95% of enterprise genAI pilots delivered no measurable P&L impact** despite $30–40B spend. Top-down estimates remain huge (McKinsey $2.6–4.4T/yr; Goldman ~7% global GDP). ([Fortune](https://fortune.com/2025/08/18/mit-report-95-percent-generative-ai-pilots-at-companies-failing-cfo/))
- **Government readiness:** Oxford Insights 2025 — **US #1, China #8** (from #23). US federal genAI use rose ~9x 2023→2024. ([Oxford Insights](https://oxfordinsights.com/ai-readiness/government-ai-readiness-index-2025/))
- **Research output (AI Index):** notable models 2024 **US 40 / China 15 / Europe 3** (2026 Index: US 50 / Europe 2); **China holds 69.7% of granted AI patents**; US-China benchmark gap near-parity; **of ~$581B global AI investment 2025, ~$344B US.** ([Stanford HAI](https://hai.stanford.edu/news/ai-index-2025-state-of-ai-in-10-charts), [IEEE Spectrum](https://spectrum.ieee.org/state-of-ai-index-2026))

---

## 6. Startups, leniency & support

- **VC by region (2025):** AI took >half of global VC. **US AI startups $222.1B (65% of US VC)**; Stanford AI Index private-AI-investment: **US ~$285.9B vs China $12.4B (~23x gap)**. Europe ~€66B total VC, AI ~39% of it. China private VC fell to <25% of prior year; **state guidance funds** filled in (gov-linked AI deals <10/yr pre-2018 → 140+ in 2025). ([PitchBook/SiliconANGLE](https://siliconangle.com/2026/01/07/pitchbook-ai-dominates-global-venture-capital-2025-deal-value-nears-record/), [Stanford HAI](https://hai.stanford.edu/ai-index/2025-ai-index-report/economy), [Fortune](https://fortune.com/2026/05/19/deepseek-china-ai-venture-capital-nvidia-pitchbook-trends-term-sheet/))
- **Mega-rounds:** OpenAI $40B@$300B; Anthropic to ~$965B; xAI ~$200–230B; Mistral €11.7B; Safe Superintelligence $5B→$32B. ([Visual Capitalist](https://www.visualcapitalist.com/ranked-the-biggest-ai-funding-rounds-of-2025-so-far/))
- **Leniency/sandboxes:** UK most explicitly pro-innovation (FCA "supercharged sandbox" + AI Live Testing; "won't come after you every time something goes wrong"). EU AI Act mandates ≥1 sandbox/state — **delayed to Aug 2027**, only ~8/27 ready. China "pilots now, statute later" (lenient on deployment, strict on speech/data). US light-touch federal (Dec 2025 preemption EO). **Most permissive to ship fast: US & China; UK among rules-based regimes; EU most constraining.** ([FCA](https://www.fca.org.uk/firms/innovation/ai-approach), [Orrick](https://www.orrick.com/en/Insights/2026/05/EUs-Digital-Omnibus-on-AI-7-Key-Changes-You-Need-to-Know))
- **Government support:** US light direct (CHIPS spillover + $500B Stargate private). EU **InvestAI €200B**; **France €109B** (largest single-event gov commitment, Paris Feb 2025) + Station F. UK **AI Growth Zones**, sovereign compute ≥20x by 2030, Sovereign AI Unit up to £500M. China **National VC Guidance Fund** (Dec 26 2025; ¥100B seed → ¥1T/~$143B over 20 yrs); guidance-fund system **2,178 funds ~$900B**. Gulf: Stargate UAE (5 GW), Saudi HUMAIN. ([EC](https://luxembourg.representation.ec.europa.eu/actualites-et-evenements/actualites/eu-launches-investai-initiative-mobilise-eu200-billion-investment-artificial-intelligence-2025-02-11_en), [Bloomberg](https://www.bloomberg.com/news/articles/2025-12-26/china-starts-state-backed-venture-funds-to-support-tech-startups))
- **Talent & immigration:** US **$100,000 H-1B petition fee** (effective Sep 21 2025) — a "startup tax" (~60% of top AI startups had an immigrant founder). China **K-visa** (Oct 1 2025): uncapped, 5-yr, no employer sponsor, STEM/AI-targeted; Qiming/"sea turtle" returnees — timed to exploit US restrictions. ([Bulletin](https://thebulletin.org/2025/10/how-trumps-new-h-1b-fee-will-hurt-silicon-valley-and-ai-startups/), [C&EN](https://cen.acs.org/careers/employment/Chinas-K-visa-targets-global/103/web/2025/10))
- **Founder-friendliness:** US = capital + speed + market, rising talent cost. China = state capital + deployment speed, but directed + walled. EU = subsidy ambition + legal certainty, heaviest compliance + shallow late-stage capital. UK = most balanced rules-based. Gulf = capital/infra, thin talent/market.

---

## 7. Cross-regional clinical-AI synthesis (course tie-in)

| | US | EU | China |
|---|---|---|---|
| **Approvals** | ~1,200–1,451 cumulative, ~295/yr; **radiology ~76%**; mostly 510(k) | High-risk under AI Act **+ MDR/IVDR** (double conformity), notified-body bottleneck | **154** (H1 2025), **79.9% Class III** w/ trials; radiology 68.8% |
| **Philosophy** | Device-by-device, enforcement-discretion; reimbursement-bottlenecked | Risk-tier stacked on device law; data space deferred to 2029 | State-coordinated approval + volume procurement |
| **Gen/LLM clinical AI** | Regulatory void (Jan 2026 CDS guidance silent on AI) | Default high-risk, 2027-28 | **Mass-deployed** (AQ 100M users; 261 hospitals ran DeepSeek) |
| **Reimbursement** | CPT Cat III, NTAP caps (the bottleneck) | Germany DiGA / France PECAN | Volume-based procurement |
| **Deployment scale** | Narrow tools, broad count | Slow, compliance-gated | **Population scale** |

**Bottom line for a precision-health audience:** the US *approves* the most discrete devices but is reimbursement-starved and has no LLM pathway; the EU is the most heavily regulated vertical (AI Act + MDR/IVDR + EHDS); China *deploys* clinical AI at population scale under lighter, state-coordinated oversight. The same regulate/build/deploy split that defines AI broadly defines clinical AI specifically.

---

## Caveats (carried from the agents)
- Several valuations are press-reported, not company-confirmed (Anthropic ~$965B & ~$47B run-rate; DeepSeek/Moonshot/Zhipu; xAI rounds).
- The **NVIDIA 15% / 25% revenue-share/tariff** terms are political understandings; NVIDIA stated the 15% was not finalized.
- China **40 GW DC / 400 GW spare power** and Ascend **TSMC-die-bank-exhausted** are analyst estimates (Rystad/Goldman/SemiAnalysis), not official.
- Adoption percentages mix methodologies: **Eurostat 20%** (formal) vs **McKinsey 56–57%** (broad survey) measure different things — do not conflate.
- Open-weight metrics differ: **OpenRouter usage (~30–45%)** vs **HF downloads (17.1%)** are distinct.
- FDA cumulative count varies by source/date (npj 1,016 mid-2025 peer-reviewed vs ~1,451 end-2025 tracker).
- NMPA-approved TCM tongue/pulse devices + specific deployed TCM LLMs were not firmly verified; treat as research-stage.

---

## 8. Asia beyond China (addendum — the pillar is not monolithic)

The third pillar is **Asia**, China-led but genuinely heterogeneous. Added 2026-05-30 from a 3-agent sweep on India, Korea/Japan, and Singapore/SE-Asia/Taiwan.

### The regulatory spectrum (Asia spans the whole range)
- **China** — state-vertical (see §3). **South Korea** — **AI Basic Act, effective Jan 22 2026**, the 2nd comprehensive regime after the EU but far lighter (fines cap **~₩30M/~$21k**) ([Cooley](https://www.cooley.com/news/insight/2026/2026-01-27-south-koreas-ai-basic-act-overview-and-key-takeaways)). **Japan** — **AI Promotion Act (2025), no penalties** ([White & Case](https://www.whitecase.com/insight-alert/japans-first-ai-legislation-becomes-law-focus-promoting-research-and-development-no)). **India** — **light-touch**, India AI Governance Guidelines (Nov 5 2025), no standalone law ([NeGD](https://negd.gov.in/press_release/meity-unveils-india-ai-governance-guidelines-under-indiaai-mission-to-ensure-safe-inclusive-and-responsible-adoption-of-artificial-intelligence-across-sectors/)). **Singapore** — voluntary Model AI Governance Framework + **AI Verify** testing toolkit ([Cambridge](https://www.cambridge.org/core/journals/cambridge-forum-on-ai-law-and-governance/article/governing-intelligence-singapores-evolving-ai-governance-framework/5E54A373E193E2D51354ADC1F509B9B4)).

### Asia owns the AI hardware (the region's collective edge)
- **TSMC (Taiwan)** fabricates ~all leading-edge AI logic; **NVIDIA booked >60% of its CoWoS** packaging, the real bottleneck ([wccftech](https://wccftech.com/nvidia-alone-has-tsmc-advanced-packaging-lines-booked-for-several-years-ahead/)). **Korea HBM:** **SK Hynix ~62% of HBM**, NVIDIA's lead supplier; with Samsung a duopoly; SK Hynix to supply **~2/3 of HBM4** for NVIDIA Vera Rubin ([CNBC](https://www.cnbc.com/2026/01/29/sk-hynix-beats-samsung-2025-profit-ai-memory-hbm.html), [TrendForce](https://www.trendforce.com/news/2026/01/28/news-sk-hynix-reportedly-to-supply-about-two-thirds-of-nvidia-hbm4-samsung-targets-early-delivery/)). **Foxconn (Taiwan)** builds **>40% of AI servers**, lead integrator for GB200/GB300 ([Foxconn](https://www.foxconn.com/en-us/press-center/press-releases/latest-news/1601)). **MediaTek** co-designs Google's TPUs.

### Models & national programs
- **India:** IndiaAI Mission **~$1.25B**, ~38k subsidized GPUs; **Sarvam** (govt-anointed sovereign LLM, open-weight Sarvam-30B/105B), **Krutrim** (pivoted to cloud May 2026); AI on **UPI/Aadhaar** at population scale ([PIB](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2012375&reg=3&lang=2), [Sarvam](https://www.sarvam.ai/blogs/indias-sovereign-llm)).
- **South Korea:** **₩100T (~$74B) "AI G3" fund**; a **₩530B sovereign-model program** (LG **EXAONE** open-weight, **Naver HyperCLOVA X**, SKT A.X, Upstage); a **260k-Blackwell-GPU / up to $10B** buy ([TechCrunch](https://techcrunch.com/2025/09/27/how-south-korea-plans-to-best-openai-google-others-with-homegrown-ai/), [Korea Herald](https://www.koreaherald.com/article/10606845)).
- **Japan:** **Sakana AI** $135M Series B at **$2.65B** (Japan's top unicorn); **SoftBank** ~$41B / **~11% of OpenAI** + Stargate co-lead; **NTT tsuzumi 2**; **Rapidus** 2nm pilot ([TechCrunch](https://techcrunch.com/2025/11/17/sakana-ai-raises-135m-series-b-at-a-2-65b-valuation-to-continue-building-ai-models-for-japan/), [Rapidus](https://www.rapidus.inc/en/)).
- **SE Asia (the battleground):** **SEA-LION switched from Llama to Alibaba Qwen** (Nov 25 2025) for Singapore's national model — a US-aligned government on a Chinese base ([TechNode](https://technode.com/2025/11/25/singapores-national-ai-program-drops-meta-model-and-switches-to-alibabas-qwen/)); **Sahabat-AI** (Indonesia, Indosat/GoTo/NVIDIA); **FPT AI Factory** (Vietnam, $200M); Malaysia ran **DeepSeek-on-Huawei** (govt later disavowed) ([NVIDIA](https://blogs.nvidia.com/blog/indonesia-tech-leaders-sovereign-ai/), [DCD](https://www.datacenterdynamics.com/en/news/malaysian-govt-retracts-launch-of-huawei-powered-sovereign-ai/)).

### Clinical AI (Asia ex-China)
- **Korea** is a clear exporter: approvals **62→108→157 (2023-25)**; **Lunit** (FDA-cleared, ~1M mammograms/yr, ₩83.1B 2025 rev), **VUNO** ([Seoul Economic Daily](https://en.sedaily.com/technology/2026/04/12/medical-ai-approvals-surge-25-fold-in-3-years-now)). **India:** **Qure.ai** qXR underpins the national TB program (NIDAAN), ~$123M raised; **Niramai** thermal breast screening ([Tracxn](https://tracxn.com/d/companies/qure.ai/__QfqJ14wDyhMbUvOibApm4cgaquEx22FBbT11aIuIIFc/funding-and-investors)). **Japan:** AI-hospital programs, diagnostics-heavy SaMD.

### Rest of world (beyond the big three)
- **Gulf = capital:** **Stargate UAE** (5 GW; G42, MGX), Saudi **HUMAIN**, MGX committed **up to $500B** to US Stargate. Capital + compute, thin talent/market ([MEI](https://mei.edu/report/ai-the-gulf-and-the-us-a-primer/)).
- **Global South = deployment battleground:** US vs Chinese open models for the next billion users; sovereign-language models (SEA-LION, Sahabat, TAIDE) on rented compute, increasingly Chinese open weights.

**Synthesis:** Three hubs (US, Europe, Asia) build and govern AI; everyone else deploys it, increasingly on Chinese open weights, which is itself a geopolitical outcome. Within Asia: China leads on models/scale, Korea/Japan/India build sovereign stacks, and the **Taiwan-Korea hardware spine** (TSMC + HBM + Foxconn) is the region's structural lock on the whole industry.
