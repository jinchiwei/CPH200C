---
title: "Three Governing Systems for AI"
eyebrow: "CPH 200C · PROBLEM SET 3 · 2026"
subtitle: "How the United States, Europe, and Asia govern, build, and deploy artificial intelligence, with deployed clinical AI as the central case"
name: "Curtis Chambers · Jinchi Wei"
org: "CPH 200C: Computational Precision Health"
date: "2026-06-04"
---

## Abstract

By 2026 the United States, Europe, and Asia are not running the same artificial-intelligence playbook at different speeds. They are running three different governing systems. The United States is market-led: light federal rules, the deepest pool of private capital, and closed frontier models. Europe is comprehensive-regulatory: a single horizontal law, the AI Act, now being softened under competitiveness pressure, paired with large sovereignty subsidies but little frontier building. Asia, led by China, is state-driven: iterative ministry rules, heavy industrial policy, an open-weight export strategy, population-scale deployment, and ownership of the physical hardware on which the entire industry runs. This report reads those three systems across four scopes, policy, funding, technical developments, and deployment landscape, and uses deployed clinical AI as the central, evidence-rich test case. The recurring finding is that the regions differ less in whether they deploy AI than in how, and that the binding constraint on real-world deployment, in medicine especially, is institutional (regulation, payment, and coordination) rather than model quality. A secondary thesis, sharpened by the 2025 chip-export regime, is that hardware scarcity may be pushing China toward genuine methodological innovation while the United States mostly scales compute.

## 1. Introduction and framing

A useful way to read the global AI landscape is through three verbs: who writes the rules (regulate), who builds the frontier (build), and who deploys at scale (deploy). Almost no region does all three well, and each is, in effect, winning a different race.

The framing is not new. In *AI Superpowers* (2018), Kai-Fu Lee argued that the field had entered an "age of implementation" in which execution and data, not research breakthroughs, would decide the winners, a thesis that favored China. In the United States, Eric Schmidt and co-authors (*The Age of AI*, 2021) and the National Security Commission on Artificial Intelligence framed AI as a national-security race the United States had to win, warning that China was closing the gap. In Europe, Mario Draghi's 2024 competitiveness report delivered the now-canonical diagnosis that Europe "regulates but does not build."

What none of them anticipated is the event that reset the board in January 2025: a Chinese open-weight model, DeepSeek R1, reached frontier quality at a fraction of the cost, while United States labs moved further toward closed models. Lee's implementation thesis has largely held, but with a twist, and the gap between demonstrated capability and realized economic value has turned out to be the dominant story of the period.

## 2. Policy and regulation

### 2.1 United States: executive whiplash and a state patchwork

United States federal policy reversed sharply at the start of the second Trump administration. On January 20, 2025, the administration rescinded Biden's Executive Order 14110, and on January 23 issued "Removing Barriers to American Leadership in Artificial Intelligence" (Wiley 2025; White House 2025a). The resulting "Winning the Race: America's AI Action Plan," released July 23, 2025, is explicitly deregulatory, organized around accelerating innovation, building infrastructure, and leading international AI diplomacy, and was accompanied by an order barring federal procurement of language models judged ideologically non-neutral (White House 2025b; Skadden 2025).

With no comprehensive federal framework, individual states legislated, producing a patchwork: Colorado's AI Act (delayed to June 2026), Texas TRAIGA (effective January 2026), California's SB 53 Transparency in Frontier AI Act (the lighter successor to the vetoed SB 1047), and New York's RAISE Act (Mintz 2025; NY Governor 2025). The federal response has been preemptive: a proposed moratorium on state AI laws failed in the Senate by 99 to 1, after which a December 2025 executive order directed federal agencies to challenge "burdensome" state statutes (StateScoop 2025; Akin Gump 2025). The net posture is a light federal touch, a thickening state patchwork, and an unresolved fight over who governs AI.

### 2.2 Europe: the AI Act, and its 2025 softening

Europe took the opposite bet. The EU AI Act (Regulation 2024/1689) entered into force in August 2024 on a phased schedule: prohibited practices from February 2025, general-purpose-model obligations from August 2025, and high-risk rules thereafter, with penalties up to 35 million euros or 7 percent of global turnover (European Commission 2024). A voluntary GPAI Code of Practice followed in July 2025, signed by most major United States labs (TTMS 2025).

The defining development of late 2025, however, was retreat. Under competitiveness pressure crystallized by the Draghi report, the Commission published a "Digital Omnibus" on November 19, 2025 that delayed the high-risk regime to December 2027 (standalone systems) and August 2028 (systems embedded in regulated products, including medical devices); the separate AI Liability Directive was scrapped entirely (White & Case 2026; Corporate Europe Observatory 2026). Critics characterized the package as a rollback of digital rights. The underlying numbers explain the panic: only about 11 percent of EU firms use AI against a 75 percent target for 2030, roughly 73 percent of foundation models are United States-origin against 15 percent for China, and Europe holds only four of the world's top fifty technology firms (Carnegie 2025).

### 2.3 Asia: one region, the full regulatory spectrum

Asia is the third governing system, but it is the least monolithic. It spans the entire regulatory range. China governs through a stack of vertical, fast-iterating ministry rules rather than an omnibus statute: the 2023 Interim Measures require public generative services to file with regulators (748 services filed by December 2025), and since September 2025 all AI-generated content must carry visible and metadata labels (China-Briefing 2023; Loeb 2025). China explicitly removed a comprehensive AI law from its 2025 agenda in favor of pilots and standards, and went on the diplomatic offensive, proposing a World AI Cooperation Organization at the 2025 World AI Conference (East Asia Forum 2025; gov.cn 2025a). South Korea took the EU path, enacting an AI Basic Act effective January 2026 (the second comprehensive regime after the EU, though far lighter, with fines capped near 21,000 US dollars) (Cooley 2026). Japan went the opposite way with an innovation-first AI Promotion Act (2025) carrying no penalties (White & Case 2025). India and Singapore are light-touch, India through governance guidelines rather than a standalone law, Singapore through a voluntary framework paired with its AI Verify testing toolkit (NeGD 2025; Cambridge 2025).

## 3. Funding and capital

The United States dominates AI capital to a degree that structures the entire field. By the Stanford AI Index, United States private AI investment in 2025 was roughly 285.9 billion dollars against China's 12.4 billion, a gap of about 23 times, and the United States captured on the order of 83 percent of global private AI investment (Stanford HAI 2025; IEEE Spectrum 2026). Within the United States, AI startups absorbed about 222 billion dollars, roughly 65 percent of all United States venture capital (PitchBook 2026). Mega-rounds defined the year: OpenAI raised at a 300-billion-dollar valuation and later higher, Anthropic's valuation climbed toward the high hundreds of billions, and xAI raised at 200 billion or more (Visual Capitalist 2025). These valuations are press-reported and should be read as directional.

Europe's problem is structural rather than one of ambition. European AI startups raised a record 21.6 billion dollars in 2025, led by Mistral (valued near 11.7 billion euros, with ASML taking roughly 11 percent), Helsing, ElevenLabs, and others, but the continent has a thin late-stage capital layer and few systemic challengers to United States cloud providers (Tech.eu 2026; CNBC 2025a). China presents a third model: private venture capital fell to under a quarter of its prior-year level, and state guidance funds filled the gap, with government-linked AI deals rising from fewer than ten per year before 2018 to more than 140 in 2025 (Fortune 2026).

Government support differs in kind, not just amount. The United States funds AI lightly and directly, relying on private capital and the privately financed Project Stargate. Europe leans on public ambition: the InvestAI initiative aims to mobilize 200 billion euros, and France alone pledged 109 billion euros at the February 2025 Paris summit, the largest single-event government commitment by any state (European Commission 2025; Bloomberg 2025). China runs the most systematic state-capital machine, a National Venture Capital Guidance Fund seeded in December 2025 that aims to mobilize roughly one trillion yuan (about 143 billion US dollars) over twenty years, atop a guidance-fund system already comprising more than 2,000 funds worth roughly 900 billion dollars.

Regulatory leniency and talent policy compound these differences. The United Kingdom is the most explicitly pro-innovation rules-based regime, with a financial-conduct sandbox and a stated tolerance for failure; the EU mandates regulatory sandboxes but has delayed them to 2027, with only about eight of 27 member states ready; China is permissive on deployment speed while strict on speech and data (FCA 2025; Orrick 2026). On talent, the two largest systems moved in opposite directions in the same year: the United States imposed a 100,000-dollar fee on each H-1B petition in September 2025, a measure widely described as a tax on startups given that roughly 60 percent of top United States AI startups had an immigrant founder, while China launched an uncapped, employer-sponsorship-free K-visa for STEM and AI talent in October 2025, timed to absorb the talent the United States was now pricing out (Bulletin of the Atomic Scientists 2025; C&EN 2025).

## 4. Technical developments

### 4.1 The chip war and the constraint it imposes

The single constraint that shapes every region's technical strategy is access to advanced compute. United States export controls escalated steadily from the first advanced-chip rules of October 2022, through the closing of the A800/H800 loophole in October 2023, to the first high-bandwidth-memory controls and 140 new entity-list designations in December 2024 (CSET 2023). The January 2025 "AI Diffusion Rule," a three-tier country framework, was rescinded in May 2025, two days before it took effect; by January 2026 the H200 and comparable parts were moved to case-by-case licensing under a 25 percent tariff, a 50 percent volume cap, and United States testing requirements (BIS 2025; Introl 2026). The NVIDIA H20 saga captured the volatility: an effective April 2025 ban triggered a multi-billion-dollar charge, after which sales resumed in July under a reported, and never formally finalized, arrangement giving the United States government 15 percent of NVIDIA's China revenue, even as China discouraged purchases and banned a workaround consumer card during Jensen Huang's May 2026 visit (Tom's Hardware 2025; PBS 2025).

### 4.2 Two paths to better models: scale versus constraint

The deeper technical story is that the constraint produced two divergent innovation paths. The United States answers the race with scale: Project Stargate (up to 500 billion dollars and 10 gigawatts by 2029), roughly 700 billion dollars of hyperscaler capital expenditure planned for 2026, and frontier models kept closed (OpenAI 2025; CNBC 2026a). With abundant GPUs there is little pressure to economize. China, capped on advanced chips, has been pushed toward methodological efficiency: mixture-of-experts architectures, distillation, DeepSeek's reported roughly 6-million-dollar training run, and cheap inference. The provocative reading, and one worth stating plainly, is that the export-control regime may be backfiring, driving genuine methodology innovation in China while the United States mostly adds hardware. Europe's technical leverage sits upstream of both: ASML's near-monopoly on extreme-ultraviolet lithography is the chokepoint through which the controls themselves operate (ASML 2026).

### 4.3 The open-weight surge

DeepSeek R1, released January 20, 2025 under an MIT license and reportedly trained for a small fraction of frontier budgets, topped the United States App Store and erased on the order of 600 billion dollars of NVIDIA market value in a single day (Britannica 2025). It catalyzed a broad Chinese open-weight ecosystem, Alibaba's Qwen, Baidu's Ernie, Zhipu's GLM, Moonshot's Kimi, ByteDance's Doubao, and Huawei's Pangu, that now outpaces the West in model count and, by some measures, in usage: Chinese open models reached roughly 30 to 45 percent of weekly traffic on a major model router by early 2026, and Chinese open-model downloads narrowly passed United States downloads in the year to August 2025 (MIT Technology Review 2026; Stanford HAI 2025b). The strategic consequence is that the Global South increasingly deploys Chinese open weights for "AI sovereignty," a point developed in Section 7.

### 4.4 Asia owns the hardware

Where Asia is genuinely unified and dominant is the physical supply chain. Taiwan's TSMC fabricates essentially all leading-edge AI logic, and its advanced-packaging capacity (CoWoS), more than 60 percent of which NVIDIA has booked, is the true bottleneck (wccftech 2026). South Korea owns the memory: SK Hynix holds roughly 62 percent of high-bandwidth memory, is NVIDIA's lead supplier, and is set to provide about two-thirds of HBM4 for NVIDIA's next platform (CNBC 2026b; TrendForce 2026). Taiwan's Foxconn assembles more than 40 percent of the world's AI servers (Foxconn 2025). The Taiwan-Korea hardware spine is the region's structural lock on the entire industry.

## 5. Deployment and adoption landscape

Capability and capital are not the same as realized value, and the 2025 evidence makes the distinction stark. McKinsey's late-2025 survey found that 88 percent of organizations use AI in at least one function, but only about a third have scaled it and only 39 percent report any earnings impact; the survey put Greater China (56 percent) and North America (57 percent) at near-parity on overall use (McKinsey 2025). Europe lags structurally: Eurostat reported that only 20 percent of EU enterprises used AI in 2025 (Eurostat 2025). The methodologies differ (Eurostat measures formal adoption, McKinsey is a broad survey), but the direction is consistent.

The deployment channel differs by region. China deploys through super-apps and the state, embedding agents in WeChat and pursuing agentic commerce. The United States deploys through the enterprise-SaaS vendor layer, where buying from a specialized vendor succeeds roughly twice as often as building in-house. Europe's deployment is gated by its own compliance regime, with around 60 percent of EU and UK developers reporting AI Act-related launch delays (CNBC 2026c; Fortune 2025).

The headline of the period is the return-on-investment gap. MIT's Project NANDA found that 95 percent of enterprise generative-AI pilots delivered no measurable profit-and-loss impact despite 30 to 40 billion dollars of spending, even as top-down estimates remained enormous (McKinsey's 2.6 to 4.4 trillion dollars per year, Goldman's roughly 7 percent of global GDP) (Fortune 2025). On the capability scoreboard the United States leads decisively, producing far more notable models than China or Europe and contributing the largest share of high-impact research and investment, while China leads on granted AI patents (Stanford HAI 2026). On government readiness, Oxford Insights ranked the United States first and China eighth in 2025, the latter up from twenty-third (Oxford Insights 2025).

## 6. The clinical-AI thread

Deployed clinical AI is the best single test of the three governing systems, both because it has the richest peer-reviewed evidence base and because the three systems diverge most sharply there. Most deployed clinical AI today is, in fact, radiology: imaging dominates the regulated landscape in every region.

### 6.1 Regulation by region

The United States regulates device by device. The Food and Drug Administration had authorized on the order of 1,200 to 1,451 AI/ML-enabled medical devices by the end of 2025 (a peer-reviewed snapshot counted 1,016 through mid-2025; industry trackers put the end-2025 figure near 1,451), roughly 76 percent of them in radiology, with a record number cleared in 2025 (Bedi et al. 2025; IntuitionLabs 2025). A separate analysis of 27 years of AI/ML device recalls confirms the scale and the design-and-software-change failure modes that dominate them (Chen et al. 2025). The Predetermined Change Control Plan framework (finalized December 2024) lets manufacturers pre-authorize model updates, and a January 2026 guidance eased oversight of clinical-decision-support software, but it is conspicuously silent on generative and large-language-model tools, leaving a regulatory void (FDA 2026; Covington 2026). Reimbursement, not approval, is the binding constraint: most AI tools carry only temporary CPT Category III codes, and a 2025 Health Tech Investment Act proposes a durable Medicare pathway (Bipartisan Policy Center 2025).

Europe applies the heaviest burden. Medical AI is automatically high-risk under the AI Act and is stacked on top of the existing Medical Device Regulation and In Vitro Diagnostic Regulation, a double conformity assessment that few notified bodies are designated to perform, creating a bottleneck; the embedded-device deadline now falls in August 2028 (Reed Smith 2025). The European Health Data Space, the eventual engine for secondary use of health data in AI training, does not fully apply until 2029 (Arnold & Porter 2025). Germany's DiGA fast-track is the clearest working reimbursement pathway, reimbursing roughly 60 digital health applications, and France has copied it (Prova Health 2025).

China deploys at a scale neither matches. Its National Medical Products Administration had approved 154 AI medical devices by mid-2025, roughly 80 percent of them the higher-risk Class III with mandatory trials, at a compound annual growth rate near 50 percent (JMIR 2026). More striking is deployment: roughly 261 hospitals locally deployed DeepSeek-R1 in the first quarter of 2025, mostly tertiary centers, and Ant Group's AQ application links more than 5,000 hospitals, about a million doctors, and over 100 million users (Nature Medicine 2025; Baidu Baike 2026). South Korea has become a clinical-AI exporter (Lunit and VUNO, with FDA clearances and roughly a million mammograms per year), and India runs population-scale screening through Qure.ai's national tuberculosis program (Seoul Economic Daily 2026; Tracxn 2026).

### 6.2 The human factor and the measurement problem

Two findings cut against techno-optimism. First, the human-AI interaction, not raw model accuracy, decides clinical value. A randomized trial found that giving physicians a large language model did not significantly improve their diagnostic reasoning, even though the model alone scored well, because clinicians underused it (Goh et al. 2024). Over-reliance carries its own hazard: endoscopists in a 2025 study detected fewer adenomas after routine exposure to AI, an automation-bias and deskilling effect (Budzyn et al. 2025). Notably, this evidence is largely Western; China's population-scale deployment has outpaced interaction research.

Second, the field is getting good at measuring what models can say, not whether they help patients. New physician-built benchmarks (HealthBench, with rubrics from 262 doctors; HealthAdminBench, for healthcare-administration agents) grade clinical capability against expert opinion (Arora et al. 2025; Bedi et al. 2026). But the broader scoping review of randomized controlled trials evaluating AI in clinical practice finds the trial evidence still thin and early-stage (Han et al. 2024). Most deployed clinical AI is validated against expert labels rather than patient outcomes, and high agreement with a clinician is not the same as being right about the patient. AI is even used against patients: insurers deploy it to deny coverage (Mello and Rose 2024).

### 6.3 The clinical synthesis

The same regulate, build, deploy split that defines AI broadly defines clinical AI specifically. The United States approves the most discrete tools but is reimbursement-starved and has no pathway for generative clinical AI. Europe is the most heavily regulated vertical. China deploys at population scale under lighter, state-coordinated oversight. The bottleneck on whether clinical AI reaches patients is institutional, regulation, payment, and coordination, more than it is a question of model quality.

## 7. Beyond the big three

Two further actors matter without rising to the level of a fourth hub. The Gulf states bring capital rather than models: Stargate UAE (5 gigawatts, led by G42 and MGX), Saudi Arabia's HUMAIN, and MGX's commitment of up to 500 billion dollars to United States Stargate represent abundant capital and compute atop thin domestic talent and markets (Middle East Institute 2025). The Global South is the deployment battleground, where United States and Chinese open models compete for the next billion users and where sovereign-language models (SEA-LION, Sahabat-AI, TAIDE) run on rented, increasingly Chinese, open weights. Singapore's national program switching its base model from Meta's Llama to Alibaba's Qwen in November 2025 is the sharpest single illustration: a United States-aligned government building on a Chinese foundation (TechNode 2025).

## 8. Synthesis and outlook

Three regions, three governing systems. The United States leads on models, capital, and device approvals but is increasingly constrained by power and, now, talent cost. Europe leads on regulation and holds one hardware chokepoint (ASML) but builds little. Asia, led by China, leads on deployment, open-weight reach, and ownership of the hardware spine, constrained mainly by access to advanced chips, a constraint it is actively engineering around. No region does regulate, build, and deploy all well.

Three implications follow. First, the open-versus-closed split is the structural story, and China's open weights are becoming global infrastructure, itself a geopolitical outcome. Second, each region's binding constraint differs, and the interesting open question is whether the United States power constraint or the China chip constraint binds harder over the next two years. Third, in medicine the lesson is sharpest: the deployed-AI bottleneck is institutional, and the field should move its success metric from agreement with expert labels toward measured patient outcomes.

## Caveats

Several figures are press-reported rather than company-confirmed, including the largest valuations (Anthropic, xAI, several Chinese labs) and the NVIDIA revenue-share and tariff terms, which are political understandings rather than signed contracts. China's data-center and spare-power projections are analyst estimates. Adoption percentages mix methodologies (formal adoption versus broad survey) and should not be conflated. Open-weight usage and download metrics measure different things. The FDA cumulative device count varies by source and date. Named deployed traditional-Chinese-medicine diagnostic systems remain largely research-stage and were not firmly verified.

## References

Policy and regulation
- Wiley, "President Trump Revokes Biden Administration's AI EO" (2025). https://www.wiley.law/alert-President-Trump-Revokes-Biden-Administrations-AI-EO-What-To-Know
- White House, "Removing Barriers to American Leadership in AI" (Jan 2025a). https://www.whitehouse.gov/presidential-actions/2025/01/removing-barriers-to-american-leadership-in-artificial-intelligence/
- White House, "America's AI Action Plan" (Jul 2025b). https://www.whitehouse.gov/wp-content/uploads/2025/07/Americas-AI-Action-Plan.pdf
- Skadden, "The White House Releases AI Action Plan" (2025). https://www.skadden.com/insights/publications/2025/07/the-white-house-releases-ai-action-plan
- StateScoop, state-AI-law moratorium / FY2026 NDAA (2025). https://statescoop.com/state-ai-law-moratorium-omitted-2026-defense-bill-trump-eo/
- Akin Gump, Dec 2025 preemption EO (2025). https://www.akingump.com/en/insights/alerts/president-trump-unveils-ai-eo-advancing-federal-preemption-of-state-laws
- Mintz, "California's SB 53" (2025). https://www.mintz.com/insights-center/viewpoints/54731/2025-10-03-charting-future-ai-governance-californias-sb-53-sets
- NY Governor, RAISE Act (Dec 2025). https://www.governor.ny.gov/news/governor-hochul-signs-nation-leading-legislation-require-ai-frameworks-ai-frontier-models
- European Commission, AI Act regulatory framework (2024). https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- TTMS, "EU AI Act Code of Practice / industry reactions" (2025). https://ttms.com/eu-ai-act-update-2025-code-of-practice-enforcement-industry-reactions/
- White & Case, "EU agrees Digital Omnibus deal" (2026). https://www.whitecase.com/insight-alert/eu-agrees-digital-omnibus-deal-simplify-ai-rules
- Corporate Europe Observatory, "How Big Tech shaped the EU's roll-back of digital rights" (2026). https://corporateeurope.org/en/2026/01/article-article-how-big-tech-shaped-eus-roll-back-digital-rights
- Carnegie Endowment, "The EU's AI Power Play" (2025). https://carnegieendowment.org/research/2025/05/the-eus-ai-power-play-between-deregulation-and-innovation
- China-Briefing, China's Interim Measures for Generative AI (2023). https://www.china-briefing.com/news/how-to-interpret-chinas-first-effort-to-regulate-generative-ai-measures/
- Loeb, China's AI labeling measures + GB 45438-2025 (2025). https://www.loeb.com/en/insights/publications/2025/03/chinas-ai-labeling-measures-and-mandatory-national-standards-take-effect-september-1
- East Asia Forum, "China resets the path to comprehensive AI governance" (2025). https://eastasiaforum.org/2025/12/25/china-resets-the-path-to-comprehensive-ai-governance/
- gov.cn, World AI Cooperation Organization proposal (2025a). https://english.www.gov.cn/news/202507/26/content_WS6884bea8c6d0868f4e8f4732.html
- Cooley, "South Korea's AI Basic Act" (2026). https://www.cooley.com/news/insight/2026/2026-01-27-south-koreas-ai-basic-act-overview-and-key-takeaways
- White & Case, "Japan's first AI legislation" (2025). https://www.whitecase.com/insight-alert/japans-first-ai-legislation-becomes-law-focus-promoting-research-and-development-no
- NeGD/MeitY, India AI Governance Guidelines (Nov 2025). https://negd.gov.in/press_release/meity-unveils-india-ai-governance-guidelines-under-indiaai-mission-to-ensure-safe-inclusive-and-responsible-adoption-of-artificial-intelligence-across-sectors/
- Cambridge Forum on AI Law and Governance, Singapore's AI governance framework (2025). https://www.cambridge.org/core/journals/cambridge-forum-on-ai-law-and-governance/article/governing-intelligence-singapores-evolving-ai-governance-framework/5E54A373E193E2D51354ADC1F509B9B4

Funding, capital, talent
- Stanford HAI, AI Index 2025, Economy chapter. https://hai.stanford.edu/ai-index/2025-ai-index-report/economy
- IEEE Spectrum, "State of AI Index 2026". https://spectrum.ieee.org/state-of-ai-index-2026
- PitchBook via SiliconANGLE, 2025 AI venture capital (2026). https://siliconangle.com/2026/01/07/pitchbook-ai-dominates-global-venture-capital-2025-deal-value-nears-record/
- Visual Capitalist, biggest AI funding rounds of 2025. https://www.visualcapitalist.com/ranked-the-biggest-ai-funding-rounds-of-2025-so-far/
- CNBC, "Mistral valued at $14B as ASML takes major stake" (2025a). https://www.cnbc.com/2025/09/09/ai-firm-mistral-valued-at-14-billion-as-asml-takes-major-stake.html
- Tech.eu, top European AI raises 2025 (2026). https://tech.eu/2026/05/21/ai-10-companies-that-raised-the-most-in-2025/
- Fortune, DeepSeek / China AI venture pullback + state funds (2026). https://fortune.com/2026/05/19/deepseek-china-ai-venture-capital-nvidia-pitchbook-trends-term-sheet/
- European Commission, InvestAI initiative (2025). https://luxembourg.representation.ec.europa.eu/actualites-et-evenements/actualites/eu-launches-investai-initiative-mobilise-eu200-billion-investment-artificial-intelligence-2025-02-11_en
- Bloomberg, China state-backed venture funds (Dec 2025). https://www.bloomberg.com/news/articles/2025-12-26/china-starts-state-backed-venture-funds-to-support-tech-startups
- FCA, AI approach and sandbox (2025). https://www.fca.org.uk/firms/innovation/ai-approach
- Orrick, "EU's Digital Omnibus on AI: 7 Key Changes" (2026). https://www.orrick.com/en/Insights/2026/05/EUs-Digital-Omnibus-on-AI-7-Key-Changes-You-Need-to-Know
- Bulletin of the Atomic Scientists, H-1B fee impact (2025). https://thebulletin.org/2025/10/how-trumps-new-h-1b-fee-will-hurt-silicon-valley-and-ai-startups/
- C&EN, China's K-visa (2025). https://cen.acs.org/careers/employment/Chinas-K-visa-targets-global/103/web/2025/10

Compute, chips, hardware
- CSET (Georgetown), BIS 2023 export-control update explainer. https://cset.georgetown.edu/article/bis-2023-update-explainer/
- BIS / Dept. of Commerce, rescission of the AI Diffusion Rule (May 2025). https://www.bis.gov/press-release/department-commerce-rescinds-biden-era-artificial-intelligence-diffusion-rule-strengthens-chip-related
- Introl, BIS H200 China export policy + AI Overwatch Act (2026). https://introl.com/blog/bis-h200-china-export-policy-ai-overwatch-act-2026
- Tom's Hardware, NVIDIA H20 write-off (2025). https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-writes-off-usd5-5-billion-in-gpus-as-us-govt-chokes-off-supply-of-h20s-to-china
- PBS NewsHour, US 15% cut of NVIDIA/AMD China sales (2025). https://www.pbs.org/newshour/politics/under-new-unusual-agreement-u-s-will-get-a-15-cut-of-nvidia-and-amd-chip-sales-to-china
- OpenAI, Project Stargate (2025). https://openai.com/index/five-new-stargate-sites/
- CNBC, hyperscaler 2026 capex (2026a). https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html
- SemiAnalysis, Huawei CloudMatrix 384. https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72
- ASML, 2025 results (2026). https://www.sec.gov/Archives/edgar/data/0000937966/000162828026003701/a2026_01x28xpresentation.htm
- wccftech, NVIDIA share of TSMC CoWoS (2026). https://wccftech.com/nvidia-alone-has-tsmc-advanced-packaging-lines-booked-for-several-years-ahead/
- CNBC, SK Hynix HBM lead (2026b). https://www.cnbc.com/2026/01/29/sk-hynix-beats-samsung-2025-profit-ai-memory-hbm.html
- TrendForce, SK Hynix HBM4 share (2026). https://www.trendforce.com/news/2026/01/28/news-sk-hynix-reportedly-to-supply-about-two-thirds-of-nvidia-hbm4-samsung-targets-early-delivery/
- Foxconn, COMPUTEX 2025 AI-server share. https://www.foxconn.com/en-us/press-center/press-releases/latest-news/1601

Models, adoption, deployment
- Britannica, DeepSeek. https://www.britannica.com/money/DeepSeek
- MIT Technology Review, China open-source models (2026). https://www.technologyreview.com/2026/04/21/1135658/china-open-source-models-ai-artificial-intelligence/
- Stanford HAI, "Beyond DeepSeek: China's open-weight ecosystem" (2025b). https://hai.stanford.edu/policy/beyond-deepseek-chinas-diverse-open-weight-ai-ecosystem-and-its-policy-implications
- McKinsey, "The State of AI" (2025). https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai
- Eurostat, EU enterprise AI use 2025. https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20251211-2
- Fortune, MIT NANDA 95% pilots report (2025). https://fortune.com/2025/08/18/mit-report-95-percent-generative-ai-pilots-at-companies-failing-cfo/
- CNBC, China super-app agentic commerce (2026c). https://www.cnbc.com/2026/01/21/china-tech-ai-agentic-commerce-super-apps-alibaba-taobao-qwen-tencent-wechat-doubbao-weixin.html
- Oxford Insights, Government AI Readiness Index 2025. https://oxfordinsights.com/ai-readiness/government-ai-readiness-index-2025/
- Stanford HAI, AI Index 2025 in 10 charts. https://hai.stanford.edu/news/ai-index-2025-state-of-ai-in-10-charts
- TechNode, Singapore SEA-LION switches to Qwen (2025). https://technode.com/2025/11/25/singapores-national-ai-program-drops-meta-model-and-switches-to-alibabas-qwen/
- Middle East Institute, "AI, the Gulf, and the US" (2025). https://mei.edu/report/ai-the-gulf-and-the-us-a-primer/

Clinical / healthcare AI
- Han et al., "Randomised controlled trials evaluating AI in clinical practice: a scoping review," Lancet Digital Health (2024). https://doi.org/10.1016/S2589-7500(24)00047-5
- Goh et al., "Large language model influence on diagnostic reasoning: a randomized clinical trial," JAMA Network Open (2024). https://doi.org/10.1001/jamanetworkopen.2024.40969
- Mello & Rose, "Denial: artificial intelligence tools and health insurance coverage decisions," JAMA Health Forum (2024). https://doi.org/10.1001/jamahealthforum.2024.0622
- Bedi et al., FDA AI/ML-enabled medical device landscape, npj Digital Medicine (2025). https://www.nature.com/articles/s41746-025-01800-1
- Chen et al., "Regulatory insights from 27 years of AI/ML-enabled medical device recalls in the US," JMIR Medical Informatics (2025). https://doi.org/10.2196/67552
- IntuitionLabs, FDA AI medical device tracker (2025). https://intuitionlabs.ai/articles/fda-ai-medical-device-tracker
- FDA, revised Clinical Decision Support guidance (Jan 2026), via Covington analysis. https://www.cov.com/en/news-and-insights/insights/2026/01/5-key-takeaways-from-fdas-revised-clinical-decision-support-cds-software-guidance
- Bipartisan Policy Center, "Paying for AI in US health care" (2025). https://bipartisanpolicy.org/issue-brief/paying-for-ai-in-u-s-health-care/
- Reed Smith, EU AI Act and medical devices (2025). https://www.reedsmith.com/our-insights/blogs/viewpoints/102kq35/the-eu-ai-act-and-medical-devices-navigating-high-risk-compliance/
- Arnold & Porter, European Health Data Space regulation (2025). https://www.arnoldporter.com/en/perspectives/advisories/2025/03/european-health-data-space-regulation-published
- Prova Health, Germany DiGA reimbursement guide (2025). https://www.provahealth.com/insights/diga-reimbursement-germany-guide
- JMIR Medical Informatics, NMPA-approved AI medical devices retrospective (2026). https://medinform.jmir.org/2026/1/e85538
- Nature Medicine, DeepSeek-R1 deployment across Chinese hospitals (2025). https://www.nature.com/articles/s41591-025-03836-y
- Seoul Economic Daily, Korea medical-AI approvals surge (2026). https://en.sedaily.com/technology/2026/04/12/medical-ai-approvals-surge-25-fold-in-3-years-now
- Tracxn, Qure.ai funding profile (2026). https://tracxn.com/d/companies/qure.ai
- Budzyn et al., endoscopist deskilling after AI exposure, Lancet Gastroenterology & Hepatology (2025).
- Arora et al., "HealthBench," (2025). arXiv:2505.08775
- Bedi et al., "HealthAdminBench," (2026). arXiv:2604.09937

Framing works
- Lee, Kai-Fu. *AI Superpowers: China, Silicon Valley, and the New World Order* (2018).
- Kissinger, Schmidt, and Huttenlocher. *The Age of AI: And Our Human Future* (2021); National Security Commission on Artificial Intelligence, Final Report (2021).
- Draghi, Mario. *The Future of European Competitiveness* (2024).

*Compiled from a parallel multi-agent web and literature sweep (2026-05-30) plus PubMed verification of selected clinical-AI sources. Press-reported figures are flagged in the Caveats. This report accompanies the slide deck "Three Governing Systems for Clinical AI."*
