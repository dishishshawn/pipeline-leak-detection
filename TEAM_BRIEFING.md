# Team Briefing: Pipeline Leak Detection System

This document is a non-technical guide to what this project does, why it matters, how it works, what is proven in the current repo, and how to talk about it accurately with investors.

The short version:

This project is an intelligent pipeline-monitoring prototype that watches SCADA-style telemetry, detects patterns consistent with leaks, and turns those patterns into leak-risk scores and operator-friendly alerts. It is not just a dashboard and it is not just one machine-learning model. It is a full demonstration stack with historical analysis, live simulation, physics-backed stress testing, and multiple model families that can be compared and retrained.

The metrics and examples in this document reflect the artifacts currently checked into the repository as of April 1, 2026.

## 1. What Problem We Are Solving

Pipelines can fail quietly at first. A small leak may begin as a subtle pressure drop or flow imbalance long before it becomes visually obvious or triggers a simple threshold alarm.

That matters because leaks can create:

- product loss
- safety risk
- environmental damage
- shutdowns and maintenance costs
- operator distrust if the alarm system cries wolf too often

The goal of this project is to detect leaks earlier and more intelligently than a basic single-threshold alarm by looking at multiple signals together over time.

## 2. Project In One Sentence

The system takes in pipeline sensor data, engineers leak-sensitive features from that data, scores the likelihood of a leak with trained models, and then converts those scores into practical alerts shown in a dashboard and simulator.

## 3. What The System Actually Does

Think of the system as a 6-step chain:

1. It receives SCADA-style readings such as pressure, flow rate, temperature, pump speed, energy consumption, and operating state.
2. It cleans and validates the data so the model is not reacting to broken rows or missing critical fields.
3. It creates higher-level features that are more informative than raw sensor numbers alone.
4. It feeds those features into trained leak-detection models.
5. It converts raw model scores into alert decisions using a policy that reduces noise and chatter.
6. It shows the results in a dashboard that can replay history or simulate live incidents.

In plain English, the system is trying to answer:

"Given what the pipeline has been doing over the last few moments, does this look like a real leak, a normal operating fluctuation, or some other disturbance?"

## 4. Why This Is Better Than A Basic Alarm Threshold

A simple alarm might say:

- if pressure drops below X, alert
- if flow changes by more than Y, alert

That approach is easy to implement, but it breaks down when:

- demand changes naturally during the day
- multiple sensors move together in non-obvious ways
- small leaks create subtle patterns instead of dramatic one-shot failures
- noisy sensors create false alarms

This project improves on that by looking at patterns, not just raw absolute values.

Examples of the patterns it uses:

- pressure change between consecutive readings
- flow-rate change between consecutive readings
- percent change instead of only absolute change
- rolling mean and rolling standard deviation
- z-scores relative to recent behavior
- pressure-to-flow ratio
- deviation from cross-segment behavior
- persistence of negative flow trends

That means the system can say:

"This is not just low pressure. This is sustained pressure drop plus flow imbalance plus a deviation from recent normal behavior on one segment."

That is much closer to how an experienced operator thinks.

## 5. What A User Sees In The Product

The dashboard has two big modes:

### Historical Analysis

This mode loads stored telemetry data and lets the team:

- inspect time-series behavior
- compare models
- review prediction details
- inspect confusion matrices and classification reports
- compare ROC curves and leaderboard metrics

This is the proof and benchmarking side of the project.

### Live Simulator

This mode simulates a running pipeline in real time and scores each new reading as it arrives.

It includes:

- multiple pipeline segments
- realistic demand cycles
- normal operating noise
- incident presets such as slow seep, demand shock, compound incident, and micro leak
- manual leak injection in the lightweight simulator
- download of confirmed alert events

This is the demo and stress-test side of the project.

## 6. The Two Simulator Backends

One of the strongest parts of this repo is that it does not rely on a single fake data source.

### Lightweight Backend

This backend is fast and demo-safe. It generates SCADA-shaped data with segment state, demand cycles, noise, and scenario logic.

It is useful for:

- live demos
- rapid testing
- manual leak injection
- generating training data for realtime models

### Physics-Backed Backend

This backend is more realistic and slower. It runs a transient flow solver based on physical equations.

At a high level, it models:

- continuity of mass flow
- momentum in the pipe
- friction losses
- compressibility effects
- leak discharge through an orifice
- sensor lag and sensor noise

This is important because it means the project is not only learning from arbitrary random numbers. It also has a route to more physically grounded synthetic data.

In investor language:

The simulator is not just a visual gimmick. It is part of the product-validation story.

## 7. Data Sources Behind The System

The project uses several layers of data:

### A. SCADA-Style Sample Data

This supports dashboard development and baseline model workflows.

### B. Simulator-Generated Realtime Training Data

This is generated from the live simulator itself. That matters because the live models are trained on the same kinds of patterns the dashboard will later show.

The training corpus intentionally includes:

- steady-state normal operation
- leak scenarios
- non-leak stress events such as demand shock
- compound incidents
- micro leaks
- rupture-style events

This is valuable because it teaches the model not only what leaks look like, but also what non-leak disturbances look like.

### C. Physics-Simulation Dataset

The repo contains a physics-sim training artifact built from:

- 90,500 total rows
- 500 scenarios

Its stronger models reached about:

- ROC-AUC 0.9815 for random forest
- ROC-AUC 0.9799 for XGBoost
- F1 around 0.88 for the best tree models

This gives the project a more realistic transient-pressure training and validation path.

### D. Petrobras 3W Real Telemetry Path

This is a major credibility point, but it must be described honestly.

The Petrobras 3W workflow is based on real oil-well telemetry, not a field-deployed liquid pipeline leak fleet. It is useful because it gives the project exposure to large-scale real industrial sensor behavior and fault/anomaly detection patterns.

The checked-in Petrobras training summary shows:

- about 2.56 million rows
- 484 total scenarios split across training and test
- 27 engineered features
- best reported ROC-AUC about 0.9425
- best reported F1 about 0.9297

That means this repo is not limited to toy examples, but it would still be inaccurate to claim it is already validated on a production pipeline operator's own live leak history.

### E. Future Multi-Source Robust Corpus

The project can also fuse simulator data with external telemetry-compatible datasets when those files are available locally.

This is the beginning of a generalization strategy:

- normalize multiple datasets into a common schema
- rebalance them so one source does not dominate
- retrain live-safe models on a broader corpus

That is a smart systems-design choice even though the live demo currently prefers a narrower model for better false-positive behavior.

## 8. Model Strategy

This project does not bet everything on one algorithm.

The repo supports several model families:

- logistic regression
- random forest
- XGBoost
- LightGBM
- isolation forest anomaly detection
- weighted ensemble models

Why that matters:

- logistic regression gives a simple baseline
- random forest is robust and interpretable at a high level
- XGBoost and LightGBM are strong performance-focused tree boosters
- isolation forest is useful when anomaly framing matters
- ensembles combine strengths from multiple models

This makes the project more credible than a single-model school demo because it allows real comparison of tradeoffs.

## 9. The Live Demo Model: ATLAS

The dashboard explicitly treats one model as the recommended live-demo model:

ATLAS = Adaptive Telemetry Leak Alert System

In the codebase, ATLAS is the `Realtime Xgboost` model.

It was selected because operational behavior mattered more than just raw offline accuracy — particularly micro-leak sensitivity after adding CUSUM and pressure-flow divergence features.

According to the current checked-in evaluation artifacts, ATLAS achieved approximately:

- 73% micro-leak sensitivity in scenario evaluation
- 100% slow-seep detection rate in scenario evaluation
- 0.8-step median slow-seep detection delay
- 0.00% false-positive rate in steady-state evaluation
- 0.064 alert toggle rate (low chatter)

The key idea:

ATLAS was chosen because it is especially good at subtle leak detection while still keeping false alarms very low.

After adding micro-leak-focused features (CUSUM cumulative pressure drop, long-window rolling stats, pressure-flow divergence), XGBoost's gradient boosting exploits these cumulative signals far better than Random Forest:

- Realtime XGBoost: 73% micro-leak sensitivity, 0% FPR
- Realtime Random Forest: 46% micro-leak sensitivity, 0.06% FPR
- Realtime Hybrid Ensemble: 64% micro-leak sensitivity, 0% FPR
- Robust models still trigger too many false alarms for a clean live demo

That is exactly the kind of design tradeoff investors like to hear explained clearly.

## 10. Score Versus Alert

Another important point: the model score is not the same thing as the final operator alert.

The system separates:

- leak score
- predicted class
- confirmed alert

Why?

Because a raw score can flicker if the signal is noisy. To make alerts more useful, the project adds an alert policy layer with ideas like:

- thresholds
- persistence windows
- confirmation over multiple ticks
- cooldown periods

That means the project is designed more like an operator-support tool than a homework classifier.

## 11. What We Can Honestly Claim Today

These claims are well supported by the current repository:

- The project ingests SCADA-style telemetry and turns it into leak-risk predictions.
- It uses engineered temporal and cross-signal features, not just raw thresholding.
- It compares multiple model families instead of relying on a single algorithm.
- It has both historical benchmarking and live simulation workflows.
- It includes a lightweight simulator and a physics-backed simulator.
- It has evidence of strong performance on synthetic realtime evaluation and meaningful performance on larger real industrial telemetry from Petrobras 3W.
- It includes logic to reduce false alarms through alert smoothing and calibrated thresholds.
- It is built so that new datasets can be normalized and retrained into the pipeline later.

## 12. What We Should Not Overclaim

These statements would go too far if we said them today:

- "This is already deployed in a real pipeline network."
- "This is certified for safety-critical operations."
- "It can precisely locate the leak anywhere in the network today."
- "It is fully validated on large amounts of real pipeline leak telemetry from commercial operators."
- "It works on every fluid, every pipe geometry, and every operator without retraining."
- "The simulator is a perfect digital twin of any real pipeline."

Better wording is:

- "This is a strong prototype and validation platform."
- "It demonstrates a credible path toward earlier leak detection."
- "It has been stress-tested with synthetic and physics-based scenarios and supported by real industrial telemetry workflows."
- "It still needs operator-specific data integration and field validation before production deployment."

## 13. Best Investor Narrative

If someone asks, "What is this really?", the strongest answer is:

This is an intelligent leak-detection and operator-decision-support platform prototype. It combines SCADA-style telemetry, machine learning, and simulation to identify subtle leak behavior earlier than simple threshold alarms. It is designed not just to score leaks offline, but to operate in a live-monitoring setting with alert stability, scenario testing, and retraining paths for new datasets.

That framing is stronger than saying:

"It is a machine-learning model that finds leaks."

The project is bigger than that.

## 14. Common Investor Questions And Good Answers

### "Is this just an alarm dashboard?"

No. The dashboard is the visible front end, but underneath it is a full pipeline of data validation, feature engineering, model scoring, alert logic, benchmarking, and simulation.

### "Why not just use pressure thresholds?"

Because real systems have noise, demand swings, and operating changes. A fixed threshold may miss subtle leaks or create nuisance alarms. This system looks at multivariate time-based patterns, not just a single number crossing a line.

### "What makes this intelligent?"

It learns from historical and simulated examples, uses engineered leak-sensitive features, and compares multiple model families to choose the best operational tradeoff.

### "How fast can it detect a leak?"

In the current scenario evaluation, the selected live-demo model reaches a median slow-seep detection delay of 0.8 simulation steps. In the default simulator configuration, one step represents one simulated minute, but deployment timing would ultimately depend on real sensor cadence and integration design.

### "How do you control false alarms?"

Three ways:

- model selection based on operational behavior, not only raw offline accuracy
- threshold calibration per model
- alert persistence and cooldown logic so a short noisy spike does not immediately become an operator alert

### "What data does it need?"

At minimum, it works with SCADA-style telemetry such as timestamp, segment, pressure, flow, temperature, and operating-state context. The architecture can normalize multiple datasets into a common schema.

### "Is it trained only on fake data?"

No, but that answer needs nuance. The live-demo model is strongly tied to simulator-generated training data, which is intentional for demo stability. The repo also includes physics-based synthetic training and a large real-telemetry Petrobras workflow. So the project is not fake-data-only, but it is also not yet fully validated on commercial pipeline leak fleets.

### "Can it pinpoint the exact leak location?"

Not precisely in the current live product. Today it is better described as segment-level leak-risk detection and early warning. Exact localization is a future extension, especially with richer sensing modalities.

### "Is this production-ready?"

Not yet. It is a sophisticated prototype with strong evidence, but field deployment would still require SCADA integration, operator-specific retraining, reliability hardening, and validation in live environments.

### "What is the commercial value?"

The value proposition is earlier warning, fewer missed leaks, fewer nuisance alarms, better operator awareness, and a training/demo environment that helps explain and validate system behavior before deployment.

### "What is technically differentiated here?"

The strongest differentiators are:

- multivariate time-series feature engineering
- model comparison instead of one-model guessing
- alert-policy separation from raw scores
- live simulator for stress testing
- physics-backed synthetic path
- data-normalization pipeline for future retraining on new sources

## 15. Talking About The Tradeoffs

A strong technical answer always includes tradeoffs.

Examples:

- The best offline accuracy model is not always the best live operating model.
- A broader robust model may generalize better across datasets but can create too many false alarms for a clean demo.
- Physics-backed simulation is more realistic, but it is slower than the lightweight simulator.
- Real telemetry adds credibility, but it may not perfectly match the final target deployment environment.

If the team can explain tradeoffs clearly, investors will trust the project more.

## 16. Simple 60-Second Pitch

We built an intelligent pipeline leak-detection prototype that watches SCADA-style telemetry and identifies leak behavior earlier and more intelligently than simple threshold alarms. The system cleans incoming sensor data, creates leak-sensitive features, scores the leak risk with several machine-learning models, and converts those scores into stable alerts. What makes it stronger than a basic demo is that it includes historical benchmarking, a live simulator, a physics-backed simulation path, and retraining workflows for new datasets. Our current live-demo model was chosen because it balances excellent accuracy with strong micro-leak sensitivity and very low false alarms, but we are careful to describe it as a strong prototype rather than a fully field-validated commercial deployment.

## 17. Technical Glossary For Non-Coders

### SCADA

Supervisory Control and Data Acquisition. In plain terms, it is the stream of operational sensor data and control-state information coming from industrial equipment.

### Feature Engineering

Turning raw measurements into smarter inputs for the model. Example: instead of only using pressure, also use how pressure is changing and whether it is drifting away from normal.

### ROC-AUC

A common model-quality metric. Higher is generally better. It measures how well a model separates positive cases from negative cases across thresholds.

### F1 Score

A metric that balances precision and recall. It is helpful when both missed leaks and false alarms matter.

### False Positive Rate

How often the system raises alarms when there is no real leak.

### Recall Or Sensitivity

How often the system successfully catches actual leak events.

### Micro Leak

A very small leak that creates only subtle signal changes. These are often the hardest cases and are very important commercially because early detection matters.

### Ensemble

A model that combines multiple models instead of relying on only one.

### Alert Policy

The rule layer that decides when a leak score becomes a real alert. This helps reduce alarm chatter.

### Physics-Backed Simulation

A simulator that uses physical equations of flow and pressure rather than only scripted patterns.

## 18. Safe Claims Versus Risky Claims

Safe claims:

- "The system detects leak-like behavior from SCADA-style telemetry."
- "It uses machine learning plus simulation, not just static thresholds."
- "It has been evaluated on multiple scenario types including slow seep and micro leak."
- "The current live-demo model was selected for low false alarms and strong subtle-leak sensitivity."
- "The architecture supports retraining on new datasets."

Risky claims:

- "It already replaces commercial leak-detection systems."
- "It is validated in the field."
- "It guarantees exact localization."
- "It is universally accurate across all pipelines."
- "The simulator exactly matches real operations."

## 19. Where These Claims Come From In The Repo

If someone on the team wants to trace the evidence:

- `dashboard/app.py` contains the live dashboard logic and the ATLAS live-demo selection.
- `src/features/engineer.py` contains the feature engineering logic.
- `src/evaluation/alert_policy.py` contains the score-to-alert smoothing logic.
- `src/simulation/core.py` and `src/simulation/scenarios.py` define the lightweight simulator and incident presets.
- `src/physics_sim/solver.py` and `src/simulation/physics_backend.py` define the physics-backed simulation path.
- `reports/evaluation_results.json` contains the scenario-based live evaluation metrics.
- `models/realtime/scada_pipeline_realtime_training_summary.json` contains the realtime training summary.
- `models/physics_sim/physics_sim_training_summary.json` contains the physics-sim training summary.
- `models/petrobras/petrobras_training_summary.json` contains the Petrobras real-telemetry training summary.

## 20. Final Takeaway

The most important thing for the team to understand is this:

This project is not just "AI for leaks" and it is not just "a dashboard." It is a prototype leak-detection platform built around telemetry, feature engineering, model selection, alert policy, and simulation. Its strongest story is not that it is already finished, but that it shows a technically credible and commercially meaningful path toward earlier, smarter, and more explainable leak detection.
