# PathSnap route reconstruction — Phase 2

Member 3 owns the candidate-road HMM, distance-based transition scoring,
Viterbi decoding, and intermediate directed-road recovery in this package.
The original model, preprocessing, backend, and frontend interfaces remain
outside this package and are not changed here.

## Architecture

1. **Candidate emissions** — `emission_probability.py` normalizes model
   probabilities among the candidate arcs for each GPS observation. It can use
   precomputed `pred_prob` values or a supplied classifier. Reference labels
   are not an inference input.
2. **Transition scoring** — `transition_probability.py` supports the existing
   connectivity mode and a directed distance mode. `distance_transition.py`
   projects each GPS point onto its candidate arc, estimates the directed
   distance between projections, and compares it with the GPS displacement.
   `road_graph.py` uses directed shortest paths; it never creates reverse
   connections that are absent from the graph.
3. **Viterbi decoding** — `viterbi.py` maximizes the sum of log emission and
   transition scores. Unreachable transitions are represented as negative
   infinity rather than being collapsed onto the probability floor. If every
   state transition across an observation pair is unreachable, decoding
   restarts at that observation instead of fabricating a connected path.
4. **Intermediate-road recovery** — `recover_continuous_route` retains its
   connectivity-only behavior when no GPS track is supplied. When `gps_track`
   is supplied, it uses `recover_constrained_route` and accepts a connection
   only if its directed candidate-road distance fits
   `max_speed_mps * time_gap_s + gps_tolerance_m`. An unreachable, invalid, or
   over-budget connection starts a separate route section.
5. **Evaluation** — `.route` references are newline-delimited integer arc IDs.
   `route_metrics.py` evaluates route order with edit distance and LCS F1, and
   measures road overlap with a length-weighted arc multiset precision, recall,
   and F1. Continuity alone is not treated as route correctness.

## Same-arc GPS jitter and reverse movement

The transition distance now treats a backward projection on the same directed
arc as zero route progress only when its projected displacement is within the
configurable `backward_tolerance_m` (default: 30 m). This avoids turning small
GPS jitter into an end-to-start network loop. Larger backward movement still
requires a directed loop in the road graph; if it is unreachable or exceeds
the time/speed budget, recovery rejects the pair and preserves separate
sections.

The distance-mode HMM uses unclipped log scores for reachable transitions.
This keeps very long finite detours distinguishable; unreachable transitions
remain impossible. Invalid or non-increasing GPS timestamps likewise make a
distance-mode HMM transition impossible and cause route recovery to split
sections.

## HMM optimization

Optimization uses the existing out-of-fold probabilities for the **80 training
trips only** (208,417 observations). The OOF generator uses five
trip-grouped folds to avoid trip leakage. Optuna TPE evaluates 25 trials with
the objective `accuracy + 0.25 * direct_continuity`; `beta` is held at 30 m
and transition weight is searched from 0 to 2.

The same-arc distance change invalidates cached errors only for backward
same-arc jitter transitions. A new cache is therefore built under
`route_reconstruction/experiments/phase2_improved/hmm_validation/cache`, then
optimized parameters and trial logs are saved beside it. The previous cache
and measured results remain untouched. The previous 80-trip cache was
reused only for mathematically unchanged values; the changed transition
distances were recalculated from training graphs and tracks. Verification
confirmed 80 trips, 208,417 observations, and 262,442 changed candidate-pair
entries, all with the same arc at both observations. A first full rebuild was
stopped at 26/80 trips after about 13 minutes; no partial results from that
attempt were used. Do not stage the generated cache directory; it is large
and reproducible.

## Measured results

The previous measured student-test baseline contains 10 trips and 19,160
observations:

| Method | Accuracy | Direct continuity |
|---|---:|---:|
| Classifier only | 87.9019% | 92.1201% |
| Connectivity HMM | 85.3758% | 98.7050% |
| Distance HMM, unoptimized | 86.5449% | 96.9504% |
| Distance HMM, previous optimized parameters | 88.7056% | 95.3525% |
| Distance HMM, corrected and reoptimized | 90.7933% | 97.8538% |

The previous optimized setting was `beta=30.0` and transition weight
`0.08866383439587917`. The previous unconstrained recovery inserted 4,447
intermediate arcs and left five unreachable connections; only eight trips
were fully connected. The previous constrained recovery inserted 944 arcs,
rejected 1,235 connections (1,230 over the speed-distance budget and five
unreachable), and fully connected zero trips. Those results are preserved.

On the previous optimized predictions, all 19,150 timestamp gaps were
positive (1–210 s; 138 exceeded 2 s). There were 13,044 same-arc forward
transitions and 757 same-arc backward projections within 30 m; none exceeded
30 m. This confirms that the former same-arc loop penalty was penalizing small
GPS jitter in this test set rather than substantial reverse movement. The
previous constrained experiment classified 523 rejected transitions as
same-road cases.

At `beta=30 m`, the prior `1e-12` probability floor began to collapse
reachable transitions with a distance error of about 829 m onto the same
log penalty as unreachable transitions. The revised HMM keeps finite
reachable costs distinct and reserves negative infinity for graph-unreachable
transitions. The reported 8,826 m-in-one-second route gap also exceeds the
configured recovery budget of 90 m (60 m/s plus 30 m GPS tolerance) and is
therefore split rather than forced.

The corrected HMM improves observation accuracy by 2.0877 percentage points
and direct continuity by 2.5013 points over the previous optimized HMM. On
the ten student-test routes, the corrected recovery inserts 540 arcs, rejects
39 transitions (36 physically implausible and three unreachable), creates 49
route sections, and fully connects two trips. Eight trips remain fragmented
where connections are implausible or unreachable.

| Recovery | Inserted arcs | Rejected/disconnected | Implausible | Sections | Fully connected trips | Sequence LCS F1 | Road-overlap F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous unconstrained | 4,447 | 5 | Not checked | 15 | 8 | 0.5675 | 0.6578 |
| Previous constrained | 944 | 1,235 | 1,230 | 1,245 | 0 | 0.6890 | 0.7177 |
| Corrected constrained | 540 | 39 | 36 | 49 | 2 | 0.8889 | 0.9455 |

These are not equivalent connectivity-only scores: sequence LCS F1 and
length-weighted road-overlap F1 (macro-averaged across trips) compare the
recovered route against the actual `.route` arc-ID sequence. The corrected
recovery has high route overlap while retaining breaks where no plausible
directed connection exists. The shared `evaluation/route_evaluation.py`
remains a `NotImplementedError` stub; these Phase 2 measurements are computed
in `route_metrics.py` without modifying Member 4's evaluation module.

The previous HMM pipeline runtime was 69.04 s; the corrected HMM pipeline
measured 153.70 s. The constrained recovery took 24.57 s compared with 2.97 s
for the legacy connectivity-only recovery when timed in the same evaluation
run. The improved checks cost additional runtime; the reported timings are
preserved rather than hidden.

The selected training-only parameters are `beta=30.0` and transition weight
`1.948614369534382` (trial 21/25). Training OOF accuracy/continuity were
80.8039%/97.9164% over the 80 training trips only.

New optimized HMM and recovery measurements are written to
`experiments/phase2_improved/overall_metrics.json` and
`experiments/phase2_improved/per_trip_comparison.csv`. Route sequences and
evaluation details are in `recovered_routes.json`; graphs are generated as
`observation_metrics.png`, `recovery_comparison.png`, `route_quality.png`, and
`runtime_comparison.png`. These are measured from all ten student-test
`.route` references and the original saved-model candidate probabilities.

## Backend integration

`reconstruct_route(candidates=..., road_network=..., gps_track=...,
transition_mode="distance", beta=..., transition_weight=...)` continues to
return the per-observation DataFrame with the `hmm_arc_id` field.

For physically constrained route recovery, pass the matching GPS track and
observation IDs:

```python
from route_reconstruction.route_recovery import recover_continuous_route

recovery = recover_continuous_route(
    predicted_arcs=route["hmm_arc_id"].astype(int).tolist(),
    graph=road_graph,
    gps_track=gps_track,
    observation_ids=route["obs_id"].to_numpy(dtype=int),
    max_speed_mps=60.0,
    gps_tolerance_m=30.0,
    backward_tolerance_m=30.0,
)
```

The result includes `recovered_arcs`, `route_sections`, `inserted_arcs`,
`rejected_connections`, and `is_continuous`. `disconnected_pairs` remains
available as an alias for the rejection list for callers of the legacy
`recover_continuous_route` interface. Without a GPS track, the existing
connectivity-only behavior is retained; time-based plausibility cannot be
evaluated in that mode.

## Reproduction

Run commands from the repository root in PowerShell. Use `.venv` for cache
preparation and optimization, and `.venv_inference` for inference/evaluation
with the compatible saved-model environment:

```powershell
.\.venv\Scripts\python.exe -m route_reconstruction.prepare_hmm_cache `
  --cache-dir route_reconstruction\experiments\phase2_improved\hmm_validation\cache `
  --reuse-cache-dir experiments\results\hmm_validation\cache

.\.venv\Scripts\python.exe -m route_reconstruction.optimise_hmm `
  --cache-dir route_reconstruction\experiments\phase2_improved\hmm_validation\cache `
  --output-dir route_reconstruction\experiments\phase2_improved\hmm_optimisation

.\.venv_inference\Scripts\python.exe -m route_reconstruction.run_phase2_evaluation

.\.venv\Scripts\python.exe -m route_reconstruction.plot_phase2_results

.\.venv\Scripts\python.exe -m pytest -q tests\test_route_reconstruction.py
```

The evaluation runner reads the already-generated original-model student-test
candidate probabilities. Labels and `.route` references are joined only after
decoding and optimization; student-test labels are never used to tune HMM
parameters. Neither evaluator-heldout data nor raw datasets are modified.

## Limitations

- A speed threshold is a configurable plausibility bound, not a learned travel
  speed model. Uncertain timing or a genuine stop can affect whether a gap is
  accepted.
- A shortest directed path is the minimum-distance graph route, not necessarily
  the route actually traveled. Long or infeasible paths are rejected rather
  than forced.
- Without timestamps, legacy recovery can establish directed connectivity but
  cannot assess physical plausibility.
- Arc-overlap scores do not distinguish parallel roads with different arc IDs
  even when their geometry is close; sequence scores operate on arc identity.
- The shared evaluator and backend service are still placeholders. Member 4
  should agree on these route metrics before wiring them into the shared
  evaluation endpoint.
