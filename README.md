# temporal-steering-repro

Reproduction of Behera et al., "Temporal steering of entanglement decay with
single-shot control," Phys. Rev. A 114, 012416 (2026).

**Phase 1:** discrete Kraus-map calculation of concurrence for two cascaded
amplitude-damping channels with an optional intermediate local NOT. See
`NOTES.md` for the full write-up (NOT-convention check, phase diagram,
comparison against Table I).

**Phase 2:** continuous-time Lindblad master-equation calculation
(independent atomic spontaneous emission, H=0), showing it reduces to
Phase 1's discrete picture under p(tau)=1-exp(-gamma tau), plus the native
continuous-time result (concurrence vs. intervention time for fixed total
observation time). See `NOTES_PHASE2.md`.

**Phase 3:** extends Phase 2 by coupling each atom to its own lossy cavity
(Jaynes-Cummings + cavity decay kappa), checking the model collapses back
to Phase 2 at g=0, and sweeping whether the optimal NOT-intervention timing
shifts with kappa/gamma. See `NOTES_PHASE3.md`.

## Layout

```
src/model.py                       Phase 1: core Kraus-map / concurrence code
src/step1_convention_check.py      Phase 1, Step 1: resolve the NOT convention
src/step2_phase_diagram.py         Phase 1, Step 2: phase diagram + Table I check
src/lindblad.py                    Phase 2: continuous-time Lindblad model
src/step3_lindblad_validation.py   Phase 2, Step 3: validate against Phase 1
src/step4_continuous_time.py       Phase 2, Step 4: C(T) vs. intervention time t1
src/cavity_lindblad.py             Phase 3: atom+cavity Lindblad model
src/step5_cavity_lindblad.py       Phase 3, Step 5: g=0 sanity check + kappa/gamma sweep
src/step6_cavity_strong_coupling.py Phase 3 follow-up: strong-coupling (g/gamma=10) stress test
figures/                           generated plots
results_step*.txt                  per-step text output
NOTES.md                           Phase 1 write-up
NOTES_PHASE2.md                    Phase 2 write-up
NOTES_PHASE3.md                    Phase 3 write-up
```

## Run

```
pip install -r requirements.txt
cd src
python step1_convention_check.py
python step2_phase_diagram.py
python step3_lindblad_validation.py
python step4_continuous_time.py
python step5_cavity_lindblad.py
python step6_cavity_strong_coupling.py
```
