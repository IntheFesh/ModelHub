# ModelHub · Incident Log

Append-only. Written by `release/incident_log.py` — never hand-edited.

---

## 2026-08-24T07:44:54.200040+00:00 · GATE_REJECTION · demo-model@v-doomed

- run_id: `demo-run`
- reason: accuracy: execution_accuracy 0.00% < required 50.00%
- detail: 1 gate(s) rejected: accuracy

---

## 2026-08-24T07:45:44.525065+00:00 · GATE_REJECTION · demo-model@v-doomed

- run_id: `demo-run`
- reason: accuracy: execution_accuracy 0.00% < required 50.00%
- detail: 1 gate(s) rejected: accuracy

---

## 2026-08-24T08:55:14.190119+00:00 · GATE_REJECTION · bad-model-drill@ckpt-a-underfit

- run_id: `drill-CKPT_A_UNDERFIT`
- reason: accuracy: execution_accuracy 20.00% < required 50.00%
- detail: 1 gate(s) rejected: accuracy

---

## 2026-08-24T08:55:14.191725+00:00 · GATE_REJECTION · bad-model-drill@ckpt-b-regression

- run_id: `drill-CKPT_B_REGRESSION`
- reason: regression: 10 regression(s) exceed the allowed 2: ['complex0', 'complex1', 'complex2', 'complex3', 'complex4', 'complex5', 'complex6', 'complex7', 'complex8', 'complex9']
- detail: 1 gate(s) rejected: regression

---

## 2026-08-24T08:55:14.192890+00:00 · GATE_REJECTION · bad-model-drill@ckpt-d-safety

- run_id: `drill-CKPT_D_SAFETY`
- reason: accuracy: execution_accuracy 30.00% < required 50.00%
- detail: 1 gate(s) rejected: accuracy

