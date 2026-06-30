# Roadmap

## Phase 1: proof object

- [x] Build synthetic key-value recall task.
- [x] Build local baseline and HPM-Lite.
- [x] Run initial 512/2048/4096/8192 distance checks.
- [x] Add null-slot retrieval option.
- [ ] Add one-command distance sweep.
- [ ] Add VRAM and tokens/sec logging.
- [ ] Add multi-seed table.

## Phase 2: ablations

- [ ] no episodic memory
- [ ] no recurrent path
- [ ] no router
- [ ] fixed router
- [ ] no null slot

## Phase 3: harder memory tasks

- [ ] near-duplicate keys
- [ ] missing-key/null queries
- [ ] multi-query recall
- [ ] multi-hop key chains
- [ ] noisy fact extraction

## Phase 4: learned memory control

- [ ] learned write score
- [ ] write budget
- [ ] write-value supervision
- [ ] learned null routing

## Phase 5: tiny language bridge

Only after exact recall metrics are stable:

- [ ] small pretrained model + frozen backbone
- [ ] memory adapter / router
- [ ] synthetic-to-text transfer task
- [ ] no large chatbot training from scratch
