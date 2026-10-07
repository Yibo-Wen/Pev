# Score the released model on the reserved test families. See README.md.

PY       ?= .venv/bin/python
VARIANTS ?= data/processed/variants.parquet
PANELS   ?= data/exp1_panels.json
MODEL    ?= checkpoints/pev

.PHONY: all check discovery clean-results

all: check discovery

# Confirm the download is the pinned one. Needs no GPU; run it first.
check:
	$(PY) scripts/check_data.py --pins $(PANELS) --variants $(VARIANTS)

# The discovery screen. Panel set and ranking head pinned before the test split was scored.
discovery:
	$(PY) scripts/run_discovery.py --checkpoint $(MODEL) --zero-shot \
	  --variants $(VARIANTS) --panels $(PANELS) \
	  --heads binding=choice,stability=choice --out results/discovery

clean-results:
	rm -rf results/discovery
