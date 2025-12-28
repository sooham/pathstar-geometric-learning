.PHONY: help setup venv sweep multi-sweep clean

# Default target
help:
	@echo "PathStar Geometric Learning - Make Targets"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create venv and install requirements"
	@echo "  make venv           - Alias for setup"
	@echo ""
	@echo "Running Sweeps:"
	@echo "  make sweep          - Run single GPU sweep"
	@echo "  make multi-sweep    - Run multi-GPU sweep"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove virtual environment"
	@echo ""
	@echo "Arguments (optional):"
	@echo "  CONFIG=<file>       - Sweep config file (default: sweep_config.yaml)"
	@echo "  PROJECT=<name>      - WandB project name (default: pathstar_sweep_dataset)"
	@echo "  ENTITY=<name>       - WandB entity name (optional)"
	@echo "  COUNT=<num>         - Number of runs for single sweep (optional, auto-calculated for grid)"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make sweep CONFIG=test_final.yaml PROJECT=my_project"
	@echo "  make multi-sweep CONFIG=test_final.yaml PROJECT=my_project ENTITY=my_team"
	@echo ""

# Variables with defaults
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python3
PIP = $(VENV_DIR)/bin/pip
CONFIG ?= sweep_config.yaml
PROJECT ?= pathstar_sweep_dataset
ENTITY ?=
COUNT ?=

# Setup target - create venv and install requirements
setup: $(VENV_DIR)/bin/activate

venv: setup

$(VENV_DIR)/bin/activate: requirements.txt
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "Installing requirements..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Setup complete! Virtual environment created at: $(VENV_DIR)"
	@echo "To activate manually: source $(VENV_DIR)/bin/activate"
	@echo ""
	@if [ -f /etc/os-release ] && grep -q "Ubuntu" /etc/os-release; then \
		echo "Detected Ubuntu - creating ~/.no_auto_tmux to disable auto-tmux..."; \
		touch ~/.no_auto_tmux; \
	fi

# Single GPU sweep
sweep: $(VENV_DIR)/bin/activate
	@echo "Running single GPU sweep..."
	@echo "  Config: $(CONFIG)"
	@echo "  Project: $(PROJECT)"
	@if [ -n "$(ENTITY)" ]; then echo "  Entity: $(ENTITY)"; fi
	@if [ -n "$(COUNT)" ]; then echo "  Count: $(COUNT)"; fi
	@echo ""
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "Error: Config file '$(CONFIG)' not found!"; \
		echo "Please specify a valid config file with CONFIG=<file>"; \
		exit 1; \
	fi
	$(PYTHON) run_sweep.py \
		--sweep_config $(CONFIG) \
		--project $(PROJECT) \
		$(if $(ENTITY),--entity $(ENTITY),) \
		$(if $(COUNT),--count $(COUNT),)

# Multi-GPU sweep
multi-sweep: $(VENV_DIR)/bin/activate
	@echo "Running multi-GPU sweep..."
	@echo "  Config: $(CONFIG)"
	@echo "  Project: $(PROJECT)"
	@if [ -n "$(ENTITY)" ]; then echo "  Entity: $(ENTITY)"; fi
	@echo ""
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "Error: Config file '$(CONFIG)' not found!"; \
		echo "Please specify a valid config file with CONFIG=<file>"; \
		exit 1; \
	fi
	@if ! command -v nvidia-smi &> /dev/null; then \
		echo "Warning: nvidia-smi not found. Multi-GPU sweep may not work as expected."; \
		echo "Continuing anyway..."; \
		echo ""; \
	fi
	./run_multi_gpu_sweep.sh $(CONFIG) $(PROJECT) $(ENTITY)

# Clean up virtual environment
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Cleanup complete!"

