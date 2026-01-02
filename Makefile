.PHONY: help setup venv sweep multi-sweep visualize clean viz

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
	@echo "Visualization:"
	@echo "  make visualize      - Visualize UMAP embeddings from checkpoint"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove virtual environment"
	@echo ""
	@echo "Arguments (optional):"
	@echo "  CONFIG=<file>       - Sweep config file (default: sweep_config.yaml)"
	@echo "  PROJECT=<name>      - WandB project name (optional, uses YAML value if not specified)"
	@echo "  ENTITY=<name>       - WandB entity name (optional, uses YAML value if not specified)"
	@echo "  COUNT=<num>         - Number of runs for single sweep (optional, auto-calculated for grid)"
	@echo ""
	@echo "Visualization Arguments (for make visualize):"
	@echo "  RUN=<name>          - WandB run name (required, e.g., 20251228T030542_2556bb8_...)"
	@echo "  DATA_DIR=<path>     - Data directory with meta.pkl (optional, auto-detected if possible)"
	@echo "  DEVICE=<device>     - Device to use: cpu, cuda, mps (default: cpu)"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make sweep CONFIG=test_final.yaml"
	@echo "  make multi-sweep CONFIG=test_final.yaml ENTITY=my_team"
	@echo "  make sweep CONFIG=test_final.yaml PROJECT=my_project  # Override YAML project"
	@echo "  make visualize RUN=20251228T030542_2556bb8_DSET_G1000L5P1PeUdirDt_L3E256H1MlpAgeluLnBiasD0WtEp10000Seed7828"
	@echo "  make visualize RUN=<run_name> DATA_DIR=data/my_dataset DEVICE=cuda"
	@echo ""

# Variables with defaults
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python3
PIP = $(VENV_DIR)/bin/pip
CONFIG ?= sweep_config.yaml
PROJECT ?=
ENTITY ?=
COUNT ?=
RUN ?=
DATA_DIR ?=
DEVICE ?= cpu

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
	@echo "Configuring git..."
	@git config user.email "rafizsooham@gmail.com"
	@git config user.name "sooham"
	@git config pull.rebase true
	@echo "Git configured with user.email=rafizsooham@gmail.com and user.name=sooham"


# Single GPU sweep
sweep: $(VENV_DIR)/bin/activate
	@echo "Running single GPU sweep..."
	@echo "  Config: $(CONFIG)"
	@if [ -n "$(PROJECT)" ]; then echo "  Project: $(PROJECT) (override)"; else echo "  Project: (from YAML)"; fi
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
		$(if $(PROJECT),--project $(PROJECT),) \
		$(if $(ENTITY),--entity $(ENTITY),) \
		$(if $(COUNT),--count $(COUNT),)

# Multi-GPU sweep
multi-sweep: $(VENV_DIR)/bin/activate
	@echo "Running multi-GPU sweep..."
	@echo "  Config: $(CONFIG)"
	@if [ -n "$(PROJECT)" ]; then echo "  Project: $(PROJECT) (override)"; else echo "  Project: (from YAML)"; fi
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
	./run_multi_gpu_sweep.sh $(CONFIG) "$(PROJECT)" "$(ENTITY)"

# Visualize UMAP embeddings from checkpoint
viz: visualize
visualize: $(VENV_DIR)/bin/activate
	@echo "Visualizing UMAP embeddings..."
	@if [ -z "$(RUN)" ]; then \
		echo "Error: RUN parameter is required!"; \
		echo "Usage: make visualize RUN=<wandb_run_name>"; \
		echo "Example: make visualize RUN=20251228T030542_2556bb8_DSET_G1000L5P1PeUdirDt_L3E256H1MlpAgeluLnBiasD0WtEp10000Seed7828"; \
		exit 1; \
	fi
	@FOLDER=$${FOLDER:-out}; \
	CKPT_PATH="$$FOLDER/ckpt_$(RUN).pt"; \
	if [ ! -f "$$CKPT_PATH" ]; then \
		echo "Error: Checkpoint file not found: $$CKPT_PATH"; \
		echo "Please check that the RUN name is correct and the checkpoint exists."; \
		exit 1; \
	fi; \
	echo "  Checkpoint: $$CKPT_PATH"; \
	if [ -n "$(DATA_DIR)" ]; then \
		echo "  Data directory: $(DATA_DIR)"; \
		echo "  Device: $(DEVICE)"; \
		$(PYTHON) visualize_embeddings_umap.py \
			--checkpoint "$$CKPT_PATH" \
			--data_dir "$(DATA_DIR)" \
			--device "$(DEVICE)" \
			--save_dir "visualizations/$(RUN)" \
			$(if $(INCLUDE_ROOT),--include-root,); \
	else \
		echo "  Data directory: (auto-detect from checkpoint)"; \
		echo "  Device: $(DEVICE)"; \
		$(PYTHON) visualize_embeddings_umap.py \
			--checkpoint "$$CKPT_PATH" \
			--device "$(DEVICE)" \
			--save_dir "visualizations/$(RUN)" \
			$(if $(INCLUDE_ROOT),--include-root,); \
	fi

# Clean up virtual environment
clean:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Cleanup complete!"

# Copy a zip file from VastAI instance via SCP
# Usage: make copy-zip NAME=<filename> HOST=<vastai_host>
# Example: make copy-zip NAME=results HOST=ssh6.vast.ai -p 29299
copy-zip:
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter is required!"; \
		echo "Usage: make copy-zip NAME=<filename>"; \
		exit 1; \
	fi
	@echo "Copying $(NAME).zip from VastAI instance..."
	@echo "  Source: root@vastai:/workspace/pathstar-geometric-learning/$(NAME).zip"
	@echo "  Destination: ./$(NAME).zip"
	scp root@vastai:/workspace/pathstar-geometric-learning/$(NAME).zip ./$(NAME).zip
	@echo "Copy complete!"
