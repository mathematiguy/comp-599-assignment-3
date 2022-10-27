REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
GIT_TAG ?= $(shell git log --oneline | head -n1 | awk '{print $$1}')
DOCKER_REGISTRY := mathematiguy
IMAGE := $(DOCKER_REGISTRY)/$(REPO_NAME)
HAS_DOCKER ?= $(shell which docker)
RUN ?= $(if $(HAS_DOCKER), docker run $(DOCKER_ARGS) --gpus all --network host --rm -v $$(pwd):/home/user -w /home/user -u $(UID):$(GID) $(IMAGE))
UID ?= user
GID ?= user
DOCKER_ARGS ?=

.PHONY: docker docker-push docker-pull enter enter-root

run: code.py data/glove/glove.6B.300d.txt
	$(RUN) python $<

run_snli_experiments: $(addprefix run_snli_, unilstm unilstm_glove shallow_bilstm shallow_bilstm_glove bilstm bilstm_glove)
run_rnn_experiments: $(addprefix run_rnn_, sherlock_charnn sherlock_lstm shakespeare_charnn shakespeare_lstm)

run_rnn_sherlock_charnn:
	$(RUN) python run_rnn_experiment.py --model_name sherlock_charnn --text_file data/sherlock.txt --num_epochs 1000 \
		--batch_size 32 --out_seq_len 200 --hidden_size 512 --embedding_size 300 --lr 0.002 --seq_len 100

run_rnn_sherlock_lstm:
	$(RUN) python run_rnn_experiment.py --model_name sherlock_lstm --text_file data/sherlock.txt --num_epochs 1000 \
		--batch_size 32 --out_seq_len 200 --hidden_size 512 --embedding_size 300 --lr 0.002 --seq_len 100

run_rnn_shakespeare_charnn:
	$(RUN) python run_rnn_experiment.py --model_name shakespeare_charnn --text_file data/shakespeare.txt --num_epochs 1000 \
		--batch_size 32 --out_seq_len 200 --hidden_size 512 --embedding_size 300 --lr 0.002 --seq_len 100

run_rnn_shakespeare_lstm:
	$(RUN) python run_rnn_experiment.py --model_name shakespeare_lstm --text_file data/shakespeare.txt --num_epochs 1000 \
		--batch_size 32 --out_seq_len 200 --hidden_size 512 --embedding_size 300 --lr 0.002 --seq_len 100

run_snli_unilstm:
	$(RUN) python run_snli_experiment.py --model_name UniLSTM --num_epochs 100 --hidden_dim 512 --num_layers 1

run_snli_unilstm_glove:
	$(RUN) python run_snli_experiment.py --model_name UniLSTM --num_epochs 100 --hidden_dim 512 --num_layers 1 \
		--use_glove True

run_snli_shallow_bilstm:
	$(RUN) python run_snli_experiment.py --model_name ShallowBiLSTM --num_epochs 100 --hidden_dim 256 --num_layers 1 \

run_snli_shallow_bilstm_glove:
	$(RUN) python run_snli_experiment.py --model_name ShallowBiLSTM --num_epochs 100 --hidden_dim 256 --num_layers 1 \
		--use_glove True

run_snli_bilstm:
	$(RUN) python run_snli_experiment.py --model_name BiLSTM --num_epochs 100 --hidden_dim 512 --num_layers 1 \

run_snli_bilstm_glove:
	$(RUN) python run_snli_experiment.py --model_name BiLSTM --num_epochs 100 --hidden_dim 512 --num_layers 1 \
		--use_glove True

data/glove/glove.6B.300d.txt:
	wget https://nlp.stanford.edu/data/glove.6B.zip -P data/glove
	unzip data/glove/glove.6B.zip -d data/glove

JUPYTER_PASSWORD ?= jupyter
JUPYTER_PORT ?= 8888
.PHONY: jupyter
jupyter: DOCKER_ARGS=-u $(UID):$(GID) --rm -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) -e NB_USER=$$USER -e NB_UID=$(UID) -e NB_GID=$(GID)
jupyter:
	$(RUN) jupyter lab \
		--port $(JUPYTER_PORT) \
		--ip 0.0.0.0 \
		--NotebookApp.password="$(shell $(RUN) \
			python3 -c \
			"from notebook.auth import passwd; print(passwd('$(JUPYTER_PASSWORD)', 'sha1'))")"

clean:
	rm -rf data

docker:
	docker build $(DOCKER_ARGS) --tag $(IMAGE):$(GIT_TAG) .
	docker tag $(IMAGE):$(GIT_TAG) $(IMAGE):latest

docker-push:
	docker push $(IMAGE):$(GIT_TAG)
	docker push $(IMAGE):latest

docker-pull:
	docker pull $(IMAGE):$(GIT_TAG)
	docker tag $(IMAGE):$(GIT_TAG) $(IMAGE):latest

enter: DOCKER_ARGS=-it
enter:
	$(RUN) bash

enter-root: DOCKER_ARGS=-it
enter-root: UID=root
enter-root: GID=root
enter-root:
	$(RUN) bash
