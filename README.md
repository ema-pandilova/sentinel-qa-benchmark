# SENTINEL-QA

A bilingual (English and Macedonian) cybersecurity question-answering benchmark and
the retrieval-augmented generation (RAG) evaluation harness built around it. The code
indexes a corpus of cybersecurity documents, runs four answer-generation pipelines
(zero-shot, classic RAG, DSPy-structured RAG, and GraphRAG) across several generator
models, and scores the answers with a panel of LLM judges on two axes: factual
correctness and grounding.

## Repository layout

```
rag/                retrieval, embeddings, graph construction and expansion, generation pipelines
bench/              benchmark runner, LLM-as-judge evaluation, analysis, figures, tables
data/
  benchmark.jsonl   the 80-item benchmark: question, reference answer, difficulty,
                    language, source-document id, and reference chunk ids
  analysis/         aggregated metrics that back the figures and tables
main.py             FastAPI service
ui_gradio.py        Gradio research UI
create_database.py  index construction entry point
run_all.sh          runs the full pipeline end to end
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the keys you need
```

Keys for OpenAI, Anthropic, Google, and DeepSeek are each only required for the
pipelines and judges you actually run.

## Building the index

The vector store is not shipped — it is large and fully regenerable from the corpus:

```bash
python rag/create_database.py --reset
```

To build the similarity graphs used by GraphRAG:

```bash
python -m bench.run_benchmark --experiment build-graphs
```

## Running the evaluation

```bash
python -m bench.run_benchmark --experiment multi-model   # generate answers
python -m bench.meta_eval                                 # judge grounding + factual correctness
python -m bench.analyze_judgements                        # aggregate
python -m bench.metrics                                   # retrieval, latency, cost
```

`run_all.sh` chains these together. Every stage is resumable, so an interrupted run
picks up where it stopped.

## Reproducing the figures and tables

The aggregated outputs in `data/analysis/` are included, so the figures and tables can
be rebuilt without rerunning the pipeline:

```bash
python -m bench.figures
python -m bench.tables
```

## Serving

```bash
uvicorn main:app --host 0.0.0.0 --port 8000   # REST API
python ui_gradio.py                           # research UI
```

## Data

`data/benchmark.jsonl` and the aggregated metrics under `data/analysis/` are included
here. The full per-configuration answer set and the raw per-judge decisions are large;
they are available from the authors on request.

## Citation

If you use the benchmark or the code, please cite the accompanying paper. Citation
details will be added once it is published.

## License

Released under the MIT License. See [LICENSE](LICENSE).
