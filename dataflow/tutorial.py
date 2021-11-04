# gcloud beta auth application-default login

# With direct runner
python -m apache_beam.examples.wordcount --output outputs

# With dataflow
python -m apache_beam.examples.wordcount \
    --region northamerica-northeast1  \
    --input gs://dataflow-samples/shakespeare/kinglear.txt \
    --output gs://df-tutorial/results/outputs \
    --runner DataflowRunner \
    --project coastal-epigram-162302 \
    --temp_location gs://df-tutorial/tmp/


# With direct runner, do the modified version
python modified_wordcount.py --output outputs