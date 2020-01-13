# govim integration tests

Files in this directory configure Cloud Build to run [govim] integration tests
against a gopls binary built from source.

## Running on GCP

To run these integration tests in Cloud Build, use the following steps.  Here
we assume that `$PROJECT` is a valid GCP project and `$BUCKET` is a cloud
storage bucket owned by that project.

- `cd` to the root directory of the tools project.
- (at least once per GCP project) Build the test harness:
```
$ gcloud builds submit \
	--project="${PROJECT}" \
	--config=gopls/integration/govim/cloudbuild.harness.yaml \
	--substitutions=_RESULT_BUCKET="${BUCKET}"
```
- Run the integration tests:
```
$ gcloud builds submit \
	--project="${PROJECT}" \
	--config=gopls/integration/govim/cloudbuild.yaml \
	--substitutions=_RESULT_BUCKET="${BUCKET}"
```

## Running locally

Run `gopls/integration/govim/run_local.sh`. This may take a while the first
time it is run, as it will require building the test harness. Currently this
script assumes that docker may be executed without `sudo`.

[govim]: https://github.com/govim/govim
