# govim integration tests

Files in this directory configure Cloud Build to run [govim] integration tests
against a gopls binary built from source.

## Running on GCP

To run these integration tests in Cloud Build, use the following steps.  Here
we assume that `$PROJECT_ID` is a valid GCP project and `$BUCKET` is a cloud
storage bucket owned by that project.

- `cd` to the root directory of the tools project.
- (at least once per GCP project) Build the test harness:
```
$ gcloud builds submit \
	--project="${PROJECT_ID}" \
	--config=gopls/integration/govim/cloudbuild.harness.yaml
```
- Run the integration tests:
```
$ gcloud builds submit \
	--project="${PROJECT_ID}" \
	--config=gopls/integration/govim/cloudbuild.yaml \
	--substitutions=_RESULT_BUCKET="${BUCKET}"
```

## Fetching Artifacts

Assuming the artifacts bucket is world readable, you can fetch integration from
GCS. They are located at:

- logs: `https://storage.googleapis.com/${BUCKET}/log-${EVALUATION_ID}.txt`
- artifact tarball: `https://storage.googleapis.com/${BUCKET}/govim/${EVALUATION_ID}/artifacts.tar.gz`

The `artifacts.go` command can be used to fetch both artifacts using an
evaluation id.

## Running locally

Run `gopls/integration/govim/run_local.sh`. This may take a while the first
time it is run, as it will require building the test harness. This script
accepts two flags to modify its behavior:

**--sudo**: run docker with `sudo`
**--short**: run `go test -short`

[govim]: https://github.com/govim/govim
