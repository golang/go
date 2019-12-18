# govim integration tests

Files in this directory configure Cloud Build to run [govim] integration tests
against a gopls binary built from source.

## Running on GCP

To run these integration tests in Cloud Build (assuming the `gcloud` command is
configured for a valid GCP project):

- `cd` to the root directory of the tools project.
- (at least once per GCP project) Build the test harness:  
```
$ gcloud builds submit --config=gopls/integration/govim/cloudbuild.harness.yaml
```
- Run the integration tests:  
```
$ gcloud builds submit --config=gopls/integration/govim/cloudbuild.yaml
```

## Running locally

Run `gopls/integration/govim/run_local.sh`. This may take a while the first
time it is run, as it will require building the test harness. Currently this
script assumes that docker may be executed without `sudo`.

[govim]: https://github.com/govim/govim
