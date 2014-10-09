set -ex

# Installs a version of the go.tools dashboard builder that runs the gccgo build
# command assuming there are 16 cores available to speed up build times.
# TODO(cmang): There should be an option in the builder to specify this.

curl -o /usr/local/bin/builder http://storage.googleapis.com/go-builder-data/gccgo_builder && chmod +x /usr/local/bin/builder
