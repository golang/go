set -ex

apt-get update
apt-get install -y --no-install-recommends ca-certificates
# For building Go's bootstrap 'dist' prog
apt-get install -y --no-install-recommends gcc libc6-dev
# For interacting with the Go source & subrepos:
apt-get install -y --no-install-recommends git-core

apt-get clean
rm -fr /var/lib/apt/lists
