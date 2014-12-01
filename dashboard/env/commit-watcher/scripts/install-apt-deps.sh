set -ex

apt-get update
# For running curl to get the hg starter tarballs (faster than hg clone).
apt-get install -y --no-install-recommends curl ca-certificates
# For building Go's bootstrap 'dist' prog
apt-get install -y --no-install-recommends gcc libc6-dev
# For interacting with the Go source & subrepos:
apt-get install -y --no-install-recommends mercurial git-core

apt-get clean
rm -fr /var/lib/apt/lists
