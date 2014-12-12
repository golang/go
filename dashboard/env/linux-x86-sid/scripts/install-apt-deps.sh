set -ex

apt-get update
apt-get install -y --no-install-recommends ca-certificates
# Optionally used by some net/http tests:
apt-get install -y --no-install-recommends strace 
# For building Go's bootstrap 'dist' prog
apt-get install -y --no-install-recommends gcc libc6-dev
# For 32-bit builds:
# TODO(bradfitz): move these into a 386 image that derives from this one.
apt-get install -y --no-install-recommends libc6-dev-i386 gcc-multilib
# For interacting with the Go source & subrepos:
apt-get install -y --no-install-recommends git-core

apt-get clean
rm -fr /var/lib/apt/lists
