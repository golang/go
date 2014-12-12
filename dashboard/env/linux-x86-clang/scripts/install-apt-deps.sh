set -ex

apt-get update
apt-get install -y --no-install-recommends ca-certificates
# Optionally used by some net/http tests:
apt-get install -y --no-install-recommends strace 
# For building Go's bootstrap 'dist' prog
apt-get install -y --no-install-recommends wget
wget -O - http://llvm.org/apt/llvm-snapshot.gpg.key | apt-key add -
apt-get update
apt-get install -y --no-install-recommends clang-3.5
# TODO(cmang): move these into a 386 image that derives from this one.
apt-get install -y --no-install-recommends libc6-dev-i386 gcc-multilib
# Remove gcc binary so it doesn't interfere with clang
rm -f /usr/bin/gcc
# For interacting with the Go source & subrepos:
apt-get install -y --no-install-recommends git-core

apt-get clean
rm -fr /var/lib/apt/lists
