set -ex

apt-get update
# For running curl to get the gccgo builder binary.
apt-get install -y --no-install-recommends curl ca-certificates
# Optionally used by some net/http tests:
apt-get install -y --no-install-recommends strace
# For using numeric libraries within GCC.
apt-get install -y --no-install-recommends libgmp10-dev libmpc-dev libmpfr-dev
# For building binutils and gcc from source.
apt-get install -y --no-install-recommends make g++ flex bison
# Same as above, but for 32-bit builds as well.
apt-get install -y --no-install-recommends libc6-dev-i386 g++-multilib
# For running the extended gccgo testsuite
apt-get install -y --no-install-recommends dejagnu
# For interacting with the gccgo source and git mirror:
apt-get install -y --no-install-recommends mercurial git-core

apt-get clean
rm -rf /var/lib/apt/lists
