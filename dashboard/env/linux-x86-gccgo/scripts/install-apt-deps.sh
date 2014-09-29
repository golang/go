set -ex

apt-get update
# For using numeric libraries within GCC.
apt-get install -y --no-install-recommends libgmp10-dev libmpc-dev libmpfr-dev
# For building binutils and gcc from source.
apt-get install -y --no-install-recommends make g++ flex bison
# For running the extended gccgo testsuite
apt-get install -y --no-install-recommends dejagnu

apt-get clean
rm -rf /var/lib/apt/lists
