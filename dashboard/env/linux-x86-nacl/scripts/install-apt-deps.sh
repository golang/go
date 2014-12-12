set -ex

apt-get update
# curl is needed to fetch the sel_ldr nacl binaries:
apt-get install -y --no-install-recommends curl ca-certificates
# For building Go's bootstrap 'dist' prog
apt-get install -y --no-install-recommends gcc libc6-dev
# For interacting with the Go source & subrepos:
apt-get install -y --no-install-recommends git-core
# For 32-bit nacl:
apt-get install -y --no-install-recommends libc6-i386 libc6-dev-i386 lib32stdc++6 gcc-multilib

apt-get clean
rm -fr /var/lib/apt/lists
