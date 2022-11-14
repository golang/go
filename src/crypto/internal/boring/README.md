This directory holds the core of the BoringCrypto implementation
as well as the build scripts for the module itself: syso/*.syso.

syso/goboringcrypto_linux_amd64.syso is built with:

	GOARCH=amd64 ./build.sh

syso/goboringcrypto_linux_arm64.syso is built with:

	GOARCH=arm64 ./build.sh

Both run on an x86 Debian Linux system using Docker.
For the arm64 build to run on an x86 system, you need

	apt-get install qemu-user-static qemu-binfmt-support

to allow the x86 kernel to run arm64 binaries via QEMU.

See build.sh for more details.
