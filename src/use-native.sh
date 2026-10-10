#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
for tool in go gofmt; do
	if [ -x "$root/native/bin/$tool-native" ]; then cp "$root/native/bin/$tool-native" "$root/bin/$tool"; fi
	done
for tool in asm cgo compile link preprofile cover export fix vet; do
	if [ -x "$root/native/pkg/tool/linux_amd64/$tool-native" ]; then cp "$root/native/pkg/tool/linux_amd64/$tool-native" "$root/pkg/tool/linux_amd64/$tool"; fi
	done
echo "Restored native Go executables under $root"
