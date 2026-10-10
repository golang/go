#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
src="$root/src"
go_bin="$root/bin/go"
if [ -x "$root/native/bin/go-native" ]; then go_bin="$root/native/bin/go-native"; fi
if [ ! -x "$go_bin" ]; then echo "Go is not built; run src/make.bash first" >&2; exit 1; fi
command -v "${CC:-cc}" >/dev/null 2>&1 || { echo "C compiler not found: ${CC:-cc}" >&2; exit 1; }
stage="$root/shared-stage"
rm -rf "$stage"
mkdir -p "$stage/bin" "$stage/pkg/tool/linux_amd64"
launcher="$src/cmd/sharedlauncher/launcher.c"

build_tool() {
	tool=$1; dir=$2
	(cd "$src" && GOOS=linux GOARCH=amd64 CGO_ENABLED=1 CC="${CC:-cc}" GOFLAGS= \
		"$go_bin" build -tags=sharedtool -buildmode=c-shared -o "$stage/$dir/$tool.so" "cmd/$tool")
	"${CC:-cc}" -Os -o "$stage/$dir/$tool" "$launcher" -ldl
	rm -f "$stage/$dir/$tool.h"
}
install_tool() {
	tool=$1; dir=$2
	mkdir -p "$root/native/$dir"
	if [ -x "$root/$dir/$tool" ]; then
		if [ ! -x "$root/native/$dir/$tool-native" ]; then mv "$root/$dir/$tool" "$root/native/$dir/$tool-native"; else rm -f "$root/$dir/$tool"; fi
	fi
	install -m 755 "$stage/$dir/$tool" "$root/$dir/$tool"
	install -m 755 "$stage/$dir/$tool.so" "$root/$dir/$tool.so"
}

for tool in go gofmt; do build_tool "$tool" bin; done
for tool in asm cgo compile link preprofile cover export fix vet; do build_tool "$tool" pkg/tool/linux_amd64; done
for tool in go gofmt; do install_tool "$tool" bin; done
for tool in asm cgo compile link preprofile cover export fix vet; do install_tool "$tool" pkg/tool/linux_amd64; done
echo "Shared Go toolchain installed under $root"
echo "Native executables are preserved under $root/native"
