#!/bin/sh
set -eu
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
root=$repo
go="$root/bin/go"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
"$go" version
"$go" env >/dev/null
"$go" tool compile -V
"$go" tool asm -V
"$go" tool link -V
"$go" build std
mkdir -p "$tmp/mathx"
cat >"$tmp/go.mod" <<'EOF'
module example.com/smoke

go 1.26
EOF
cat >"$tmp/mathx/mathx.go" <<'EOF'
package mathx
func Add(a, b int) int { return a + b }
EOF
cat >"$tmp/mathx/mathx_test.go" <<'EOF'
package mathx
import "testing"
func TestAdd(t *testing.T) { if Add(20, 22) != 42 { t.Fatal("bad sum") } }
EOF
cat >"$tmp/main.go" <<'EOF'
package main
import ("fmt"; "example.com/smoke/mathx")
func main() { fmt.Println(mathx.Add(20, 22)) }
EOF
(
	cd "$tmp"
	[ "$("$go" run .)" = 42 ]
	"$go" test ./...
	"$go" build -o hello .
	[ "$(./hello)" = 42 ]
	GOOS=windows GOARCH=amd64 CGO_ENABLED=0 "$go" build -o hello.exe .
)
"${CC:-cc}" "$repo/src/cmd/sharedlauncher/ffi_smoke.c" -O2 -ldl -o "$tmp/ffi_smoke"
"$tmp/ffi_smoke" "$root/bin/go.so" dll_go_main go version
"$tmp/ffi_smoke" "$root/bin/gofmt.so" dll_gofmt_main gofmt "$tmp/main.go"
for tool in asm cgo compile link preprofile cover export fix vet; do
	"$tmp/ffi_smoke" "$root/pkg/tool/linux_amd64/$tool.so" "dll_${tool}_main" "$tool" -V
	done
echo 'Linux amd64 shared-toolchain smoke tests passed.'
