// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu_test

import (
	. "internal/cpu"
	"runtime"
	"testing"
	"unsafe"
)

func TestLogosCacheLinePadSize(t *testing.T) {
	var want uintptr
	switch runtime.GOARCH {
	case "arm64":
		want = 128
	case "amd64", "386":
		want = 64
	case "arm":
		want = 32
	case "loong64":
		want = 64
	case "mips", "mipsle", "mips64", "mips64le":
		want = 32
	case "ppc64", "ppc64le":
		want = 128
	case "riscv64":
		want = 64
	case "s390x":
		want = 256
	case "wasm":
		want = 64
	default:
		t.Skipf("unknown GOARCH %s", runtime.GOARCH)
	}
	if CacheLinePadSize != want {
		t.Fatalf("CacheLinePadSize = %d, want %d", CacheLinePadSize, want)
	}
	if got := unsafe.Sizeof(CacheLinePad{}); got != want {
		t.Fatalf("Sizeof(CacheLinePad) = %d, want %d", got, want)
	}
}
