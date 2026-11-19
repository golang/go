// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"os"
	"runtime"
	"testing"
)

func requireSysRSS(t *testing.T) {
	t.Helper()
	switch runtime.GOOS {
	case "darwin":
	case "linux":
		if _, err := os.Stat("/proc/self/statm"); err != nil {
			t.Skipf("SysRSS unavailable: %v", err)
		}
	default:
		t.Skipf("SysRSS unsupported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

func TestSysRSS(t *testing.T) {
	requireSysRSS(t)
	bytes, ok := runtime.SysRSS()
	if !ok {
		t.Fatalf("SysRSS unavailable on supported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	if bytes == 0 {
		t.Errorf("SysRSS returned ok=true but bytes=0")
	}
	t.Logf("SysRSS = %d bytes (%d MiB)", bytes, bytes>>20)
}

func TestSysRSSNoAlloc(t *testing.T) {
	requireSysRSS(t)
	if _, ok := runtime.SysRSS(); !ok {
		t.Fatalf("SysRSS unavailable on supported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	allocs := testing.AllocsPerRun(100, func() {
		runtime.SysRSS()
	})
	if allocs != 0 {
		t.Errorf("SysRSS allocated %v objects per call; want 0", allocs)
	}
}
