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
	bytes, shared, hasShared := runtime.SysRSS()
	if bytes == 0 {
		t.Fatalf("SysRSS unavailable on supported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	if runtime.GOOS == "linux" && !hasShared {
		t.Error("SysRSS did not report shared RSS on Linux")
	}
	if shared > bytes || !hasShared && shared != 0 {
		t.Errorf("SysRSS returned inconsistent counts: rss=%d, shared=%d, hasShared=%t", bytes, shared, hasShared)
	}
	t.Logf("SysRSS = %d bytes (%d MiB), shared = %d bytes", bytes, bytes>>20, shared)
}

func TestSysRSSNoAlloc(t *testing.T) {
	requireSysRSS(t)
	if rss, _, _ := runtime.SysRSS(); rss == 0 {
		t.Fatalf("SysRSS unavailable on supported platform %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	allocs := testing.AllocsPerRun(100, func() {
		runtime.SysRSS()
	})
	if allocs != 0 {
		t.Errorf("SysRSS allocated %v objects per call; want 0", allocs)
	}
}
