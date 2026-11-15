// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !goexperiment.logosfastcgo

package runtime_test

import (
	"runtime"
	"sync/atomic"
	"testing"
	"unsafe"
)

func TestLogosFastcallOffPreservesCallbackFlag(t *testing.T) {
	var observed atomic.Bool
	runtime.SetLogosFastcallOffHookForTest(func(fn, arg unsafe.Pointer) int32 {
		observed.Store(runtime.LogosNocgocallbackForTest())
		return 0
	})
	defer runtime.SetLogosFastcallOffHookForTest(nil)
	var probe int
	runtime.LogosCgocallFastForTest(unsafe.Pointer(&probe), unsafe.Pointer(&probe))
	if observed.Load() {
		t.Fatal("off path entered delegated call with nocgocallback set")
	}
	if runtime.LogosNocgocallbackForTest() {
		t.Fatal("off path left nocgocallback set")
	}
}
