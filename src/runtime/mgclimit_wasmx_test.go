// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasm || wasm32

package runtime_test

import (
	"runtime"
	"runtime/debug"
	"testing"
)

func TestGCAssistUpdatesLimiter(t *testing.T) {
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.GC()
	if !runtime.GCAssistUpdatesLimiter() {
		t.Fatal("GC assist left the CPU limiter enabled with a stale timestamp")
	}
}
