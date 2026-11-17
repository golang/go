// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package testdata_test

import (
	// Alias "fmt" as bridge and "simd" as s to verify midway's injected
	// @bridge/@simd imports do not collide with user import aliases.
	bridge "fmt"
	s "simd"
	"testing"
)

// Package variable named simd must not collide with midway's injected @simd import.
var simd = 42

// Verify that @simdinit proxy rewriting for a package-variable initializer
// uses @simd/@bridge rather than colliding with user `bridge "fmt"` imports,
// package-level `simd`, or closure-local `bridge` bindings.
var aliasInit = len(bridge.Sprint(s.Int8s{}.Len())) + simd + func() int {
	bridge := 10
	return bridge + s.Int8s{}.Len()
}()

// Parameter named simd must not shadow midway's @simd references inside the
// dispatcher or specialized bodies.
func shadowSimdParam(simd int) int {
	return simd + s.Int8s{}.Len()
}

// Local variable named bridge must not shadow midway's @bridge references.
func shadowBridgeLocal() int {
	bridge := 5
	return bridge + s.BroadcastInt8s(1).Len()
}

// TestAliasedAndShadowedImports validates that midway isolates its injected
// "simd" and "simd/internal/bridge" imports under compiler-private aliases
// (@simd and @bridge) so user code can alias "simd", alias another package as
// bridge, and shadow simd/bridge at package, parameter, and local scopes
// without colliding with rewritten SIMD references.
func TestAliasedAndShadowedImports(t *testing.T) {
	n := (s.Int8s{}).Len()
	if got := len(bridge.Sprint(n)) + simd; got != len(bridge.Sprint(n))+42 {
		t.Errorf("unexpected formatted sum %d", got)
	}
	if got := shadowSimdParam(7); got != 7+n {
		t.Errorf("shadowSimdParam(7) = %d, want %d", got, 7+n)
	}
	if got := shadowBridgeLocal(); got != 5+n {
		t.Errorf("shadowBridgeLocal() = %d, want %d", got, 5+n)
	}
}

// TestAliasedAndShadowedPackageVarInit validates that midway's init-proxy
// rewriting of a SIMD-dependent package-variable initializer (aliasInit)
// preserves user import aliases (bridge "fmt", s "simd") and shadowed simd and
// bridge identifiers.
func TestAliasedAndShadowedPackageVarInit(t *testing.T) {
	n := (s.Int8s{}).Len()
	wantInit := len(bridge.Sprint(n)) + 42 + 10 + n
	if aliasInit != wantInit {
		t.Errorf("aliasInit = %d, want %d", aliasInit, wantInit)
	}
}
