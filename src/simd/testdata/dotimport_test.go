// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package testdata_test

import (
	. "simd"
	"testing"
)

// Package variables named simd and bridge must not collide with midway's
// injected @simd/@bridge imports when "simd" is dot-imported.
var simd = 42
var bridge = 99

// Package-level SIMD variables and initializers using bare dot-imported names
// (BroadcastInt8s, LoadInt8s, multi-return LoadInt8sPart, method expression
// Int8s.Len, and an init-proxied int referencing shadowing vars simd/bridge).
var dotV1 = BroadcastInt8s(7)
var dotV2 = LoadInt8s(make([]int8, 64))
var dotV3, dotN3 = LoadInt8sPart([]int8{1, 2, 3})
var dotLenExpr = Int8s.Len
var dotTotal = dotV1.Len() + dotV2.Len() + dotV3.Len() + dotN3 + dotLenExpr(dotV1) + simd + bridge

// dotHelper exercises bare dot-imported SIMD functions (BroadcastInt8s,
// LoadInt8s, LoadInt8sPart) and method expressions (Int8s.Len) alongside
// parameters named simd and bridge that shadow the unprefixed package names.
func dotHelper(simd, bridge int) int {
	v1 := BroadcastInt8s(7)
	v2 := LoadInt8s(make([]int8, 64))
	v3, n3 := LoadInt8sPart([]int8{1, 2, 3})
	lenExpr := Int8s.Len
	return v1.Len() + v2.Len() + v3.Len() + n3 + lenExpr(v1) + simd + bridge
}

// TestDotImportSimd validates that `import . "simd"` rewrites bare SIMD type
// names, Broadcast/Load calls, and method expressions to @bridge (rather than
// suffixing them into nonexistent @simd* identifiers) while keeping the dot
// import marked used and allowing user identifiers named simd and bridge.
func TestDotImportSimd(t *testing.T) {
	n := (Int8s{}).Len()
	want := 4*n + 3 + 10 + 20
	if got := dotHelper(10, 20); got != want {
		t.Errorf("dotHelper(10, 20) = %d, want %d", got, want)
	}
	if simd != 42 || bridge != 99 {
		t.Errorf("unexpected package vars: simd=%d bridge=%d", simd, bridge)
	}
}

// TestDotImportPackageVarInit validates package-level variable rewriting
// (per-size splitting, N:1 multi-return LoadInt8sPart, method-expression
// initializers, and init proxies) when "simd" is dot-imported alongside
// shadowing package variables named simd and bridge.
func TestDotImportPackageVarInit(t *testing.T) {
	n := (Int8s{}).Len()
	wantTotal := 4*n + 3 + 42 + 99
	if dotTotal != wantTotal {
		t.Errorf("dotTotal = %d, want %d", dotTotal, wantTotal)
	}
}
