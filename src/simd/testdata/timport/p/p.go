// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package p provides SIMD-initialized package variables and types with
// SIMD-dependent method sets for main2 (which does not directly import "simd").
package p

import "simd"

// Non-SIMD-typed exported package variable initialized via SIMD, plus clean
// accessors and a pointer taken during package init to verify single-variable
// storage and address identity across package boundaries.
var N = simd.Int8s{}.Len()

func GetN() int { return N }

func SetN(x int) { N = x }

var PN = &N

// Cross-package types whose SIMD dependence is only in their method sets
// (no SIMD struct fields), used by main2 to verify cross-package method-set
// SIMD dependence detection in package-level variable declarations.
type HasSimdMethod struct {
	X int
}

func (h HasSimdMethod) AddLen(v simd.Int8s) int {
	return h.X + v.Len()
}

func MakeHasSimdMethod() HasSimdMethod {
	return HasSimdMethod{X: 42}
}

type SimdMethodIface interface {
	Vec() simd.Int8s
}

type simdMethodImpl struct{}

func (simdMethodImpl) Vec() simd.Int8s { return simd.Int8s{} }

func MakeSimdMethodIface() SimdMethodIface {
	return simdMethodImpl{}
}
