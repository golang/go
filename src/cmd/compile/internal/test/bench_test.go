// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"fmt"
	"testing"
)

var globl int64
var globl32 int32

func BenchmarkLoadAdd(b *testing.B) {
	x := make([]int64, 1024)
	y := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s ^= x[i] + y[i]
		}
		globl = s
	}
}

// Added for ppc64 extswsli on power9
func BenchmarkExtShift(b *testing.B) {
	x := make([]int32, 1024)
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s ^= int64(x[i]+32) * 8
		}
		globl = s
	}
}

func BenchmarkModify(b *testing.B) {
	a := make([]int64, 1024)
	v := globl
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] += v
		}
	}
}

func BenchmarkMullImm(b *testing.B) {
	x := make([]int32, 1024)
	for i := 0; i < b.N; i++ {
		var s int32
		for i := range x {
			s += x[i] * 100
		}
		globl32 = s
	}
}

func BenchmarkConstModify(b *testing.B) {
	a := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] += 3
		}
	}
}

func BenchmarkBitSet(b *testing.B) {
	const N = 64 * 8
	a := make([]uint64, N/64)
	for i := 0; i < b.N; i++ {
		for j := uint64(0); j < N; j++ {
			a[j/64] |= 1 << (j % 64)
		}
	}
}

func BenchmarkBitClear(b *testing.B) {
	const N = 64 * 8
	a := make([]uint64, N/64)
	for i := 0; i < b.N; i++ {
		for j := uint64(0); j < N; j++ {
			a[j/64] &^= 1 << (j % 64)
		}
	}
}

func BenchmarkBitToggle(b *testing.B) {
	const N = 64 * 8
	a := make([]uint64, N/64)
	for i := 0; i < b.N; i++ {
		for j := uint64(0); j < N; j++ {
			a[j/64] ^= 1 << (j % 64)
		}
	}
}

func BenchmarkBitSetConst(b *testing.B) {
	const N = 64
	a := make([]uint64, N)
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] |= 1 << 37
		}
	}
}

func BenchmarkBitClearConst(b *testing.B) {
	const N = 64
	a := make([]uint64, N)
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] &^= 1 << 37
		}
	}
}

func BenchmarkBitToggleConst(b *testing.B) {
	const N = 64
	a := make([]uint64, N)
	for i := 0; i < b.N; i++ {
		for j := range a {
			a[j] ^= 1 << 37
		}
	}
}

func BenchmarkMulNeg(b *testing.B) {
	x := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s = (-x[i]) * 11
		}
		globl = s
	}
}

func BenchmarkMul2Neg(b *testing.B) {
	x := make([]int64, 1024)
	y := make([]int64, 1024)
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s = (-x[i]) * (-y[i])
		}
		globl = s
	}
}

func BenchmarkSimplifyNegMul(b *testing.B) {
	x := make([]int64, 1024)
	y := make([]int64, 1024)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s = -(-x[i] * y[i])
		}
		globl = s
	}
}

func BenchmarkSimplifyNegDiv(b *testing.B) {
	x := make([]int64, 1024)
	y := make([]int64, 1024)
	for i := range y {
		y[i] = 42
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var s int64
		for i := range x {
			s = -(-x[i] / y[i])
		}
		globl = s
	}
}

var globbool bool

// --- Struct literal assignment benchmarks ---

// benchStruct is a large-ish struct (8 fields, 104 bytes on 64-bit)
// used to benchmark struct literal assignment codegen.
type benchStruct struct {
	A int
	B string
	C float64
	D int64
	E bool
	F string
	G int
	H string
}

var benchStructSink benchStruct

// Nested/embedded struct types for benchmarking.
type benchInner struct {
	X int
	Y string
}

type benchNested struct {
	I benchInner
	Z float64
	W string
}

type benchEmbedded struct {
	benchInner
	Z float64
	W string
}


// --- noinline helper functions for benchmarks ---

//go:noinline
func slAssignLiteral(s []benchStruct, idx int, a int, b string, c float64, d int64, e bool, f string, g int, h string) {
	s[idx] = benchStruct{A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h}
}

//go:noinline
func slAssignFieldByField(s []benchStruct, idx int, a int, b string, c float64, d int64, e bool, f string, g int, h string) {
	s[idx].A = a
	s[idx].B = b
	s[idx].C = c
	s[idx].D = d
	s[idx].E = e
	s[idx].F = f
	s[idx].G = g
	s[idx].H = h
}

//go:noinline
func slAssignGlobal(a int, b string, c float64, d int64, e bool, f string, g int, h string) {
	benchStructSink = benchStruct{A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h}
}

//go:noinline
func slAssignPtr(p *benchStruct, a int, b string, c float64, d int64, e bool, f string, g int, h string) {
	*p = benchStruct{A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h}
}

//go:noinline
func slAssignNested(s []benchNested, idx int, x int, y string, z float64, w string) {
	s[idx] = benchNested{I: benchInner{X: x, Y: y}, Z: z, W: w}
}

//go:noinline
func slAssignEmbedded(s []benchEmbedded, idx int, x int, y string, z float64, w string) {
	s[idx] = benchEmbedded{benchInner: benchInner{X: x, Y: y}, Z: z, W: w}
}

//go:noinline
func slGetInt() int { return 42 }

//go:noinline
func slGetString() string { return "hello" }

//go:noinline
func slAssignFuncCall(s []benchStruct, idx int, c float64, d int64, e bool, g int) {
	s[idx] = benchStruct{
		A: slGetInt(), B: slGetString(), C: c, D: d,
		E: e, F: slGetString(), G: g, H: slGetString(),
	}
}

var slGlobalInt int = 99

//go:noinline
func slAssignAlias(s []benchStruct, idx int, b string, c float64, d int64) {
	// RHS reads a global (slGlobalInt) — may alias with destination,
	// so the compiler must use a stack temporary (not optimized).
	s[idx] = benchStruct{A: slGlobalInt, B: b, C: c, D: d, E: true, F: b, G: 7, H: b}
}

//go:noinline
func slAssignPartial(s []benchStruct, idx int, a int) {
	// Only one field set — zero + one store.
	s[idx] = benchStruct{A: a}
}

//go:noinline
func slAssignPartialHalf(s []benchStruct, idx int, a int, b string, c float64, d int64) {
	// Half the fields set — zero + four stores.
	s[idx] = benchStruct{A: a, B: b, C: c, D: d}
}

//go:noinline
func slAssignZeroValue(s []benchStruct, idx int) {
	// Zero-value literal — just zero the destination.
	s[idx] = benchStruct{}
}

//go:noinline
func slAssignLHSAlias(s []benchStruct, idx int, a int, c float64, d int64) {
	// RHS reads directly from the destination slice element.
	// exprSafeForDirectStore rejects these (field access through slice index),
	// so the compiler falls back to the stack temporary path.
	s[idx] = benchStruct{A: a, B: s[idx].B, C: c, D: d, E: s[idx].E, F: s[idx].F, G: s[idx].G, H: s[idx].H}
}

//go:noinline
func slAssignNestedPartial(s []benchNested, idx int, x int, z float64, w string) {
	// Outer fully set, inner benchInner partial (only X, not Y).
	// Optimization fires but must zero destination first (compLitAllFieldsSet).
	s[idx] = benchNested{I: benchInner{X: x}, Z: z, W: w}
}

//go:noinline
func slAssignRHSGrowsSlice(s *[]benchStruct, idx int, b string, c float64, d int64) {
	// RHS lambda grows the slice via append.
	// The order pass extracts the call to an autotemp, so the
	// optimization fires (direct stores, no stack temporary).
	(*s)[idx] = benchStruct{A: func() int { *s = append(*s, benchStruct{}); return 42 }(), B: b, C: c, D: d, E: true, F: b, G: 7, H: b}
}

// BenchmarkStructLitAssign measures the performance impact of struct literal
// assignment codegen. When the compiler decomposes struct literals into direct
// field stores (avoiding DUFFZERO+DUFFCOPY via a stack temporary), this should
// match or beat field-by-field assignment.
func BenchmarkStructLitAssign(b *testing.B) {
	s := make([]benchStruct, 64)
	str1 := fmt.Sprintf("%s", "hello")
	str2 := fmt.Sprintf("%s", "world")
	str3 := fmt.Sprintf("%s", "test!")

	b.Run("SliceLiteral", func(b *testing.B) {
		for b.Loop() {
			slAssignLiteral(s, 0, 42, str1, 3.14, 100, true, str2, 7, str3)
		}
	})
	b.Run("SliceFieldByField", func(b *testing.B) {
		for b.Loop() {
			slAssignFieldByField(s, 0, 42, str1, 3.14, 100, true, str2, 7, str3)
		}
	})
	b.Run("Global", func(b *testing.B) {
		for b.Loop() {
			slAssignGlobal(42, str1, 3.14, 100, true, str2, 7, str3)
		}
	})
	b.Run("PtrDeref", func(b *testing.B) {
		p := &s[0]
		for b.Loop() {
			slAssignPtr(p, 42, str1, 3.14, 100, true, str2, 7, str3)
		}
	})
	b.Run("Nested", func(b *testing.B) {
		ns := make([]benchNested, 64)
		for b.Loop() {
			slAssignNested(ns, 0, 42, str1, 3.14, str2)
		}
	})
	b.Run("Embedded", func(b *testing.B) {
		es := make([]benchEmbedded, 64)
		for b.Loop() {
			slAssignEmbedded(es, 0, 42, str1, 3.14, str2)
		}
	})
	b.Run("FuncCallRHS", func(b *testing.B) {
		for b.Loop() {
			slAssignFuncCall(s, 0, 3.14, 100, true, 7)
		}
	})
	b.Run("AliasRHS", func(b *testing.B) {
		for b.Loop() {
			slAssignAlias(s, 0, str1, 3.14, 100)
		}
	})
	b.Run("PartialOne", func(b *testing.B) {
		for b.Loop() {
			slAssignPartial(s, 0, 42)
		}
	})
	b.Run("PartialHalf", func(b *testing.B) {
		for b.Loop() {
			slAssignPartialHalf(s, 0, 42, str1, 3.14, 100)
		}
	})
	b.Run("ZeroValue", func(b *testing.B) {
		for b.Loop() {
			slAssignZeroValue(s, 0)
		}
	})
	b.Run("LHSAlias", func(b *testing.B) {
		for b.Loop() {
			slAssignLHSAlias(s, 0, 42, 3.14, 100)
		}
	})
	b.Run("NestedPartial", func(b *testing.B) {
		ns := make([]benchNested, 64)
		for b.Loop() {
			slAssignNestedPartial(ns, 0, 42, 3.14, str1)
		}
	})
	b.Run("RHSGrowsSlice", func(b *testing.B) {
		ms := make([]benchStruct, 1, 1024)
		for b.Loop() {
			ms = ms[:1]
			slAssignRHSGrowsSlice(&ms, 0, str1, 3.14, 100)
		}
	})
}

// containsRight compares strs[i] == str (slice element on left).
//
//go:noinline
func containsRight(strs []string, str string) bool {
	for i := range strs {
		if strs[i] == str {
			return true
		}
	}
	return false
}

// containsLeft compares str == strs[i] (parameter on left).
//
//go:noinline
func containsLeft(strs []string, str string) bool {
	for i := range strs {
		if str == strs[i] {
			return true
		}
	}
	return false
}

// BenchmarkStringEqParamOrder tests that the operand order of string
// equality comparisons does not affect performance. See issue #74471.
func BenchmarkStringEqParamOrder(b *testing.B) {
	strs := []string{
		"12312312", "abcsdsfw", "abcdefgh", "qereqwre",
		"gwertdsg", "hellowod", "iamgroot", "theiswer",
		"dg323sdf", "gadsewwe", "g42dg4t3", "4hre2323",
		"23eg4325", "13234234", "32dfgsdg", "23fgre34",
		"43rerrer", "hh2s2443", "hhwesded", "1swdf23d",
		"gwcdrwer", "bfgwertd", "badgwe3g", "lhoejyop",
	}
	target := fmt.Sprintf("%s", "notfound")
	b.Run("ParamRight", func(b *testing.B) {
		for b.Loop() {
			globbool = containsRight(strs, target)
		}
	})
	b.Run("ParamLeft", func(b *testing.B) {
		for b.Loop() {
			globbool = containsLeft(strs, target)
		}
	})
}
