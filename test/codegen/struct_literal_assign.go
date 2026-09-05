// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that fully-initialized struct literal assignments do not generate
// unnecessary zeroing and copying via a stack temporary. When every field is
// set in the literal, the compiler should write fields directly to the
// destination without zeroing or copying an intermediate.

package codegen

type BigStruct struct {
	A int
	B string
	C float64
	D int64
	E bool
	F string
	G int
	H string
}

var bsSink BigStruct
var bsSlice = make([]BigStruct, 16)
var bsIdx int

// --- Basic LHS variants (optimized) ---

// Slice index LHS
func AssignFullLiteralToSlice(a int, b string, c float64, d int64, e bool, f string, g int, h string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	bsSlice[bsIdx] = BigStruct{
		A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h,
	}
}

// Global variable LHS
func AssignFullLiteralToGlobal(a int, b string, c float64, d int64, e bool, f string, g int, h string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	bsSink = BigStruct{
		A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h,
	}
}

// Pointer dereference LHS
func AssignFullLiteralToDeref(p *BigStruct, a int, b string, c float64, d int64, e bool, f string, g int, h string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	*p = BigStruct{
		A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h,
	}
}

type OuterNamed struct {
	Inner BigStruct
	X     int
}

// Struct field through pointer LHS
func AssignFullLiteralToFieldViaPtr(o *OuterNamed, a int, b string, c float64, d int64, e bool, f string, g int, h string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	o.Inner = BigStruct{
		A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h,
	}
}

// Local variable LHS — not affected by the addressable-target optimization
// (locals go through the existing isSimpleName path in oaslit).
func AssignFullLiteralToLocal(a int, b string, c float64, d int64, e bool, f string, g int, h string) BigStruct {
	x := BigStruct{
		A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h,
	}
	return x
}

// --- Nested struct literal ---

type InnerSmall struct {
	X int
	Y string
}

type OuterWithInner struct {
	I InnerSmall
	Z float64
	W string
}

var owSlice = make([]OuterWithInner, 16)

// Nested struct literal with all fields set
func AssignNestedLiteralToSlice(x int, y string, z float64, w string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	owSlice[0] = OuterWithInner{
		I: InnerSmall{X: x, Y: y},
		Z: z,
		W: w,
	}
}

// Nested struct literal to global
func AssignNestedLiteralToGlobal(x int, y string, z float64, w string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	bsNested = OuterWithInner{
		I: InnerSmall{X: x, Y: y},
		Z: z,
		W: w,
	}
}

var bsNested OuterWithInner

// --- Anonymous/embedded fields ---

type EmbedBase struct {
	X int
	Y string
}

type WithEmbed struct {
	EmbedBase
	Z float64
	W string
}

var weSlice = make([]WithEmbed, 16)

// Anonymous embedded struct, all fields set
func AssignEmbeddedLiteralToSlice(x int, y string, z float64, w string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	weSlice[0] = WithEmbed{
		EmbedBase: EmbedBase{X: x, Y: y},
		Z:         z,
		W:         w,
	}
}

// Multiple levels of embedding
type DeepBase struct {
	A int
	B string
}

type MidEmbed struct {
	DeepBase
	C float64
}

type DeepEmbed struct {
	MidEmbed
	D string
}

var deSlice = make([]DeepEmbed, 16)

func AssignDeepEmbeddedLiteralToSlice(a int, b string, c float64, d string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	deSlice[0] = DeepEmbed{
		MidEmbed: MidEmbed{
			DeepBase: DeepBase{A: a, B: b},
			C:        c,
		},
		D: d,
	}
}

// --- Function call results in RHS ---

//go:noinline
func bsGetInt() int { return 42 }

//go:noinline
func bsGetString() string { return "hello" }

// Function call results are placed into autotemps by the order pass,
// so they should be safe for the optimization.
func AssignFuncCallResultToSlice(c float64, d int64, e bool, g int) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	bsSlice[0] = BigStruct{
		A: bsGetInt(), B: bsGetString(), C: c, D: d,
		E: e, F: bsGetString(), G: g, H: bsGetString(),
	}
}

// --- Type conversion in RHS ---

func AssignWithConvToSlice(a int32, b string, c float64, d int64, e bool, f string, g int, h string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	bsSlice[0] = BigStruct{
		A: int(a), B: b, C: c, D: d, E: e, F: f, G: g, H: h,
	}
}

// --- RHS function call that grows the LHS slice ---

// Function call in RHS that grows the destination slice via append.
// The order pass extracts the call to an autotemp before the assignment,
// so the optimization can fire safely.
func AssignRHSGrowsSlice(s []BigStruct, b string, c float64, d int64, e bool, f string, g int, h string) {
	// 386:-"DUFFZERO"
	// 386:-"DUFFCOPY"
	// 386:-"runtime.wbMove"
	s[0] = BigStruct{
		A: func() int { s = append(s, BigStruct{}); return 99 }(),
		B: b, C: c, D: d, E: e, F: f, G: g, H: h,
	}
}

// --- RHS aliases LHS (must NOT be optimized) ---

func AssignRHSAliasesLHS(a int, c float64, d int64, e bool, g int) {
	// RHS reads from the same slice element being written.
	// The optimization must NOT fire — exprSafeForDirectStore rejects
	// slice index field accesses. The compiler falls back to the
	// stack temporary path to avoid corrupting RHS values.
	bsSlice[bsIdx] = BigStruct{
		A: a, B: bsSlice[bsIdx].B, C: c, D: d,
		E: e, F: bsSlice[bsIdx].F, G: g, H: bsSlice[bsIdx].H,
	}
}
