// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains code generation tests related to the handling of
// map types.

// ------------------- //
//     Access Const    //
// ------------------- //

// Direct use of constants in fast map access calls (Issue #19015).

func AccessInt1(m map[int]int) int {
	// amd64:"MOV[LQ] [$]5"
	return m[5]
}

func AccessInt2(m map[int]int) bool {
	// amd64:"MOV[LQ] [$]5"
	_, ok := m[5]
	return ok
}

func AccessString1(m map[string]int) int {
	// amd64:`.*"abc"`
	return m["abc"]
}

func AccessString2(m map[string]int) bool {
	// amd64:`.*"abc"`
	_, ok := m["abc"]
	return ok
}

func AccessStringIntArray2(m map[string][16]int, k string) bool {
	// amd64:-"MOVUPS"
	_, ok := m[k]
	return ok
}

type Struct struct {
	A, B, C, D, E, F, G, H, I, J int
}

func AccessStringStruct2(m map[string]Struct, k string) bool {
	// amd64:-"MOVUPS"
	_, ok := m[k]
	return ok
}

func AccessIntArrayLarge2(m map[int][512]int, k int) bool {
	// amd64:-"REP",-"MOVSQ"
	_, ok := m[k]
	return ok
}

// ------------------- //
//  String Conversion  //
// ------------------- //

func LookupStringConversionSimple(m map[string]int, bytes []byte) int {
	// amd64:-`.*runtime\.slicebytetostring\(`
	return m[string(bytes)]
}

func LookupStringConversionStructLit(m map[struct{ string }]int, bytes []byte) int {
	// amd64:-`.*runtime\.slicebytetostring\(`
	return m[struct{ string }{string(bytes)}]
}

func LookupStringConversionArrayLit(m map[[2]string]int, bytes []byte) int {
	// amd64:-`.*runtime\.slicebytetostring\(`
	return m[[2]string{string(bytes), string(bytes)}]
}

func LookupStringConversionNestedLit(m map[[1]struct{ s [1]string }]int, bytes []byte) int {
	// amd64:-`.*runtime\.slicebytetostring\(`
	return m[[1]struct{ s [1]string }{struct{ s [1]string }{s: [1]string{string(bytes)}}}]
}

func LookupStringConversionKeyedArrayLit(m map[[2]string]int, bytes []byte) int {
	// amd64:-`.*runtime\.slicebytetostring\(`
	return m[[2]string{0: string(bytes)}]
}

func LookupStringConversion1(m map[string]int, bytes []byte) int {
	// amd64:-`.*runtime\.slicebytetostring\(`
	s := string(bytes)
	return m[s]
}
func LookupStringConversion2(m *map[string]int, bytes []byte) int {
	// amd64:-`.*runtime\.slicebytetostring\(`
	s := string(bytes)
	return (*m)[s]
}
func LookupStringConversion3(m map[string]int, bytes []byte) (int, bool) {
	// amd64:-`.*runtime\.slicebytetostring\(`
	s := string(bytes)
	r, ok := m[s]
	return r, ok
}
func DeleteStringConversion(m map[string]int, bytes []byte) {
	// amd64:-`.*runtime\.slicebytetostring\(`
	s := string(bytes)
	delete(m, s)
}

// ------------------- //
//     Map Clear       //
// ------------------- //

// Optimization of map clear idiom (Issue #20138).

func MapClearReflexive(m map[int]int) {
	// amd64:`.*runtime\.mapclear`
	// amd64:-`.*runtime\.(mapiterinit|mapIterStart)`
	for k := range m {
		delete(m, k)
	}
}

func MapClearIndirect(m map[int]int) {
	s := struct{ m map[int]int }{m: m}
	// amd64:`.*runtime\.mapclear`
	// amd64:-`.*runtime\.(mapiterinit|mapIterStart)`
	for k := range s.m {
		delete(s.m, k)
	}
}

func MapClearPointer(m map[*byte]int) {
	// amd64:`.*runtime\.mapclear`
	// amd64:-`.*runtime\.(mapiterinit|mapIterStart)`
	for k := range m {
		delete(m, k)
	}
}

func MapClearNotReflexive(m map[float64]int) {
	// amd64:`.*runtime\.(mapiterinit|mapIterStart)`
	// amd64:-`.*runtime\.mapclear`
	for k := range m {
		delete(m, k)
	}
}

func MapClearInterface(m map[interface{}]int) {
	// amd64:`.*runtime\.(mapiterinit|mapIterStart)`
	// amd64:-`.*runtime\.mapclear`
	for k := range m {
		delete(m, k)
	}
}

func MapClearSideEffect(m map[int]int) int {
	k := 0
	// amd64:`.*runtime\.(mapiterinit|mapIterStart)`
	// amd64:-`.*runtime\.mapclear`
	for k = range m {
		delete(m, k)
	}
	return k
}

func MapLiteralSizing(x int) (map[int]int, map[int]int) {
	// This is tested for internal/abi/maps.go:MapBucketCountBits={3,4,5}
	// amd64:"MOVL [$]33,"
	m := map[int]int{
		0:  0,
		1:  1,
		2:  2,
		3:  3,
		4:  4,
		5:  5,
		6:  6,
		7:  7,
		8:  8,
		9:  9,
		10: 10,
		11: 11,
		12: 12,
		13: 13,
		14: 14,
		15: 15,
		16: 16,
		17: 17,
		18: 18,
		19: 19,
		20: 20,
		21: 21,
		22: 22,
		23: 23,
		24: 24,
		25: 25,
		26: 26,
		27: 27,
		28: 28,
		29: 29,
		30: 30,
		31: 32,
		32: 32,
	}
	// amd64:"MOVL [$]33,"
	n := map[int]int{
		0:  0,
		1:  1,
		2:  2,
		3:  3,
		4:  4,
		5:  5,
		6:  6,
		7:  7,
		8:  8,
		9:  9,
		10: 10,
		11: 11,
		12: 12,
		13: 13,
		14: 14,
		15: 15,
		16: 16,
		17: 17,
		18: 18,
		19: 19,
		20: 20,
		21: 21,
		22: 22,
		23: 23,
		24: 24,
		25: 25,
		26: 26,
		27: 27,
		28: 28,
		29: 29,
		30: 30,
		31: 32,
		32: 32,
	}
	return m, n
}
