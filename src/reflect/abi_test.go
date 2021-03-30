// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build goexperiment.regabi
//go:build goexperiment.regabi

package reflect_test

import (
	"internal/abi"
	"math/rand"
	"reflect"
	"runtime"
	"testing"
	"testing/quick"
)

func TestReflectValueCallABI(t *testing.T) {
	// Enable register-based reflect.Call and ensure we don't
	// use potentially incorrect cached versions by clearing
	// the cache before we start and after we're done.
	var oldRegs struct {
		ints, floats int
		floatSize    uintptr
	}
	oldRegs.ints = *reflect.IntArgRegs
	oldRegs.floats = *reflect.FloatArgRegs
	oldRegs.floatSize = *reflect.FloatRegSize
	*reflect.IntArgRegs = abi.IntArgRegs
	*reflect.FloatArgRegs = abi.FloatArgRegs
	*reflect.FloatRegSize = uintptr(abi.EffectiveFloatRegSize)
	reflect.ClearLayoutCache()
	defer func() {
		*reflect.IntArgRegs = oldRegs.ints
		*reflect.FloatArgRegs = oldRegs.floats
		*reflect.FloatRegSize = oldRegs.floatSize
		reflect.ClearLayoutCache()
	}()

	// Execute the functions defined below which all have the
	// same form and perform the same function: pass all arguments
	// to return values. The purpose is to test the call boundary
	// and make sure it works.
	r := rand.New(rand.NewSource(genValueRandSeed))
	for _, fn := range []interface{}{
		passNone,
		passInt,
		passInt8,
		passInt16,
		passInt32,
		passInt64,
		passUint,
		passUint8,
		passUint16,
		passUint32,
		passUint64,
		passFloat32,
		passFloat64,
		passComplex64,
		passComplex128,
		passManyInt,
		passManyFloat64,
		passArray1,
		passArray,
		passArray1Mix,
		passString,
		// TODO(mknyszek): Test passing interface values.
		passSlice,
		passPointer,
		passStruct1,
		passStruct2,
		passStruct3,
		passStruct4,
		passStruct5,
		passStruct6,
		passStruct7,
		passStruct8,
		passStruct9,
		passStruct10,
		// TODO(mknyszek): Test passing unsafe.Pointer values.
		// TODO(mknyszek): Test passing chan values.
		passStruct11,
		passStruct12,
		passStruct13,
		pass2Struct1,
		passEmptyStruct,
		passStruct10AndSmall,
	} {
		fn := reflect.ValueOf(fn)
		t.Run(runtime.FuncForPC(fn.Pointer()).Name(), func(t *testing.T) {
			typ := fn.Type()
			if typ.Kind() != reflect.Func {
				t.Fatalf("test case is not a function, has type: %s", typ.String())
			}
			if typ.NumIn() != typ.NumOut() {
				t.Fatalf("test case has different number of inputs and outputs: %d in, %d out", typ.NumIn(), typ.NumOut())
			}
			var args []reflect.Value
			for i := 0; i < typ.NumIn(); i++ {
				args = append(args, genValue(t, typ.In(i), r))
			}
			results := fn.Call(args)
			for i := range results {
				x, y := args[i].Interface(), results[i].Interface()
				if reflect.DeepEqual(x, y) {
					continue
				}
				t.Errorf("arg and result %d differ: got %+v, want %+v", i, x, y)
			}
		})
	}
}

// Functions for testing reflect.Value.Call.

//go:registerparams
//go:noinline
func passNone() {}

//go:registerparams
//go:noinline
func passInt(a int) int {
	return a
}

//go:registerparams
//go:noinline
func passInt8(a int8) int8 {
	return a
}

//go:registerparams
//go:noinline
func passInt16(a int16) int16 {
	return a
}

//go:registerparams
//go:noinline
func passInt32(a int32) int32 {
	return a
}

//go:registerparams
//go:noinline
func passInt64(a int64) int64 {
	return a
}

//go:registerparams
//go:noinline
func passUint(a uint) uint {
	return a
}

//go:registerparams
//go:noinline
func passUint8(a uint8) uint8 {
	return a
}

//go:registerparams
//go:noinline
func passUint16(a uint16) uint16 {
	return a
}

//go:registerparams
//go:noinline
func passUint32(a uint32) uint32 {
	return a
}

//go:registerparams
//go:noinline
func passUint64(a uint64) uint64 {
	return a
}

//go:registerparams
//go:noinline
func passFloat32(a float32) float32 {
	return a
}

//go:registerparams
//go:noinline
func passFloat64(a float64) float64 {
	return a
}

//go:registerparams
//go:noinline
func passComplex64(a complex64) complex64 {
	return a
}

//go:registerparams
//go:noinline
func passComplex128(a complex128) complex128 {
	return a
}

//go:registerparams
//go:noinline
func passArray1(a [1]uint32) [1]uint32 {
	return a
}

//go:registerparams
//go:noinline
func passArray(a [2]uintptr) [2]uintptr {
	return a
}

//go:registerparams
//go:noinline
func passArray1Mix(a int, b [1]uint32, c float64) (int, [1]uint32, float64) {
	return a, b, c
}

//go:registerparams
//go:noinline
func passString(a string) string {
	return a
}

//go:registerparams
//go:noinline
func passSlice(a []byte) []byte {
	return a
}

//go:registerparams
//go:noinline
func passPointer(a *byte) *byte {
	return a
}

//go:registerparams
//go:noinline
func passManyInt(a, b, c, d, e, f, g, h, i, j int) (int, int, int, int, int, int, int, int, int, int) {
	return a, b, c, d, e, f, g, h, i, j
}

//go:registerparams
//go:noinline
func passManyFloat64(a, b, c, d, e, f, g, h, i, j, l, m, n, o, p, q, r, s, t float64) (float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64) {
	return a, b, c, d, e, f, g, h, i, j, l, m, n, o, p, q, r, s, t
}

//go:registerparams
//go:noinline
func passStruct1(a Struct1) Struct1 {
	return a
}

//go:registerparams
//go:noinline
func passStruct2(a Struct2) Struct2 {
	return a
}

//go:registerparams
//go:noinline
func passStruct3(a Struct3) Struct3 {
	return a
}

//go:registerparams
//go:noinline
func passStruct4(a Struct4) Struct4 {
	return a
}

//go:registerparams
//go:noinline
func passStruct5(a Struct5) Struct5 {
	return a
}

//go:registerparams
//go:noinline
func passStruct6(a Struct6) Struct6 {
	return a
}

//go:registerparams
//go:noinline
func passStruct7(a Struct7) Struct7 {
	return a
}

//go:registerparams
//go:noinline
func passStruct8(a Struct8) Struct8 {
	return a
}

//go:registerparams
//go:noinline
func passStruct9(a Struct9) Struct9 {
	return a
}

//go:registerparams
//go:noinline
func passStruct10(a Struct10) Struct10 {
	return a
}

//go:registerparams
//go:noinline
func passStruct11(a Struct11) Struct11 {
	return a
}

//go:registerparams
//go:noinline
func passStruct12(a Struct12) Struct12 {
	return a
}

//go:registerparams
//go:noinline
func passStruct13(a Struct13) Struct13 {
	return a
}

//go:registerparams
//go:noinline
func pass2Struct1(a, b Struct1) (x, y Struct1) {
	return a, b
}

//go:registerparams
//go:noinline
func passEmptyStruct(a int, b struct{}, c float64) (int, struct{}, float64) {
	return a, b, c
}

// This test case forces a large argument to the stack followed by more
// in-register arguments.
//go:registerparams
//go:noinline
func passStruct10AndSmall(a Struct10, b byte, c uint) (Struct10, byte, uint) {
	return a, b, c
}

// Struct1 is a simple integer-only aggregate struct.
type Struct1 struct {
	A, B, C uint
}

// Struct2 is Struct1 but with an array-typed field that will
// force it to get passed on the stack.
type Struct2 struct {
	A, B, C uint
	D       [2]uint32
}

// Struct3 is Struct2 but with an anonymous array-typed field.
// This should act identically to Struct2.
type Struct3 struct {
	A, B, C uint
	D       [2]uint32
}

// Struct4 has byte-length fields that should
// each use up a whole registers.
type Struct4 struct {
	A, B int8
	C, D uint8
	E    bool
}

// Struct5 is a relatively large struct
// with both integer and floating point values.
type Struct5 struct {
	A             uint16
	B             int16
	C, D          uint32
	E             int32
	F, G, H, I, J float32
}

// Struct6 has a nested struct.
type Struct6 struct {
	Struct1
}

// Struct7 is a struct with a nested array-typed field
// that cannot be passed in registers as a result.
type Struct7 struct {
	Struct1
	Struct2
}

// Struct8 is large aggregate struct type that may be
// passed in registers.
type Struct8 struct {
	Struct5
	Struct1
}

// Struct9 is a type that has an array type nested
// 2 layers deep, and as a result needs to be passed
// on the stack.
type Struct9 struct {
	Struct1
	Struct7
}

// Struct10 is a struct type that is too large to be
// passed in registers.
type Struct10 struct {
	Struct5
	Struct8
}

// Struct11 is a struct type that has several reference
// types in it.
type Struct11 struct {
	X map[string]int
}

// Struct12 has Struct11 embedded into it to test more
// paths.
type Struct12 struct {
	A int
	Struct11
}

// Struct13 tests an empty field.
type Struct13 struct {
	A int
	X struct{}
	B int
}

const genValueRandSeed = 0

// genValue generates a pseudorandom reflect.Value with type t.
// The reflect.Value produced by this function is always the same
// for the same type.
func genValue(t *testing.T, typ reflect.Type, r *rand.Rand) reflect.Value {
	// Re-seed and reset the PRNG because we want each value with the
	// same type to be the same random value.
	r.Seed(genValueRandSeed)
	v, ok := quick.Value(typ, r)
	if !ok {
		t.Fatal("failed to generate value")
	}
	return v
}
