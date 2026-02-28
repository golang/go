// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.regabiargs

package reflect_test

import (
	"internal/abi"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"testing"
	"testing/quick"
)

// As of early May 2021 this is no longer necessary for amd64,
// but it remains in case this is needed for the next register abi port.
// TODO (1.18) If enabling register ABI on additional architectures turns out not to need this, remove it.
type MagicLastTypeNameForTestingRegisterABI struct{}

func TestMethodValueCallABI(t *testing.T) {
	// Enable register-based reflect.Call and ensure we don't
	// use potentially incorrect cached versions by clearing
	// the cache before we start and after we're done.
	defer reflect.SetArgRegs(reflect.SetArgRegs(abi.IntArgRegs, abi.FloatArgRegs, abi.EffectiveFloatRegSize))

	// This test is simple. Calling a method value involves
	// pretty much just plumbing whatever arguments in whichever
	// location through to reflectcall. They're already set up
	// for us, so there isn't a whole lot to do. Let's just
	// make sure that we can pass register and stack arguments
	// through. The exact combination is not super important.
	makeMethodValue := func(method string) (*StructWithMethods, any) {
		s := new(StructWithMethods)
		v := reflect.ValueOf(s).MethodByName(method)
		return s, v.Interface()
	}

	a0 := StructFewRegs{
		10, 11, 12, 13,
		20.0, 21.0, 22.0, 23.0,
	}
	a1 := [4]uint64{100, 101, 102, 103}
	a2 := StructFillRegs{
		1, 2, 3, 4, 5, 6, 7, 8, 9,
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
	}

	s, i := makeMethodValue("AllRegsCall")
	f0 := i.(func(StructFewRegs, MagicLastTypeNameForTestingRegisterABI) StructFewRegs)
	r0 := f0(a0, MagicLastTypeNameForTestingRegisterABI{})
	if r0 != a0 {
		t.Errorf("bad method value call: got %#v, want %#v", r0, a0)
	}
	if s.Value != 1 {
		t.Errorf("bad method value call: failed to set s.Value: got %d, want %d", s.Value, 1)
	}

	s, i = makeMethodValue("RegsAndStackCall")
	f1 := i.(func(StructFewRegs, [4]uint64, MagicLastTypeNameForTestingRegisterABI) (StructFewRegs, [4]uint64))
	r0, r1 := f1(a0, a1, MagicLastTypeNameForTestingRegisterABI{})
	if r0 != a0 {
		t.Errorf("bad method value call: got %#v, want %#v", r0, a0)
	}
	if r1 != a1 {
		t.Errorf("bad method value call: got %#v, want %#v", r1, a1)
	}
	if s.Value != 2 {
		t.Errorf("bad method value call: failed to set s.Value: got %d, want %d", s.Value, 2)
	}

	s, i = makeMethodValue("SpillStructCall")
	f2 := i.(func(StructFillRegs, MagicLastTypeNameForTestingRegisterABI) StructFillRegs)
	r2 := f2(a2, MagicLastTypeNameForTestingRegisterABI{})
	if r2 != a2 {
		t.Errorf("bad method value call: got %#v, want %#v", r2, a2)
	}
	if s.Value != 3 {
		t.Errorf("bad method value call: failed to set s.Value: got %d, want %d", s.Value, 3)
	}

	s, i = makeMethodValue("ValueRegMethodSpillInt")
	f3 := i.(func(StructFillRegs, int, MagicLastTypeNameForTestingRegisterABI) (StructFillRegs, int))
	r3a, r3b := f3(a2, 42, MagicLastTypeNameForTestingRegisterABI{})
	if r3a != a2 {
		t.Errorf("bad method value call: got %#v, want %#v", r3a, a2)
	}
	if r3b != 42 {
		t.Errorf("bad method value call: got %#v, want %#v", r3b, 42)
	}
	if s.Value != 4 {
		t.Errorf("bad method value call: failed to set s.Value: got %d, want %d", s.Value, 4)
	}

	s, i = makeMethodValue("ValueRegMethodSpillPtr")
	f4 := i.(func(StructFillRegs, *byte, MagicLastTypeNameForTestingRegisterABI) (StructFillRegs, *byte))
	vb := byte(10)
	r4a, r4b := f4(a2, &vb, MagicLastTypeNameForTestingRegisterABI{})
	if r4a != a2 {
		t.Errorf("bad method value call: got %#v, want %#v", r4a, a2)
	}
	if r4b != &vb {
		t.Errorf("bad method value call: got %#v, want %#v", r4b, &vb)
	}
	if s.Value != 5 {
		t.Errorf("bad method value call: failed to set s.Value: got %d, want %d", s.Value, 5)
	}
}

type StructWithMethods struct {
	Value int
}

type StructFewRegs struct {
	a0, a1, a2, a3 int
	f0, f1, f2, f3 float64
}

type StructFillRegs struct {
	a0, a1, a2, a3, a4, a5, a6, a7, a8                              int
	f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14 float64
}

func (m *StructWithMethods) AllRegsCall(s StructFewRegs, _ MagicLastTypeNameForTestingRegisterABI) StructFewRegs {
	m.Value = 1
	return s
}

func (m *StructWithMethods) RegsAndStackCall(s StructFewRegs, a [4]uint64, _ MagicLastTypeNameForTestingRegisterABI) (StructFewRegs, [4]uint64) {
	m.Value = 2
	return s, a
}

func (m *StructWithMethods) SpillStructCall(s StructFillRegs, _ MagicLastTypeNameForTestingRegisterABI) StructFillRegs {
	m.Value = 3
	return s
}

// When called as a method value, i is passed on the stack.
// When called as a method, i is passed in a register.
func (m *StructWithMethods) ValueRegMethodSpillInt(s StructFillRegs, i int, _ MagicLastTypeNameForTestingRegisterABI) (StructFillRegs, int) {
	m.Value = 4
	return s, i
}

// When called as a method value, i is passed on the stack.
// When called as a method, i is passed in a register.
func (m *StructWithMethods) ValueRegMethodSpillPtr(s StructFillRegs, i *byte, _ MagicLastTypeNameForTestingRegisterABI) (StructFillRegs, *byte) {
	m.Value = 5
	return s, i
}

func TestReflectCallABI(t *testing.T) {
	// Enable register-based reflect.Call and ensure we don't
	// use potentially incorrect cached versions by clearing
	// the cache before we start and after we're done.
	defer reflect.SetArgRegs(reflect.SetArgRegs(abi.IntArgRegs, abi.FloatArgRegs, abi.EffectiveFloatRegSize))

	// Execute the functions defined below which all have the
	// same form and perform the same function: pass all arguments
	// to return values. The purpose is to test the call boundary
	// and make sure it works.
	r := rand.New(rand.NewSource(genValueRandSeed))
	for _, fn := range abiCallTestCases {
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
			for arg := range typ.Ins() {
				args = append(args, genValue(t, arg, r))
			}
			results := fn.Call(args)
			for i := range results {
				x, y := args[i].Interface(), results[i].Interface()
				if reflect.DeepEqual(x, y) {
					continue
				}
				t.Errorf("arg and result %d differ: got %+v, want %+v", i, y, x)
			}
		})
	}
}

func TestReflectMakeFuncCallABI(t *testing.T) {
	// Enable register-based reflect.MakeFunc and ensure we don't
	// use potentially incorrect cached versions by clearing
	// the cache before we start and after we're done.
	defer reflect.SetArgRegs(reflect.SetArgRegs(abi.IntArgRegs, abi.FloatArgRegs, abi.EffectiveFloatRegSize))

	// Execute the functions defined below which all have the
	// same form and perform the same function: pass all arguments
	// to return values. The purpose is to test the call boundary
	// and make sure it works.
	r := rand.New(rand.NewSource(genValueRandSeed))
	makeFuncHandler := func(args []reflect.Value) []reflect.Value {
		if len(args) == 0 {
			return []reflect.Value{}
		}
		return args[:len(args)-1] // The last Value is an empty magic value.
	}
	for _, callFn := range abiMakeFuncTestCases {
		fnTyp := reflect.TypeOf(callFn).In(0)
		fn := reflect.MakeFunc(fnTyp, makeFuncHandler)
		callFn := reflect.ValueOf(callFn)
		t.Run(runtime.FuncForPC(callFn.Pointer()).Name(), func(t *testing.T) {
			args := []reflect.Value{fn}
			for i := 0; i < fnTyp.NumIn()-1; /* last one is magic type */ i++ {
				args = append(args, genValue(t, fnTyp.In(i), r))
			}
			results := callFn.Call(args)
			for i := range results {
				x, y := args[i+1].Interface(), results[i].Interface()
				if reflect.DeepEqual(x, y) {
					continue
				}
				t.Errorf("arg and result %d differ: got %+v, want %+v", i, y, x)
			}
		})
	}
	t.Run("OnlyPointerInRegisterGC", func(t *testing.T) {
		// This test attempts to induce a failure wherein
		// the last pointer to an object is passed via registers.
		// If makeFuncStub doesn't successfully store the pointer
		// to a location visible to the GC, the object should be
		// freed and then the next GC should notice that an object
		// was inexplicably revived.
		var f func(b *uint64, _ MagicLastTypeNameForTestingRegisterABI) *uint64
		mkfn := reflect.MakeFunc(reflect.TypeOf(f), func(args []reflect.Value) []reflect.Value {
			*(args[0].Interface().(*uint64)) = 5
			return args[:1]
		})
		fn := mkfn.Interface().(func(*uint64, MagicLastTypeNameForTestingRegisterABI) *uint64)

		// Call the MakeFunc'd function while trying pass the only pointer
		// to a new heap-allocated uint64.
		*reflect.CallGC = true
		x := fn(new(uint64), MagicLastTypeNameForTestingRegisterABI{})
		*reflect.CallGC = false

		// Check for bad pointers (which should be x if things went wrong).
		runtime.GC()

		// Sanity check x.
		if *x != 5 {
			t.Fatalf("failed to set value in object")
		}
	})
}

var abiCallTestCases = []any{
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
	passStruct14,
	passStruct15,
	pass2Struct1,
	passEmptyStruct,
	passStruct10AndSmall,
}

// Functions for testing reflect function call functionality.

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
func passStruct14(a Struct14) Struct14 {
	return a
}

//go:registerparams
//go:noinline
func passStruct15(a Struct15) Struct15 {
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
//
//go:registerparams
//go:noinline
func passStruct10AndSmall(a Struct10, b byte, c uint) (Struct10, byte, uint) {
	return a, b, c
}

var abiMakeFuncTestCases = []any{
	callArgsNone,
	callArgsInt,
	callArgsInt8,
	callArgsInt16,
	callArgsInt32,
	callArgsInt64,
	callArgsUint,
	callArgsUint8,
	callArgsUint16,
	callArgsUint32,
	callArgsUint64,
	callArgsFloat32,
	callArgsFloat64,
	callArgsComplex64,
	callArgsComplex128,
	callArgsManyInt,
	callArgsManyFloat64,
	callArgsArray1,
	callArgsArray,
	callArgsArray1Mix,
	callArgsString,
	// TODO(mknyszek): Test callArgsing interface values.
	callArgsSlice,
	callArgsPointer,
	callArgsStruct1,
	callArgsStruct2,
	callArgsStruct3,
	callArgsStruct4,
	callArgsStruct5,
	callArgsStruct6,
	callArgsStruct7,
	callArgsStruct8,
	callArgsStruct9,
	callArgsStruct10,
	// TODO(mknyszek): Test callArgsing unsafe.Pointer values.
	// TODO(mknyszek): Test callArgsing chan values.
	callArgsStruct11,
	callArgsStruct12,
	callArgsStruct13,
	callArgsStruct14,
	callArgsStruct15,
	callArgs2Struct1,
	callArgsEmptyStruct,
}

//go:registerparams
//go:noinline
func callArgsNone(f func(MagicLastTypeNameForTestingRegisterABI)) {
	f(MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsInt(f func(int, MagicLastTypeNameForTestingRegisterABI) int, a0 int) int {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsInt8(f func(int8, MagicLastTypeNameForTestingRegisterABI) int8, a0 int8) int8 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsInt16(f func(int16, MagicLastTypeNameForTestingRegisterABI) int16, a0 int16) int16 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsInt32(f func(int32, MagicLastTypeNameForTestingRegisterABI) int32, a0 int32) int32 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsInt64(f func(int64, MagicLastTypeNameForTestingRegisterABI) int64, a0 int64) int64 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsUint(f func(uint, MagicLastTypeNameForTestingRegisterABI) uint, a0 uint) uint {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsUint8(f func(uint8, MagicLastTypeNameForTestingRegisterABI) uint8, a0 uint8) uint8 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsUint16(f func(uint16, MagicLastTypeNameForTestingRegisterABI) uint16, a0 uint16) uint16 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsUint32(f func(uint32, MagicLastTypeNameForTestingRegisterABI) uint32, a0 uint32) uint32 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsUint64(f func(uint64, MagicLastTypeNameForTestingRegisterABI) uint64, a0 uint64) uint64 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsFloat32(f func(float32, MagicLastTypeNameForTestingRegisterABI) float32, a0 float32) float32 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsFloat64(f func(float64, MagicLastTypeNameForTestingRegisterABI) float64, a0 float64) float64 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsComplex64(f func(complex64, MagicLastTypeNameForTestingRegisterABI) complex64, a0 complex64) complex64 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsComplex128(f func(complex128, MagicLastTypeNameForTestingRegisterABI) complex128, a0 complex128) complex128 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsArray1(f func([1]uint32, MagicLastTypeNameForTestingRegisterABI) [1]uint32, a0 [1]uint32) [1]uint32 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsArray(f func([2]uintptr, MagicLastTypeNameForTestingRegisterABI) [2]uintptr, a0 [2]uintptr) [2]uintptr {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsArray1Mix(f func(int, [1]uint32, float64, MagicLastTypeNameForTestingRegisterABI) (int, [1]uint32, float64), a0 int, a1 [1]uint32, a2 float64) (int, [1]uint32, float64) {
	return f(a0, a1, a2, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsString(f func(string, MagicLastTypeNameForTestingRegisterABI) string, a0 string) string {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsSlice(f func([]byte, MagicLastTypeNameForTestingRegisterABI) []byte, a0 []byte) []byte {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsPointer(f func(*byte, MagicLastTypeNameForTestingRegisterABI) *byte, a0 *byte) *byte {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsManyInt(f func(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 int, x MagicLastTypeNameForTestingRegisterABI) (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9 int), a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 int) (int, int, int, int, int, int, int, int, int, int) {
	return f(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsManyFloat64(f func(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18 float64, x MagicLastTypeNameForTestingRegisterABI) (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18 float64), a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18 float64) (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18 float64) {
	return f(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct1(f func(Struct1, MagicLastTypeNameForTestingRegisterABI) Struct1, a0 Struct1) Struct1 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct2(f func(Struct2, MagicLastTypeNameForTestingRegisterABI) Struct2, a0 Struct2) Struct2 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct3(f func(Struct3, MagicLastTypeNameForTestingRegisterABI) Struct3, a0 Struct3) Struct3 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct4(f func(Struct4, MagicLastTypeNameForTestingRegisterABI) Struct4, a0 Struct4) Struct4 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct5(f func(Struct5, MagicLastTypeNameForTestingRegisterABI) Struct5, a0 Struct5) Struct5 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct6(f func(Struct6, MagicLastTypeNameForTestingRegisterABI) Struct6, a0 Struct6) Struct6 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct7(f func(Struct7, MagicLastTypeNameForTestingRegisterABI) Struct7, a0 Struct7) Struct7 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct8(f func(Struct8, MagicLastTypeNameForTestingRegisterABI) Struct8, a0 Struct8) Struct8 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct9(f func(Struct9, MagicLastTypeNameForTestingRegisterABI) Struct9, a0 Struct9) Struct9 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct10(f func(Struct10, MagicLastTypeNameForTestingRegisterABI) Struct10, a0 Struct10) Struct10 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct11(f func(Struct11, MagicLastTypeNameForTestingRegisterABI) Struct11, a0 Struct11) Struct11 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct12(f func(Struct12, MagicLastTypeNameForTestingRegisterABI) Struct12, a0 Struct12) Struct12 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct13(f func(Struct13, MagicLastTypeNameForTestingRegisterABI) Struct13, a0 Struct13) Struct13 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct14(f func(Struct14, MagicLastTypeNameForTestingRegisterABI) Struct14, a0 Struct14) Struct14 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsStruct15(f func(Struct15, MagicLastTypeNameForTestingRegisterABI) Struct15, a0 Struct15) Struct15 {
	return f(a0, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgs2Struct1(f func(Struct1, Struct1, MagicLastTypeNameForTestingRegisterABI) (Struct1, Struct1), a0, a1 Struct1) (r0, r1 Struct1) {
	return f(a0, a1, MagicLastTypeNameForTestingRegisterABI{})
}

//go:registerparams
//go:noinline
func callArgsEmptyStruct(f func(int, struct{}, float64, MagicLastTypeNameForTestingRegisterABI) (int, struct{}, float64), a0 int, a1 struct{}, a2 float64) (int, struct{}, float64) {
	return f(a0, a1, a2, MagicLastTypeNameForTestingRegisterABI{})
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

// Struct14 tests a non-zero-sized (and otherwise register-assignable)
// struct with a field that is a non-zero length array with zero-sized members.
type Struct14 struct {
	A uintptr
	X [3]struct{}
	B float64
}

// Struct15 tests a non-zero-sized (and otherwise register-assignable)
// struct with a struct field that is zero-sized but contains a
// non-zero length array with zero-sized members.
type Struct15 struct {
	A uintptr
	X struct {
		Y [3]struct{}
	}
	B float64
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

func TestSignalingNaNArgument(t *testing.T) {
	v := reflect.ValueOf(func(x float32) {
		// make sure x is a signaling NaN.
		u := math.Float32bits(x)
		if u != snan {
			t.Fatalf("signaling NaN not correct: %x\n", u)
		}
	})
	v.Call([]reflect.Value{reflect.ValueOf(math.Float32frombits(snan))})
}

func TestSignalingNaNReturn(t *testing.T) {
	v := reflect.ValueOf(func() float32 {
		return math.Float32frombits(snan)
	})
	var x float32
	reflect.ValueOf(&x).Elem().Set(v.Call(nil)[0])
	// make sure x is a signaling NaN.
	u := math.Float32bits(x)
	if u != snan {
		t.Fatalf("signaling NaN not correct: %x\n", u)
	}
}
