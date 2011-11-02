// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quick

import (
	"rand"
	"reflect"
	"testing"
)

func fBool(a bool) bool { return a }

func fFloat32(a float32) float32 { return a }

func fFloat64(a float64) float64 { return a }

func fComplex64(a complex64) complex64 { return a }

func fComplex128(a complex128) complex128 { return a }

func fInt16(a int16) int16 { return a }

func fInt32(a int32) int32 { return a }

func fInt64(a int64) int64 { return a }

func fInt8(a int8) int8 { return a }

func fInt(a int) int { return a }

func fUInt8(a uint8) uint8 { return a }

func fMap(a map[int]int) map[int]int { return a }

func fSlice(a []byte) []byte { return a }

func fString(a string) string { return a }

type TestStruct struct {
	A int
	B string
}

func fStruct(a TestStruct) TestStruct { return a }

func fUint16(a uint16) uint16 { return a }

func fUint32(a uint32) uint32 { return a }

func fUint64(a uint64) uint64 { return a }

func fUint8(a uint8) uint8 { return a }

func fUint(a uint) uint { return a }

func fUintptr(a uintptr) uintptr { return a }

func fIntptr(a *int) *int {
	b := *a
	return &b
}

func reportError(property string, err error, t *testing.T) {
	if err != nil {
		t.Errorf("%s: %s", property, err)
	}
}

func TestCheckEqual(t *testing.T) {
	reportError("fBool", CheckEqual(fBool, fBool, nil), t)
	reportError("fFloat32", CheckEqual(fFloat32, fFloat32, nil), t)
	reportError("fFloat64", CheckEqual(fFloat64, fFloat64, nil), t)
	reportError("fComplex64", CheckEqual(fComplex64, fComplex64, nil), t)
	reportError("fComplex128", CheckEqual(fComplex128, fComplex128, nil), t)
	reportError("fInt16", CheckEqual(fInt16, fInt16, nil), t)
	reportError("fInt32", CheckEqual(fInt32, fInt32, nil), t)
	reportError("fInt64", CheckEqual(fInt64, fInt64, nil), t)
	reportError("fInt8", CheckEqual(fInt8, fInt8, nil), t)
	reportError("fInt", CheckEqual(fInt, fInt, nil), t)
	reportError("fUInt8", CheckEqual(fUInt8, fUInt8, nil), t)
	reportError("fInt32", CheckEqual(fInt32, fInt32, nil), t)
	reportError("fMap", CheckEqual(fMap, fMap, nil), t)
	reportError("fSlice", CheckEqual(fSlice, fSlice, nil), t)
	reportError("fString", CheckEqual(fString, fString, nil), t)
	reportError("fStruct", CheckEqual(fStruct, fStruct, nil), t)
	reportError("fUint16", CheckEqual(fUint16, fUint16, nil), t)
	reportError("fUint32", CheckEqual(fUint32, fUint32, nil), t)
	reportError("fUint64", CheckEqual(fUint64, fUint64, nil), t)
	reportError("fUint8", CheckEqual(fUint8, fUint8, nil), t)
	reportError("fUint", CheckEqual(fUint, fUint, nil), t)
	reportError("fUintptr", CheckEqual(fUintptr, fUintptr, nil), t)
	reportError("fIntptr", CheckEqual(fIntptr, fIntptr, nil), t)
}

// This tests that ArbitraryValue is working by checking that all the arbitrary
// values of type MyStruct have x = 42.
type myStruct struct {
	x int
}

func (m myStruct) Generate(r *rand.Rand, _ int) reflect.Value {
	return reflect.ValueOf(myStruct{x: 42})
}

func myStructProperty(in myStruct) bool { return in.x == 42 }

func TestCheckProperty(t *testing.T) {
	reportError("myStructProperty", Check(myStructProperty, nil), t)
}

func TestFailure(t *testing.T) {
	f := func(x int) bool { return false }
	err := Check(f, nil)
	if err == nil {
		t.Errorf("Check didn't return an error")
	}
	if _, ok := err.(*CheckError); !ok {
		t.Errorf("Error was not a CheckError: %s", err)
	}

	err = CheckEqual(fUint, fUint32, nil)
	if err == nil {
		t.Errorf("#1 CheckEqual didn't return an error")
	}
	if _, ok := err.(SetupError); !ok {
		t.Errorf("#1 Error was not a SetupError: %s", err)
	}

	err = CheckEqual(func(x, y int) {}, func(x int) {}, nil)
	if err == nil {
		t.Errorf("#2 CheckEqual didn't return an error")
	}
	if _, ok := err.(SetupError); !ok {
		t.Errorf("#2 Error was not a SetupError: %s", err)
	}

	err = CheckEqual(func(x int) int { return 0 }, func(x int) int32 { return 0 }, nil)
	if err == nil {
		t.Errorf("#3 CheckEqual didn't return an error")
	}
	if _, ok := err.(SetupError); !ok {
		t.Errorf("#3 Error was not a SetupError: %s", err)
	}
}
