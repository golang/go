// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quick

import (
	"math/rand"
	"reflect"
	"testing"
)

func fArray(a [4]byte) [4]byte { return a }

type TestArrayAlias [4]byte

func fArrayAlias(a TestArrayAlias) TestArrayAlias { return a }

func fBool(a bool) bool { return a }

type TestBoolAlias bool

func fBoolAlias(a TestBoolAlias) TestBoolAlias { return a }

func fFloat32(a float32) float32 { return a }

type TestFloat32Alias float32

func fFloat32Alias(a TestFloat32Alias) TestFloat32Alias { return a }

func fFloat64(a float64) float64 { return a }

type TestFloat64Alias float64

func fFloat64Alias(a TestFloat64Alias) TestFloat64Alias { return a }

func fComplex64(a complex64) complex64 { return a }

type TestComplex64Alias complex64

func fComplex64Alias(a TestComplex64Alias) TestComplex64Alias { return a }

func fComplex128(a complex128) complex128 { return a }

type TestComplex128Alias complex128

func fComplex128Alias(a TestComplex128Alias) TestComplex128Alias { return a }

func fInt16(a int16) int16 { return a }

type TestInt16Alias int16

func fInt16Alias(a TestInt16Alias) TestInt16Alias { return a }

func fInt32(a int32) int32 { return a }

type TestInt32Alias int32

func fInt32Alias(a TestInt32Alias) TestInt32Alias { return a }

func fInt64(a int64) int64 { return a }

type TestInt64Alias int64

func fInt64Alias(a TestInt64Alias) TestInt64Alias { return a }

func fInt8(a int8) int8 { return a }

type TestInt8Alias int8

func fInt8Alias(a TestInt8Alias) TestInt8Alias { return a }

func fInt(a int) int { return a }

type TestIntAlias int

func fIntAlias(a TestIntAlias) TestIntAlias { return a }

func fMap(a map[int]int) map[int]int { return a }

type TestMapAlias map[int]int

func fMapAlias(a TestMapAlias) TestMapAlias { return a }

func fPtr(a *int) *int {
	if a == nil {
		return nil
	}
	b := *a
	return &b
}

type TestPtrAlias *int

func fPtrAlias(a TestPtrAlias) TestPtrAlias { return a }

func fSlice(a []byte) []byte { return a }

type TestSliceAlias []byte

func fSliceAlias(a TestSliceAlias) TestSliceAlias { return a }

func fString(a string) string { return a }

type TestStringAlias string

func fStringAlias(a TestStringAlias) TestStringAlias { return a }

type TestStruct struct {
	A int
	B string
}

func fStruct(a TestStruct) TestStruct { return a }

type TestStructAlias TestStruct

func fStructAlias(a TestStructAlias) TestStructAlias { return a }

func fUint16(a uint16) uint16 { return a }

type TestUint16Alias uint16

func fUint16Alias(a TestUint16Alias) TestUint16Alias { return a }

func fUint32(a uint32) uint32 { return a }

type TestUint32Alias uint32

func fUint32Alias(a TestUint32Alias) TestUint32Alias { return a }

func fUint64(a uint64) uint64 { return a }

type TestUint64Alias uint64

func fUint64Alias(a TestUint64Alias) TestUint64Alias { return a }

func fUint8(a uint8) uint8 { return a }

type TestUint8Alias uint8

func fUint8Alias(a TestUint8Alias) TestUint8Alias { return a }

func fUint(a uint) uint { return a }

type TestUintAlias uint

func fUintAlias(a TestUintAlias) TestUintAlias { return a }

func fUintptr(a uintptr) uintptr { return a }

type TestUintptrAlias uintptr

func fUintptrAlias(a TestUintptrAlias) TestUintptrAlias { return a }

func reportError(property string, err error, t *testing.T) {
	if err != nil {
		t.Errorf("%s: %s", property, err)
	}
}

func TestCheckEqual(t *testing.T) {
	reportError("fArray", CheckEqual(fArray, fArray, nil), t)
	reportError("fArrayAlias", CheckEqual(fArrayAlias, fArrayAlias, nil), t)
	reportError("fBool", CheckEqual(fBool, fBool, nil), t)
	reportError("fBoolAlias", CheckEqual(fBoolAlias, fBoolAlias, nil), t)
	reportError("fFloat32", CheckEqual(fFloat32, fFloat32, nil), t)
	reportError("fFloat32Alias", CheckEqual(fFloat32Alias, fFloat32Alias, nil), t)
	reportError("fFloat64", CheckEqual(fFloat64, fFloat64, nil), t)
	reportError("fFloat64Alias", CheckEqual(fFloat64Alias, fFloat64Alias, nil), t)
	reportError("fComplex64", CheckEqual(fComplex64, fComplex64, nil), t)
	reportError("fComplex64Alias", CheckEqual(fComplex64Alias, fComplex64Alias, nil), t)
	reportError("fComplex128", CheckEqual(fComplex128, fComplex128, nil), t)
	reportError("fComplex128Alias", CheckEqual(fComplex128Alias, fComplex128Alias, nil), t)
	reportError("fInt16", CheckEqual(fInt16, fInt16, nil), t)
	reportError("fInt16Alias", CheckEqual(fInt16Alias, fInt16Alias, nil), t)
	reportError("fInt32", CheckEqual(fInt32, fInt32, nil), t)
	reportError("fInt32Alias", CheckEqual(fInt32Alias, fInt32Alias, nil), t)
	reportError("fInt64", CheckEqual(fInt64, fInt64, nil), t)
	reportError("fInt64Alias", CheckEqual(fInt64Alias, fInt64Alias, nil), t)
	reportError("fInt8", CheckEqual(fInt8, fInt8, nil), t)
	reportError("fInt8Alias", CheckEqual(fInt8Alias, fInt8Alias, nil), t)
	reportError("fInt", CheckEqual(fInt, fInt, nil), t)
	reportError("fIntAlias", CheckEqual(fIntAlias, fIntAlias, nil), t)
	reportError("fInt32", CheckEqual(fInt32, fInt32, nil), t)
	reportError("fInt32Alias", CheckEqual(fInt32Alias, fInt32Alias, nil), t)
	reportError("fMap", CheckEqual(fMap, fMap, nil), t)
	reportError("fMapAlias", CheckEqual(fMapAlias, fMapAlias, nil), t)
	reportError("fPtr", CheckEqual(fPtr, fPtr, nil), t)
	reportError("fPtrAlias", CheckEqual(fPtrAlias, fPtrAlias, nil), t)
	reportError("fSlice", CheckEqual(fSlice, fSlice, nil), t)
	reportError("fSliceAlias", CheckEqual(fSliceAlias, fSliceAlias, nil), t)
	reportError("fString", CheckEqual(fString, fString, nil), t)
	reportError("fStringAlias", CheckEqual(fStringAlias, fStringAlias, nil), t)
	reportError("fStruct", CheckEqual(fStruct, fStruct, nil), t)
	reportError("fStructAlias", CheckEqual(fStructAlias, fStructAlias, nil), t)
	reportError("fUint16", CheckEqual(fUint16, fUint16, nil), t)
	reportError("fUint16Alias", CheckEqual(fUint16Alias, fUint16Alias, nil), t)
	reportError("fUint32", CheckEqual(fUint32, fUint32, nil), t)
	reportError("fUint32Alias", CheckEqual(fUint32Alias, fUint32Alias, nil), t)
	reportError("fUint64", CheckEqual(fUint64, fUint64, nil), t)
	reportError("fUint64Alias", CheckEqual(fUint64Alias, fUint64Alias, nil), t)
	reportError("fUint8", CheckEqual(fUint8, fUint8, nil), t)
	reportError("fUint8Alias", CheckEqual(fUint8Alias, fUint8Alias, nil), t)
	reportError("fUint", CheckEqual(fUint, fUint, nil), t)
	reportError("fUintAlias", CheckEqual(fUintAlias, fUintAlias, nil), t)
	reportError("fUintptr", CheckEqual(fUintptr, fUintptr, nil), t)
	reportError("fUintptrAlias", CheckEqual(fUintptrAlias, fUintptrAlias, nil), t)
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

// Recursive data structures didn't terminate.
// Issues 8818 and 11148.
func TestRecursive(t *testing.T) {
	type R struct {
		Ptr      *R
		SliceP   []*R
		Slice    []R
		Map      map[int]R
		MapP     map[int]*R
		MapR     map[*R]*R
		SliceMap []map[int]R
	}

	f := func(r R) bool { return true }
	Check(f, nil)
}

func TestEmptyStruct(t *testing.T) {
	f := func(struct{}) bool { return true }
	Check(f, nil)
}

type (
	A struct{ B *B }
	B struct{ A *A }
)

func TestMutuallyRecursive(t *testing.T) {
	f := func(a A) bool { return true }
	Check(f, nil)
}

// Some serialization formats (e.g. encoding/pem) cannot distinguish
// between a nil and an empty map or slice, so avoid generating the
// zero value for these.
func TestNonZeroSliceAndMap(t *testing.T) {
	type Q struct {
		M map[int]int
		S []int
	}
	f := func(q Q) bool {
		return q.M != nil && q.S != nil
	}
	err := Check(f, nil)
	if err != nil {
		t.Fatal(err)
	}
}
