// build -goexperiment regabi,regabiargs

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"reflect"
)

func main() {
	// Only print if there is a problem
	Caller2()
	if FailCount != 0 {
		fmt.Fprintf(os.Stderr, "FAILURES: %d\n", FailCount)
		os.Exit(2)
	}
}

var ParamFailCount int

var ReturnFailCount int

var FailCount int

var Mode string

type UtilsType int

//go:noinline
func NoteFailure(cm int, pidx int, fidx int, pkg string, pref string, parmNo int, isret bool, _ uint64) {
	if isret {
		if ParamFailCount != 0 {
			return
		}
		ReturnFailCount++
	} else {
		ParamFailCount++
	}
	fmt.Fprintf(os.Stderr, "Error: fail %s |%d|%d|%d| =%s.Test%d= %s %d\n", Mode, cm, pidx, fidx, pkg, fidx, pref, parmNo)

	if ParamFailCount+FailCount+ReturnFailCount > 9999 {
		os.Exit(1)
	}
}

//go:noinline
func NoteFailureElem(cm int, pidx int, fidx int, pkg string, pref string, parmNo int, elem int, isret bool, _ uint64) {

	if isret {
		if ParamFailCount != 0 {
			return
		}
		ReturnFailCount++
	} else {
		ParamFailCount++
	}
	fmt.Fprintf(os.Stderr, "Error: fail %s |%d|%d|%d| =%s.Test%d= %s %d elem %d\n", Mode, cm, pidx, fidx, pkg, fidx, pref, parmNo, elem)

	if ParamFailCount+FailCount+ReturnFailCount > 9999 {
		os.Exit(1)
	}
}

func BeginFcn() {
	ParamFailCount = 0
	ReturnFailCount = 0
}

func EndFcn() {
	FailCount += ParamFailCount
	FailCount += ReturnFailCount
}

func Caller2() {
	BeginFcn()
	c0 := StructF2S0{F0: ArrayF2S1E1{New_3(float64(-0.4418990509835844))}}
	c1 := ArrayF2S2E1{StructF2S1{ /* _: "񊶿(z̽|" */ F1: "􂊇񊶿"}}
	c2 := int16(4162)
	c3 := float32(-7.667096e+37)
	c4 := int64(3202175648847048679)
	var p0 ArrayF2S0E0
	p0 = ArrayF2S0E0{}
	var p1 uint8
	p1 = uint8(57)
	var p2 uint16
	p2 = uint16(10920)
	var p3 float64
	p3 = float64(-1.597256501942112)
	Mode = ""
	// 5 returns 4 params
	r0, r1, r2, r3, r4 := Test2(p0, p1, p2, p3)
	if !EqualStructF2S0(r0, c0) {
		NoteFailure(9, 42, 2, "genChecker42", "return", 0, true, uint64(0))
	}
	if r1 != c1 {
		NoteFailure(9, 42, 2, "genChecker42", "return", 1, true, uint64(0))
	}
	if r2 != c2 {
		NoteFailure(9, 42, 2, "genChecker42", "return", 2, true, uint64(0))
	}
	if r3 != c3 {
		NoteFailure(9, 42, 2, "genChecker42", "return", 3, true, uint64(0))
	}
	if r4 != c4 {
		NoteFailure(9, 42, 2, "genChecker42", "return", 4, true, uint64(0))
	}
	// same call via reflection
	Mode = "reflect"
	rc := reflect.ValueOf(Test2)
	rvslice := rc.Call([]reflect.Value{reflect.ValueOf(p0), reflect.ValueOf(p1), reflect.ValueOf(p2), reflect.ValueOf(p3)})
	rr0i := rvslice[0].Interface()
	rr0v := rr0i.(StructF2S0)
	if !EqualStructF2S0(rr0v, c0) {
		NoteFailure(9, 42, 2, "genChecker42", "return", 0, true, uint64(0))
	}
	rr1i := rvslice[1].Interface()
	rr1v := rr1i.(ArrayF2S2E1)
	if rr1v != c1 {
		NoteFailure(9, 42, 2, "genChecker42", "return", 1, true, uint64(0))
	}
	rr2i := rvslice[2].Interface()
	rr2v := rr2i.(int16)
	if rr2v != c2 {
		NoteFailure(9, 42, 2, "genChecker42", "return", 2, true, uint64(0))
	}
	rr3i := rvslice[3].Interface()
	rr3v := rr3i.(float32)
	if rr3v != c3 {
		NoteFailure(9, 42, 2, "genChecker42", "return", 3, true, uint64(0))
	}
	rr4i := rvslice[4].Interface()
	rr4v := rr4i.(int64)
	if rr4v != c4 {
		NoteFailure(9, 42, 2, "genChecker42", "return", 4, true, uint64(0))
	}
	EndFcn()
}

type StructF0S0 struct {
}

type ArrayF0S0E2 [2]int16

type ArrayF0S1E1 [1]StructF0S0

type StructF1S0 struct {
	F0 StructF1S1
	_  ArrayF1S0E4
}

type StructF1S1 struct {
}

type StructF1S2 struct {
	F0 uint32
	F1 uint8
	F2 string
	F3 string
	F4 ArrayF1S1E1
}

type StructF1S3 struct {
	F0 float64
}

type StructF1S4 struct {
	_  int32
	F1 float32
}

type StructF1S5 struct {
	F0 uint16
}

type StructF1S6 struct {
	F0 uint8
	F1 uint32
}

type ArrayF1S0E4 [4]float64

type ArrayF1S1E1 [1]StructF1S3

type ArrayF1S2E2 [2]StructF1S4

type ArrayF1S3E2 [2]StructF1S5

type ArrayF1S4E4 [4]ArrayF1S5E3

type ArrayF1S5E3 [3]string

type ArrayF1S6E1 [1]float64

type StructF2S0 struct {
	F0 ArrayF2S1E1
}

// equal func for StructF2S0
//go:noinline
func EqualStructF2S0(left StructF2S0, right StructF2S0) bool {
	return EqualArrayF2S1E1(left.F0, right.F0)
}

type StructF2S1 struct {
	_  string
	F1 string
}

type ArrayF2S0E0 [0]int8

type ArrayF2S1E1 [1]*float64

// equal func for ArrayF2S1E1
//go:noinline
func EqualArrayF2S1E1(left ArrayF2S1E1, right ArrayF2S1E1) bool {
	return *left[0] == *right[0]
}

type ArrayF2S2E1 [1]StructF2S1

// 5 returns 4 params
//go:registerparams
//go:noinline
func Test2(p0 ArrayF2S0E0, p1 uint8, _ uint16, p3 float64) (r0 StructF2S0, r1 ArrayF2S2E1, r2 int16, r3 float32, r4 int64) {
	// consume some stack space, so as to trigger morestack
	var pad [16]uint64
	pad[FailCount&0x1]++
	rc0 := StructF2S0{F0: ArrayF2S1E1{New_3(float64(-0.4418990509835844))}}
	rc1 := ArrayF2S2E1{StructF2S1{ /* _: "񊶿(z̽|" */ F1: "􂊇񊶿"}}
	rc2 := int16(4162)
	rc3 := float32(-7.667096e+37)
	rc4 := int64(3202175648847048679)
	p1f0c := uint8(57)
	if p1 != p1f0c {
		NoteFailureElem(9, 42, 2, "genChecker42", "parm", 1, 0, false, pad[0])
		return
	}
	_ = uint16(10920)
	p3f0c := float64(-1.597256501942112)
	if p3 != p3f0c {
		NoteFailureElem(9, 42, 2, "genChecker42", "parm", 3, 0, false, pad[0])
		return
	}
	defer func(p0 ArrayF2S0E0, p1 uint8) {
		// check parm passed
		// check parm passed
		if p1 != p1f0c {
			NoteFailureElem(9, 42, 2, "genChecker42", "parm", 1, 0, false, pad[0])
			return
		}
		// check parm captured
		if p3 != p3f0c {
			NoteFailureElem(9, 42, 2, "genChecker42", "parm", 3, 0, false, pad[0])
			return
		}
	}(p0, p1)

	return rc0, rc1, rc2, rc3, rc4
	// 0 addr-taken params, 0 addr-taken returns
}

//go:noinline
func New_3(i float64) *float64 {
	x := new(float64)
	*x = i
	return x
}
