// errorcheck -0 -d=ssa/late_fuse/debug=1

//go:build (amd64 && !gcflags_noopt) || (arm64 && !gcflags_noopt)

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

const Cf2 = 2.0

func fEqEq(a int, f float64) bool {
	return a == 0 && f > Cf2 || a == 0 && f < -Cf2 // ERROR "Redirect Eq64 based on Eq64$"
}

func fEqNeq(a int32, f float64) bool {
	return a == 0 && f > Cf2 || a != 0 && f < -Cf2 // ERROR "Redirect Neq32 based on Eq32$"
}

func fEqLess(a int8, f float64) bool {
	return a == 0 && f > Cf2 || a < 0 && f < -Cf2
}

func fEqLeq(a float64, f float64) bool {
	return a == 0 && f > Cf2 || a <= 0 && f < -Cf2
}

func fEqLessU(a uint, f float64) bool {
	return a == 0 && f > Cf2 || a < 0 && f < -Cf2
}

func fEqLeqU(a uint64, f float64) bool {
	return a == 0 && f > Cf2 || a <= 0 && f < -Cf2 // ERROR "Redirect Leq64U based on Eq64$"
}

func fNeqEq(a int, f float64) bool {
	return a != 0 && f > Cf2 || a == 0 && f < -Cf2 // ERROR "Redirect Eq64 based on Neq64$"
}

func fNeqNeq(a int32, f float64) bool {
	return a != 0 && f > Cf2 || a != 0 && f < -Cf2 // ERROR "Redirect Neq32 based on Neq32$"
}

func fNeqLess(a float32, f float64) bool {
	// TODO: Add support for floating point numbers in prove
	return a != 0 && f > Cf2 || a < 0 && f < -Cf2
}

func fNeqLeq(a int16, f float64) bool {
	return a != 0 && f > Cf2 || a <= 0 && f < -Cf2 // ERROR "Redirect Leq16 based on Neq16$"
}

func fNeqLessU(a uint, f float64) bool {
	return a != 0 && f > Cf2 || a < 0 && f < -Cf2
}

func fNeqLeqU(a uint32, f float64) bool {
	return a != 0 && f > Cf2 || a <= 0 && f < -Cf2 // ERROR "Redirect Leq32U based on Neq32$"
}

func fLessEq(a int, f float64) bool {
	return a < 0 && f > Cf2 || a == 0 && f < -Cf2
}

func fLessNeq(a int32, f float64) bool {
	return a < 0 && f > Cf2 || a != 0 && f < -Cf2
}

func fLessLess(a float32, f float64) bool {
	return a < 0 && f > Cf2 || a < 0 && f < -Cf2 // ERROR "Redirect Less32F based on Less32F$"
}

func fLessLeq(a float64, f float64) bool {
	return a < 0 && f > Cf2 || a <= 0 && f < -Cf2
}

func fLeqEq(a float64, f float64) bool {
	return a <= 0 && f > Cf2 || a == 0 && f < -Cf2
}

func fLeqNeq(a int16, f float64) bool {
	return a <= 0 && f > Cf2 || a != 0 && f < -Cf2 // ERROR "Redirect Neq16 based on Leq16$"
}

func fLeqLess(a float32, f float64) bool {
	return a <= 0 && f > Cf2 || a < 0 && f < -Cf2
}

func fLeqLeq(a int8, f float64) bool {
	return a <= 0 && f > Cf2 || a <= 0 && f < -Cf2 // ERROR "Redirect Leq8 based on Leq8$"
}

func fLessUEq(a uint8, f float64) bool {
	return a < 0 && f > Cf2 || a == 0 && f < -Cf2
}

func fLessUNeq(a uint16, f float64) bool {
	return a < 0 && f > Cf2 || a != 0 && f < -Cf2
}

func fLessULessU(a uint32, f float64) bool {
	return a < 0 && f > Cf2 || a < 0 && f < -Cf2
}

func fLessULeqU(a uint64, f float64) bool {
	return a < 0 && f > Cf2 || a <= 0 && f < -Cf2
}

func fLeqUEq(a uint8, f float64) bool {
	return a <= 0 && f > Cf2 || a == 0 && f < -Cf2 // ERROR "Redirect Eq8 based on Leq8U$"
}

func fLeqUNeq(a uint16, f float64) bool {
	return a <= 0 && f > Cf2 || a != 0 && f < -Cf2 // ERROR "Redirect Neq16 based on Leq16U$"
}

func fLeqLessU(a uint32, f float64) bool {
	return a <= 0 && f > Cf2 || a < 0 && f < -Cf2
}

func fLeqLeqU(a uint64, f float64) bool {
	return a <= 0 && f > Cf2 || a <= 0 && f < -Cf2 // ERROR "Redirect Leq64U based on Leq64U$"
}

// Arg tests are disabled because the op name is different on amd64 and arm64.

func fEqPtrEqPtr(a, b *int, f float64) bool {
	return a == b && f > Cf2 || a == b && f < -Cf2 // ERROR "Redirect EqPtr based on EqPtr$"
}

func fEqPtrNeqPtr(a, b *int, f float64) bool {
	return a == b && f > Cf2 || a != b && f < -Cf2 // ERROR "Redirect NeqPtr based on EqPtr$"
}

func fNeqPtrEqPtr(a, b *int, f float64) bool {
	return a != b && f > Cf2 || a == b && f < -Cf2 // ERROR "Redirect EqPtr based on NeqPtr$"
}

func fNeqPtrNeqPtr(a, b *int, f float64) bool {
	return a != b && f > Cf2 || a != b && f < -Cf2 // ERROR "Redirect NeqPtr based on NeqPtr$"
}

func fEqInterEqInter(a interface{}, f float64) bool {
	return a == nil && f > Cf2 || a == nil && f < -Cf2 // ERROR "Redirect IsNonNil based on IsNonNil$"
}

func fEqInterNeqInter(a interface{}, f float64) bool {
	return a == nil && f > Cf2 || a != nil && f < -Cf2
}

func fNeqInterEqInter(a interface{}, f float64) bool {
	return a != nil && f > Cf2 || a == nil && f < -Cf2
}

func fNeqInterNeqInter(a interface{}, f float64) bool {
	return a != nil && f > Cf2 || a != nil && f < -Cf2 // ERROR "Redirect IsNonNil based on IsNonNil$"
}

func fEqSliceEqSlice(a []int, f float64) bool {
	return a == nil && f > Cf2 || a == nil && f < -Cf2 // ERROR "Redirect IsNonNil based on IsNonNil$"
}

func fEqSliceNeqSlice(a []int, f float64) bool {
	return a == nil && f > Cf2 || a != nil && f < -Cf2
}

func fNeqSliceEqSlice(a []int, f float64) bool {
	return a != nil && f > Cf2 || a == nil && f < -Cf2
}

func fNeqSliceNeqSlice(a []int, f float64) bool {
	return a != nil && f > Cf2 || a != nil && f < -Cf2 // ERROR "Redirect IsNonNil based on IsNonNil$"
}

func fPhi(a, b string) string {
	aslash := strings.HasSuffix(a, "/") // ERROR "Redirect Phi based on Phi$"
	bslash := strings.HasPrefix(b, "/")
	switch {
	case aslash && bslash:
		return a + b[1:]
	case !aslash && !bslash:
		return a + "/" + b
	}
	return a + b
}

func main() {
}
