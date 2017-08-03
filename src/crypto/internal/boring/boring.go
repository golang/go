// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64
// +build !cmd_go_bootstrap

package boring

// #include "goboringcrypto.h"
import "C"

const available = true

func init() {
	C._goboringcrypto_BORINGSSL_bcm_power_on_self_test()
	if C._goboringcrypto_FIPS_mode() != 1 {
		panic("boringcrypto: not in FIPS mode")
	}
}

// Unreachable marks code that should be unreachable
// when BoringCrypto is in use. It panics.
func Unreachable() {
	panic("boringcrypto: invalid code execution")
}

// provided by runtime to avoid os import
func runtime_arg0() string

// UnreachableExceptTests marks code that should be unreachable
// when BoringCrypto is in use. It panics.
func UnreachableExceptTests() {
	arg0 := runtime_arg0()
	if len(arg0) < 5 || arg0[len(arg0)-5:] != ".test" {
		println("ARG0", arg0)
		panic("boringcrypto: invalid code execution")
	}
}

type fail string

func (e fail) Error() string { return "boringcrypto: " + string(e) + " failed" }
