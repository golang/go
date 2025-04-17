// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Converting constants between types must introduce rounding.

package main

import "fmt"

const (
    F32 = 0.00999999977648258209228515625
    F64 = 0.01000000000000000020816681711721685132943093776702880859375
)

var F = float64(float32(0.01))

func main() {
	// 0.01 rounded to float32 then to float64 is F32.
	// 0.01 represented directly in float64 is F64.
	if F != F32 {
		panic(fmt.Sprintf("F=%.1000g, want %.1000g", F, F32))
	}
}
