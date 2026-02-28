// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func cmovClobberAX64(v1, v2 int64, x1, x2 float64) int64 {
	r := v1
	if x1 == x2 {
		r = v2
	}
	return r
}

//go:noinline
func cmovClobberAX32(v1, v2 int32, x1, x2 float64) int32 {
	r := v1
	if x1 == x2 {
		r = v2
	}
	return r
}

//go:noinline
func cmovClobberAX16(v1, v2 int16, x1, x2 float64) int16 {
	r := v1
	if x1 == x2 {
		r = v2
	}
	return r
}

func main() {
	if cmovClobberAX16(1, 2, 4.0, 5.0) != 1 {
		panic("CMOVQEQF causes incorrect code")
	}
	if cmovClobberAX32(1, 2, 4.0, 5.0) != 1 {
		panic("CMOVQEQF causes incorrect code")
	}
	if cmovClobberAX64(1, 2, 4.0, 5.0) != 1 {
		panic("CMOVQEQF causes incorrect code")
	}

}
