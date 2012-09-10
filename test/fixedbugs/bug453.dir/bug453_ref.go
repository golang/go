// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64,!386

package main

// These functions are standins for the functions in the .s file on other platforms.
func bug453a() float64 {
	return -1
}
func bug453b() float64 {
	return 1
}
