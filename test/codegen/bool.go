// asmcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to boolean simplifications/optimizations.

func convertNeq0B(x uint8, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETNE"
	b := x&1 != 0
	return c && b
}

func convertNeq0W(x uint16, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETNE"
	b := x&1 != 0
	return c && b
}

func convertNeq0L(x uint32, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETB"
	b := x&1 != 0
	return c && b
}

func convertNeq0Q(x uint64, c bool) bool {
	// amd64:"ANDQ\t[$]1",-"SETB"
	b := x&1 != 0
	return c && b
}
