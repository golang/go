// asmcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to boolean simplifications/optimizations.

func convertNeq0B(x uint8, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETNE"
	// ppc64:"ANDCC",-"CMPW",-"ISEL"
	// ppc64le:"ANDCC",-"CMPW",-"ISEL"
	// ppc64le/power9:"ANDCC",-"CMPW",-"ISEL"
	b := x&1 != 0
	return c && b
}

func convertNeq0W(x uint16, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETNE"
	// ppc64:"ANDCC",-"CMPW",-"ISEL"
	// ppc64le:"ANDCC",-"CMPW",-"ISEL"
	// ppc64le/power9:"ANDCC",-CMPW",-"ISEL"
	b := x&1 != 0
	return c && b
}

func convertNeq0L(x uint32, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETB"
	// ppc64:"ANDCC",-"CMPW",-"ISEL"
	// ppc64le:"ANDCC",-"CMPW",-"ISEL"
	// ppc64le/power9:"ANDCC",-"CMPW",-"ISEL"
	b := x&1 != 0
	return c && b
}

func convertNeq0Q(x uint64, c bool) bool {
	// amd64:"ANDL\t[$]1",-"SETB"
	// ppc64:"ANDCC",-"CMP",-"ISEL"
	// ppc64le:"ANDCC",-"CMP",-"ISEL"
	// ppc64le/power9:"ANDCC",-"CMP",-"ISEL"
	b := x&1 != 0
	return c && b
}

func convertNeqBool32(x uint32) bool {
        // ppc64:"ANDCC",-"CMPW",-"ISEL"
        // ppc64le:"ANDCC",-"CMPW",-"ISEL"
        // ppc64le/power9:"ANDCC",-"CMPW",-"ISEL"
        return x&1 != 0
}

func convertEqBool32(x uint32) bool {
        // ppc64:"ANDCC",-"CMPW","XOR",-"ISEL"
        // ppc64le:"ANDCC",-"CMPW","XOR",-"ISEL"
        // ppc64le/power9:"ANDCC","XOR",-"CMPW",-"ISEL"
        return x&1 == 0
}

func convertNeqBool64(x uint64) bool {
        // ppc64:"ANDCC",-"CMP",-"ISEL"
        // ppc64le:"ANDCC",-"CMP",-"ISEL"
        // ppc64le/power9:"ANDCC",-"CMP",-"ISEL"
        return x&1 != 0
}

func convertEqBool64(x uint64) bool {
        // ppc64:"ANDCC","XOR",-"CMP",-"ISEL"
        // ppc64le:"ANDCC","XOR",-"CMP",-"ISEL"
        // ppc64le/power9:"ANDCC","XOR",-"CMP",-"ISEL"
        return x&1 == 0
}
