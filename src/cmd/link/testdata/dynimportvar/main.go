// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that we can access dynamically imported variables.
// We ues mach_task_self_ from darwin's system library.
// Check that loading the variable from C and Go gets the
// same result.

//go:build darwin

package main

/*
#include <mach/mach_init.h>

unsigned int Mach_task_self(void) {
	return mach_task_self();
}
*/
import "C"

import "cmd/link/testdata/dynimportvar/asm"

func main() {
	c := uint32(C.Mach_task_self())
	a := asm.Mach_task_self()
	if a != c {
		println("got", a, "want", c)
		panic("FAIL")
	}
}
