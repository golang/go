// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program will abort.

/*
#include <stdlib.h>
*/
import "C"

func init() {
	register("Abort", Abort)
}

func Abort() {
	C.abort()
}
