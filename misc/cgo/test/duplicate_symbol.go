// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains test cases for cgo.

package cgotest

/*
int base_symbol = 0;

#define alias_one base_symbol
#define alias_two base_symbol
*/
import "C"

import "fmt"

func duplicateSymbols() {
	fmt.Printf("%v %v %v\n", C.base_symbol, C.alias_one, C.alias_two)
}
