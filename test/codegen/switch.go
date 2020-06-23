// asmcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of switch statements.

package codegen

// see issue 33934
func f(x string) int {
	// amd64:-`cmpstring`
	switch x {
	case "":
		return -1
	case "1", "2", "3":
		return -2
	default:
		return -3
	}
}
