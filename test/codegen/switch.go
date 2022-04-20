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

// use jump tables for 8+ int cases
func square(x int) int {
	// amd64:`JMP\s\(.*\)\(.*\)$`
	switch x {
	case 1:
		return 1
	case 2:
		return 4
	case 3:
		return 9
	case 4:
		return 16
	case 5:
		return 25
	case 6:
		return 36
	case 7:
		return 49
	case 8:
		return 64
	default:
		return x * x
	}
}

// use jump tables for 8+ string lengths
func length(x string) int {
	// amd64:`JMP\s\(.*\)\(.*\)$`
	switch x {
	case "a":
		return 1
	case "bb":
		return 2
	case "ccc":
		return 3
	case "dddd":
		return 4
	case "eeeee":
		return 5
	case "ffffff":
		return 6
	case "ggggggg":
		return 7
	case "hhhhhhhh":
		return 8
	default:
		return len(x)
	}
}
