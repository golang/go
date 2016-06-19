// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that switch statements with duplicate cases are detected by the compiler.
// Does not compile.

package main

import "fmt"

func f0(x int) {
	switch x {
	case 0:
	case 0: // ERROR "duplicate case 0 in switch"
	}

	switch x {
	case 0:
	case int(0): // ERROR "duplicate case 0 in switch"
	}
}

func f1(x float32) {
	switch x {
	case 5:
	case 5: // ERROR "duplicate case 5 in switch"
	case 5.0: // ERROR "duplicate case 5 in switch"
	}
}

func f2(s string) {
	switch s {
	case "":
	case "": // ERROR "duplicate case .. in switch"
	case "abc":
	case "abc": // ERROR "duplicate case .abc. in switch"
	}
}

func f3(e interface{}) {
	switch e {
	case 0:
	case 0: // ERROR "duplicate case 0 in switch"
	case int64(0):
	case float32(10):
	case float32(10): // ERROR "duplicate case float32\(10\) in switch"
	case float64(10):
	case float64(10): // ERROR "duplicate case float64\(10\) in switch"
	}
}

func f4(e interface{}) {
	switch e.(type) {
	case int:
	case int: // ERROR "duplicate case int in type switch"
	case int64:
	case error: // ERROR "duplicate case error in type switch"
	case error:
	case fmt.Stringer:
	case fmt.Stringer: // ERROR "duplicate case fmt.Stringer in type switch"
	case struct {
		i int "tag1"
	}:
	case struct {
		i int "tag2"
	}:
	case struct {
		i int "tag1"
	}: // ERROR "duplicate case struct { i int .tag1. } in type switch"
	}
}

func f5(a [1]int) {
	switch a {
	case [1]int{0}:
	case [1]int{0}: // OK -- see issue 15896
	}
}
