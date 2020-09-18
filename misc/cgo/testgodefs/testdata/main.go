// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"reflect"
)

// Test that the struct field in anonunion.go was promoted.
var v1 T
var v2 = v1.L

// Test that P, Q, and R all point to byte.
var v3 = Issue8478{P: (*byte)(nil), Q: (**byte)(nil), R: (***byte)(nil)}

// Test that N, A and B are fully defined
var v4 = N{}
var v5 = A{}
var v6 = B{}

// Test that S is fully defined
var v7 = S{}

// Test that #define'd type is fully defined
var _ = issue38649{X: 0}

func main() {
	pass := true

	// The Go translation of bitfields should not have any of the
	// bitfield types. The order in which bitfields are laid out
	// in memory is implementation defined, so we can't easily
	// know how a bitfield should correspond to a Go type, even if
	// it appears to be aligned correctly.
	bitfieldType := reflect.TypeOf(bitfields{})
	check := func(name string) {
		_, ok := bitfieldType.FieldByName(name)
		if ok {
			fmt.Fprintf(os.Stderr, "found unexpected bitfields field %s\n", name)
			pass = false
		}
	}
	check("Short1")
	check("Short2")
	check("Short3")

	if !pass {
		os.Exit(1)
	}
}
