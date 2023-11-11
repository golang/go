// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test cases for symbol name mangling.

package main

import (
	"fmt"
	"strings"
)

// Issue 58800:
// Instantiated function name may contain weird characters
// that confuse the external linker, so it needs to be
// mangled.
type S struct {
	X int `parser:"|@@)"`
}

//go:noinline
func F[T any]() {}

func P() {
	F[S]()
}

// Issue 62098: the name mangling code doesn't handle some string
// symbols correctly.
func G(id string) error {
	if strings.ContainsAny(id, "&$@;/:+,?\\{^}%`]\">[~<#|") {
		return fmt.Errorf("invalid")
	}
	return nil
}

func main() {}
