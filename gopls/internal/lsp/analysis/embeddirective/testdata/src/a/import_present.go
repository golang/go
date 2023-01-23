// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// Misplaced, above imports.
//go:embed embedText // want "go:embed directives must precede a \"var\" declaration"

import (
	"fmt"

	_ "embed"
)

//go:embed embedText // ok
var s string

// The analyzer does not check for many directives using the same var.
//
//go:embed embedText // ok
//go:embed embedText // ok
var s string

// Comments and blank lines between are OK.
//
//go:embed embedText // ok
//
// foo

var s string

// Followed by wrong kind of decl.
//
//go:embed embedText // want "go:embed directives must precede a \"var\" declaration"
func foo()

// Multiple variable specs.
//
//go:embed embedText // want "declarations following go:embed directives must define a single variable"
var foo, bar []byte

// Specifying a value is not allowed.
//
//go:embed embedText // want "declarations following go:embed directives must not specify a value"
var s string = "foo"

// TODO: This should not be OK, misplaced according to compiler.
//
//go:embed embedText // ok
var (
	s string
	x string
)

// var blocks are OK as long as the variable following the directive is OK.
var (
	x, y, z string
	//go:embed embedText // ok
	s       string
	q, r, t string
)

//go:embed embedText // want "go:embed directives must precede a \"var\" declaration"
var ()

// This is main function
func main() {
	fmt.Println(s)
}

// No declaration following.
//go:embed embedText // want "go:embed directives must precede a \"var\" declaration"
