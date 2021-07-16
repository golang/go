// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that various kinds of "imported and not used"
// errors are caught by the compiler.
// Does not compile.

package main

// standard
import "fmt" // ERROR "imported but not used.*fmt"

// renamed
import X "math" // ERROR "imported but not used.*math"

// import dot
import . "bufio" // ERROR "imported but not used.*bufio"

// again, package without anything in it
import "./empty"   // ERROR "imported but not used.*empty"
import Z "./empty" // ERROR "imported but not used.*empty"
import . "./empty" // ERROR "imported but not used.*empty"
