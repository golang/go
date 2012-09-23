// $G $D/empty.go && errchk $G $D/$F.go

// NOTE: This test is not run by 'run.go' and so not run by all.bash.
// To run this test you must use the ./run shell script.

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that various kinds of "imported and not used"
// errors are caught by the compiler.
// Does not compile.

package main

// standard
import "fmt"	// ERROR "imported and not used.*fmt"

// renamed
import X "math"	// ERROR "imported and not used.*math"

// import dot
import . "bufio"	// ERROR "imported and not used.*bufio"

// again, package without anything in it
import "./empty"	// ERROR "imported and not used.*empty"
import Z "./empty"	// ERROR "imported and not used.*empty"
import . "./empty"	// ERROR "imported and not used.*empty"

