// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that various kinds of "imported and not used"
// errors are caught by the compiler.
// Does not compile.

package main

// standard
import "fmt"	// ERROR "imported and not used.*fmt|\x22fmt\x22 imported and not used"

// renamed
import X "math"	// ERROR "imported and not used.*math|\x22math\x22 imported as X and not used"

// import dot
import . "bufio"	// ERROR "imported and not used.*bufio|imported and not used"

// again, package without anything in it
import "./empty"	// ERROR "imported and not used.*empty|imported and not used"
import Z "./empty"	// ERROR "imported and not used.*empty|imported as Z and not used"
import . "./empty"	// ERROR "imported and not used.*empty|imported and not used"

