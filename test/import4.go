// $G $D/empty.go && errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// various kinds of imported and not used

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

