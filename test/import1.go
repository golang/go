// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// check for import conflicts

package main

import "bufio"	// GCCGO_ERROR "previous|not used"
import bufio "os"	// ERROR "redeclared|redefinition|incompatible" "imported and not used"

import (
	"fmt"	// GCCGO_ERROR "previous|not used"
	fmt "math"	// ERROR "redeclared|redefinition|incompatible" "imported and not used"
)
