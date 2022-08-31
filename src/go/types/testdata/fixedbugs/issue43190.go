// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The errors below are produced by the parser, but we check
// them here for consistency with the types2 tests.

package p

import ; // ERROR missing import path
import ';' // ERROR import path must be a string
// TODO(gri) The parser should accept mixing imports with other
//           top-level declarations for better error recovery.
// var _ int
import . ; //  ERROR missing import path

import ()
import (.) // ERROR missing import path
import (
	"fmt"
	.
) // ERROR missing import path

var _ = fmt.Println
