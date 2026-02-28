// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Most of the errors below are actually produced by the parser, but we check
// them here for consistency with the types2 tests.

package p

import ; /* ERROR invalid import path */ /* ERROR expected 'STRING' */
import // ERROR expected ';'
var _ int
import /* ERROR expected declaration */ .;

import ()
import (.)
import (
	"fmt"
	.
)

var _ = fmt /* ERROR "undeclared name" */ .Println
