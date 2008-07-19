// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST

import Globals "globals"
import Universe "universe"


export Expr
type Expr struct {
	typ *Globals.Type;
	op int;
	x, y *Expr;
}


export Stat
type Stat struct {
	// To be completed
}
