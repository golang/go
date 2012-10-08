// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of conversions.

package types

import (
	"go/ast"
)

// conversion typechecks the type conversion conv to type typ. iota is the current
// value of iota or -1 if iota doesn't have a value in the current context. The result
// of the conversion is returned via x. If the conversion has type errors, the returned
// x is marked as invalid (x.mode == invalid).
//
func (check *checker) conversion(x *operand, conv *ast.CallExpr, typ Type, iota int) {
	// all conversions have one argument
	if len(conv.Args) != 1 {
		check.invalidOp(conv.Pos(), "%s conversion requires exactly one argument", conv)
		goto Error
	}

	// evaluate argument
	check.expr(x, conv.Args[0], nil, iota)
	if x.mode == invalid {
		goto Error
	}

	// TODO(gri) fix this - implement all checks and constant evaluation
	x.mode = value
	x.expr = conv
	x.typ = typ
	return

Error:
	x.mode = invalid
}
