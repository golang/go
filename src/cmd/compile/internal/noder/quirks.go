// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"fmt"

	"cmd/compile/internal/syntax"
)

// typeExprEndPos returns the position that noder would leave base.Pos
// after parsing the given type expression.
//
// Deprecated: This function exists to emulate position semantics from
// Go 1.17, necessary for compatibility with the backend DWARF
// generation logic that assigns variables to their appropriate scope.
func typeExprEndPos(expr0 syntax.Expr) syntax.Pos {
	for {
		switch expr := expr0.(type) {
		case *syntax.Name:
			return expr.Pos()
		case *syntax.SelectorExpr:
			return expr.X.Pos()

		case *syntax.ParenExpr:
			expr0 = expr.X

		case *syntax.Operation:
			assert(expr.Op == syntax.Mul)
			assert(expr.Y == nil)
			expr0 = expr.X

		case *syntax.ArrayType:
			expr0 = expr.Elem
		case *syntax.ChanType:
			expr0 = expr.Elem
		case *syntax.DotsType:
			expr0 = expr.Elem
		case *syntax.MapType:
			expr0 = expr.Value
		case *syntax.SliceType:
			expr0 = expr.Elem

		case *syntax.StructType:
			return expr.Pos()

		case *syntax.InterfaceType:
			expr0 = lastFieldType(expr.MethodList)
			if expr0 == nil {
				return expr.Pos()
			}

		case *syntax.FuncType:
			expr0 = lastFieldType(expr.ResultList)
			if expr0 == nil {
				expr0 = lastFieldType(expr.ParamList)
				if expr0 == nil {
					return expr.Pos()
				}
			}

		case *syntax.IndexExpr: // explicit type instantiation
			targs := syntax.UnpackListExpr(expr.Index)
			expr0 = targs[len(targs)-1]

		default:
			panic(fmt.Sprintf("%s: unexpected type expression %v", expr.Pos(), syntax.String(expr)))
		}
	}
}

func lastFieldType(fields []*syntax.Field) syntax.Expr {
	if len(fields) == 0 {
		return nil
	}
	return fields[len(fields)-1].Type
}
