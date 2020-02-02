// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"go/ast"
	"go/types"
)

// builtinArgKind determines the expected object kind for a builtin
// argument. It attempts to use the AST hints from builtin.go where
// possible.
func (c *completer) builtinArgKind(obj types.Object, call *ast.CallExpr) objKind {
	astObj, err := c.snapshot.View().LookupBuiltin(c.ctx, obj.Name())
	if err != nil {
		return 0
	}
	exprIdx := exprAtPos(c.pos, call.Args)

	decl, ok := astObj.Decl.(*ast.FuncDecl)
	if !ok || exprIdx >= len(decl.Type.Params.List) {
		return 0
	}

	switch ptyp := decl.Type.Params.List[exprIdx].Type.(type) {
	case *ast.ChanType:
		return kindChan
	case *ast.ArrayType:
		return kindSlice
	case *ast.MapType:
		return kindMap
	case *ast.Ident:
		switch ptyp.Name {
		case "Type":
			switch obj.Name() {
			case "make":
				return kindChan | kindSlice | kindMap
			case "len":
				return kindSlice | kindMap | kindArray | kindString | kindChan
			case "cap":
				return kindSlice | kindArray | kindChan
			}
		}
	}

	return 0
}

// builtinArgType infers the type of an argument to a builtin
// function. "parentType" is the inferred type for the builtin call's
// parent node.
func (c *completer) builtinArgType(obj types.Object, call *ast.CallExpr, parentType types.Type) (infType types.Type, wantType, variadic bool) {
	exprIdx := exprAtPos(c.pos, call.Args)

	switch obj.Name() {
	case "append":
		// Check if we are completing the variadic append() param.
		variadic = exprIdx == 1 && len(call.Args) <= 2
		infType = parentType

		// If we are completing an individual element of the variadic
		// param, "deslice" the expected type.
		if !variadic && exprIdx > 0 {
			if slice, ok := parentType.(*types.Slice); ok {
				infType = slice.Elem()
			}
		}
	case "delete":
		if exprIdx > 0 && len(call.Args) > 0 {
			// Try to fill in expected type of map key.
			firstArgType := c.pkg.GetTypesInfo().TypeOf(call.Args[0])
			if firstArgType != nil {
				if mt, ok := firstArgType.Underlying().(*types.Map); ok {
					infType = mt.Key()
				}
			}
		}
	case "copy":
		var t1, t2 types.Type
		if len(call.Args) > 0 {
			t1 = c.pkg.GetTypesInfo().TypeOf(call.Args[0])
			if len(call.Args) > 1 {
				t2 = c.pkg.GetTypesInfo().TypeOf(call.Args[1])
			}
		}

		// Fill in expected type of either arg if the other is already present.
		if exprIdx == 1 && t1 != nil {
			infType = t1
		} else if exprIdx == 0 && t2 != nil {
			infType = t2
		}
	case "new":
		wantType = true
		if parentType != nil {
			// Expected type for "new" is the de-pointered parent type.
			if ptr, ok := parentType.Underlying().(*types.Pointer); ok {
				infType = ptr.Elem()
			}
		}
	case "make":
		if exprIdx == 0 {
			wantType = true
			infType = parentType
		} else {
			infType = types.Typ[types.Int]
		}
	}

	return infType, wantType, variadic
}
