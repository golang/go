// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"context"
	"go/ast"
	"go/types"
)

// builtinArgKind determines the expected object kind for a builtin
// argument. It attempts to use the AST hints from builtin.go where
// possible.
func (c *completer) builtinArgKind(ctx context.Context, obj types.Object, call *ast.CallExpr) objKind {
	builtin, err := c.snapshot.BuiltinPackage(ctx)
	if err != nil {
		return 0
	}
	exprIdx := exprAtPos(c.pos, call.Args)

	builtinObj := builtin.Package.Scope.Lookup(obj.Name())
	if builtinObj == nil {
		return 0
	}
	decl, ok := builtinObj.Decl.(*ast.FuncDecl)
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
// function. parentInf is the inferred type info for the builtin
// call's parent node.
func (c *completer) builtinArgType(obj types.Object, call *ast.CallExpr, parentInf candidateInference) candidateInference {
	var (
		exprIdx = exprAtPos(c.pos, call.Args)

		// Propagate certain properties from our parent's inference.
		inf = candidateInference{
			typeName:  parentInf.typeName,
			modifiers: parentInf.modifiers,
		}
	)

	switch obj.Name() {
	case "append":
		if parentInf.objType == nil {
			break
		}

		inf.objType = parentInf.objType

		if exprIdx <= 0 {
			break
		}

		inf.objType = deslice(inf.objType)

		// Check if we are completing the variadic append() param.
		inf.variadic = exprIdx == 1 && len(call.Args) <= 2

		// Penalize the first append() argument as a candidate. You
		// don't normally append a slice to itself.
		if sliceChain := objChain(c.pkg.GetTypesInfo(), call.Args[0]); len(sliceChain) > 0 {
			inf.penalized = append(inf.penalized, penalizedObj{objChain: sliceChain, penalty: 0.9})
		}
	case "delete":
		if exprIdx > 0 && len(call.Args) > 0 {
			// Try to fill in expected type of map key.
			firstArgType := c.pkg.GetTypesInfo().TypeOf(call.Args[0])
			if firstArgType != nil {
				if mt, ok := firstArgType.Underlying().(*types.Map); ok {
					inf.objType = mt.Key()
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
			inf.objType = t1
		} else if exprIdx == 0 && t2 != nil {
			inf.objType = t2
		}
	case "new":
		inf.typeName.wantTypeName = true
		if parentInf.objType != nil {
			// Expected type for "new" is the de-pointered parent type.
			if ptr, ok := parentInf.objType.Underlying().(*types.Pointer); ok {
				inf.objType = ptr.Elem()
			}
		}
	case "make":
		if exprIdx == 0 {
			inf.typeName.wantTypeName = true
			inf.objType = parentInf.objType
		} else {
			inf.objType = types.Typ[types.UntypedInt]
		}
	}

	return inf
}
