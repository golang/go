// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

// TODO(gri): absorb this into go/types.

import (
	"fmt"
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

// PackageInfo holds the ASTs and facts derived by the type-checker
// for a single package.
//
// Not mutated once constructed.
//
type PackageInfo struct {
	Pkg        *types.Package
	Importable bool        // true if 'import "Pkg.Path()"' would resolve to this
	Err        error       // non-nil if the package had static errors
	Files      []*ast.File // abstract syntax for the package's files
	types.Info             // type-checker deductions.
}

func (info *PackageInfo) String() string {
	return fmt.Sprintf("PackageInfo(%s)", info.Pkg.Path())
}

// TypeOf returns the type of expression e.
// Precondition: e belongs to the package's ASTs.
//
func (info *PackageInfo) TypeOf(e ast.Expr) types.Type {
	if t, ok := info.Types[e]; ok {
		return t
	}
	// Defining ast.Idents (id := expr) get only Ident callbacks
	// but not Expr callbacks.
	if id, ok := e.(*ast.Ident); ok {
		return info.ObjectOf(id).Type()
	}
	panic("no type for expression")
}

// ValueOf returns the value of expression e if it is a constant, nil
// otherwise.
// Precondition: e belongs to the package's ASTs.
//
func (info *PackageInfo) ValueOf(e ast.Expr) exact.Value {
	return info.Values[e]
}

// ObjectOf returns the typechecker object denoted by the specified id.
// Precondition: id belongs to the package's ASTs.
//
func (info *PackageInfo) ObjectOf(id *ast.Ident) types.Object {
	return info.Objects[id]
}

// IsType returns true iff expression e denotes a type.
// Precondition: e belongs to the package's ASTs.
//
// TODO(gri): move this into go/types.
//
func (info *PackageInfo) IsType(e ast.Expr) bool {
	switch e := e.(type) {
	case *ast.SelectorExpr: // pkg.Type
		if sel := info.Selections[e]; sel.Kind() == types.PackageObj {
			_, isType := sel.Obj().(*types.TypeName)
			return isType
		}
	case *ast.StarExpr: // *T
		return info.IsType(e.X)
	case *ast.Ident:
		_, isType := info.ObjectOf(e).(*types.TypeName)
		return isType
	case *ast.ArrayType, *ast.StructType, *ast.FuncType, *ast.InterfaceType, *ast.MapType, *ast.ChanType:
		return true
	case *ast.ParenExpr:
		return info.IsType(e.X)
	}
	return false
}

// TypeCaseVar returns the implicit variable created by a single-type
// case clause in a type switch, or nil if not found.
//
func (info *PackageInfo) TypeCaseVar(cc *ast.CaseClause) *types.Var {
	if v := info.Implicits[cc]; v != nil {
		return v.(*types.Var)
	}
	return nil
}

var (
	tEface      = new(types.Interface)
	tComplex64  = types.Typ[types.Complex64]
	tComplex128 = types.Typ[types.Complex128]
	tFloat32    = types.Typ[types.Float32]
	tFloat64    = types.Typ[types.Float64]
)

// BuiltinCallSignature returns a new Signature describing the
// effective type of a builtin operator for the particular call e.
//
// This requires ad-hoc typing rules for all variadic (append, print,
// println) and polymorphic (append, copy, delete, close) built-ins.
// This logic could be part of the typechecker, and should arguably
// be moved there and made accessible via an additional types.Context
// callback.
//
func (info *PackageInfo) BuiltinCallSignature(e *ast.CallExpr) *types.Signature {
	var params []*types.Var
	var isVariadic bool

	switch builtin := unparen(e.Fun).(*ast.Ident).Name; builtin {
	case "append":
		var t0, t1 types.Type
		t0 = info.TypeOf(e) // infer arg[0] type from result type
		if e.Ellipsis != 0 {
			// append(tslice, tslice...) []T
			// append(byteslice, "foo"...) []byte
			t1 = info.TypeOf(e.Args[1]) // no conversion
		} else {
			// append([]T, x, y, z) []T
			t1 = t0.Underlying()
			isVariadic = true
		}
		params = append(params,
			types.NewVar(token.NoPos, nil, "", t0),
			types.NewVar(token.NoPos, nil, "", t1))

	case "print", "println": // print{,ln}(any, ...interface{})
		isVariadic = true
		// Note, arg0 may have any type, not necessarily tEface.
		params = append(params,
			types.NewVar(token.NoPos, nil, "", info.TypeOf(e.Args[0])),
			types.NewVar(token.NoPos, nil, "", types.NewSlice(tEface)))

	case "close":
		params = append(params, types.NewVar(token.NoPos, nil, "", info.TypeOf(e.Args[0])))

	case "copy":
		// copy([]T, []T) int
		// Infer arg types from each other.  Sleazy.
		var st *types.Slice
		if t, ok := info.TypeOf(e.Args[0]).Underlying().(*types.Slice); ok {
			st = t
		} else if t, ok := info.TypeOf(e.Args[1]).Underlying().(*types.Slice); ok {
			st = t
		} else {
			panic("cannot infer types in call to copy()")
		}
		stvar := types.NewVar(token.NoPos, nil, "", st)
		params = append(params, stvar, stvar)

	case "delete":
		// delete(map[K]V, K)
		tmap := info.TypeOf(e.Args[0])
		tkey := tmap.Underlying().(*types.Map).Key()
		params = append(params,
			types.NewVar(token.NoPos, nil, "", tmap),
			types.NewVar(token.NoPos, nil, "", tkey))

	case "len", "cap":
		params = append(params, types.NewVar(token.NoPos, nil, "", info.TypeOf(e.Args[0])))

	case "real", "imag":
		// Reverse conversion to "complex" case below.
		var argType types.Type
		switch info.TypeOf(e).(*types.Basic).Kind() {
		case types.UntypedFloat:
			argType = types.Typ[types.UntypedComplex]
		case types.Float64:
			argType = tComplex128
		case types.Float32:
			argType = tComplex64
		default:
			unreachable()
		}
		params = append(params, types.NewVar(token.NoPos, nil, "", argType))

	case "complex":
		var argType types.Type
		switch info.TypeOf(e).(*types.Basic).Kind() {
		case types.UntypedComplex:
			argType = types.Typ[types.UntypedFloat]
		case types.Complex128:
			argType = tFloat64
		case types.Complex64:
			argType = tFloat32
		default:
			unreachable()
		}
		v := types.NewVar(token.NoPos, nil, "", argType)
		params = append(params, v, v)

	case "panic":
		params = append(params, types.NewVar(token.NoPos, nil, "", tEface))

	case "recover":
		// no params

	default:
		panic("unknown builtin: " + builtin)
	}

	return types.NewSignature(nil, nil, types.NewTuple(params...), nil, isVariadic)
}
