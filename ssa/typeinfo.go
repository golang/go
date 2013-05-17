package ssa

// This file defines utilities for querying the results of typechecker:
// types of expressions, values of constant expressions, referents of identifiers.

import (
	"code.google.com/p/go.tools/go/types"
	"fmt"
	"go/ast"
)

// TypeInfo contains information provided by the type checker about
// the abstract syntax for a single package.
type TypeInfo struct {
	types     map[ast.Expr]types.Type     // inferred types of expressions
	constants map[ast.Expr]*Literal       // values of constant expressions
	idents    map[*ast.Ident]types.Object // canonical type objects for named entities
}

// TypeOf returns the type of expression e.
// Precondition: e belongs to the package's ASTs.
func (info *TypeInfo) TypeOf(e ast.Expr) types.Type {
	// For Ident, b.types may be more specific than
	// b.obj(id.(*ast.Ident)).GetType(),
	// e.g. in the case of typeswitch.
	if t, ok := info.types[e]; ok {
		return t
	}
	// The typechecker doesn't notify us of all Idents,
	// e.g. s.Key and s.Value in a RangeStmt.
	// So we have this fallback.
	// TODO(gri): This is a typechecker bug.  When fixed,
	// eliminate this case and panic.
	if id, ok := e.(*ast.Ident); ok {
		return info.ObjectOf(id).Type()
	}
	panic("no type for expression")
}

// ValueOf returns the value of expression e if it is a constant,
// nil otherwise.
//
func (info *TypeInfo) ValueOf(e ast.Expr) *Literal {
	return info.constants[e]
}

// ObjectOf returns the typechecker object denoted by the specified id.
// Precondition: id belongs to the package's ASTs.
//
func (info *TypeInfo) ObjectOf(id *ast.Ident) types.Object {
	if obj, ok := info.idents[id]; ok {
		return obj
	}
	panic(fmt.Sprintf("no types.Object for ast.Ident %s @ %p", id.Name, id))
}

// IsType returns true iff expression e denotes a type.
// Precondition: e belongs to the package's ASTs.
// e must be a true expression, not a KeyValueExpr, or an Ident
// appearing in a SelectorExpr or declaration.
//
func (info *TypeInfo) IsType(e ast.Expr) bool {
	switch e := e.(type) {
	case *ast.SelectorExpr: // pkg.Type
		if obj := info.isPackageRef(e); obj != nil {
			return objKind(obj) == ast.Typ
		}
	case *ast.StarExpr: // *T
		return info.IsType(e.X)
	case *ast.Ident:
		return objKind(info.ObjectOf(e)) == ast.Typ
	case *ast.ArrayType, *ast.StructType, *ast.FuncType, *ast.InterfaceType, *ast.MapType, *ast.ChanType:
		return true
	case *ast.ParenExpr:
		return info.IsType(e.X)
	}
	return false
}

// isPackageRef returns the identity of the object if sel is a
// package-qualified reference to a named const, var, func or type.
// Otherwise it returns nil.
// Precondition: sel belongs to the package's ASTs.
//
func (info *TypeInfo) isPackageRef(sel *ast.SelectorExpr) types.Object {
	if id, ok := sel.X.(*ast.Ident); ok {
		if obj := info.ObjectOf(id); objKind(obj) == ast.Pkg {
			return obj.(*types.Package).Scope().Lookup(sel.Sel.Name)
		}
	}
	return nil
}

// builtinCallSignature returns a new Signature describing the
// effective type of a builtin operator for the particular call e.
//
// This requires ad-hoc typing rules for all variadic (append, print,
// println) and polymorphic (append, copy, delete, close) built-ins.
// This logic could be part of the typechecker, and should arguably
// be moved there and made accessible via an additional types.Context
// callback.
//
// The returned Signature is degenerate and only intended for use by
// emitCallArgs.
//
func builtinCallSignature(info *TypeInfo, e *ast.CallExpr) *types.Signature {
	var params []*types.Var
	var isVariadic bool

	switch builtin := noparens(e.Fun).(*ast.Ident).Name; builtin {
	case "append":
		var t0, t1 types.Type
		t0 = info.TypeOf(e) // infer arg[0] type from result type
		if e.Ellipsis != 0 {
			// append([]T, []T) []T
			// append([]byte, string) []byte
			t1 = info.TypeOf(e.Args[1]) // no conversion
		} else {
			// append([]T, ...T) []T
			t1 = t0.Underlying().(*types.Slice).Elem()
			isVariadic = true
		}
		params = append(params,
			types.NewVar(nil, "", t0),
			types.NewVar(nil, "", t1))

	case "print", "println": // print{,ln}(any, ...interface{})
		isVariadic = true
		// Note, arg0 may have any type, not necessarily tEface.
		params = append(params,
			types.NewVar(nil, "", info.TypeOf(e.Args[0])),
			types.NewVar(nil, "", tEface))

	case "close":
		params = append(params, types.NewVar(nil, "", info.TypeOf(e.Args[0])))

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
		stvar := types.NewVar(nil, "", st)
		params = append(params, stvar, stvar)

	case "delete":
		// delete(map[K]V, K)
		tmap := info.TypeOf(e.Args[0])
		tkey := tmap.Underlying().(*types.Map).Key()
		params = append(params,
			types.NewVar(nil, "", tmap),
			types.NewVar(nil, "", tkey))

	case "len", "cap":
		params = append(params, types.NewVar(nil, "", info.TypeOf(e.Args[0])))

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
		params = append(params, types.NewVar(nil, "", argType))

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
		v := types.NewVar(nil, "", argType)
		params = append(params, v, v)

	case "panic":
		params = append(params, types.NewVar(nil, "", tEface))

	case "recover":
		// no params

	default:
		panic("unknown builtin: " + builtin)
	}

	return types.NewSignature(nil, types.NewTuple(params...), nil, isVariadic)
}
