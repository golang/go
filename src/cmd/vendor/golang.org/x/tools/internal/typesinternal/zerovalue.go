// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strconv"
	"strings"
)

// ZeroString returns the string representation of the "zero" value of the type t.
// This string can be used on the right-hand side of an assignment where the
// left-hand side has that explicit type.
// Exception: This does not apply to tuples. Their string representation is
// informational only and cannot be used in an assignment.
// When assigning to a wider type (such as 'any'), it's the caller's
// responsibility to handle any necessary type conversions.
// See [ZeroExpr] for a variant that returns an [ast.Expr].
func ZeroString(t types.Type, qf types.Qualifier) string {
	switch t := t.(type) {
	case *types.Basic:
		switch {
		case t.Info()&types.IsBoolean != 0:
			return "false"
		case t.Info()&types.IsNumeric != 0:
			return "0"
		case t.Info()&types.IsString != 0:
			return `""`
		case t.Kind() == types.UnsafePointer:
			fallthrough
		case t.Kind() == types.UntypedNil:
			return "nil"
		default:
			panic(fmt.Sprint("ZeroString for unexpected type:", t))
		}

	case *types.Pointer, *types.Slice, *types.Interface, *types.Chan, *types.Map, *types.Signature:
		return "nil"

	case *types.Named, *types.Alias:
		switch under := t.Underlying().(type) {
		case *types.Struct, *types.Array:
			return types.TypeString(t, qf) + "{}"
		default:
			return ZeroString(under, qf)
		}

	case *types.Array, *types.Struct:
		return types.TypeString(t, qf) + "{}"

	case *types.TypeParam:
		// Assumes func new is not shadowed.
		return "*new(" + types.TypeString(t, qf) + ")"

	case *types.Tuple:
		// Tuples are not normal values.
		// We are currently format as "(t[0], ..., t[n])". Could be something else.
		components := make([]string, t.Len())
		for i := 0; i < t.Len(); i++ {
			components[i] = ZeroString(t.At(i).Type(), qf)
		}
		return "(" + strings.Join(components, ", ") + ")"

	case *types.Union:
		// Variables of these types cannot be created, so it makes
		// no sense to ask for their zero value.
		panic(fmt.Sprintf("invalid type for a variable: %v", t))

	default:
		panic(t) // unreachable.
	}
}

// ZeroExpr returns the ast.Expr representation of the "zero" value of the type t.
// ZeroExpr is defined for types that are suitable for variables.
// It may panic for other types such as Tuple or Union.
// See [ZeroString] for a variant that returns a string.
func ZeroExpr(f *ast.File, pkg *types.Package, typ types.Type) ast.Expr {
	switch t := typ.(type) {
	case *types.Basic:
		switch {
		case t.Info()&types.IsBoolean != 0:
			return &ast.Ident{Name: "false"}
		case t.Info()&types.IsNumeric != 0:
			return &ast.BasicLit{Kind: token.INT, Value: "0"}
		case t.Info()&types.IsString != 0:
			return &ast.BasicLit{Kind: token.STRING, Value: `""`}
		case t.Kind() == types.UnsafePointer:
			fallthrough
		case t.Kind() == types.UntypedNil:
			return ast.NewIdent("nil")
		default:
			panic(fmt.Sprint("ZeroExpr for unexpected type:", t))
		}

	case *types.Pointer, *types.Slice, *types.Interface, *types.Chan, *types.Map, *types.Signature:
		return ast.NewIdent("nil")

	case *types.Named, *types.Alias:
		switch under := t.Underlying().(type) {
		case *types.Struct, *types.Array:
			return &ast.CompositeLit{
				Type: TypeExpr(f, pkg, typ),
			}
		default:
			return ZeroExpr(f, pkg, under)
		}

	case *types.Array, *types.Struct:
		return &ast.CompositeLit{
			Type: TypeExpr(f, pkg, typ),
		}

	case *types.TypeParam:
		return &ast.StarExpr{ // *new(T)
			X: &ast.CallExpr{
				// Assumes func new is not shadowed.
				Fun: ast.NewIdent("new"),
				Args: []ast.Expr{
					ast.NewIdent(t.Obj().Name()),
				},
			},
		}

	case *types.Tuple:
		// Unlike ZeroString, there is no ast.Expr can express tuple by
		// "(t[0], ..., t[n])".
		panic(fmt.Sprintf("invalid type for a variable: %v", t))

	case *types.Union:
		// Variables of these types cannot be created, so it makes
		// no sense to ask for their zero value.
		panic(fmt.Sprintf("invalid type for a variable: %v", t))

	default:
		panic(t) // unreachable.
	}
}

// IsZeroExpr uses simple syntactic heuristics to report whether expr
// is a obvious zero value, such as 0, "", nil, or false.
// It cannot do better without type information.
func IsZeroExpr(expr ast.Expr) bool {
	switch e := expr.(type) {
	case *ast.BasicLit:
		return e.Value == "0" || e.Value == `""`
	case *ast.Ident:
		return e.Name == "nil" || e.Name == "false"
	default:
		return false
	}
}

// TypeExpr returns syntax for the specified type. References to named types
// from packages other than pkg are qualified by an appropriate package name, as
// defined by the import environment of file.
// It may panic for types such as Tuple or Union.
func TypeExpr(f *ast.File, pkg *types.Package, typ types.Type) ast.Expr {
	switch t := typ.(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.UnsafePointer:
			// TODO(hxjiang): replace the implementation with types.Qualifier.
			return &ast.SelectorExpr{X: ast.NewIdent("unsafe"), Sel: ast.NewIdent("Pointer")}
		default:
			return ast.NewIdent(t.Name())
		}

	case *types.Pointer:
		return &ast.UnaryExpr{
			Op: token.MUL,
			X:  TypeExpr(f, pkg, t.Elem()),
		}

	case *types.Array:
		return &ast.ArrayType{
			Len: &ast.BasicLit{
				Kind:  token.INT,
				Value: fmt.Sprintf("%d", t.Len()),
			},
			Elt: TypeExpr(f, pkg, t.Elem()),
		}

	case *types.Slice:
		return &ast.ArrayType{
			Elt: TypeExpr(f, pkg, t.Elem()),
		}

	case *types.Map:
		return &ast.MapType{
			Key:   TypeExpr(f, pkg, t.Key()),
			Value: TypeExpr(f, pkg, t.Elem()),
		}

	case *types.Chan:
		dir := ast.ChanDir(t.Dir())
		if t.Dir() == types.SendRecv {
			dir = ast.SEND | ast.RECV
		}
		return &ast.ChanType{
			Dir:   dir,
			Value: TypeExpr(f, pkg, t.Elem()),
		}

	case *types.Signature:
		var params []*ast.Field
		for i := 0; i < t.Params().Len(); i++ {
			params = append(params, &ast.Field{
				Type: TypeExpr(f, pkg, t.Params().At(i).Type()),
				Names: []*ast.Ident{
					{
						Name: t.Params().At(i).Name(),
					},
				},
			})
		}
		if t.Variadic() {
			last := params[len(params)-1]
			last.Type = &ast.Ellipsis{Elt: last.Type.(*ast.ArrayType).Elt}
		}
		var returns []*ast.Field
		for i := 0; i < t.Results().Len(); i++ {
			returns = append(returns, &ast.Field{
				Type: TypeExpr(f, pkg, t.Results().At(i).Type()),
			})
		}
		return &ast.FuncType{
			Params: &ast.FieldList{
				List: params,
			},
			Results: &ast.FieldList{
				List: returns,
			},
		}

	case interface{ Obj() *types.TypeName }: // *types.{Alias,Named,TypeParam}
		switch t.Obj().Pkg() {
		case pkg, nil:
			return ast.NewIdent(t.Obj().Name())
		}
		pkgName := t.Obj().Pkg().Name()

		// TODO(hxjiang): replace the implementation with types.Qualifier.
		// If the file already imports the package under another name, use that.
		for _, cand := range f.Imports {
			if path, _ := strconv.Unquote(cand.Path.Value); path == t.Obj().Pkg().Path() {
				if cand.Name != nil && cand.Name.Name != "" {
					pkgName = cand.Name.Name
				}
			}
		}
		if pkgName == "." {
			return ast.NewIdent(t.Obj().Name())
		}
		return &ast.SelectorExpr{
			X:   ast.NewIdent(pkgName),
			Sel: ast.NewIdent(t.Obj().Name()),
		}

	case *types.Struct:
		return ast.NewIdent(t.String())

	case *types.Interface:
		return ast.NewIdent(t.String())

	case *types.Union:
		// TODO(hxjiang): handle the union through syntax (~A | ... | ~Z).
		// Remove nil check when calling typesinternal.TypeExpr.
		return nil

	case *types.Tuple:
		panic("invalid input type types.Tuple")

	default:
		panic("unreachable")
	}
}
