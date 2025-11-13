// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strings"
)

// ZeroString returns the string representation of the zero value for any type t.
// The boolean result indicates whether the type is or contains an invalid type
// or a non-basic (constraint) interface type.
//
// Even for invalid input types, ZeroString may return a partially correct
// string representation. The caller should use the returned isValid boolean
// to determine the validity of the expression.
//
// When assigning to a wider type (such as 'any'), it's the caller's
// responsibility to handle any necessary type conversions.
//
// This string can be used on the right-hand side of an assignment where the
// left-hand side has that explicit type.
// References to named types are qualified by an appropriate (optional)
// qualifier function.
// Exception: This does not apply to tuples. Their string representation is
// informational only and cannot be used in an assignment.
//
// See [ZeroExpr] for a variant that returns an [ast.Expr].
func ZeroString(t types.Type, qual types.Qualifier) (_ string, isValid bool) {
	switch t := t.(type) {
	case *types.Basic:
		switch {
		case t.Info()&types.IsBoolean != 0:
			return "false", true
		case t.Info()&types.IsNumeric != 0:
			return "0", true
		case t.Info()&types.IsString != 0:
			return `""`, true
		case t.Kind() == types.UnsafePointer:
			fallthrough
		case t.Kind() == types.UntypedNil:
			return "nil", true
		case t.Kind() == types.Invalid:
			return "invalid", false
		default:
			panic(fmt.Sprintf("ZeroString for unexpected type %v", t))
		}

	case *types.Pointer, *types.Slice, *types.Chan, *types.Map, *types.Signature:
		return "nil", true

	case *types.Interface:
		if !t.IsMethodSet() {
			return "invalid", false
		}
		return "nil", true

	case *types.Named:
		switch under := t.Underlying().(type) {
		case *types.Struct, *types.Array:
			return types.TypeString(t, qual) + "{}", true
		default:
			return ZeroString(under, qual)
		}

	case *types.Alias:
		switch t.Underlying().(type) {
		case *types.Struct, *types.Array:
			return types.TypeString(t, qual) + "{}", true
		default:
			// A type parameter can have alias but alias type's underlying type
			// can never be a type parameter.
			// Use types.Unalias to preserve the info of type parameter instead
			// of call Underlying() going right through and get the underlying
			// type of the type parameter which is always an interface.
			return ZeroString(types.Unalias(t), qual)
		}

	case *types.Array, *types.Struct:
		return types.TypeString(t, qual) + "{}", true

	case *types.TypeParam:
		// Assumes func new is not shadowed.
		return "*new(" + types.TypeString(t, qual) + ")", true

	case *types.Tuple:
		// Tuples are not normal values.
		// We are currently format as "(t[0], ..., t[n])". Could be something else.
		isValid := true
		components := make([]string, t.Len())
		for i := 0; i < t.Len(); i++ {
			comp, ok := ZeroString(t.At(i).Type(), qual)

			components[i] = comp
			isValid = isValid && ok
		}
		return "(" + strings.Join(components, ", ") + ")", isValid

	case *types.Union:
		// Variables of these types cannot be created, so it makes
		// no sense to ask for their zero value.
		panic(fmt.Sprintf("invalid type for a variable: %v", t))

	default:
		panic(t) // unreachable.
	}
}

// ZeroExpr returns the ast.Expr representation of the zero value for any type t.
// The boolean result indicates whether the type is or contains an invalid type
// or a non-basic (constraint) interface type.
//
// Even for invalid input types, ZeroExpr may return a partially correct ast.Expr
// representation. The caller should use the returned isValid boolean to determine
// the validity of the expression.
//
// This function is designed for types suitable for variables and should not be
// used with Tuple or Union types.References to named types are qualified by an
// appropriate (optional) qualifier function.
//
// See [ZeroString] for a variant that returns a string.
func ZeroExpr(t types.Type, qual types.Qualifier) (_ ast.Expr, isValid bool) {
	switch t := t.(type) {
	case *types.Basic:
		switch {
		case t.Info()&types.IsBoolean != 0:
			return &ast.Ident{Name: "false"}, true
		case t.Info()&types.IsNumeric != 0:
			return &ast.BasicLit{Kind: token.INT, Value: "0"}, true
		case t.Info()&types.IsString != 0:
			return &ast.BasicLit{Kind: token.STRING, Value: `""`}, true
		case t.Kind() == types.UnsafePointer:
			fallthrough
		case t.Kind() == types.UntypedNil:
			return ast.NewIdent("nil"), true
		case t.Kind() == types.Invalid:
			return &ast.BasicLit{Kind: token.STRING, Value: `"invalid"`}, false
		default:
			panic(fmt.Sprintf("ZeroExpr for unexpected type %v", t))
		}

	case *types.Pointer, *types.Slice, *types.Chan, *types.Map, *types.Signature:
		return ast.NewIdent("nil"), true

	case *types.Interface:
		if !t.IsMethodSet() {
			return &ast.BasicLit{Kind: token.STRING, Value: `"invalid"`}, false
		}
		return ast.NewIdent("nil"), true

	case *types.Named:
		switch under := t.Underlying().(type) {
		case *types.Struct, *types.Array:
			return &ast.CompositeLit{
				Type: TypeExpr(t, qual),
			}, true
		default:
			return ZeroExpr(under, qual)
		}

	case *types.Alias:
		switch t.Underlying().(type) {
		case *types.Struct, *types.Array:
			return &ast.CompositeLit{
				Type: TypeExpr(t, qual),
			}, true
		default:
			return ZeroExpr(types.Unalias(t), qual)
		}

	case *types.Array, *types.Struct:
		return &ast.CompositeLit{
			Type: TypeExpr(t, qual),
		}, true

	case *types.TypeParam:
		return &ast.StarExpr{ // *new(T)
			X: &ast.CallExpr{
				// Assumes func new is not shadowed.
				Fun: ast.NewIdent("new"),
				Args: []ast.Expr{
					ast.NewIdent(t.Obj().Name()),
				},
			},
		}, true

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

// TypeExpr returns syntax for the specified type. References to named types
// are qualified by an appropriate (optional) qualifier function.
// It may panic for types such as Tuple or Union.
//
// See also https://go.dev/issues/75604, which will provide a robust
// Type-to-valid-Go-syntax formatter.
func TypeExpr(t types.Type, qual types.Qualifier) ast.Expr {
	switch t := t.(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.UnsafePointer:
			return &ast.SelectorExpr{X: ast.NewIdent(qual(types.NewPackage("unsafe", "unsafe"))), Sel: ast.NewIdent("Pointer")}
		default:
			return ast.NewIdent(t.Name())
		}

	case *types.Pointer:
		return &ast.UnaryExpr{
			Op: token.MUL,
			X:  TypeExpr(t.Elem(), qual),
		}

	case *types.Array:
		return &ast.ArrayType{
			Len: &ast.BasicLit{
				Kind:  token.INT,
				Value: fmt.Sprintf("%d", t.Len()),
			},
			Elt: TypeExpr(t.Elem(), qual),
		}

	case *types.Slice:
		return &ast.ArrayType{
			Elt: TypeExpr(t.Elem(), qual),
		}

	case *types.Map:
		return &ast.MapType{
			Key:   TypeExpr(t.Key(), qual),
			Value: TypeExpr(t.Elem(), qual),
		}

	case *types.Chan:
		dir := ast.ChanDir(t.Dir())
		if t.Dir() == types.SendRecv {
			dir = ast.SEND | ast.RECV
		}
		return &ast.ChanType{
			Dir:   dir,
			Value: TypeExpr(t.Elem(), qual),
		}

	case *types.Signature:
		var params []*ast.Field
		for i := 0; i < t.Params().Len(); i++ {
			params = append(params, &ast.Field{
				Type: TypeExpr(t.Params().At(i).Type(), qual),
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
				Type: TypeExpr(t.Results().At(i).Type(), qual),
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

	case *types.TypeParam:
		pkgName := qual(t.Obj().Pkg())
		if pkgName == "" || t.Obj().Pkg() == nil {
			return ast.NewIdent(t.Obj().Name())
		}
		return &ast.SelectorExpr{
			X:   ast.NewIdent(pkgName),
			Sel: ast.NewIdent(t.Obj().Name()),
		}

	// types.TypeParam also implements interface NamedOrAlias. To differentiate,
	// case TypeParam need to be present before case NamedOrAlias.
	// TODO(hxjiang): remove this comment once TypeArgs() is added to interface
	// NamedOrAlias.
	case NamedOrAlias:
		var expr ast.Expr = ast.NewIdent(t.Obj().Name())
		if pkgName := qual(t.Obj().Pkg()); pkgName != "." && pkgName != "" {
			expr = &ast.SelectorExpr{
				X:   ast.NewIdent(pkgName),
				Sel: expr.(*ast.Ident),
			}
		}

		// TODO(hxjiang): call t.TypeArgs after adding method TypeArgs() to
		// typesinternal.NamedOrAlias.
		if hasTypeArgs, ok := t.(interface{ TypeArgs() *types.TypeList }); ok {
			if typeArgs := hasTypeArgs.TypeArgs(); typeArgs != nil && typeArgs.Len() > 0 {
				var indices []ast.Expr
				for i := range typeArgs.Len() {
					indices = append(indices, TypeExpr(typeArgs.At(i), qual))
				}
				expr = &ast.IndexListExpr{
					X:       expr,
					Indices: indices,
				}
			}
		}

		return expr

	case *types.Struct:
		return ast.NewIdent(t.String())

	case *types.Interface:
		return ast.NewIdent(t.String())

	case *types.Union:
		if t.Len() == 0 {
			panic("Union type should have at least one term")
		}
		// Same as go/ast, the return expression will put last term in the
		// Y field at topmost level of BinaryExpr.
		// For union of type "float32 | float64 | int64", the structure looks
		// similar to:
		// {
		// 	X: {
		// 		X: float32,
		// 		Op: |
		// 		Y: float64,
		// 	}
		// 	Op: |,
		// 	Y: int64,
		// }
		var union ast.Expr
		for i := range t.Len() {
			term := t.Term(i)
			termExpr := TypeExpr(term.Type(), qual)
			if term.Tilde() {
				termExpr = &ast.UnaryExpr{
					Op: token.TILDE,
					X:  termExpr,
				}
			}
			if i == 0 {
				union = termExpr
			} else {
				union = &ast.BinaryExpr{
					X:  union,
					Op: token.OR,
					Y:  termExpr,
				}
			}
		}
		return union

	case *types.Tuple:
		panic("invalid input type types.Tuple")

	default:
		panic("unreachable")
	}
}
