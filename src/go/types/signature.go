// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/internal/typeparams"
	"go/token"
)

// funcType type-checks a function or method type.
func (check *Checker) funcType(sig *Signature, recvPar *ast.FieldList, ftyp *ast.FuncType) {
	check.openScope(ftyp, "function")
	check.scope.isFunc = true
	check.recordScope(ftyp, check.scope)
	sig.scope = check.scope
	defer check.closeScope()

	var recvTyp ast.Expr // rewritten receiver type; valid if != nil
	if recvPar != nil && len(recvPar.List) > 0 {
		// collect generic receiver type parameters, if any
		// - a receiver type parameter is like any other type parameter, except that it is declared implicitly
		// - the receiver specification acts as local declaration for its type parameters, which may be blank
		_, rname, rparams := check.unpackRecv(recvPar.List[0].Type, true)
		if len(rparams) > 0 {
			// Blank identifiers don't get declared and regular type-checking of the instantiated
			// parameterized receiver type expression fails in Checker.collectParams of receiver.
			// Identify blank type parameters and substitute each with a unique new identifier named
			// "n_" (where n is the parameter index) and which cannot conflict with any user-defined
			// name.
			var smap map[*ast.Ident]*ast.Ident // substitution map from "_" to "n_" identifiers
			for i, p := range rparams {
				if p.Name == "_" {
					new := *p
					new.Name = fmt.Sprintf("%d_", i)
					rparams[i] = &new // use n_ identifier instead of _ so it can be looked up
					if smap == nil {
						smap = make(map[*ast.Ident]*ast.Ident)
					}
					smap[p] = &new
				}
			}
			if smap != nil {
				// blank identifiers were found => use rewritten receiver type
				recvTyp = isubst(recvPar.List[0].Type, smap)
			}
			sig.rparams = check.declareTypeParams(nil, rparams)
			// determine receiver type to get its type parameters
			// and the respective type parameter bounds
			var recvTParams []*TypeName
			if rname != nil {
				// recv should be a Named type (otherwise an error is reported elsewhere)
				// Also: Don't report an error via genericType since it will be reported
				//       again when we type-check the signature.
				// TODO(gri) maybe the receiver should be marked as invalid instead?
				if recv := asNamed(check.genericType(rname, false)); recv != nil {
					recvTParams = recv.tparams
				}
			}
			// provide type parameter bounds
			// - only do this if we have the right number (otherwise an error is reported elsewhere)
			if len(sig.rparams) == len(recvTParams) {
				// We have a list of *TypeNames but we need a list of Types.
				list := make([]Type, len(sig.rparams))
				for i, t := range sig.rparams {
					list[i] = t.typ
				}
				smap := makeSubstMap(recvTParams, list)
				for i, tname := range sig.rparams {
					bound := recvTParams[i].typ.(*_TypeParam).bound
					// bound is (possibly) parameterized in the context of the
					// receiver type declaration. Substitute parameters for the
					// current context.
					// TODO(gri) should we assume now that bounds always exist?
					//           (no bound == empty interface)
					if bound != nil {
						bound = check.subst(tname.pos, bound, smap)
						tname.typ.(*_TypeParam).bound = bound
					}
				}
			}
		}
	}

	if tparams := typeparams.Get(ftyp); tparams != nil {
		sig.tparams = check.collectTypeParams(tparams)
		// Always type-check method type parameters but complain that they are not allowed.
		// (A separate check is needed when type-checking interface method signatures because
		// they don't have a receiver specification.)
		if recvPar != nil {
			check.errorf(tparams, _Todo, "methods cannot have type parameters")
		}
	}

	// Value (non-type) parameters' scope starts in the function body. Use a temporary scope for their
	// declarations and then squash that scope into the parent scope (and report any redeclarations at
	// that time).
	scope := NewScope(check.scope, token.NoPos, token.NoPos, "function body (temp. scope)")
	recvList, _ := check.collectParams(scope, recvPar, recvTyp, false) // use rewritten receiver type, if any
	params, variadic := check.collectParams(scope, ftyp.Params, nil, true)
	results, _ := check.collectParams(scope, ftyp.Results, nil, false)
	scope.squash(func(obj, alt Object) {
		check.errorf(obj, _DuplicateDecl, "%s redeclared in this block", obj.Name())
		check.reportAltDecl(alt)
	})

	if recvPar != nil {
		// recv parameter list present (may be empty)
		// spec: "The receiver is specified via an extra parameter section preceding the
		// method name. That parameter section must declare a single parameter, the receiver."
		var recv *Var
		switch len(recvList) {
		case 0:
			// error reported by resolver
			recv = NewParam(0, nil, "", Typ[Invalid]) // ignore recv below
		default:
			// more than one receiver
			check.error(recvList[len(recvList)-1], _BadRecv, "method must have exactly one receiver")
			fallthrough // continue with first receiver
		case 1:
			recv = recvList[0]
		}

		// TODO(gri) We should delay rtyp expansion to when we actually need the
		//           receiver; thus all checks here should be delayed to later.
		rtyp, _ := deref(recv.typ)
		rtyp = expand(rtyp)

		// spec: "The receiver type must be of the form T or *T where T is a type name."
		// (ignore invalid types - error was reported before)
		if t := rtyp; t != Typ[Invalid] {
			var err string
			if T := asNamed(t); T != nil {
				// spec: "The type denoted by T is called the receiver base type; it must not
				// be a pointer or interface type and it must be declared in the same package
				// as the method."
				if T.obj.pkg != check.pkg {
					err = "type not defined in this package"
				} else {
					switch u := optype(T).(type) {
					case *Basic:
						// unsafe.Pointer is treated like a regular pointer
						if u.kind == UnsafePointer {
							err = "unsafe.Pointer"
						}
					case *Pointer, *Interface:
						err = "pointer or interface type"
					}
				}
			} else {
				err = "basic or unnamed type"
			}
			if err != "" {
				check.errorf(recv, _InvalidRecv, "invalid receiver %s (%s)", recv.typ, err)
				// ok to continue
			}
		}
		sig.recv = recv
	}

	sig.params = NewTuple(params...)
	sig.results = NewTuple(results...)
	sig.variadic = variadic
}

// collectParams declares the parameters of list in scope and returns the corresponding
// variable list. If type0 != nil, it is used instead of the first type in list.
func (check *Checker) collectParams(scope *Scope, list *ast.FieldList, type0 ast.Expr, variadicOk bool) (params []*Var, variadic bool) {
	if list == nil {
		return
	}

	var named, anonymous bool
	for i, field := range list.List {
		ftype := field.Type
		if i == 0 && type0 != nil {
			ftype = type0
		}
		if t, _ := ftype.(*ast.Ellipsis); t != nil {
			ftype = t.Elt
			if variadicOk && i == len(list.List)-1 && len(field.Names) <= 1 {
				variadic = true
			} else {
				check.softErrorf(t, _MisplacedDotDotDot, "can only use ... with final parameter in list")
				// ignore ... and continue
			}
		}
		typ := check.varType(ftype)
		// The parser ensures that f.Tag is nil and we don't
		// care if a constructed AST contains a non-nil tag.
		if len(field.Names) > 0 {
			// named parameter
			for _, name := range field.Names {
				if name.Name == "" {
					check.invalidAST(name, "anonymous parameter")
					// ok to continue
				}
				par := NewParam(name.Pos(), check.pkg, name.Name, typ)
				check.declare(scope, name, par, scope.pos)
				params = append(params, par)
			}
			named = true
		} else {
			// anonymous parameter
			par := NewParam(ftype.Pos(), check.pkg, "", typ)
			check.recordImplicit(field, par)
			params = append(params, par)
			anonymous = true
		}
	}

	if named && anonymous {
		check.invalidAST(list, "list contains both named and anonymous parameters")
		// ok to continue
	}

	// For a variadic function, change the last parameter's type from T to []T.
	// Since we type-checked T rather than ...T, we also need to retro-actively
	// record the type for ...T.
	if variadic {
		last := params[len(params)-1]
		last.typ = &Slice{elem: last.typ}
		check.recordTypeAndValue(list.List[len(list.List)-1].Type, typexpr, last.typ, nil)
	}

	return
}

// isubst returns an x with identifiers substituted per the substitution map smap.
// isubst only handles the case of (valid) method receiver type expressions correctly.
func isubst(x ast.Expr, smap map[*ast.Ident]*ast.Ident) ast.Expr {
	switch n := x.(type) {
	case *ast.Ident:
		if alt := smap[n]; alt != nil {
			return alt
		}
	case *ast.StarExpr:
		X := isubst(n.X, smap)
		if X != n.X {
			new := *n
			new.X = X
			return &new
		}
	case *ast.IndexExpr:
		elems := typeparams.UnpackExpr(n.Index)
		var newElems []ast.Expr
		for i, elem := range elems {
			new := isubst(elem, smap)
			if new != elem {
				if newElems == nil {
					newElems = make([]ast.Expr, len(elems))
					copy(newElems, elems)
				}
				newElems[i] = new
			}
		}
		if newElems != nil {
			index := typeparams.PackExpr(newElems)
			new := *n
			new.Index = index
			return &new
		}
	case *ast.ParenExpr:
		return isubst(n.X, smap) // no need to keep parentheses
	default:
		// Other receiver type expressions are invalid.
		// It's fine to ignore those here as they will
		// be checked elsewhere.
	}
	return x
}
