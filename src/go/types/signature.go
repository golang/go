// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/token"
	. "internal/types/errors"
)

// ----------------------------------------------------------------------------
// API

// A Signature represents a (non-builtin) function or method type.
// The receiver is ignored when comparing signatures for identity.
type Signature struct {
	// We need to keep the scope in Signature (rather than passing it around
	// and store it in the Func Object) because when type-checking a function
	// literal we call the general type checker which returns a general Type.
	// We then unpack the *Signature and use the scope for the literal body.
	rparams  *TypeParamList // receiver type parameters from left to right, or nil
	tparams  *TypeParamList // type parameters from left to right, or nil
	scope    *Scope         // function scope for package-local and non-instantiated signatures; nil otherwise
	recv     *Var           // nil if not a method
	params   *Tuple         // (incoming) parameters from left to right; or nil
	results  *Tuple         // (outgoing) results from left to right; or nil
	variadic bool           // true if the last parameter's type is of the form ...T (or string, for append built-in only)
}

// NewSignature returns a new function type for the given receiver, parameters,
// and results, either of which may be nil. If variadic is set, the function
// is variadic, it must have at least one parameter, and the last parameter
// must be of unnamed slice type.
//
// Deprecated: Use [NewSignatureType] instead which allows for type parameters.
func NewSignature(recv *Var, params, results *Tuple, variadic bool) *Signature {
	return NewSignatureType(recv, nil, nil, params, results, variadic)
}

// NewSignatureType creates a new function type for the given receiver,
// receiver type parameters, type parameters, parameters, and results. If
// variadic is set, params must hold at least one parameter and the last
// parameter's core type must be of unnamed slice or bytestring type.
// If recv is non-nil, typeParams must be empty. If recvTypeParams is
// non-empty, recv must be non-nil.
func NewSignatureType(recv *Var, recvTypeParams, typeParams []*TypeParam, params, results *Tuple, variadic bool) *Signature {
	if variadic {
		n := params.Len()
		if n == 0 {
			panic("variadic function must have at least one parameter")
		}
		core := coreString(params.At(n - 1).typ)
		if _, ok := core.(*Slice); !ok && !isString(core) {
			panic(fmt.Sprintf("got %s, want variadic parameter with unnamed slice type or string as core type", core.String()))
		}
	}
	sig := &Signature{recv: recv, params: params, results: results, variadic: variadic}
	if len(recvTypeParams) != 0 {
		if recv == nil {
			panic("function with receiver type parameters must have a receiver")
		}
		sig.rparams = bindTParams(recvTypeParams)
	}
	if len(typeParams) != 0 {
		if recv != nil {
			panic("function with type parameters cannot have a receiver")
		}
		sig.tparams = bindTParams(typeParams)
	}
	return sig
}

// Recv returns the receiver of signature s (if a method), or nil if a
// function. It is ignored when comparing signatures for identity.
//
// For an abstract method, Recv returns the enclosing interface either
// as a *[Named] or an *[Interface]. Due to embedding, an interface may
// contain methods whose receiver type is a different interface.
func (s *Signature) Recv() *Var { return s.recv }

// TypeParams returns the type parameters of signature s, or nil.
func (s *Signature) TypeParams() *TypeParamList { return s.tparams }

// RecvTypeParams returns the receiver type parameters of signature s, or nil.
func (s *Signature) RecvTypeParams() *TypeParamList { return s.rparams }

// Params returns the parameters of signature s, or nil.
func (s *Signature) Params() *Tuple { return s.params }

// Results returns the results of signature s, or nil.
func (s *Signature) Results() *Tuple { return s.results }

// Variadic reports whether the signature s is variadic.
func (s *Signature) Variadic() bool { return s.variadic }

func (t *Signature) Underlying() Type { return t }
func (t *Signature) String() string   { return TypeString(t, nil) }

// ----------------------------------------------------------------------------
// Implementation

// funcType type-checks a function or method type.
func (check *Checker) funcType(sig *Signature, recvPar *ast.FieldList, ftyp *ast.FuncType) {
	check.openScope(ftyp, "function")
	check.scope.isFunc = true
	check.recordScope(ftyp, check.scope)
	sig.scope = check.scope
	defer check.closeScope()

	if recvPar != nil && len(recvPar.List) > 0 {
		// collect generic receiver type parameters, if any
		// - a receiver type parameter is like any other type parameter, except that it is declared implicitly
		// - the receiver specification acts as local declaration for its type parameters, which may be blank
		_, rname, rparams := check.unpackRecv(recvPar.List[0].Type, true)
		if len(rparams) > 0 {
			// The scope of the type parameter T in "func (r T[T]) f()"
			// starts after f, not at "r"; see #52038.
			scopePos := ftyp.Params.Pos()
			tparams := check.declareTypeParams(nil, rparams, scopePos)
			sig.rparams = bindTParams(tparams)
			// Blank identifiers don't get declared, so naive type-checking of the
			// receiver type expression would fail in Checker.collectParams below,
			// when Checker.ident cannot resolve the _ to a type.
			//
			// Checker.recvTParamMap maps these blank identifiers to their type parameter
			// types, so that they may be resolved in Checker.ident when they fail
			// lookup in the scope.
			for i, p := range rparams {
				if p.Name == "_" {
					if check.recvTParamMap == nil {
						check.recvTParamMap = make(map[*ast.Ident]*TypeParam)
					}
					check.recvTParamMap[p] = tparams[i]
				}
			}
			// determine receiver type to get its type parameters
			// and the respective type parameter bounds
			var recvTParams []*TypeParam
			if rname != nil {
				// recv should be a Named type (otherwise an error is reported elsewhere)
				// Also: Don't report an error via genericType since it will be reported
				//       again when we type-check the signature.
				// TODO(gri) maybe the receiver should be marked as invalid instead?
				if recv := asNamed(check.genericType(rname, nil)); recv != nil {
					recvTParams = recv.TypeParams().list()
				}
			}
			// provide type parameter bounds
			if len(tparams) == len(recvTParams) {
				smap := makeRenameMap(recvTParams, tparams)
				for i, tpar := range tparams {
					recvTPar := recvTParams[i]
					check.mono.recordCanon(tpar, recvTPar)
					// recvTPar.bound is (possibly) parameterized in the context of the
					// receiver type declaration. Substitute parameters for the current
					// context.
					tpar.bound = check.subst(tpar.obj.pos, recvTPar.bound, smap, nil, check.context())
				}
			} else if len(tparams) < len(recvTParams) {
				// Reporting an error here is a stop-gap measure to avoid crashes in the
				// compiler when a type parameter/argument cannot be inferred later. It
				// may lead to follow-on errors (see issues go.dev/issue/51339, go.dev/issue/51343).
				// TODO(gri) find a better solution
				got := measure(len(tparams), "type parameter")
				check.errorf(recvPar, BadRecv, "got %s, but receiver base type declares %d", got, len(recvTParams))
			}
		}
	}

	if ftyp.TypeParams != nil {
		check.collectTypeParams(&sig.tparams, ftyp.TypeParams)
		// Always type-check method type parameters but complain that they are not allowed.
		// (A separate check is needed when type-checking interface method signatures because
		// they don't have a receiver specification.)
		if recvPar != nil {
			check.error(ftyp.TypeParams, InvalidMethodTypeParams, "methods cannot have type parameters")
		}
	}

	// Use a temporary scope for all parameter declarations and then
	// squash that scope into the parent scope (and report any
	// redeclarations at that time).
	//
	// TODO(adonovan): now that each declaration has the correct
	// scopePos, there should be no need for scope squashing.
	// Audit to ensure all lookups honor scopePos and simplify.
	scope := NewScope(check.scope, nopos, nopos, "function body (temp. scope)")
	scopePos := ftyp.End() // all parameters' scopes start after the signature
	recvList, _ := check.collectParams(scope, recvPar, false, scopePos)
	params, variadic := check.collectParams(scope, ftyp.Params, true, scopePos)
	results, _ := check.collectParams(scope, ftyp.Results, false, scopePos)
	scope.squash(func(obj, alt Object) {
		check.errorf(obj, DuplicateDecl, "%s redeclared in this block", obj.Name())
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
			recv = NewParam(nopos, nil, "", Typ[Invalid]) // ignore recv below
		default:
			// more than one receiver
			check.error(recvList[len(recvList)-1], InvalidRecv, "method has multiple receivers")
			fallthrough // continue with first receiver
		case 1:
			recv = recvList[0]
		}
		sig.recv = recv

		// Delay validation of receiver type as it may cause premature expansion
		// of types the receiver type is dependent on (see issues go.dev/issue/51232, go.dev/issue/51233).
		check.later(func() {
			// spec: "The receiver type must be of the form T or *T where T is a type name."
			rtyp, _ := deref(recv.typ)
			atyp := Unalias(rtyp)
			if !isValid(atyp) {
				return // error was reported before
			}
			// spec: "The type denoted by T is called the receiver base type; it must not
			// be a pointer or interface type and it must be declared in the same package
			// as the method."
			switch T := atyp.(type) {
			case *Named:
				// The receiver type may be an instantiated type referred to
				// by an alias (which cannot have receiver parameters for now).
				if T.TypeArgs() != nil && sig.RecvTypeParams() == nil {
					check.errorf(recv, InvalidRecv, "cannot define new methods on instantiated type %s", rtyp)
					break
				}
				if T.obj.pkg != check.pkg {
					check.errorf(recv, InvalidRecv, "cannot define new methods on non-local type %s", rtyp)
					break
				}
				var cause string
				switch u := T.under().(type) {
				case *Basic:
					// unsafe.Pointer is treated like a regular pointer
					if u.kind == UnsafePointer {
						cause = "unsafe.Pointer"
					}
				case *Pointer, *Interface:
					cause = "pointer or interface type"
				case *TypeParam:
					// The underlying type of a receiver base type cannot be a
					// type parameter: "type T[P any] P" is not a valid declaration.
					unreachable()
				}
				if cause != "" {
					check.errorf(recv, InvalidRecv, "invalid receiver type %s (%s)", rtyp, cause)
				}
			case *Basic:
				check.errorf(recv, InvalidRecv, "cannot define new methods on non-local type %s", rtyp)
			default:
				check.errorf(recv, InvalidRecv, "invalid receiver type %s", recv.typ)
			}
		}).describef(recv, "validate receiver %s", recv)
	}

	sig.params = NewTuple(params...)
	sig.results = NewTuple(results...)
	sig.variadic = variadic
}

// collectParams declares the parameters of list in scope and returns the corresponding
// variable list.
func (check *Checker) collectParams(scope *Scope, list *ast.FieldList, variadicOk bool, scopePos token.Pos) (params []*Var, variadic bool) {
	if list == nil {
		return
	}

	var named, anonymous bool
	for i, field := range list.List {
		ftype := field.Type
		if t, _ := ftype.(*ast.Ellipsis); t != nil {
			ftype = t.Elt
			if variadicOk && i == len(list.List)-1 && len(field.Names) <= 1 {
				variadic = true
			} else {
				check.softErrorf(t, MisplacedDotDotDot, "can only use ... with final parameter in list")
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
					check.error(name, InvalidSyntaxTree, "anonymous parameter")
					// ok to continue
				}
				par := NewParam(name.Pos(), check.pkg, name.Name, typ)
				check.declare(scope, name, par, scopePos)
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
		check.error(list, InvalidSyntaxTree, "list contains both named and anonymous parameters")
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
