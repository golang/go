// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"go/token"
	. "internal/types/errors"
	"path/filepath"
	"strings"
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
//
//go:fix inline
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
			panic(fmt.Sprintf("got %s, want variadic parameter with unnamed slice type or string as common underlying type", core.String()))
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

func (s *Signature) Underlying() Type { return s }
func (s *Signature) String() string   { return TypeString(s, nil) }

// ----------------------------------------------------------------------------
// Implementation

// funcType type-checks a function or method type.
func (check *Checker) funcType(sig *Signature, recvPar *ast.FieldList, ftyp *ast.FuncType) {
	check.openScope(ftyp, "function")
	check.scope.isFunc = true
	check.recordScope(ftyp, check.scope)
	sig.scope = check.scope
	defer check.closeScope()

	// collect method receiver, if any
	var recv *Var
	var rparams *TypeParamList
	if recvPar != nil && recvPar.NumFields() > 0 {
		// We have at least one receiver; make sure we don't have more than one.
		if n := len(recvPar.List); n > 1 {
			check.error(recvPar.List[n-1], InvalidRecv, "method has multiple receivers")
			// continue with first one
		}
		// all type parameters' scopes start after the method name
		scopePos := ftyp.Pos()
		recv, rparams = check.collectRecv(recvPar.List[0], scopePos)
	}

	// collect and declare function type parameters
	if ftyp.TypeParams != nil {
		// Always type-check method type parameters but complain that they are not allowed.
		// (A separate check is needed when type-checking interface method signatures because
		// they don't have a receiver specification.)
		if recvPar != nil {
			check.error(ftyp.TypeParams, InvalidMethodTypeParams, "methods cannot have type parameters")
		}
		check.collectTypeParams(&sig.tparams, ftyp.TypeParams)
	}

	// collect ordinary and result parameters
	pnames, params, variadic := check.collectParams(ftyp.Params, true)
	rnames, results, _ := check.collectParams(ftyp.Results, false)

	// declare named receiver, ordinary, and result parameters
	scopePos := ftyp.End() // all parameter's scopes start after the signature
	if recv != nil && recv.name != "" {
		check.declare(check.scope, recvPar.List[0].Names[0], recv, scopePos)
	}
	check.declareParams(pnames, params, scopePos)
	check.declareParams(rnames, results, scopePos)

	sig.recv = recv
	sig.rparams = rparams
	sig.params = NewTuple(params...)
	sig.results = NewTuple(results...)
	sig.variadic = variadic
}

// collectRecv extracts the method receiver and its type parameters (if any) from rparam.
// It declares the type parameters (but not the receiver) in the current scope, and
// returns the receiver variable and its type parameter list (if any).
func (check *Checker) collectRecv(rparam *ast.Field, scopePos token.Pos) (*Var, *TypeParamList) {
	// Unpack the receiver parameter which is of the form
	//
	//	"(" [rfield] ["*"] rbase ["[" rtparams "]"] ")"
	//
	// The receiver name rname, the pointer indirection, and the
	// receiver type parameters rtparams may not be present.
	rptr, rbase, rtparams := check.unpackRecv(rparam.Type, true)

	// Determine the receiver base type.
	var recvType Type = Typ[Invalid]
	var recvTParamsList *TypeParamList
	if rtparams == nil {
		// If there are no type parameters, we can simply typecheck rparam.Type.
		// If that is a generic type, varType will complain.
		// Further receiver constraints will be checked later, with validRecv.
		// We use rparam.Type (rather than base) to correctly record pointer
		// and parentheses in types.Info (was bug, see go.dev/issue/68639).
		recvType = check.varType(rparam.Type)
		// Defining new methods on instantiated (alias or defined) types is not permitted.
		// Follow literal pointer/alias type chain and check.
		// (Correct code permits at most one pointer indirection, but for this check it
		// doesn't matter if we have multiple pointers.)
		a, _ := unpointer(recvType).(*Alias) // recvType is not generic per above
		for a != nil {
			baseType := unpointer(a.fromRHS)
			if g, _ := baseType.(genericType); g != nil && g.TypeParams() != nil {
				check.errorf(rbase, InvalidRecv, "cannot define new methods on instantiated type %s", g)
				recvType = Typ[Invalid] // avoid follow-on errors by Checker.validRecv
				break
			}
			a, _ = baseType.(*Alias)
		}
	} else {
		// If there are type parameters, rbase must denote a generic base type.
		// Important: rbase must be resolved before declaring any receiver type
		// parameters (which may have the same name, see below).
		var baseType *Named // nil if not valid
		var cause string
		if t := check.genericType(rbase, &cause); isValid(t) {
			switch t := t.(type) {
			case *Named:
				baseType = t
			case *Alias:
				// Methods on generic aliases are not permitted.
				// Only report an error if the alias type is valid.
				if isValid(unalias(t)) {
					check.errorf(rbase, InvalidRecv, "cannot define new methods on generic alias type %s", t)
				}
				// Ok to continue but do not set basetype in this case so that
				// recvType remains invalid (was bug, see go.dev/issue/70417).
			default:
				panic("unreachable")
			}
		} else {
			if cause != "" {
				check.errorf(rbase, InvalidRecv, "%s", cause)
			}
			// Ok to continue but do not set baseType (see comment above).
		}

		// Collect the type parameters declared by the receiver (see also
		// Checker.collectTypeParams). The scope of the type parameter T in
		// "func (r T[T]) f() {}" starts after f, not at r, so we declare it
		// after typechecking rbase (see go.dev/issue/52038).
		recvTParams := make([]*TypeParam, len(rtparams))
		for i, rparam := range rtparams {
			tpar := check.declareTypeParam(rparam, scopePos)
			recvTParams[i] = tpar
			// For historic reasons, type parameters in receiver type expressions
			// are considered both definitions and uses and thus must be recorded
			// in the Info.Uses and Info.Types maps (see go.dev/issue/68670).
			check.recordUse(rparam, tpar.obj)
			check.recordTypeAndValue(rparam, typexpr, tpar, nil)
		}
		recvTParamsList = bindTParams(recvTParams)

		// Get the type parameter bounds from the receiver base type
		// and set them for the respective (local) receiver type parameters.
		if baseType != nil {
			baseTParams := baseType.TypeParams().list()
			if len(recvTParams) == len(baseTParams) {
				smap := makeRenameMap(baseTParams, recvTParams)
				for i, recvTPar := range recvTParams {
					baseTPar := baseTParams[i]
					check.mono.recordCanon(recvTPar, baseTPar)
					// baseTPar.bound is possibly parameterized by other type parameters
					// defined by the generic base type. Substitute those parameters with
					// the receiver type parameters declared by the current method.
					recvTPar.bound = check.subst(recvTPar.obj.pos, baseTPar.bound, smap, nil, check.context())
				}
			} else {
				got := measure(len(recvTParams), "type parameter")
				check.errorf(rbase, BadRecv, "receiver declares %s, but receiver base type declares %d", got, len(baseTParams))
			}

			// The type parameters declared by the receiver also serve as
			// type arguments for the receiver type. Instantiate the receiver.
			check.verifyVersionf(rbase, go1_18, "type instantiation")
			targs := make([]Type, len(recvTParams))
			for i, targ := range recvTParams {
				targs[i] = targ
			}
			recvType = check.instance(rparam.Type.Pos(), baseType, targs, nil, check.context())
			check.recordInstance(rbase, targs, recvType)

			// Reestablish pointerness if needed (but avoid a pointer to an invalid type).
			if rptr && isValid(recvType) {
				recvType = NewPointer(recvType)
			}

			check.recordParenthesizedRecvTypes(rparam.Type, recvType)
		}
	}

	// Make sure we have no more than one receiver name.
	var rname *ast.Ident
	if n := len(rparam.Names); n >= 1 {
		if n > 1 {
			check.error(rparam.Names[n-1], InvalidRecv, "method has multiple receivers")
		}
		rname = rparam.Names[0]
	}

	// Create the receiver parameter.
	// recvType is invalid if baseType was never set.
	var recv *Var
	if rname != nil && rname.Name != "" {
		// named receiver
		recv = NewParam(rname.Pos(), check.pkg, rname.Name, recvType)
		// In this case, the receiver is declared by the caller
		// because it must be declared after any type parameters
		// (otherwise it might shadow one of them).
	} else {
		// anonymous receiver
		recv = NewParam(rparam.Pos(), check.pkg, "", recvType)
		check.recordImplicit(rparam, recv)
	}

	// Delay validation of receiver type as it may cause premature expansion of types
	// the receiver type is dependent on (see go.dev/issue/51232, go.dev/issue/51233).
	check.later(func() {
		check.validRecv(rbase, recv)
	}).describef(recv, "validRecv(%s)", recv)

	return recv, recvTParamsList
}

func unpointer(t Type) Type {
	for {
		p, _ := t.(*Pointer)
		if p == nil {
			return t
		}
		t = p.base
	}
}

// recordParenthesizedRecvTypes records parenthesized intermediate receiver type
// expressions that all map to the same type, by recursively unpacking expr and
// recording the corresponding type for it. Example:
//
//	expression  -->  type
//	----------------------
//	(*(T[P]))        *T[P]
//	 *(T[P])         *T[P]
//	  (T[P])          T[P]
//	   T[P]           T[P]
func (check *Checker) recordParenthesizedRecvTypes(expr ast.Expr, typ Type) {
	for {
		check.recordTypeAndValue(expr, typexpr, typ, nil)
		switch e := expr.(type) {
		case *ast.ParenExpr:
			expr = e.X
		case *ast.StarExpr:
			expr = e.X
			// In a correct program, typ must be an unnamed
			// pointer type. But be careful and don't panic.
			ptr, _ := typ.(*Pointer)
			if ptr == nil {
				return // something is wrong
			}
			typ = ptr.base
		default:
			return // cannot unpack any further
		}
	}
}

// collectParams collects (but does not declare) all parameters of list and returns
// the list of parameter names, corresponding parameter variables, and whether the
// parameter list is variadic. Anonymous parameters are recorded with nil names.
func (check *Checker) collectParams(list *ast.FieldList, variadicOk bool) (names []*ast.Ident, params []*Var, variadic bool) {
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
				check.softErrorf(t, InvalidSyntaxTree, "invalid use of ...")
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
				// named parameter is declared by caller
				names = append(names, name)
				params = append(params, par)
			}
			named = true
		} else {
			// anonymous parameter
			par := NewParam(ftype.Pos(), check.pkg, "", typ)
			check.recordImplicit(field, par)
			names = append(names, nil)
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

// declareParams declares each named parameter in the current scope.
func (check *Checker) declareParams(names []*ast.Ident, params []*Var, scopePos token.Pos) {
	for i, name := range names {
		if name != nil && name.Name != "" {
			check.declare(check.scope, name, params[i], scopePos)
		}
	}
}

// validRecv verifies that the receiver satisfies its respective spec requirements
// and reports an error otherwise.
func (check *Checker) validRecv(pos positioner, recv *Var) {
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
		if T.obj.pkg != check.pkg || isCGoTypeObj(check.fset, T.obj) {
			check.errorf(pos, InvalidRecv, "cannot define new methods on non-local type %s", rtyp)
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
			panic("unreachable")
		}
		if cause != "" {
			check.errorf(pos, InvalidRecv, "invalid receiver type %s (%s)", rtyp, cause)
		}
	case *Basic:
		check.errorf(pos, InvalidRecv, "cannot define new methods on non-local type %s", rtyp)
	default:
		check.errorf(pos, InvalidRecv, "invalid receiver type %s", recv.typ)
	}
}

// isCGoTypeObj reports whether the given type name was created by cgo.
func isCGoTypeObj(fset *token.FileSet, obj *TypeName) bool {
	return strings.HasPrefix(obj.name, "_Ctype_") ||
		strings.HasPrefix(filepath.Base(fset.File(obj.pos).Name()), "_cgo_")
}
