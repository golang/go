// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type-checking of identifiers and type expressions.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"go/constant"
	"sort"
	"strconv"
	"strings"
)

// ident type-checks identifier e and initializes x with the value or type of e.
// If an error occurred, x.mode is set to invalid.
// For the meaning of def, see Checker.definedType, below.
// If wantType is set, the identifier e is expected to denote a type.
//
func (check *Checker) ident(x *operand, e *syntax.Name, def *Named, wantType bool) {
	x.mode = invalid
	x.expr = e

	// Note that we cannot use check.lookup here because the returned scope
	// may be different from obj.Parent(). See also Scope.LookupParent doc.
	scope, obj := check.scope.LookupParent(e.Value, check.pos)
	if obj == nil {
		if e.Value == "_" {
			check.errorf(e, "cannot use _ as value or type")
		} else {
			check.errorf(e, "undeclared name: %s", e.Value)
		}
		return
	}
	check.recordUse(e, obj)

	// Type-check the object.
	// Only call Checker.objDecl if the object doesn't have a type yet
	// (in which case we must actually determine it) or the object is a
	// TypeName and we also want a type (in which case we might detect
	// a cycle which needs to be reported). Otherwise we can skip the
	// call and avoid a possible cycle error in favor of the more
	// informative "not a type/value" error that this function's caller
	// will issue (see issue #25790).
	typ := obj.Type()
	if _, gotType := obj.(*TypeName); typ == nil || gotType && wantType {
		check.objDecl(obj, def)
		typ = obj.Type() // type must have been assigned by Checker.objDecl
	}
	assert(typ != nil)

	// The object may be dot-imported: If so, remove its package from
	// the map of unused dot imports for the respective file scope.
	// (This code is only needed for dot-imports. Without them,
	// we only have to mark variables, see *Var case below).
	if pkg := obj.Pkg(); pkg != check.pkg && pkg != nil {
		delete(check.unusedDotImports[scope], pkg)
	}

	switch obj := obj.(type) {
	case *PkgName:
		check.errorf(e, "use of package %s not in selector", obj.name)
		return

	case *Const:
		check.addDeclDep(obj)
		if typ == Typ[Invalid] {
			return
		}
		if obj == universeIota {
			if check.iota == nil {
				check.errorf(e, "cannot use iota outside constant declaration")
				return
			}
			x.val = check.iota
		} else {
			x.val = obj.val
		}
		assert(x.val != nil)
		x.mode = constant_

	case *TypeName:
		x.mode = typexpr

	case *Var:
		// It's ok to mark non-local variables, but ignore variables
		// from other packages to avoid potential race conditions with
		// dot-imported variables.
		if obj.pkg == check.pkg {
			obj.used = true
		}
		check.addDeclDep(obj)
		if typ == Typ[Invalid] {
			return
		}
		x.mode = variable

	case *Func:
		check.addDeclDep(obj)
		x.mode = value

	case *Builtin:
		x.id = obj.id
		x.mode = builtin

	case *Nil:
		x.mode = value

	default:
		unreachable()
	}

	x.typ = typ
}

// typ type-checks the type expression e and returns its type, or Typ[Invalid].
// The type must not be an (uninstantiated) generic type.
func (check *Checker) typ(e syntax.Expr) Type {
	return check.definedType(e, nil)
}

// varType type-checks the type expression e and returns its type, or Typ[Invalid].
// The type must not be an (uninstantiated) generic type and it must be ordinary
// (see ordinaryType).
func (check *Checker) varType(e syntax.Expr) Type {
	typ := check.definedType(e, nil)
	check.ordinaryType(leftPos(e), typ)
	return typ
}

// ordinaryType reports an error if typ is an interface type containing
// type lists or is (or embeds) the predeclared type comparable.
func (check *Checker) ordinaryType(pos syntax.Pos, typ Type) {
	// We don't want to call Under() (via Interface) or complete interfaces while we
	// are in the middle of type-checking parameter declarations that might belong to
	// interface methods. Delay this check to the end of type-checking.
	check.atEnd(func() {
		if t := typ.Interface(); t != nil {
			check.completeInterface(pos, t) // TODO(gri) is this the correct position?
			if t.allTypes != nil {
				check.softErrorf(pos, "interface contains type constraints (%s)", t.allTypes)
				return
			}
			if t.IsComparable() {
				check.softErrorf(pos, "interface is (or embeds) comparable")
			}
		}
	})
}

// anyType type-checks the type expression e and returns its type, or Typ[Invalid].
// The type may be generic or instantiated.
func (check *Checker) anyType(e syntax.Expr) Type {
	typ := check.typInternal(e, nil)
	assert(isTyped(typ))
	check.recordTypeAndValue(e, typexpr, typ, nil)
	return typ
}

// definedType is like typ but also accepts a type name def.
// If def != nil, e is the type specification for the defined type def, declared
// in a type declaration, and def.underlying will be set to the type of e before
// any components of e are type-checked.
//
func (check *Checker) definedType(e syntax.Expr, def *Named) Type {
	typ := check.typInternal(e, def)
	assert(isTyped(typ))
	if isGeneric(typ) {
		check.errorf(e, "cannot use generic type %s without instantiation", typ)
		typ = Typ[Invalid]
	}
	check.recordTypeAndValue(e, typexpr, typ, nil)
	return typ
}

// genericType is like typ but the type must be an (uninstantiated) generic type.
func (check *Checker) genericType(e syntax.Expr, reportErr bool) Type {
	typ := check.typInternal(e, nil)
	assert(isTyped(typ))
	if typ != Typ[Invalid] && !isGeneric(typ) {
		if reportErr {
			check.errorf(e, "%s is not a generic type", typ)
		}
		typ = Typ[Invalid]
	}
	// TODO(gri) what is the correct call below?
	check.recordTypeAndValue(e, typexpr, typ, nil)
	return typ
}

// isubst returns an x with identifiers substituted per the substitution map smap.
// isubst only handles the case of (valid) method receiver type expressions correctly.
func isubst(x syntax.Expr, smap map[*syntax.Name]*syntax.Name) syntax.Expr {
	switch n := x.(type) {
	case *syntax.Name:
		if alt := smap[n]; alt != nil {
			return alt
		}
	// case *syntax.StarExpr:
	// 	X := isubst(n.X, smap)
	// 	if X != n.X {
	// 		new := *n
	// 		n.X = X
	// 		return &new
	// 	}
	case *syntax.Operation:
		if n.Op == syntax.Mul && n.Y == nil {
			X := isubst(n.X, smap)
			if X != n.X {
				new := *n
				n.X = X
				return &new
			}
		}
	case *syntax.CallExpr:
		var args []syntax.Expr
		for i, arg := range n.ArgList {
			Arg := isubst(arg, smap)
			if Arg != arg {
				if args == nil {
					args = make([]syntax.Expr, len(n.ArgList))
					copy(args, n.ArgList)
				}
				args[i] = Arg
			}
		}
		if args != nil {
			new := *n
			new.ArgList = args
			return &new
		}
	case *syntax.ParenExpr:
		X := isubst(n.X, smap)
		if X != n.X {
			return X // no need to recreate the parentheses
		}
	default:
		// Other receiver type expressions are invalid.
		// It's fine to ignore those here as they will
		// be checked elsewhere.
	}
	return x
}

// funcType type-checks a function or method type.
func (check *Checker) funcType(sig *Signature, recvPar *syntax.Field, tparams []*syntax.Field, ftyp *syntax.FuncType) {
	check.openScope(ftyp, "function")
	check.scope.isFunc = true
	check.recordScope(ftyp, check.scope)
	sig.scope = check.scope
	defer check.closeScope()

	var recvTyp syntax.Expr // rewritten receiver type; valid if != nil
	if recvPar != nil {
		// collect generic receiver type parameters, if any
		// - a receiver type parameter is like any other type parameter, except that it is declared implicitly
		// - the receiver specification acts as local declaration for its type parameters, which may be blank
		_, rname, rparams := check.unpackRecv(recvPar.Type, true)
		if len(rparams) > 0 {
			// Blank identifiers don't get declared and regular type-checking of the instantiated
			// parameterized receiver type expression fails in Checker.collectParams of receiver.
			// Identify blank type parameters and substitute each with a unique new identifier named
			// "n_" (where n is the parameter index) and which cannot conflict with any user-defined
			// name.
			var smap map[*syntax.Name]*syntax.Name // substitution map from "_" to "!n" identifiers
			for i, p := range rparams {
				if p.Value == "_" {
					new := *p
					new.Value = fmt.Sprintf("%d_", i)
					rparams[i] = &new // use n_ identifier instead of _ so it can be looked up
					if smap == nil {
						smap = make(map[*syntax.Name]*syntax.Name)
					}
					smap[p] = &new
				}
			}
			if smap != nil {
				// blank identifiers were found => use rewritten receiver type
				recvTyp = isubst(recvPar.Type, smap)
			}
			// TODO(gri) rework declareTypeParams
			sig.rparams = nil
			for _, rparam := range rparams {
				sig.rparams = check.declareTypeParam(sig.rparams, rparam)
			}
			// determine receiver type to get its type parameters
			// and the respective type parameter bounds
			var recvTParams []*TypeName
			if rname != nil {
				// recv should be a Named type (otherwise an error is reported elsewhere)
				// Also: Don't report an error via genericType since it will be reported
				//       again when we type-check the signature.
				// TODO(gri) maybe the receiver should be marked as invalid instead?
				if recv := check.genericType(rname, false).Named(); recv != nil {
					recvTParams = recv.tparams
				}
			}
			// provide type parameter bounds
			// - only do this if we have the right number (otherwise an error is reported elsewhere)
			if len(sig.rparams) == len(recvTParams) {
				// We have a list of *TypeNames but we need a list of Types.
				// While creating this list, also update type parameter pointer designation
				// for each (*TypeParam) list entry, by copying the information from the
				// receiver base type's type parameters.
				list := make([]Type, len(sig.rparams))
				for i, t := range sig.rparams {
					t.typ.(*TypeParam).ptr = recvTParams[i].typ.(*TypeParam).ptr
					list[i] = t.typ
				}
				for i, tname := range sig.rparams {
					bound := recvTParams[i].typ.(*TypeParam).bound
					// bound is (possibly) parameterized in the context of the
					// receiver type declaration. Substitute parameters for the
					// current context.
					// TODO(gri) should we assume now that bounds always exist?
					//           (no bound == empty interface)
					if bound != nil {
						bound = check.subst(tname.pos, bound, makeSubstMap(recvTParams, list))
						tname.typ.(*TypeParam).bound = bound
					}
				}
			}
		}
	}

	if tparams != nil {
		sig.tparams = check.collectTypeParams(tparams)
		// Always type-check method type parameters but complain if they are not enabled.
		// (A separate check is needed when type-checking interface method signatures because
		// they don't have a receiver specification.)
		if recvPar != nil && !check.conf.AcceptMethodTypeParams {
			check.errorf(ftyp, "methods cannot have type parameters")
		}
	}

	// Value (non-type) parameters' scope starts in the function body. Use a temporary scope for their
	// declarations and then squash that scope into the parent scope (and report any redeclarations at
	// that time).
	scope := NewScope(check.scope, nopos, nopos, "function body (temp. scope)")
	var recvList []*Var // TODO(gri) remove the need for making a list here
	if recvPar != nil {
		recvList, _ = check.collectParams(scope, []*syntax.Field{recvPar}, recvTyp, false) // use rewritten receiver type, if any
	}
	params, variadic := check.collectParams(scope, ftyp.ParamList, nil, true)
	results, _ := check.collectParams(scope, ftyp.ResultList, nil, false)
	scope.Squash(func(obj, alt Object) {
		check.errorf(obj, "%s redeclared in this block", obj.Name())
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
			check.error(recvList[len(recvList)-1].Pos(), "method must have exactly one receiver")
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
			if T := t.Named(); T != nil {
				// spec: "The type denoted by T is called the receiver base type; it must not
				// be a pointer or interface type and it must be declared in the same package
				// as the method."
				if T.obj.pkg != check.pkg {
					err = "type not defined in this package"
				} else {
					switch u := optype(T.Under()).(type) {
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
				check.errorf(recv.pos, "invalid receiver %s (%s)", recv.typ, err)
				// ok to continue
			}
		}
		sig.recv = recv
	}

	sig.params = NewTuple(params...)
	sig.results = NewTuple(results...)
	sig.variadic = variadic
}

// goTypeName returns the Go type name for typ and
// removes any occurences of "types." from that name.
func goTypeName(typ Type) string {
	return strings.ReplaceAll(fmt.Sprintf("%T", typ), "types.", "")
}

// typInternal drives type checking of types.
// Must only be called by definedType or genericType.
//
func (check *Checker) typInternal(e0 syntax.Expr, def *Named) (T Type) {
	if check.conf.Trace {
		check.trace(e0.Pos(), "type %s", e0)
		check.indent++
		defer func() {
			check.indent--
			var under Type
			if T != nil {
				// Calling Under() here may lead to endless instantiations.
				// Test case: type T[P any] *T[P]
				// TODO(gri) investigate if that's a bug or to be expected
				// (see also analogous comment in Checker.instantiate).
				under = T.Underlying()
			}
			if T == under {
				check.trace(e0.Pos(), "=> %s // %s", T, goTypeName(T))
			} else {
				check.trace(e0.Pos(), "=> %s (under = %s) // %s", T, under, goTypeName(T))
			}
		}()
	}

	switch e := e0.(type) {
	case *syntax.BadExpr:
		// ignore - error reported before

	case *syntax.Name:
		var x operand
		check.ident(&x, e, def, true)

		switch x.mode {
		case typexpr:
			typ := x.typ
			def.setUnderlying(typ)
			return typ
		case invalid:
			// ignore - error reported before
		case novalue:
			check.errorf(&x, "%s used as type", &x)
		default:
			check.errorf(&x, "%s is not a type", &x)
		}

	case *syntax.SelectorExpr:
		var x operand
		check.selector(&x, e)

		switch x.mode {
		case typexpr:
			typ := x.typ
			def.setUnderlying(typ)
			return typ
		case invalid:
			// ignore - error reported before
		case novalue:
			check.errorf(&x, "%s used as type", &x)
		default:
			check.errorf(&x, "%s is not a type", &x)
		}

	case *syntax.IndexExpr:
		return check.instantiatedType(e.X, []syntax.Expr{e.Index}, def)

	case *syntax.CallExpr:
		return check.instantiatedType(e.Fun, e.ArgList, def)

	case *syntax.ParenExpr:
		// Generic types must be instantiated before they can be used in any form.
		// Consequently, generic types cannot be parenthesized.
		return check.definedType(e.X, def)

	case *syntax.ArrayType:
		typ := new(Array)
		def.setUnderlying(typ)
		if e.Len != nil {
			typ.len = check.arrayLength(e.Len)
		} else {
			// [...]array
			check.errorf(e, "invalid use of [...] array (outside a composite literal)")
			typ.len = -1
		}
		typ.elem = check.varType(e.Elem)
		return typ

	case *syntax.SliceType:
		typ := new(Slice)
		def.setUnderlying(typ)
		typ.elem = check.varType(e.Elem)
		return typ

	case *syntax.StructType:
		typ := new(Struct)
		def.setUnderlying(typ)
		check.structType(typ, e)
		return typ

	case *syntax.Operation:
		if e.Op == syntax.Mul && e.Y == nil {
			typ := new(Pointer)
			def.setUnderlying(typ)
			typ.base = check.varType(e.X)
			return typ
		}

	case *syntax.FuncType:
		typ := new(Signature)
		def.setUnderlying(typ)
		check.funcType(typ, nil, nil, e)
		return typ

	case *syntax.InterfaceType:
		typ := new(Interface)
		def.setUnderlying(typ)
		if def != nil {
			typ.obj = def.obj
		}
		check.interfaceType(typ, e, def)
		return typ

	case *syntax.MapType:
		typ := new(Map)
		def.setUnderlying(typ)

		typ.key = check.varType(e.Key)
		typ.elem = check.varType(e.Value)

		// spec: "The comparison operators == and != must be fully defined
		// for operands of the key type; thus the key type must not be a
		// function, map, or slice."
		//
		// Delay this check because it requires fully setup types;
		// it is safe to continue in any case (was issue 6667).
		check.atEnd(func() {
			if !Comparable(typ.key) {
				var why string
				if typ.key.TypeParam() != nil {
					why = " (missing comparable constraint)"
				}
				check.errorf(e.Key, "invalid map key type %s%s", typ.key, why)
			}
		})

		return typ

	case *syntax.ChanType:
		typ := new(Chan)
		def.setUnderlying(typ)

		dir := SendRecv
		switch e.Dir {
		case 0:
			// nothing to do
		case syntax.SendOnly:
			dir = SendOnly
		case syntax.RecvOnly:
			dir = RecvOnly
		default:
			check.invalidASTf(e, "unknown channel direction %d", e.Dir)
			// ok to continue
		}

		typ.dir = dir
		typ.elem = check.varType(e.Elem)
		return typ

	default:
		check.errorf(e0, "%s is not a type", e0)
	}

	typ := Typ[Invalid]
	def.setUnderlying(typ)
	return typ
}

// typeOrNil type-checks the type expression (or nil value) e
// and returns the type of e, or nil. If e is a type, it must
// not be an (uninstantiated) generic type.
// If e is neither a type nor nil, typeOrNil returns Typ[Invalid].
// TODO(gri) should we also disallow non-var types?
func (check *Checker) typOrNil(e syntax.Expr) Type {
	var x operand
	check.rawExpr(&x, e, nil)
	switch x.mode {
	case invalid:
		// ignore - error reported before
	case novalue:
		check.errorf(&x, "%s used as type", &x)
	case typexpr:
		check.instantiatedOperand(&x)
		return x.typ
	case value:
		if x.isNil() {
			return nil
		}
		fallthrough
	default:
		check.errorf(&x, "%s is not a type", &x)
	}
	return Typ[Invalid]
}

func (check *Checker) instantiatedType(x syntax.Expr, targs []syntax.Expr, def *Named) Type {
	b := check.genericType(x, true) // TODO(gri) what about cycles?
	if b == Typ[Invalid] {
		return b // error already reported
	}
	base := b.Named()
	if base == nil {
		unreachable() // should have been caught by genericType
	}

	// create a new type Instance rather than instantiate the type
	// TODO(gri) should do argument number check here rather than
	// when instantiating the type?
	typ := new(instance)
	def.setUnderlying(typ)

	typ.check = check
	typ.pos = x.Pos()
	typ.base = base

	// evaluate arguments (always)
	typ.targs = check.typeList(targs)
	if typ.targs == nil {
		def.setUnderlying(Typ[Invalid]) // avoid later errors due to lazy instantiation
		return Typ[Invalid]
	}

	// determine argument positions (for error reporting)
	typ.poslist = make([]syntax.Pos, len(targs))
	for i, arg := range targs {
		typ.poslist[i] = arg.Pos()
	}

	// make sure we check instantiation works at least once
	// and that the resulting type is valid
	check.atEnd(func() {
		t := typ.expand()
		check.validType(t, nil)
	})

	return typ
}

// arrayLength type-checks the array length expression e
// and returns the constant length >= 0, or a value < 0
// to indicate an error (and thus an unknown length).
func (check *Checker) arrayLength(e syntax.Expr) int64 {
	var x operand
	check.expr(&x, e)
	if x.mode != constant_ {
		if x.mode != invalid {
			check.errorf(&x, "array length %s must be constant", &x)
		}
		return -1
	}
	if isUntyped(x.typ) || isInteger(x.typ) {
		if val := constant.ToInt(x.val); val.Kind() == constant.Int {
			if representableConst(val, check, Typ[Int], nil) {
				if n, ok := constant.Int64Val(val); ok && n >= 0 {
					return n
				}
				check.errorf(&x, "invalid array length %s", &x)
				return -1
			}
		}
	}
	check.errorf(&x, "array length %s must be integer", &x)
	return -1
}

// typeList provides the list of types corresponding to the incoming expression list.
// If an error occured, the result is nil, but all list elements were type-checked.
func (check *Checker) typeList(list []syntax.Expr) []Type {
	res := make([]Type, len(list)) // res != nil even if len(list) == 0
	for i, x := range list {
		t := check.varType(x)
		if t == Typ[Invalid] {
			res = nil
		}
		if res != nil {
			res[i] = t
		}
	}
	return res
}

// collectParams declares the parameters of list in scope and returns the corresponding
// variable list. If type0 != nil, it is used instead of the the first type in list.
func (check *Checker) collectParams(scope *Scope, list []*syntax.Field, type0 syntax.Expr, variadicOk bool) (params []*Var, variadic bool) {
	if list == nil {
		return
	}

	var named, anonymous bool

	var typ Type
	var prev syntax.Expr
	for i, field := range list {
		ftype := field.Type
		if i == 0 && type0 != nil {
			ftype = type0
		}
		if t, _ := ftype.(*syntax.DotsType); t != nil {
			ftype = t.Elem
			if variadicOk && i == len(list)-1 {
				variadic = true
			} else {
				check.softErrorf(t, "can only use ... with final parameter in list")
				// ignore ... and continue
			}
		}
		// type-check type of grouped fields only once
		if ftype != prev {
			typ = check.varType(ftype)
			prev = ftype
		}
		// The parser ensures that f.Tag is nil and we don't
		// care if a constructed AST contains a non-nil tag.
		if field.Name != nil {
			// named parameter
			name := field.Name.Value
			if name == "" {
				check.invalidASTf(field.Name, "anonymous parameter")
				// ok to continue
			}
			par := NewParam(field.Name.Pos(), check.pkg, name, typ)
			check.declare(scope, field.Name, par, scope.pos)
			params = append(params, par)
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
		check.invalidASTf(list[0], "list contains both named and anonymous parameters")
		// ok to continue
	}

	// For a variadic function, change the last parameter's type from T to []T.
	// Since we type-checked T rather than ...T, we also need to retro-actively
	// record the type for ...T.
	if variadic {
		last := params[len(params)-1]
		last.typ = &Slice{elem: last.typ}
		check.recordTypeAndValue(list[len(list)-1].Type, typexpr, last.typ, nil)
	}

	return
}

func (check *Checker) declareInSet(oset *objset, pos syntax.Pos, obj Object) bool {
	if alt := oset.insert(obj); alt != nil {
		check.errorf(pos, "%s redeclared", obj.Name())
		check.reportAltDecl(alt)
		return false
	}
	return true
}

func (check *Checker) interfaceType(ityp *Interface, iface *syntax.InterfaceType, def *Named) {
	var types []syntax.Expr
	for _, f := range iface.MethodList {
		if f.Name != nil {
			// We have a method with name f.Name, or a type
			// of a type list (f.Name.Value == "type").
			name := f.Name.Value
			if name == "_" {
				check.errorf(f.Name, "invalid method name _")
				continue // ignore
			}

			if name == "type" {
				types = append(types, f.Type)
				continue
			}

			typ := check.typ(f.Type)
			sig, _ := typ.(*Signature)
			if sig == nil {
				if typ != Typ[Invalid] {
					check.invalidASTf(f.Type, "%s is not a method signature", typ)
				}
				continue // ignore
			}

			// Always type-check method type parameters but complain if they are not enabled.
			// (This extra check is needed here because interface method signatures don't have
			// a receiver specification.)
			if sig.tparams != nil && !check.conf.AcceptMethodTypeParams {
				check.errorf(f.Type, "methods cannot have type parameters")
			}

			// use named receiver type if available (for better error messages)
			var recvTyp Type = ityp
			if def != nil {
				recvTyp = def
			}
			sig.recv = NewVar(f.Name.Pos(), check.pkg, "", recvTyp)

			m := NewFunc(f.Name.Pos(), check.pkg, name, sig)
			check.recordDef(f.Name, m)
			ityp.methods = append(ityp.methods, m)
		} else {
			// We have an embedded type. completeInterface will
			// eventually verify that we have an interface.
			ityp.embeddeds = append(ityp.embeddeds, check.typ(f.Type))
			check.posMap[ityp] = append(check.posMap[ityp], f.Type.Pos())
		}
	}

	// type constraints
	ityp.types = NewSum(check.collectTypeConstraints(iface.Pos(), types))

	if len(ityp.methods) == 0 && ityp.types == nil && len(ityp.embeddeds) == 0 {
		// empty interface
		ityp.allMethods = markComplete
		return
	}

	// sort for API stability
	sort.Sort(byUniqueMethodName(ityp.methods))
	sort.Stable(byUniqueTypeName(ityp.embeddeds))

	check.later(func() { check.completeInterface(iface.Pos(), ityp) })
}

func (check *Checker) completeInterface(pos syntax.Pos, ityp *Interface) {
	if ityp.allMethods != nil {
		return
	}

	// completeInterface may be called via the LookupFieldOrMethod,
	// MissingMethod, Identical, or IdenticalIgnoreTags external API
	// in which case check will be nil. In this case, type-checking
	// must be finished and all interfaces should have been completed.
	if check == nil {
		panic("internal error: incomplete interface")
	}

	if check.conf.Trace {
		// Types don't generally have position information.
		// If we don't have a valid pos provided, try to use
		// one close enough.
		if !pos.IsKnown() && len(ityp.methods) > 0 {
			pos = ityp.methods[0].pos
		}

		check.trace(pos, "complete %s", ityp)
		check.indent++
		defer func() {
			check.indent--
			check.trace(pos, "=> %s (methods = %v, types = %v)", ityp, ityp.allMethods, ityp.allTypes)
		}()
	}

	// An infinitely expanding interface (due to a cycle) is detected
	// elsewhere (Checker.validType), so here we simply assume we only
	// have valid interfaces. Mark the interface as complete to avoid
	// infinite recursion if the validType check occurs later for some
	// reason.
	ityp.allMethods = markComplete

	// Methods of embedded interfaces are collected unchanged; i.e., the identity
	// of a method I.m's Func Object of an interface I is the same as that of
	// the method m in an interface that embeds interface I. On the other hand,
	// if a method is embedded via multiple overlapping embedded interfaces, we
	// don't provide a guarantee which "original m" got chosen for the embedding
	// interface. See also issue #34421.
	//
	// If we don't care to provide this identity guarantee anymore, instead of
	// reusing the original method in embeddings, we can clone the method's Func
	// Object and give it the position of a corresponding embedded interface. Then
	// we can get rid of the mpos map below and simply use the cloned method's
	// position.

	var seen objset
	var methods []*Func
	mpos := make(map[*Func]syntax.Pos) // method specification or method embedding position, for good error messages
	addMethod := func(pos syntax.Pos, m *Func, explicit bool) {
		switch other := seen.insert(m); {
		case other == nil:
			methods = append(methods, m)
			mpos[m] = pos
		case explicit:
			check.errorf(pos, "duplicate method %s", m.name)
			check.errorf(mpos[other.(*Func)], "\tother declaration of %s", m.name) // secondary error, \t indented
		default:
			// check method signatures after all types are computed (issue #33656)
			check.atEnd(func() {
				if !check.identical(m.typ, other.Type()) {
					check.errorf(pos, "duplicate method %s", m.name)
					check.errorf(mpos[other.(*Func)], "\tother declaration of %s", m.name) // secondary error, \t indented
				}
			})
		}
	}

	for _, m := range ityp.methods {
		addMethod(m.pos, m, true)
	}

	// collect types
	allTypes := ityp.types

	posList := check.posMap[ityp]
	for i, typ := range ityp.embeddeds {
		pos := posList[i] // embedding position
		utyp := typ.Under()
		etyp := utyp.Interface()
		if etyp == nil {
			if utyp != Typ[Invalid] {
				var format string
				if _, ok := utyp.(*TypeParam); ok {
					format = "%s is a type parameter, not an interface"
				} else {
					format = "%s is not an interface"
				}
				check.errorf(pos, format, typ)
			}
			continue
		}
		check.completeInterface(pos, etyp)
		for _, m := range etyp.allMethods {
			addMethod(pos, m, false) // use embedding position pos rather than m.pos
		}
		allTypes = intersect(allTypes, etyp.allTypes)
	}

	if methods != nil {
		sort.Sort(byUniqueMethodName(methods))
		ityp.allMethods = methods
	}
	ityp.allTypes = allTypes
}

// intersect computes the intersection of the types x and y.
// Note: A incomming nil type stands for the top type. A top
// type result is returned as nil.
func intersect(x, y Type) (r Type) {
	defer func() {
		if r == theTop {
			r = nil
		}
	}()

	switch {
	case x == theBottom || y == theBottom:
		return theBottom
	case x == nil || x == theTop:
		return y
	case y == nil || x == theTop:
		return x
	}

	xtypes := unpack(x)
	ytypes := unpack(y)
	// Compute the list rtypes which includes only
	// types that are in both xtypes and ytypes.
	// Quadratic algorithm, but good enough for now.
	// TODO(gri) fix this
	var rtypes []Type
	for _, x := range xtypes {
		if includes(ytypes, x) {
			rtypes = append(rtypes, x)
		}
	}

	if rtypes == nil {
		return theBottom
	}
	return NewSum(rtypes)
}

// byUniqueTypeName named type lists can be sorted by their unique type names.
type byUniqueTypeName []Type

func (a byUniqueTypeName) Len() int           { return len(a) }
func (a byUniqueTypeName) Less(i, j int) bool { return sortName(a[i]) < sortName(a[j]) }
func (a byUniqueTypeName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func sortName(t Type) string {
	if named := t.Named(); named != nil {
		return named.obj.Id()
	}
	return ""
}

// byUniqueMethodName method lists can be sorted by their unique method names.
type byUniqueMethodName []*Func

func (a byUniqueMethodName) Len() int           { return len(a) }
func (a byUniqueMethodName) Less(i, j int) bool { return a[i].Id() < a[j].Id() }
func (a byUniqueMethodName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func (check *Checker) tag(t *syntax.BasicLit) string {
	if t != nil {
		if t.Kind == syntax.StringLit {
			if val, err := strconv.Unquote(t.Value); err == nil {
				return val
			}
		}
		check.invalidASTf(t, "incorrect tag syntax: %q", t.Value)
	}
	return ""
}

func (check *Checker) structType(styp *Struct, e *syntax.StructType) {
	if e.FieldList == nil {
		return
	}

	// struct fields and tags
	var fields []*Var
	var tags []string

	// for double-declaration checks
	var fset objset

	// current field typ and tag
	var typ Type
	var tag string
	add := func(ident *syntax.Name, embedded bool, pos syntax.Pos) {
		if tag != "" && tags == nil {
			tags = make([]string, len(fields))
		}
		if tags != nil {
			tags = append(tags, tag)
		}

		name := ident.Value
		fld := NewField(pos, check.pkg, name, typ, embedded)
		// spec: "Within a struct, non-blank field names must be unique."
		if name == "_" || check.declareInSet(&fset, pos, fld) {
			fields = append(fields, fld)
			check.recordDef(ident, fld)
		}
	}

	// addInvalid adds an embedded field of invalid type to the struct for
	// fields with errors; this keeps the number of struct fields in sync
	// with the source as long as the fields are _ or have different names
	// (issue #25627).
	addInvalid := func(ident *syntax.Name, pos syntax.Pos) {
		typ = Typ[Invalid]
		tag = ""
		add(ident, true, pos)
	}

	for i, f := range e.FieldList {
		typ = check.varType(f.Type)
		if i < len(e.TagList) {
			tag = check.tag(e.TagList[i])
		}
		if f.Name != nil {
			// named field
			add(f.Name, false, f.Name.Pos())
		} else {
			// embedded field
			// spec: "An embedded type must be specified as a (possibly parenthesized) type name T or
			// as a pointer to a non-interface type name *T, and T itself may not be a pointer type."
			pos := leftPos(f.Type)
			name := embeddedFieldIdent(f.Type)
			if name == nil {
				check.errorf(pos, "invalid embedded field type %s", f.Type)
				name = &syntax.Name{Value: "_"} // TODO(gri) need to set position to pos
				addInvalid(name, pos)
				continue
			}
			add(name, true, pos)
			// Because we have a name, typ must be of the form T or *T, where T is the name
			// of a (named or alias) type, and t (= deref(typ)) must be the type of T.
			// We must delay this check to the end because we don't want to instantiate
			// (via t.Under()) a possibly incomplete type.
			embeddedTyp := typ // for closure below
			embeddedPos := pos
			check.atEnd(func() {
				t, isPtr := deref(embeddedTyp)
				switch t := optype(t.Under()).(type) {
				case *Basic:
					if t == Typ[Invalid] {
						// error was reported before
						return
					}
					// unsafe.Pointer is treated like a regular pointer
					if t.kind == UnsafePointer {
						check.errorf(embeddedPos, "embedded field type cannot be unsafe.Pointer")
					}
				case *Pointer:
					check.errorf(embeddedPos, "embedded field type cannot be a pointer")
				case *Interface:
					if isPtr {
						check.errorf(embeddedPos, "embedded field type cannot be a pointer to an interface")
					}
				}
			})
		}
	}

	styp.fields = fields
	styp.tags = tags
}

func embeddedFieldIdent(e syntax.Expr) *syntax.Name {
	switch e := e.(type) {
	case *syntax.Name:
		return e
	case *syntax.Operation:
		if base := ptrBase(e); base != nil {
			// *T is valid, but **T is not
			if op, _ := base.(*syntax.Operation); op == nil || ptrBase(op) == nil {
				return embeddedFieldIdent(e.X)
			}
		}
	case *syntax.SelectorExpr:
		return e.Sel
	case *syntax.CallExpr:
		return embeddedFieldIdent(e.Fun)
	case *syntax.ParenExpr:
		return embeddedFieldIdent(e.X)
	}
	return nil // invalid embedded field
}

func (check *Checker) collectTypeConstraints(pos syntax.Pos, types []syntax.Expr) []Type {
	list := make([]Type, 0, len(types)) // assume all types are correct
	for _, texpr := range types {
		if texpr == nil {
			check.invalidASTf(pos, "missing type constraint")
			continue
		}
		typ := check.varType(texpr)
		// A type constraint may be a predeclared type or a
		// composite type composed of only predeclared types.
		// TODO(gri) If we enable this again it also must run
		// at the end.
		const restricted = false
		var why string
		if restricted && !check.typeConstraint(typ, &why) {
			check.errorf(texpr, "invalid type constraint %s (%s)", typ, why)
			continue
		}
		list = append(list, typ)
	}

	// Ensure that each type is only present once in the type list.
	// Types may be interfaces, which may not be complete yet. It's
	// ok to do this check at the end because it's not a requirement
	// for correctness of the code.
	check.atEnd(func() {
		uniques := make([]Type, 0, len(list)) // assume all types are unique
		for i, t := range list {
			if t := t.Interface(); t != nil {
				check.completeInterface(types[i].Pos(), t)
			}
			if includes(uniques, t) {
				check.softErrorf(types[i], "duplicate type %s in type list", t)
			}
			uniques = append(uniques, t)
		}
	})

	return list
}

// includes reports whether typ is in list
func includes(list []Type, typ Type) bool {
	for _, e := range list {
		if Identical(typ, e) {
			return true
		}
	}
	return false
}

// typeConstraint checks that typ may be used in a type list.
// For now this just checks for the absence of defined (*Named) types.
func (check *Checker) typeConstraint(typ Type, why *string) bool {
	switch t := typ.(type) {
	case *Basic:
		// ok
	case *Array:
		return check.typeConstraint(t.elem, why)
	case *Slice:
		return check.typeConstraint(t.elem, why)
	case *Struct:
		for _, f := range t.fields {
			if !check.typeConstraint(f.typ, why) {
				return false
			}
		}
	case *Pointer:
		return check.typeConstraint(t.base, why)
	case *Tuple:
		if t == nil {
			return true
		}
		for _, v := range t.vars {
			if !check.typeConstraint(v.typ, why) {
				return false
			}
		}
	case *Signature:
		if len(t.tparams) != 0 {
			panic("type parameter in function type")
		}
		return (t.recv == nil || check.typeConstraint(t.recv.typ, why)) &&
			check.typeConstraint(t.params, why) &&
			check.typeConstraint(t.results, why)
	case *Interface:
		t.assertCompleteness()
		for _, m := range t.allMethods {
			if !check.typeConstraint(m.typ, why) {
				return false
			}
		}
	case *Map:
		return check.typeConstraint(t.key, why) && check.typeConstraint(t.elem, why)
	case *Chan:
		return check.typeConstraint(t.elem, why)
	case *Named:
		*why = check.sprintf("contains defined type %s", t)
		return false
	case *TypeParam:
		// ok, e.g.: func f (type T interface { type T }) ()
	default:
		unreachable()
	}
	return true
}

func ptrBase(x *syntax.Operation) syntax.Expr {
	if x.Op == syntax.Mul && x.Y == nil {
		return x.X
	}
	return nil
}
