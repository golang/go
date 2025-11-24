// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type-checking of identifiers and type expressions.

package types

import (
	"fmt"
	"go/ast"
	"go/constant"
	. "internal/types/errors"
	"strings"
)

// ident type-checks identifier e and initializes x with the value or type of e.
// If an error occurred, x.mode is set to invalid.
// If wantType is set, the identifier e is expected to denote a type.
func (check *Checker) ident(x *operand, e *ast.Ident, wantType bool) {
	x.mode = invalid
	x.expr = e

	scope, obj := check.lookupScope(e.Name)
	switch obj {
	case nil:
		if e.Name == "_" {
			check.error(e, InvalidBlank, "cannot use _ as value or type")
		} else if isValidName(e.Name) {
			check.errorf(e, UndeclaredName, "undefined: %s", e.Name)
		}
		return
	case universeComparable:
		if !check.verifyVersionf(e, go1_18, "predeclared %s", e.Name) {
			return // avoid follow-on errors
		}
	}
	// Because the representation of any depends on gotypesalias, we don't check
	// pointer identity here.
	if obj.Name() == "any" && obj.Parent() == Universe {
		if !check.verifyVersionf(e, go1_18, "predeclared %s", e.Name) {
			return // avoid follow-on errors
		}
	}
	check.recordUse(e, obj)

	// If we want a type but don't have one, stop right here and avoid potential problems
	// with missing underlying types. This also gives better error messages in some cases
	// (see go.dev/issue/65344).
	_, gotType := obj.(*TypeName)
	if !gotType && wantType {
		check.errorf(e, NotAType, "%s is not a type", obj.Name())
		// avoid "declared but not used" errors
		// (don't use Checker.use - we don't want to evaluate too much)
		if v, _ := obj.(*Var); v != nil && v.pkg == check.pkg /* see Checker.use1 */ {
			check.usedVars[v] = true
		}
		return
	}

	// Type-check the object.
	// Only call Checker.objDecl if the object doesn't have a type yet
	// (in which case we must actually determine it) or the object is a
	// TypeName from the current package and we also want a type (in which case
	// we might detect a cycle which needs to be reported). Otherwise we can skip
	// the call and avoid a possible cycle error in favor of the more informative
	// "not a type/value" error that this function's caller will issue (see
	// go.dev/issue/25790).
	//
	// Note that it is important to avoid calling objDecl on objects from other
	// packages, to avoid races: see issue #69912.
	typ := obj.Type()
	if typ == nil || (gotType && wantType && obj.Pkg() == check.pkg) {
		check.objDecl(obj)
		typ = obj.Type() // type must have been assigned by Checker.objDecl
	}
	assert(typ != nil)

	// The object may have been dot-imported.
	// If so, mark the respective package as used.
	// (This code is only needed for dot-imports. Without them,
	// we only have to mark variables, see *Var case below).
	if pkgName := check.dotImportMap[dotImportKey{scope, obj.Name()}]; pkgName != nil {
		check.usedPkgNames[pkgName] = true
	}

	switch obj := obj.(type) {
	case *PkgName:
		check.errorf(e, InvalidPkgUse, "use of package %s not in selector", obj.name)
		return

	case *Const:
		check.addDeclDep(obj)
		if !isValid(typ) {
			return
		}
		if obj == universeIota {
			if check.iota == nil {
				check.error(e, InvalidIota, "cannot use iota outside constant declaration")
				return
			}
			x.val = check.iota
		} else {
			x.val = obj.val
		}
		assert(x.val != nil)
		x.mode = constant_

	case *TypeName:
		if !check.conf._EnableAlias && check.isBrokenAlias(obj) {
			check.errorf(e, InvalidDeclCycle, "invalid use of type alias %s in recursive type (see go.dev/issue/50729)", obj.name)
			return
		}
		x.mode = typexpr

	case *Var:
		// It's ok to mark non-local variables, but ignore variables
		// from other packages to avoid potential race conditions with
		// dot-imported variables.
		if obj.pkg == check.pkg {
			check.usedVars[obj] = true
		}
		check.addDeclDep(obj)
		if !isValid(typ) {
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
		panic("unreachable")
	}

	x.typ = typ
}

// typ type-checks the type expression e and returns its type, or Typ[Invalid].
// The type must not be an (uninstantiated) generic type.
func (check *Checker) typ(e ast.Expr) Type {
	return check.declaredType(e, nil)
}

// varType type-checks the type expression e and returns its type, or Typ[Invalid].
// The type must not be an (uninstantiated) generic type and it must not be a
// constraint interface.
func (check *Checker) varType(e ast.Expr) Type {
	typ := check.declaredType(e, nil)
	check.validVarType(e, typ)
	return typ
}

// validVarType reports an error if typ is a constraint interface.
// The expression e is used for error reporting, if any.
func (check *Checker) validVarType(e ast.Expr, typ Type) {
	// If we have a type parameter there's nothing to do.
	if isTypeParam(typ) {
		return
	}

	// We don't want to call typ.Underlying() or complete interfaces while we are in
	// the middle of type-checking parameter declarations that might belong
	// to interface methods. Delay this check to the end of type-checking.
	check.later(func() {
		if t, _ := typ.Underlying().(*Interface); t != nil {
			tset := computeInterfaceTypeSet(check, e.Pos(), t) // TODO(gri) is this the correct position?
			if !tset.IsMethodSet() {
				if tset.comparable {
					check.softErrorf(e, MisplacedConstraintIface, "cannot use type %s outside a type constraint: interface is (or embeds) comparable", typ)
				} else {
					check.softErrorf(e, MisplacedConstraintIface, "cannot use type %s outside a type constraint: interface contains type constraints", typ)
				}
			}
		}
	}).describef(e, "check var type %s", typ)
}

// declaredType is like typ but also accepts a type name def.
// If def != nil, e is the type specification for the [Alias] or [Named] type
// named def, and def.typ.fromRHS will be set to the [Type] of e immediately
// after its creation.
func (check *Checker) declaredType(e ast.Expr, def *TypeName) Type {
	typ := check.typInternal(e, def)
	assert(isTyped(typ))
	if isGeneric(typ) {
		check.errorf(e, WrongTypeArgCount, "cannot use generic type %s without instantiation", typ)
		typ = Typ[Invalid]
	}
	check.recordTypeAndValue(e, typexpr, typ, nil)
	return typ
}

// genericType is like typ but the type must be an (uninstantiated) generic
// type. If cause is non-nil and the type expression was a valid type but not
// generic, cause will be populated with a message describing the error.
//
// Note: If the type expression was invalid and an error was reported before,
// cause will not be populated; thus cause alone cannot be used to determine
// if an error occurred.
func (check *Checker) genericType(e ast.Expr, cause *string) Type {
	typ := check.typInternal(e, nil)
	assert(isTyped(typ))
	if isValid(typ) && !isGeneric(typ) {
		if cause != nil {
			*cause = check.sprintf("%s is not a generic type", typ)
		}
		typ = Typ[Invalid]
	}
	// TODO(gri) what is the correct call below?
	check.recordTypeAndValue(e, typexpr, typ, nil)
	return typ
}

// goTypeName returns the Go type name for typ and
// removes any occurrences of "types." from that name.
func goTypeName(typ Type) string {
	return strings.ReplaceAll(fmt.Sprintf("%T", typ), "types.", "")
}

// typInternal drives type checking of types.
// Must only be called by declaredType or genericType.
func (check *Checker) typInternal(e0 ast.Expr, def *TypeName) (T Type) {
	if check.conf._Trace {
		check.trace(e0.Pos(), "-- type %s", e0)
		check.indent++
		defer func() {
			check.indent--
			var under Type
			if T != nil {
				// Calling T.Underlying() here may lead to endless instantiations.
				// Test case: type T[P any] *T[P]
				under = safeUnderlying(T)
			}
			if T == under {
				check.trace(e0.Pos(), "=> %s // %s", T, goTypeName(T))
			} else {
				check.trace(e0.Pos(), "=> %s (under = %s) // %s", T, under, goTypeName(T))
			}
		}()
	}

	switch e := e0.(type) {
	case *ast.BadExpr:
		// ignore - error reported before

	case *ast.Ident:
		var x operand
		check.ident(&x, e, true)

		switch x.mode {
		case typexpr:
			return x.typ
		case invalid:
			// ignore - error reported before
		case novalue:
			check.errorf(&x, NotAType, "%s used as type", &x)
		default:
			check.errorf(&x, NotAType, "%s is not a type", &x)
		}

	case *ast.SelectorExpr:
		var x operand
		check.selector(&x, e, true)

		switch x.mode {
		case typexpr:
			return x.typ
		case invalid:
			// ignore - error reported before
		case novalue:
			check.errorf(&x, NotAType, "%s used as type", &x)
		default:
			check.errorf(&x, NotAType, "%s is not a type", &x)
		}

	case *ast.IndexExpr, *ast.IndexListExpr:
		ix := unpackIndexedExpr(e)
		check.verifyVersionf(inNode(e, ix.lbrack), go1_18, "type instantiation")
		return check.instantiatedType(ix)

	case *ast.ParenExpr:
		// Generic types must be instantiated before they can be used in any form.
		// Consequently, generic types cannot be parenthesized.
		return check.declaredType(e.X, def)

	case *ast.ArrayType:
		if e.Len == nil {
			typ := new(Slice)
			typ.elem = check.varType(e.Elt)
			return typ
		}

		typ := new(Array)
		// Provide a more specific error when encountering a [...] array
		// rather than leaving it to the handling of the ... expression.
		if _, ok := e.Len.(*ast.Ellipsis); ok {
			check.error(e.Len, BadDotDotDotSyntax, "invalid use of [...] array (outside a composite literal)")
			typ.len = -1
		} else {
			typ.len = check.arrayLength(e.Len)
		}
		typ.elem = check.varType(e.Elt)
		if typ.len >= 0 {
			return typ
		}
		// report error if we encountered [...]

	case *ast.Ellipsis:
		// dots are handled explicitly where they are valid
		check.error(e, InvalidSyntaxTree, "invalid use of ...")

	case *ast.StructType:
		typ := new(Struct)
		check.structType(typ, e)
		return typ

	case *ast.StarExpr:
		typ := new(Pointer)
		typ.base = Typ[Invalid] // avoid nil base in invalid recursive type declaration
		typ.base = check.varType(e.X)
		// If typ.base is invalid, it's unlikely that *base is particularly
		// useful - even a valid dereferenciation will lead to an invalid
		// type again, and in some cases we get unexpected follow-on errors
		// (e.g., go.dev/issue/49005). Return an invalid type instead.
		if !isValid(typ.base) {
			return Typ[Invalid]
		}
		return typ

	case *ast.FuncType:
		typ := new(Signature)
		check.funcType(typ, nil, e)
		return typ

	case *ast.InterfaceType:
		typ := check.newInterface()
		check.interfaceType(typ, e, def)
		return typ

	case *ast.MapType:
		typ := new(Map)
		typ.key = check.varType(e.Key)
		typ.elem = check.varType(e.Value)

		// spec: "The comparison operators == and != must be fully defined
		// for operands of the key type; thus the key type must not be a
		// function, map, or slice."
		//
		// Delay this check because it requires fully setup types;
		// it is safe to continue in any case (was go.dev/issue/6667).
		check.later(func() {
			if !Comparable(typ.key) {
				var why string
				if isTypeParam(typ.key) {
					why = " (missing comparable constraint)"
				}
				check.errorf(e.Key, IncomparableMapKey, "invalid map key type %s%s", typ.key, why)
			}
		}).describef(e.Key, "check map key %s", typ.key)

		return typ

	case *ast.ChanType:
		typ := new(Chan)

		dir := SendRecv
		switch e.Dir {
		case ast.SEND | ast.RECV:
			// nothing to do
		case ast.SEND:
			dir = SendOnly
		case ast.RECV:
			dir = RecvOnly
		default:
			check.errorf(e, InvalidSyntaxTree, "unknown channel direction %d", e.Dir)
			// ok to continue
		}

		typ.dir = dir
		typ.elem = check.varType(e.Value)
		return typ

	default:
		check.errorf(e0, NotAType, "%s is not a type", e0)
		check.use(e0)
	}

	typ := Typ[Invalid]
	return typ
}

func (check *Checker) instantiatedType(ix *indexedExpr) (res Type) {
	if check.conf._Trace {
		check.trace(ix.Pos(), "-- instantiating type %s with %s", ix.x, ix.indices)
		check.indent++
		defer func() {
			check.indent--
			// Don't format the underlying here. It will always be nil.
			check.trace(ix.Pos(), "=> %s", res)
		}()
	}

	var cause string
	typ := check.genericType(ix.x, &cause)
	if cause != "" {
		check.errorf(ix.orig, NotAGenericType, invalidOp+"%s (%s)", ix.orig, cause)
	}
	if !isValid(typ) {
		return typ // error already reported
	}
	// typ must be a generic Alias or Named type (but not a *Signature)
	if _, ok := typ.(*Signature); ok {
		panic("unexpected generic signature")
	}
	gtyp := typ.(genericType)

	// evaluate arguments
	targs := check.typeList(ix.indices)
	if targs == nil {
		return Typ[Invalid]
	}

	// create instance
	// The instance is not generic anymore as it has type arguments, but unless
	// instantiation failed, it still satisfies the genericType interface because
	// it has type parameters, too.
	ityp := check.instance(ix.Pos(), gtyp, targs, nil, check.context())
	inst, _ := ityp.(genericType)
	if inst == nil {
		return Typ[Invalid]
	}

	// For Named types, orig.tparams may not be set up, so we need to do expansion later.
	check.later(func() {
		// This is an instance from the source, not from recursive substitution,
		// and so it must be resolved during type-checking so that we can report
		// errors.
		check.recordInstance(ix.orig, targs, inst)

		name := inst.(interface{ Obj() *TypeName }).Obj().name
		tparams := inst.TypeParams().list()
		if check.validateTArgLen(ix.Pos(), name, len(tparams), len(targs)) {
			// check type constraints
			if i, err := check.verify(ix.Pos(), inst.TypeParams().list(), targs, check.context()); err != nil {
				// best position for error reporting
				pos := ix.Pos()
				if i < len(ix.indices) {
					pos = ix.indices[i].Pos()
				}
				check.softErrorf(atPos(pos), InvalidTypeArg, "%v", err)
			} else {
				check.mono.recordInstance(check.pkg, ix.Pos(), tparams, targs, ix.indices)
			}
		}
	}).describef(ix, "verify instantiation %s", inst)

	return inst
}

// arrayLength type-checks the array length expression e
// and returns the constant length >= 0, or a value < 0
// to indicate an error (and thus an unknown length).
func (check *Checker) arrayLength(e ast.Expr) int64 {
	// If e is an identifier, the array declaration might be an
	// attempt at a parameterized type declaration with missing
	// constraint. Provide an error message that mentions array
	// length.
	if name, _ := e.(*ast.Ident); name != nil {
		obj := check.lookup(name.Name)
		if obj == nil {
			check.errorf(name, InvalidArrayLen, "undefined array length %s or missing type constraint", name.Name)
			return -1
		}
		if _, ok := obj.(*Const); !ok {
			check.errorf(name, InvalidArrayLen, "invalid array length %s", name.Name)
			return -1
		}
	}

	var x operand
	check.expr(nil, &x, e)
	if x.mode != constant_ {
		if x.mode != invalid {
			check.errorf(&x, InvalidArrayLen, "array length %s must be constant", &x)
		}
		return -1
	}

	if isUntyped(x.typ) || isInteger(x.typ) {
		if val := constant.ToInt(x.val); val.Kind() == constant.Int {
			if representableConst(val, check, Typ[Int], nil) {
				if n, ok := constant.Int64Val(val); ok && n >= 0 {
					return n
				}
			}
		}
	}

	var msg string
	if isInteger(x.typ) {
		msg = "invalid array length %s"
	} else {
		msg = "array length %s must be integer"
	}
	check.errorf(&x, InvalidArrayLen, msg, &x)
	return -1
}

// typeList provides the list of types corresponding to the incoming expression list.
// If an error occurred, the result is nil, but all list elements were type-checked.
func (check *Checker) typeList(list []ast.Expr) []Type {
	res := make([]Type, len(list)) // res != nil even if len(list) == 0
	for i, x := range list {
		t := check.varType(x)
		if !isValid(t) {
			res = nil
		}
		if res != nil {
			res[i] = t
		}
	}
	return res
}
