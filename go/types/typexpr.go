// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of identifiers and type expressions.

package types

import (
	"go/ast"
	"go/token"
	"strconv"

	"code.google.com/p/go.tools/go/exact"
)

// ident typechecks identifier e and initializes x with the value or type of e.
// If an error occurred, x.mode is set to invalid.
// For the meaning of def and cycleOk, see check.typ, below.
//
func (check *checker) ident(x *operand, e *ast.Ident, def *Named, cycleOk bool) {
	x.mode = invalid
	x.expr = e

	obj := check.topScope.LookupParent(e.Name)
	check.callIdent(e, obj)
	if obj == nil {
		if e.Name == "_" {
			check.errorf(e.Pos(), "cannot use _ as value or type")
		} else {
			check.errorf(e.Pos(), "undeclared name: %s", e.Name)
		}
		return
	}

	typ := obj.Type()
	if typ == nil {
		// object not yet declared
		if check.objMap == nil {
			check.dump("%s: %s should have been declared (we are inside a function)", e.Pos(), e)
			unreachable()
		}
		check.declareObject(obj, def, cycleOk)
		typ = obj.Type()
	}
	assert(typ != nil)

	switch obj := obj.(type) {
	case *Package:
		check.errorf(e.Pos(), "use of package %s not in selector", obj.name)
		return

	case *Const:
		if typ == Typ[Invalid] {
			return
		}
		if obj == universeIota {
			if check.iota == nil {
				check.errorf(e.Pos(), "cannot use iota outside constant declaration")
				return
			}
			x.val = check.iota
		} else {
			x.val = obj.val // may be nil if we don't know the constant value
		}
		x.mode = constant

	case *TypeName:
		x.mode = typexpr
		named, _ := typ.(*Named)
		if !cycleOk && named != nil && !named.complete {
			check.errorf(obj.pos, "illegal cycle in declaration of %s", obj.name)
			// maintain x.mode == typexpr despite error
			typ = Typ[Invalid]
		}
		if def != nil {
			def.underlying = typ
		}

	case *Var:
		x.mode = variable

	case *Func:
		x.mode = value

	default:
		unreachable()
	}

	x.typ = typ
}

// typ typechecks the type expression e and initializes x with the type of e.
// If an error occurred, x.mode is set to invalid.
// If def != nil, e is the type specification for the named type def, declared
// in a type declaration, and def.underlying will be set to the type of e before
// any components of e are typechecked.
// If cycleOk is set, e (or elements of e) may refer to a named type that is not
// yet completely set up.
//
func (check *checker) typ(e ast.Expr, def *Named, cycleOk bool) (res Type) {
	if trace {
		check.trace(e.Pos(), "%s", e)
		defer check.untrace("=> %s", res)
	}

	// notify clients of type
	if f := check.ctxt.Expr; f != nil {
		defer func() {
			assert(e != nil && res != nil && !isUntyped(res))
			f(e, res, nil)
		}()
	}

	switch e := e.(type) {
	case *ast.Ident:
		var x operand
		check.ident(&x, e, def, cycleOk)

		switch x.mode {
		case typexpr:
			return x.typ
		case invalid:
			// ignore - error reported before
		case novalue:
			check.errorf(x.pos(), "%s used as type", &x)
		default:
			check.errorf(x.pos(), "%s is not a type", &x)
		}

	case *ast.SelectorExpr:
		var x operand
		check.selector(&x, e)

		switch x.mode {
		case typexpr:
			return x.typ
		case invalid:
			// ignore - error reported before
		case novalue:
			check.errorf(x.pos(), "%s used as type", &x)
		default:
			check.errorf(x.pos(), "%s is not a type", &x)
		}

	case *ast.ParenExpr:
		return check.typ(e.X, def, cycleOk)

	case *ast.ArrayType:
		if e.Len != nil {
			var x operand
			check.expr(&x, e.Len)
			if x.mode != constant {
				if x.mode != invalid {
					check.errorf(x.pos(), "array length %s must be constant", &x)
				}
				break
			}
			if !x.isInteger() {
				check.errorf(x.pos(), "array length %s must be integer", &x)
				break
			}
			n, ok := exact.Int64Val(x.val)
			if !ok || n < 0 {
				check.errorf(x.pos(), "invalid array length %s", &x)
				break
			}

			typ := new(Array)
			if def != nil {
				def.underlying = typ
			}

			typ.len = n
			typ.elt = check.typ(e.Elt, nil, cycleOk)
			return typ

		} else {
			typ := new(Slice)
			if def != nil {
				def.underlying = typ
			}

			typ.elt = check.typ(e.Elt, nil, true)
			return typ
		}

	case *ast.StructType:
		typ := new(Struct)
		if def != nil {
			def.underlying = typ
		}

		typ.fields, typ.tags = check.collectFields(e.Fields, cycleOk)
		return typ

	case *ast.StarExpr:
		typ := new(Pointer)
		if def != nil {
			def.underlying = typ
		}

		typ.base = check.typ(e.X, nil, true)
		return typ

	case *ast.FuncType:
		typ := new(Signature)
		if def != nil {
			def.underlying = typ
		}

		scope := NewScope(check.topScope)
		if retainASTLinks {
			scope.node = e
		}
		typ.scope = scope
		params, isVariadic := check.collectParams(scope, e.Params, true)
		results, _ := check.collectParams(scope, e.Results, false)
		typ.params = NewTuple(params...)
		typ.results = NewTuple(results...)
		typ.isVariadic = isVariadic
		return typ

	case *ast.InterfaceType:
		typ := new(Interface)
		if def != nil {
			def.underlying = typ
		}

		typ.methods = check.collectMethods(e.Methods, cycleOk)
		return typ

	case *ast.MapType:
		typ := new(Map)
		if def != nil {
			def.underlying = typ
		}

		typ.key = check.typ(e.Key, nil, true)
		typ.elt = check.typ(e.Value, nil, true)
		return typ

	case *ast.ChanType:
		typ := new(Chan)
		if def != nil {
			def.underlying = typ
		}

		typ.dir = e.Dir
		typ.elt = check.typ(e.Value, nil, true)
		return typ

	default:
		check.errorf(e.Pos(), "%s is not a type", e)
	}

	return Typ[Invalid]
}

// typeOrNil typechecks the type expression (or nil value) e
// and returns the typ of e, or nil.
// If e is neither a type nor nil, typOrNil returns Typ[Invalid].
//
func (check *checker) typOrNil(e ast.Expr) Type {
	var x operand
	check.rawExpr(&x, e, nil)
	switch x.mode {
	case invalid:
		// ignore - error reported before
	case novalue:
		check.errorf(x.pos(), "%s used as type", &x)
	case typexpr:
		return x.typ
	case constant:
		if x.isNil() {
			return nil
		}
		fallthrough
	default:
		check.errorf(x.pos(), "%s is not a type", &x)
	}
	return Typ[Invalid]
}

func (check *checker) collectParams(scope *Scope, list *ast.FieldList, variadicOk bool) (params []*Var, isVariadic bool) {
	if list == nil {
		return
	}
	var last *Var
	for i, field := range list.List {
		ftype := field.Type
		if t, _ := ftype.(*ast.Ellipsis); t != nil {
			ftype = t.Elt
			if variadicOk && i == len(list.List)-1 {
				isVariadic = true
			} else {
				check.invalidAST(field.Pos(), "... not permitted")
				// ok to continue
			}
		}
		// the parser ensures that f.Tag is nil and we don't
		// care if a constructed AST contains a non-nil tag
		typ := check.typ(ftype, nil, true)
		if len(field.Names) > 0 {
			// named parameter
			for _, name := range field.Names {
				par := NewVar(name.Pos(), check.pkg, name.Name, typ)
				check.declare(scope, name, par)

				last = par
				copy := *par
				params = append(params, &copy)
			}
		} else {
			// anonymous parameter
			par := NewVar(ftype.Pos(), check.pkg, "", typ)
			check.callImplicitObj(field, par)

			last = nil // not accessible inside function
			params = append(params, par)
		}
	}
	// For a variadic function, change the last parameter's object type
	// from T to []T (this is the type used inside the function), but
	// keep the params list unchanged (this is the externally visible type).
	if isVariadic && last != nil {
		last.typ = &Slice{elt: last.typ}
	}
	return
}

func (check *checker) collectMethods(list *ast.FieldList, cycleOk bool) (methods []*Func) {
	if list == nil {
		return nil
	}
	scope := NewScope(nil)
	for _, f := range list.List {
		typ := check.typ(f.Type, nil, cycleOk)
		// the parser ensures that f.Tag is nil and we don't
		// care if a constructed AST contains a non-nil tag
		if len(f.Names) > 0 {
			// methods (the parser ensures that there's only one
			// and we don't care if a constructed AST has more)
			sig, ok := typ.(*Signature)
			if !ok {
				check.invalidAST(f.Type.Pos(), "%s is not a method signature", typ)
				continue
			}
			for _, name := range f.Names {
				m := NewFunc(name.Pos(), check.pkg, name.Name, sig)
				check.declare(scope, name, m)
				methods = append(methods, m)
			}
		} else {
			// embedded interface
			switch t := typ.Underlying().(type) {
			case nil:
				// The underlying type is in the process of being defined
				// but we need it in order to complete this type. For now
				// complain with an "unimplemented" error. This requires
				// a bit more work.
				// TODO(gri) finish this.
				check.errorf(f.Type.Pos(), "reference to incomplete type %s - unimplemented", f.Type)
			case *Interface:
				for _, m := range t.methods {
					check.declare(scope, nil, m)
					methods = append(methods, m)
				}
			default:
				if t != Typ[Invalid] {
					check.errorf(f.Type.Pos(), "%s is not an interface type", typ)
				}
			}
		}
	}
	return
}

func (check *checker) tag(t *ast.BasicLit) string {
	if t != nil {
		if t.Kind == token.STRING {
			if val, err := strconv.Unquote(t.Value); err == nil {
				return val
			}
		}
		check.invalidAST(t.Pos(), "incorrect tag syntax: %q", t.Value)
	}
	return ""
}

func (check *checker) collectFields(list *ast.FieldList, cycleOk bool) (fields []*Field, tags []string) {
	if list == nil {
		return
	}

	scope := NewScope(nil)

	var typ Type   // current field typ
	var tag string // current field tag
	add := func(field *ast.Field, ident *ast.Ident, name string, anonymous bool, pos token.Pos) {
		if tag != "" && tags == nil {
			tags = make([]string, len(fields))
		}
		if tags != nil {
			tags = append(tags, tag)
		}

		fld := NewField(pos, check.pkg, name, typ, anonymous)
		check.declare(scope, ident, fld)
		fields = append(fields, fld)
	}

	for _, f := range list.List {
		typ = check.typ(f.Type, nil, cycleOk)
		tag = check.tag(f.Tag)
		if len(f.Names) > 0 {
			// named fields
			for _, name := range f.Names {
				add(f, name, name.Name, false, name.Pos())
			}
		} else {
			// anonymous field
			pos := f.Type.Pos()
			t, isPtr := deref(typ)
			switch t := t.(type) {
			case *Basic:
				add(f, nil, t.name, true, pos)
			case *Named:
				// spec: "An embedded type must be specified as a type name
				// T or as a pointer to a non-interface type name *T, and T
				// itself may not be a pointer type."
				switch t.Underlying().(type) {
				case *Pointer:
					check.errorf(pos, "anonymous field type cannot be a pointer")
					continue // ignore this field
				case *Interface:
					if isPtr {
						check.errorf(pos, "anonymous field type cannot be a pointer to an interface")
						continue // ignore this field
					}
				}
				add(f, nil, t.obj.name, true, pos)
			default:
				if typ != Typ[Invalid] {
					check.invalidAST(pos, "anonymous field type %s must be named", typ)
				}
			}
		}
	}

	return
}
