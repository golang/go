// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type-checking of identifiers and type expressions.

package types

import (
	"go/ast"
	"go/token"
	"sort"
	"strconv"

	"code.google.com/p/go.tools/go/exact"
)

// ident type-checks identifier e and initializes x with the value or type of e.
// If an error occurred, x.mode is set to invalid.
// For the meaning of def and cycleOk, see check.typ, below.
//
func (check *checker) ident(x *operand, e *ast.Ident, def *Named, cycleOk bool) {
	x.mode = invalid
	x.expr = e

	obj := check.topScope.LookupParent(e.Name)
	if obj == nil {
		if e.Name == "_" {
			check.errorf(e.Pos(), "cannot use _ as value or type")
		} else {
			check.errorf(e.Pos(), "undeclared name: %s", e.Name)
		}
		return
	}
	check.recordObject(e, obj)

	typ := obj.Type()
	if typ == nil {
		// object not yet declared
		if check.objMap == nil {
			check.dump("%s: %s should have been declared (we are inside a function)", e.Pos(), e)
			unreachable()
		}
		check.objDecl(obj, def, cycleOk)
		typ = obj.Type()
	}
	assert(typ != nil)

	switch obj := obj.(type) {
	case *PkgName:
		check.errorf(e.Pos(), "use of package %s not in selector", obj.name)
		return

	case *Const:
		// The constant may be dot-imported. Mark it as used so that
		// later we can determine if the corresponding dot-imported
		// package was used. Same applies for other objects, below.
		// (This code is only used for dot-imports. Without them, we
		// would only have to mark Vars.)
		obj.used = true
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
			x.val = obj.val
		}
		assert(x.val != nil)
		x.mode = constant

	case *TypeName:
		obj.used = true
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
		obj.used = true
		x.mode = variable
		if typ != Typ[Invalid] {
			check.addDeclDep(obj)
		}

	case *Func:
		obj.used = true
		x.mode = value
		if typ != Typ[Invalid] {
			check.addDeclDep(obj)
		}

	case *Builtin:
		obj.used = true // for built-ins defined by package unsafe
		x.mode = builtin
		x.id = obj.id

	case *Nil:
		// no need to "use" the nil object
		x.mode = value

	default:
		unreachable()
	}

	x.typ = typ
}

// typ type-checks the type expression e and returns its type, or Typ[Invalid].
// If def != nil, e is the type specification for the named type def, declared
// in a type declaration, and def.underlying will be set to the type of e before
// any components of e are type-checked.
// If cycleOk is set, e (or elements of e) may refer to a named type that is not
// yet completely set up.
//
func (check *checker) typ(e ast.Expr, def *Named, cycleOk bool) Type {
	if trace {
		check.trace(e.Pos(), "%s", e)
		check.indent++
	}

	t := check.typInternal(e, def, cycleOk)
	assert(e != nil && t != nil && isTyped(t))

	check.recordTypeAndValue(e, t, nil)

	if trace {
		check.indent--
		check.trace(e.Pos(), "=> %s", t)
	}

	return t
}

// funcType type-checks a function or method type and returns its signature.
func (check *checker) funcType(recv *ast.FieldList, ftyp *ast.FuncType, def *Named) *Signature {
	sig := new(Signature)
	if def != nil {
		def.underlying = sig
	}

	scope := NewScope(check.topScope)
	check.recordScope(ftyp, scope)

	recv_, _ := check.collectParams(scope, recv, false)
	params, isVariadic := check.collectParams(scope, ftyp.Params, true)
	results, _ := check.collectParams(scope, ftyp.Results, false)

	if len(recv_) > 0 {
		// There must be exactly one receiver.
		if len(recv_) > 1 {
			check.invalidAST(recv_[1].Pos(), "method must have exactly one receiver")
			// ok to continue
		}
		recv := recv_[0]
		// spec: "The receiver type must be of the form T or *T where T is a type name."
		// (ignore invalid types - error was reported before)
		if t, _ := deref(recv.typ); t != Typ[Invalid] {
			var err string
			if T, _ := t.(*Named); T != nil {
				// spec: "The type denoted by T is called the receiver base type; it must not
				// be a pointer or interface type and it must be declared in the same package
				// as the method."
				if T.obj.pkg != check.pkg {
					err = "type not defined in this package"
				} else {
					// TODO(gri) This is not correct if the underlying type is unknown yet.
					switch u := T.underlying.(type) {
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

	sig.scope = scope
	sig.params = NewTuple(params...)
	sig.results = NewTuple(results...)
	sig.isVariadic = isVariadic

	return sig
}

// typInternal contains the core of type checking of types.
// Must only be called by typ.
//
func (check *checker) typInternal(e ast.Expr, def *Named, cycleOk bool) Type {
	switch e := e.(type) {
	case *ast.BadExpr:
		// ignore - error reported before

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
			typ.elem = check.typ(e.Elt, nil, cycleOk)
			return typ

		} else {
			typ := new(Slice)
			if def != nil {
				def.underlying = typ
			}

			typ.elem = check.typ(e.Elt, nil, true)
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
		return check.funcType(nil, e, def)

	case *ast.InterfaceType:
		return check.interfaceType(e, def, cycleOk)

	case *ast.MapType:
		typ := new(Map)
		if def != nil {
			def.underlying = typ
		}

		typ.key = check.typ(e.Key, nil, true)
		typ.elem = check.typ(e.Value, nil, true)

		// spec: "The comparison operators == and != must be fully defined
		// for operands of the key type; thus the key type must not be a
		// function, map, or slice."
		//
		// Delay this check because it requires fully setup types;
		// it is safe to continue in any case (was issue 6667).
		check.delay(func() {
			if !isComparable(typ.key) {
				check.errorf(e.Key.Pos(), "invalid map key type %s", typ.key)
			}
		})

		return typ

	case *ast.ChanType:
		typ := new(Chan)
		if def != nil {
			def.underlying = typ
		}

		dir := SendRecv
		switch e.Dir {
		case ast.SEND | ast.RECV:
			// nothing to do
		case ast.SEND:
			dir = SendOnly
		case ast.RECV:
			dir = RecvOnly
		default:
			check.invalidAST(e.Pos(), "unknown channel direction %d", e.Dir)
			// ok to continue
		}
		typ.dir = dir
		typ.elem = check.typ(e.Value, nil, true)
		return typ

	default:
		check.errorf(e.Pos(), "%s is not a type", e)
	}

	return Typ[Invalid]
}

// typeOrNil type-checks the type expression (or nil value) e
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
	case value:
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

	for i, field := range list.List {
		ftype := field.Type
		if t, _ := ftype.(*ast.Ellipsis); t != nil {
			ftype = t.Elt
			if variadicOk && i == len(list.List)-1 {
				isVariadic = true
			} else {
				check.invalidAST(field.Pos(), "... not permitted")
				// ignore ... and continue
			}
		}
		typ := check.typ(ftype, nil, true)
		// The parser ensures that f.Tag is nil and we don't
		// care if a constructed AST contains a non-nil tag.
		if len(field.Names) > 0 {
			// named parameter
			for _, name := range field.Names {
				par := NewParam(name.Pos(), check.pkg, name.Name, typ)
				check.declare(scope, name, par)
				params = append(params, par)
			}
		} else {
			// anonymous parameter
			par := NewParam(ftype.Pos(), check.pkg, "", typ)
			check.recordImplicit(field, par)
			params = append(params, par)
		}
	}

	// For a variadic function, change the last parameter's type from T to []T.
	if isVariadic && len(params) > 0 {
		last := params[len(params)-1]
		last.typ = &Slice{elem: last.typ}
	}

	return
}

func (check *checker) declareInSet(oset *objset, pos token.Pos, obj Object) bool {
	if alt := oset.insert(obj); alt != nil {
		check.errorf(pos, "%s redeclared", obj.Name())
		check.reportAltDecl(alt)
		return false
	}
	return true
}

func (check *checker) interfaceType(ityp *ast.InterfaceType, def *Named, cycleOk bool) *Interface {
	iface := new(Interface)
	if def != nil {
		def.underlying = iface
	}

	// empty interface: common case
	if ityp.Methods == nil {
		return iface
	}

	// The parser ensures that field tags are nil and we don't
	// care if a constructed AST contains non-nil tags.

	// Phase 1: Collect explicitly declared methods, the corresponding
	//          signature (AST) expressions, and the list of embedded
	//          type (AST) expressions. Do not resolve signatures or
	//          embedded types yet to avoid cycles referring to this
	//          interface.

	var (
		mset       objset
		signatures []ast.Expr // list of corresponding method signatures
		embedded   []ast.Expr // list of embedded types
	)
	for _, f := range ityp.Methods.List {
		if len(f.Names) > 0 {
			// The parser ensures that there's only one method
			// and we don't care if a constructed AST has more.
			name := f.Names[0]
			pos := name.Pos()
			// Don't type-check signature yet - use an
			// empty signature now and update it later.
			m := NewFunc(pos, check.pkg, name.Name, new(Signature))
			// spec: "As with all method sets, in an interface type,
			// each method must have a unique name."
			// (The spec does not exclude blank _ identifiers for
			// interface methods.)
			if check.declareInSet(&mset, pos, m) {
				iface.methods = append(iface.methods, m)
				iface.allMethods = append(iface.allMethods, m)
				signatures = append(signatures, f.Type)
				check.recordObject(name, m)
			}
		} else {
			// embedded type
			embedded = append(embedded, f.Type)
		}
	}

	// Phase 2: Resolve embedded interfaces. Because an interface must not
	//          embed itself (directly or indirectly), each embedded interface
	//          can be fully resolved without depending on any method of this
	//          interface (if there is a cycle or another error, the embedded
	//          type resolves to an invalid type and is ignored).
	//          In particular, the list of methods for each embedded interface
	//          must be complete (it cannot depend on this interface), and so
	//          those methods can be added to the list of all methods of this
	//          interface.

	for _, e := range embedded {
		pos := e.Pos()
		typ := check.typ(e, nil, cycleOk)
		if typ == Typ[Invalid] {
			continue
		}
		named, _ := typ.(*Named)
		if named == nil {
			check.invalidAST(pos, "%s is not named type", typ)
			continue
		}
		// determine underlying (possibly incomplete) type
		// by following its forward chain
		// TODO(gri) should this be part of Underlying()?
		u := named.underlying
		for {
			n, _ := u.(*Named)
			if n == nil {
				break
			}
			u = n.underlying
		}
		if u == Typ[Invalid] {
			continue
		}
		embed, _ := u.(*Interface)
		if embed == nil {
			check.errorf(pos, "%s is not an interface", named)
			continue
		}
		iface.embeddeds = append(iface.embeddeds, named)
		// collect embedded methods
		for _, m := range embed.allMethods {
			if check.declareInSet(&mset, pos, m) {
				iface.allMethods = append(iface.allMethods, m)
			}
		}
	}

	// Phase 3: At this point all methods have been collected for this interface.
	//          It is now safe to type-check the signatures of all explicitly
	//          declared methods, even if they refer to this interface via a cycle
	//          and embed the methods of this interface in a parameter of interface
	//          type.

	// determine receiver type
	var recv Type = iface
	if def != nil {
		def.underlying = iface
		recv = def // use named receiver type if available
	}

	for i, m := range iface.methods {
		expr := signatures[i]
		typ := check.typ(expr, nil, true)
		if typ == Typ[Invalid] {
			continue // keep method with empty method signature
		}
		sig, _ := typ.(*Signature)
		if sig == nil {
			check.invalidAST(expr.Pos(), "%s is not a method signature", typ)
			continue // keep method with empty method signature
		}
		sig.recv = NewVar(m.pos, check.pkg, "", recv)
		*m.typ.(*Signature) = *sig // update signature (don't replace it!)
	}

	// TODO(gri) The list of explicit methods is only sorted for now to
	// produce the same Interface as NewInterface. We may be able to
	// claim source order in the future. Revisit.
	sort.Sort(byUniqueMethodName(iface.methods))

	// TODO(gri) The list of embedded types is only sorted for now to
	// produce the same Interface as NewInterface. We may be able to
	// claim source order in the future. Revisit.
	sort.Sort(byUniqueTypeName(iface.embeddeds))

	sort.Sort(byUniqueMethodName(iface.allMethods))

	return iface
}

// byUniqueTypeName named type lists can be sorted by their unique type names.
type byUniqueTypeName []*Named

func (a byUniqueTypeName) Len() int           { return len(a) }
func (a byUniqueTypeName) Less(i, j int) bool { return a[i].obj.Id() < a[j].obj.Id() }
func (a byUniqueTypeName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// byUniqueMethodName method lists can be sorted by their unique method names.
type byUniqueMethodName []*Func

func (a byUniqueMethodName) Len() int           { return len(a) }
func (a byUniqueMethodName) Less(i, j int) bool { return a[i].Id() < a[j].Id() }
func (a byUniqueMethodName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

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

func (check *checker) collectFields(list *ast.FieldList, cycleOk bool) (fields []*Var, tags []string) {
	if list == nil {
		return
	}

	var fset objset

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
		// spec: "Within a struct, non-blank field names must be unique."
		if name == "_" || check.declareInSet(&fset, pos, fld) {
			fields = append(fields, fld)
			if ident != nil {
				check.recordObject(ident, fld)
			}
		}
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
				if t == Typ[Invalid] {
					// error was reported before
					continue
				}
				// unsafe.Pointer is treated like a regular pointer
				if t.kind == UnsafePointer {
					check.errorf(pos, "anonymous field type cannot be unsafe.Pointer")
					continue
				}
				add(f, nil, t.name, true, pos)

			case *Named:
				// spec: "An embedded type must be specified as a type name
				// T or as a pointer to a non-interface type name *T, and T
				// itself may not be a pointer type."
				switch u := t.Underlying().(type) {
				case *Basic:
					// unsafe.Pointer is treated like a regular pointer
					if u.kind == UnsafePointer {
						check.errorf(pos, "anonymous field type cannot be unsafe.Pointer")
						continue
					}
				case *Pointer:
					check.errorf(pos, "anonymous field type cannot be a pointer")
					continue
				case *Interface:
					if isPtr {
						check.errorf(pos, "anonymous field type cannot be a pointer to an interface")
						continue
					}
				}
				add(f, nil, t.obj.name, true, pos)

			default:
				check.invalidAST(pos, "anonymous field type %s must be named", typ)
			}
		}
	}

	return
}
