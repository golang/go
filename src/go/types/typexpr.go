// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type-checking of identifiers and type expressions.

package types

import (
	"go/ast"
	"go/constant"
	"go/token"
	"sort"
	"strconv"
)

// ident type-checks identifier e and initializes x with the value or type of e.
// If an error occurred, x.mode is set to invalid.
// For the meaning of def, see Checker.definedType, below.
// If wantType is set, the identifier e is expected to denote a type.
//
func (check *Checker) ident(x *operand, e *ast.Ident, def *Named, wantType bool) {
	x.mode = invalid
	x.expr = e

	// Note that we cannot use check.lookup here because the returned scope
	// may be different from obj.Parent(). See also Scope.LookupParent doc.
	scope, obj := check.scope.LookupParent(e.Name, check.pos)
	if obj == nil {
		if e.Name == "_" {
			check.errorf(e, _InvalidBlank, "cannot use _ as value or type")
		} else {
			check.errorf(e, _UndeclaredName, "undeclared name: %s", e.Name)
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
		check.errorf(e, _InvalidPkgUse, "use of package %s not in selector", obj.name)
		return

	case *Const:
		check.addDeclDep(obj)
		if typ == Typ[Invalid] {
			return
		}
		if obj == universeIota {
			if check.iota == nil {
				check.errorf(e, _InvalidIota, "cannot use iota outside constant declaration")
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
func (check *Checker) typ(e ast.Expr) Type {
	return check.definedType(e, nil)
}

// definedType is like typ but also accepts a type name def.
// If def != nil, e is the type specification for the defined type def, declared
// in a type declaration, and def.underlying will be set to the type of e before
// any components of e are type-checked.
//
func (check *Checker) definedType(e ast.Expr, def *Named) (T Type) {
	if trace {
		check.trace(e.Pos(), "%s", e)
		check.indent++
		defer func() {
			check.indent--
			check.trace(e.Pos(), "=> %s", T)
		}()
	}

	T = check.typInternal(e, def)
	assert(isTyped(T))
	check.recordTypeAndValue(e, typexpr, T, nil)

	return
}

// funcType type-checks a function or method type.
func (check *Checker) funcType(sig *Signature, recvPar *ast.FieldList, ftyp *ast.FuncType) {
	scope := NewScope(check.scope, token.NoPos, token.NoPos, "function")
	scope.isFunc = true
	check.recordScope(ftyp, scope)

	recvList, _ := check.collectParams(scope, recvPar, false)
	params, variadic := check.collectParams(scope, ftyp.Params, true)
	results, _ := check.collectParams(scope, ftyp.Results, false)

	if recvPar != nil {
		// recv parameter list present (may be empty)
		// spec: "The receiver is specified via an extra parameter section preceding the
		// method name. That parameter section must declare a single parameter, the receiver."
		var recv *Var
		switch len(recvList) {
		case 0:
			check.error(recvPar, _BadRecv, "method is missing receiver")
			recv = NewParam(0, nil, "", Typ[Invalid]) // ignore recv below
		default:
			// more than one receiver
			check.error(recvList[len(recvList)-1], _BadRecv, "method must have exactly one receiver")
			fallthrough // continue with first receiver
		case 1:
			recv = recvList[0]
		}
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
				check.errorf(recv, _InvalidRecv, "invalid receiver %s (%s)", recv.typ, err)
				// ok to continue
			}
		}
		sig.recv = recv
	}

	sig.scope = scope
	sig.params = NewTuple(params...)
	sig.results = NewTuple(results...)
	sig.variadic = variadic
}

// typInternal drives type checking of types.
// Must only be called by definedType.
//
func (check *Checker) typInternal(e ast.Expr, def *Named) Type {
	switch e := e.(type) {
	case *ast.BadExpr:
		// ignore - error reported before

	case *ast.Ident:
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
			check.errorf(&x, _NotAType, "%s used as type", &x)
		default:
			check.errorf(&x, _NotAType, "%s is not a type", &x)
		}

	case *ast.SelectorExpr:
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
			check.errorf(&x, _NotAType, "%s used as type", &x)
		default:
			check.errorf(&x, _NotAType, "%s is not a type", &x)
		}

	case *ast.ParenExpr:
		return check.definedType(e.X, def)

	case *ast.ArrayType:
		if e.Len != nil {
			typ := new(Array)
			def.setUnderlying(typ)
			typ.len = check.arrayLength(e.Len)
			typ.elem = check.typ(e.Elt)
			return typ

		} else {
			typ := new(Slice)
			def.setUnderlying(typ)
			typ.elem = check.typ(e.Elt)
			return typ
		}

	case *ast.StructType:
		typ := new(Struct)
		def.setUnderlying(typ)
		check.structType(typ, e)
		return typ

	case *ast.StarExpr:
		typ := new(Pointer)
		def.setUnderlying(typ)
		typ.base = check.typ(e.X)
		return typ

	case *ast.FuncType:
		typ := new(Signature)
		def.setUnderlying(typ)
		check.funcType(typ, nil, e)
		return typ

	case *ast.InterfaceType:
		typ := new(Interface)
		def.setUnderlying(typ)
		check.interfaceType(typ, e, def)
		return typ

	case *ast.MapType:
		typ := new(Map)
		def.setUnderlying(typ)

		typ.key = check.typ(e.Key)
		typ.elem = check.typ(e.Value)

		// spec: "The comparison operators == and != must be fully defined
		// for operands of the key type; thus the key type must not be a
		// function, map, or slice."
		//
		// Delay this check because it requires fully setup types;
		// it is safe to continue in any case (was issue 6667).
		check.atEnd(func() {
			if !Comparable(typ.key) {
				check.errorf(e.Key, _IncomparableMapKey, "incomparable map key type %s", typ.key)
			}
		})

		return typ

	case *ast.ChanType:
		typ := new(Chan)
		def.setUnderlying(typ)

		dir := SendRecv
		switch e.Dir {
		case ast.SEND | ast.RECV:
			// nothing to do
		case ast.SEND:
			dir = SendOnly
		case ast.RECV:
			dir = RecvOnly
		default:
			check.invalidAST(e, "unknown channel direction %d", e.Dir)
			// ok to continue
		}

		typ.dir = dir
		typ.elem = check.typ(e.Value)
		return typ

	default:
		check.errorf(e, _NotAType, "%s is not a type", e)
	}

	typ := Typ[Invalid]
	def.setUnderlying(typ)
	return typ
}

// typeOrNil type-checks the type expression (or nil value) e
// and returns the typ of e, or nil.
// If e is neither a type nor nil, typOrNil returns Typ[Invalid].
//
func (check *Checker) typOrNil(e ast.Expr) Type {
	var x operand
	check.rawExpr(&x, e, nil)
	switch x.mode {
	case invalid:
		// ignore - error reported before
	case novalue:
		check.errorf(&x, _NotAType, "%s used as type", &x)
	case typexpr:
		return x.typ
	case value:
		if x.isNil() {
			return nil
		}
		fallthrough
	default:
		check.errorf(&x, _NotAType, "%s is not a type", &x)
	}
	return Typ[Invalid]
}

// arrayLength type-checks the array length expression e
// and returns the constant length >= 0, or a value < 0
// to indicate an error (and thus an unknown length).
func (check *Checker) arrayLength(e ast.Expr) int64 {
	var x operand
	check.expr(&x, e)
	if x.mode != constant_ {
		if x.mode != invalid {
			check.errorf(&x, _InvalidArrayLen, "array length %s must be constant", &x)
		}
		return -1
	}
	if isUntyped(x.typ) || isInteger(x.typ) {
		if val := constant.ToInt(x.val); val.Kind() == constant.Int {
			if representableConst(val, check, Typ[Int], nil) {
				if n, ok := constant.Int64Val(val); ok && n >= 0 {
					return n
				}
				check.errorf(&x, _InvalidArrayLen, "invalid array length %s", &x)
				return -1
			}
		}
	}
	check.errorf(&x, _InvalidArrayLen, "array length %s must be integer", &x)
	return -1
}

func (check *Checker) collectParams(scope *Scope, list *ast.FieldList, variadicOk bool) (params []*Var, variadic bool) {
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
				check.softErrorf(t, _MisplacedDotDotDot, "can only use ... with final parameter in list")
				// ignore ... and continue
			}
		}
		typ := check.typ(ftype)
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

func (check *Checker) declareInSet(oset *objset, pos token.Pos, obj Object) bool {
	if alt := oset.insert(obj); alt != nil {
		check.errorf(atPos(pos), _DuplicateDecl, "%s redeclared", obj.Name())
		check.reportAltDecl(alt)
		return false
	}
	return true
}

func (check *Checker) interfaceType(ityp *Interface, iface *ast.InterfaceType, def *Named) {
	for _, f := range iface.Methods.List {
		if len(f.Names) > 0 {
			// We have a method with name f.Names[0].
			// (The parser ensures that there's only one method
			// and we don't care if a constructed AST has more.)
			name := f.Names[0]
			if name.Name == "_" {
				check.errorf(name, _BlankIfaceMethod, "invalid method name _")
				continue // ignore
			}

			typ := check.typ(f.Type)
			sig, _ := typ.(*Signature)
			if sig == nil {
				if typ != Typ[Invalid] {
					check.invalidAST(f.Type, "%s is not a method signature", typ)
				}
				continue // ignore
			}

			// use named receiver type if available (for better error messages)
			var recvTyp Type = ityp
			if def != nil {
				recvTyp = def
			}
			sig.recv = NewVar(name.Pos(), check.pkg, "", recvTyp)

			m := NewFunc(name.Pos(), check.pkg, name.Name, sig)
			check.recordDef(name, m)
			ityp.methods = append(ityp.methods, m)
		} else {
			// We have an embedded interface and f.Type is its
			// (possibly qualified) embedded type name. Collect
			// it if it's a valid interface.
			typ := check.typ(f.Type)

			utyp := check.underlying(typ)
			if _, ok := utyp.(*Interface); !ok {
				if utyp != Typ[Invalid] {
					check.errorf(f.Type, _InvalidIfaceEmbed, "%s is not an interface", typ)
				}
				continue
			}

			ityp.embeddeds = append(ityp.embeddeds, typ)
			check.posMap[ityp] = append(check.posMap[ityp], f.Type.Pos())
		}
	}

	if len(ityp.methods) == 0 && len(ityp.embeddeds) == 0 {
		// empty interface
		ityp.allMethods = markComplete
		return
	}

	// sort for API stability
	sort.Sort(byUniqueMethodName(ityp.methods))
	sort.Stable(byUniqueTypeName(ityp.embeddeds))

	check.later(func() { check.completeInterface(ityp) })
}

func (check *Checker) completeInterface(ityp *Interface) {
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

	if trace {
		check.trace(token.NoPos, "complete %s", ityp)
		check.indent++
		defer func() {
			check.indent--
			check.trace(token.NoPos, "=> %s", ityp)
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
	mpos := make(map[*Func]token.Pos) // method specification or method embedding position, for good error messages
	addMethod := func(pos token.Pos, m *Func, explicit bool) {
		switch other := seen.insert(m); {
		case other == nil:
			methods = append(methods, m)
			mpos[m] = pos
		case explicit:
			check.errorf(atPos(pos), _DuplicateDecl, "duplicate method %s", m.name)
			check.errorf(atPos(mpos[other.(*Func)]), _DuplicateDecl, "\tother declaration of %s", m.name) // secondary error, \t indented
		default:
			// check method signatures after all types are computed (issue #33656)
			check.atEnd(func() {
				if !check.identical(m.typ, other.Type()) {
					check.errorf(atPos(pos), _DuplicateDecl, "duplicate method %s", m.name)
					check.errorf(atPos(mpos[other.(*Func)]), _DuplicateDecl, "\tother declaration of %s", m.name) // secondary error, \t indented
				}
			})
		}
	}

	for _, m := range ityp.methods {
		addMethod(m.pos, m, true)
	}

	posList := check.posMap[ityp]
	for i, typ := range ityp.embeddeds {
		pos := posList[i] // embedding position
		typ, ok := check.underlying(typ).(*Interface)
		if !ok {
			// An error was reported when collecting the embedded types.
			// Ignore it.
			continue
		}
		check.completeInterface(typ)
		for _, m := range typ.allMethods {
			addMethod(pos, m, false) // use embedding position pos rather than m.pos
		}
	}

	if methods != nil {
		sort.Sort(byUniqueMethodName(methods))
		ityp.allMethods = methods
	}
}

// byUniqueTypeName named type lists can be sorted by their unique type names.
type byUniqueTypeName []Type

func (a byUniqueTypeName) Len() int           { return len(a) }
func (a byUniqueTypeName) Less(i, j int) bool { return sortName(a[i]) < sortName(a[j]) }
func (a byUniqueTypeName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func sortName(t Type) string {
	if named, _ := t.(*Named); named != nil {
		return named.obj.Id()
	}
	return ""
}

// byUniqueMethodName method lists can be sorted by their unique method names.
type byUniqueMethodName []*Func

func (a byUniqueMethodName) Len() int           { return len(a) }
func (a byUniqueMethodName) Less(i, j int) bool { return a[i].Id() < a[j].Id() }
func (a byUniqueMethodName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func (check *Checker) tag(t *ast.BasicLit) string {
	if t != nil {
		if t.Kind == token.STRING {
			if val, err := strconv.Unquote(t.Value); err == nil {
				return val
			}
		}
		check.invalidAST(t, "incorrect tag syntax: %q", t.Value)
	}
	return ""
}

func (check *Checker) structType(styp *Struct, e *ast.StructType) {
	list := e.Fields
	if list == nil {
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
	add := func(ident *ast.Ident, embedded bool, pos token.Pos) {
		if tag != "" && tags == nil {
			tags = make([]string, len(fields))
		}
		if tags != nil {
			tags = append(tags, tag)
		}

		name := ident.Name
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
	addInvalid := func(ident *ast.Ident, pos token.Pos) {
		typ = Typ[Invalid]
		tag = ""
		add(ident, true, pos)
	}

	for _, f := range list.List {
		typ = check.typ(f.Type)
		tag = check.tag(f.Tag)
		if len(f.Names) > 0 {
			// named fields
			for _, name := range f.Names {
				add(name, false, name.Pos())
			}
		} else {
			// embedded field
			// spec: "An embedded type must be specified as a type name T or as a pointer
			// to a non-interface type name *T, and T itself may not be a pointer type."
			pos := f.Type.Pos()
			name := embeddedFieldIdent(f.Type)
			if name == nil {
				check.invalidAST(f.Type, "embedded field type %s has no name", f.Type)
				name = ast.NewIdent("_")
				name.NamePos = pos
				addInvalid(name, pos)
				continue
			}
			t, isPtr := deref(typ)
			// Because we have a name, typ must be of the form T or *T, where T is the name
			// of a (named or alias) type, and t (= deref(typ)) must be the type of T.
			switch t := t.Underlying().(type) {
			case *Basic:
				if t == Typ[Invalid] {
					// error was reported before
					addInvalid(name, pos)
					continue
				}

				// unsafe.Pointer is treated like a regular pointer
				if t.kind == UnsafePointer {
					check.errorf(f.Type, _InvalidPtrEmbed, "embedded field type cannot be unsafe.Pointer")
					addInvalid(name, pos)
					continue
				}

			case *Pointer:
				check.errorf(f.Type, _InvalidPtrEmbed, "embedded field type cannot be a pointer")
				addInvalid(name, pos)
				continue

			case *Interface:
				if isPtr {
					check.errorf(f.Type, _InvalidPtrEmbed, "embedded field type cannot be a pointer to an interface")
					addInvalid(name, pos)
					continue
				}
			}
			add(name, true, pos)
		}
	}

	styp.fields = fields
	styp.tags = tags
}

func embeddedFieldIdent(e ast.Expr) *ast.Ident {
	switch e := e.(type) {
	case *ast.Ident:
		return e
	case *ast.StarExpr:
		// *T is valid, but **T is not
		if _, ok := e.X.(*ast.StarExpr); !ok {
			return embeddedFieldIdent(e.X)
		}
	case *ast.SelectorExpr:
		return e.Sel
	}
	return nil // invalid embedded field
}
