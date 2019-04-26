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
			check.errorf(e.Pos(), "cannot use _ as value or type")
		} else {
			check.errorf(e.Pos(), "undeclared name: %s", e.Name)
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
		check.errorf(e.Pos(), "use of package %s not in selector", obj.name)
		return

	case *Const:
		check.addDeclDep(obj)
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

// indirectType is like typ but it also breaks the (otherwise) infinite size of recursive
// types by introducing an indirection. It should be called for components of types that
// are not laid out in place in memory, such as pointer base types, slice or map element
// types, function parameter types, etc.
func (check *Checker) indirectType(e ast.Expr) Type {
	check.push(indir)
	defer check.pop()
	return check.definedType(e, nil)
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
			check.error(recvPar.Pos(), "method is missing receiver")
			recv = NewParam(0, nil, "", Typ[Invalid]) // ignore recv below
		default:
			// more than one receiver
			check.error(recvList[len(recvList)-1].Pos(), "method must have exactly one receiver")
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
				check.errorf(recv.pos, "invalid receiver %s (%s)", recv.typ, err)
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
			check.errorf(x.pos(), "%s used as type", &x)
		default:
			check.errorf(x.pos(), "%s is not a type", &x)
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
			check.errorf(x.pos(), "%s used as type", &x)
		default:
			check.errorf(x.pos(), "%s is not a type", &x)
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
			typ.elem = check.indirectType(e.Elt)
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
		typ.base = check.indirectType(e.X)
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

		typ.key = check.indirectType(e.Key)
		typ.elem = check.indirectType(e.Value)

		// spec: "The comparison operators == and != must be fully defined
		// for operands of the key type; thus the key type must not be a
		// function, map, or slice."
		//
		// Delay this check because it requires fully setup types;
		// it is safe to continue in any case (was issue 6667).
		check.later(func() {
			if !Comparable(typ.key) {
				check.errorf(e.Key.Pos(), "invalid map key type %s", typ.key)
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
			check.invalidAST(e.Pos(), "unknown channel direction %d", e.Dir)
			// ok to continue
		}

		typ.dir = dir
		typ.elem = check.indirectType(e.Value)
		return typ

	default:
		check.errorf(e.Pos(), "%s is not a type", e)
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

// arrayLength type-checks the array length expression e
// and returns the constant length >= 0, or a value < 0
// to indicate an error (and thus an unknown length).
func (check *Checker) arrayLength(e ast.Expr) int64 {
	var x operand
	check.expr(&x, e)
	if x.mode != constant_ {
		if x.mode != invalid {
			check.errorf(x.pos(), "array length %s must be constant", &x)
		}
		return -1
	}
	if isUntyped(x.typ) || isInteger(x.typ) {
		if val := constant.ToInt(x.val); val.Kind() == constant.Int {
			if representableConst(val, check, Typ[Int], nil) {
				if n, ok := constant.Int64Val(val); ok && n >= 0 {
					return n
				}
				check.errorf(x.pos(), "invalid array length %s", &x)
				return -1
			}
		}
	}
	check.errorf(x.pos(), "array length %s must be integer", &x)
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
				check.softErrorf(t.Pos(), "can only use ... with final parameter in list")
				// ignore ... and continue
			}
		}
		typ := check.indirectType(ftype)
		// The parser ensures that f.Tag is nil and we don't
		// care if a constructed AST contains a non-nil tag.
		if len(field.Names) > 0 {
			// named parameter
			for _, name := range field.Names {
				if name.Name == "" {
					check.invalidAST(name.Pos(), "anonymous parameter")
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
		check.invalidAST(list.Pos(), "list contains both named and anonymous parameters")
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
		check.errorf(pos, "%s redeclared", obj.Name())
		check.reportAltDecl(alt)
		return false
	}
	return true
}

func (check *Checker) interfaceType(ityp *Interface, iface *ast.InterfaceType, def *Named) {
	// fast-track empty interface
	if iface.Methods.List == nil {
		ityp.allMethods = markComplete
		return
	}

	// collect embedded interfaces
	// Only needed for printing and API. Delay collection
	// to end of type-checking (for package-global interfaces)
	// when all types are complete. Local interfaces are handled
	// after each statement (as each statement processes delayed
	// functions).
	interfaceContext := check.context // capture for use in closure below
	check.later(func() {
		if trace {
			check.trace(iface.Pos(), "-- delayed checking embedded interfaces of %v", iface)
			check.indent++
			defer func() {
				check.indent--
			}()
		}

		// The context must be restored since for local interfaces
		// delayed functions are processed after each statement
		// (was issue #24140).
		defer func(ctxt context) {
			check.context = ctxt
		}(check.context)
		check.context = interfaceContext

		for _, f := range iface.Methods.List {
			if len(f.Names) == 0 {
				typ := check.indirectType(f.Type)
				// typ should be a named type denoting an interface
				// (the parser will make sure it's a named type but
				// constructed ASTs may be wrong).
				if typ == Typ[Invalid] {
					continue // error reported before
				}
				embed, _ := typ.Underlying().(*Interface)
				if embed == nil {
					check.errorf(f.Type.Pos(), "%s is not an interface", typ)
					continue
				}
				// Correct embedded interfaces must be complete -
				// don't just assert, but report error since this
				// used to be the underlying cause for issue #18395.
				if embed.allMethods == nil {
					check.dump("%v: incomplete embedded interface %s", f.Type.Pos(), typ)
					unreachable()
				}
				// collect interface
				ityp.embeddeds = append(ityp.embeddeds, typ)
			}
		}
		// sort to match NewInterface/NewInterface2
		// TODO(gri) we may be able to switch to source order
		sort.Stable(byUniqueTypeName(ityp.embeddeds))
	})

	// compute method set
	var tname *TypeName
	var path []*TypeName
	if def != nil {
		tname = def.obj
		path = []*TypeName{tname}
	}
	info := check.infoFromTypeLit(check.scope, iface, tname, path)
	if info == nil || info == &emptyIfaceInfo {
		// we got an error or the empty interface - exit early
		ityp.allMethods = markComplete
		return
	}

	// use named receiver type if available (for better error messages)
	var recvTyp Type = ityp
	if def != nil {
		recvTyp = def
	}

	// Correct receiver type for all methods explicitly declared
	// by this interface after we're done with type-checking at
	// this level. See comment below for details.
	check.later(func() {
		for _, m := range ityp.methods {
			m.typ.(*Signature).recv.typ = recvTyp
		}
	})

	// collect methods
	var sigfix []*methodInfo
	for i, minfo := range info.methods {
		fun := minfo.fun
		if fun == nil {
			name := minfo.src.Names[0]
			pos := name.Pos()
			// Don't type-check signature yet - use an
			// empty signature now and update it later.
			// But set up receiver since we know it and
			// its position, and because interface method
			// signatures don't get a receiver via regular
			// type-checking (there isn't a receiver in the
			// method's AST). Setting the receiver type is
			// also important for ptrRecv() (see methodset.go).
			//
			// Note: For embedded methods, the receiver type
			// should be the type of the interface that declared
			// the methods in the first place. Since we get the
			// methods here via methodInfo, which may be computed
			// before we have all relevant interface types, we use
			// the current interface's type (recvType). This may be
			// the type of the interface embedding the interface that
			// declared the methods. This doesn't matter for type-
			// checking (we only care about the receiver type for
			// the ptrRecv predicate, and it's never a pointer recv
			// for interfaces), but it matters for go/types clients
			// and for printing. We correct the receiver after type-
			// checking.
			//
			// TODO(gri) Consider marking methods signatures
			// as incomplete, for better error messages. See
			// also the T4 and T5 tests in testdata/cycles2.src.
			sig := new(Signature)
			sig.recv = NewVar(pos, check.pkg, "", recvTyp)
			fun = NewFunc(pos, check.pkg, name.Name, sig)
			minfo.fun = fun
			check.recordDef(name, fun)
			sigfix = append(sigfix, minfo)
		}
		// fun != nil
		if i < info.explicits {
			ityp.methods = append(ityp.methods, fun)
		}
		ityp.allMethods = append(ityp.allMethods, fun)
	}

	// fix signatures now that we have collected all methods
	savedContext := check.context
	for _, minfo := range sigfix {
		// (possibly embedded) methods must be type-checked within their scope and
		// type-checking them must not affect the current context (was issue #23914)
		check.context = context{scope: minfo.scope}
		typ := check.indirectType(minfo.src.Type)
		sig, _ := typ.(*Signature)
		if sig == nil {
			if typ != Typ[Invalid] {
				check.invalidAST(minfo.src.Type.Pos(), "%s is not a method signature", typ)
			}
			continue // keep method with empty method signature
		}
		// update signature, but keep recv that was set up before
		old := minfo.fun.typ.(*Signature)
		sig.recv = old.recv
		*old = *sig // update signature (don't replace pointer!)
	}
	check.context = savedContext

	// sort to match NewInterface/NewInterface2
	// TODO(gri) we may be able to switch to source order
	sort.Sort(byUniqueMethodName(ityp.methods))

	if ityp.allMethods == nil {
		ityp.allMethods = markComplete
	} else {
		sort.Sort(byUniqueMethodName(ityp.allMethods))
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
		check.invalidAST(t.Pos(), "incorrect tag syntax: %q", t.Value)
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
				check.invalidAST(pos, "embedded field type %s has no name", f.Type)
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
					check.errorf(pos, "embedded field type cannot be unsafe.Pointer")
					addInvalid(name, pos)
					continue
				}

			case *Pointer:
				check.errorf(pos, "embedded field type cannot be a pointer")
				addInvalid(name, pos)
				continue

			case *Interface:
				if isPtr {
					check.errorf(pos, "embedded field type cannot be a pointer to an interface")
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
