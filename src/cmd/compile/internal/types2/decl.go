// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"go/constant"
)

func (check *Checker) reportAltDecl(obj Object) {
	if pos := obj.Pos(); pos.IsKnown() {
		// We use "other" rather than "previous" here because
		// the first declaration seen may not be textually
		// earlier in the source.
		check.errorf(pos, "\tother declaration of %s", obj.Name()) // secondary error, \t indented
	}
}

func (check *Checker) declare(scope *Scope, id *syntax.Name, obj Object, pos syntax.Pos) {
	// spec: "The blank identifier, represented by the underscore
	// character _, may be used in a declaration like any other
	// identifier but the declaration does not introduce a new
	// binding."
	if obj.Name() != "_" {
		if alt := scope.Insert(obj); alt != nil {
			check.errorf(obj.Pos(), "%s redeclared in this block", obj.Name())
			check.reportAltDecl(alt)
			return
		}
		obj.setScopePos(pos)
	}
	if id != nil {
		check.recordDef(id, obj)
	}
}

// pathString returns a string of the form a->b-> ... ->g for a path [a, b, ... g].
func pathString(path []Object) string {
	var s string
	for i, p := range path {
		if i > 0 {
			s += "->"
		}
		s += p.Name()
	}
	return s
}

// objDecl type-checks the declaration of obj in its respective (file) context.
// For the meaning of def, see Checker.definedType, in typexpr.go.
func (check *Checker) objDecl(obj Object, def *Named) {
	if check.conf.Trace && obj.Type() == nil {
		if check.indent == 0 {
			fmt.Println() // empty line between top-level objects for readability
		}
		check.trace(obj.Pos(), "-- checking %s (%s, objPath = %s)", obj, obj.color(), pathString(check.objPath))
		check.indent++
		defer func() {
			check.indent--
			check.trace(obj.Pos(), "=> %s (%s)", obj, obj.color())
		}()
	}

	// Checking the declaration of obj means inferring its type
	// (and possibly its value, for constants).
	// An object's type (and thus the object) may be in one of
	// three states which are expressed by colors:
	//
	// - an object whose type is not yet known is painted white (initial color)
	// - an object whose type is in the process of being inferred is painted grey
	// - an object whose type is fully inferred is painted black
	//
	// During type inference, an object's color changes from white to grey
	// to black (pre-declared objects are painted black from the start).
	// A black object (i.e., its type) can only depend on (refer to) other black
	// ones. White and grey objects may depend on white and black objects.
	// A dependency on a grey object indicates a cycle which may or may not be
	// valid.
	//
	// When objects turn grey, they are pushed on the object path (a stack);
	// they are popped again when they turn black. Thus, if a grey object (a
	// cycle) is encountered, it is on the object path, and all the objects
	// it depends on are the remaining objects on that path. Color encoding
	// is such that the color value of a grey object indicates the index of
	// that object in the object path.

	// During type-checking, white objects may be assigned a type without
	// traversing through objDecl; e.g., when initializing constants and
	// variables. Update the colors of those objects here (rather than
	// everywhere where we set the type) to satisfy the color invariants.
	if obj.color() == white && obj.Type() != nil {
		obj.setColor(black)
		return
	}

	switch obj.color() {
	case white:
		assert(obj.Type() == nil)
		// All color values other than white and black are considered grey.
		// Because black and white are < grey, all values >= grey are grey.
		// Use those values to encode the object's index into the object path.
		obj.setColor(grey + color(check.push(obj)))
		defer func() {
			check.pop().setColor(black)
		}()

	case black:
		assert(obj.Type() != nil)
		return

	default:
		// Color values other than white or black are considered grey.
		fallthrough

	case grey:
		// We have a cycle.
		// In the existing code, this is marked by a non-nil type
		// for the object except for constants and variables whose
		// type may be non-nil (known), or nil if it depends on the
		// not-yet known initialization value.
		// In the former case, set the type to Typ[Invalid] because
		// we have an initialization cycle. The cycle error will be
		// reported later, when determining initialization order.
		// TODO(gri) Report cycle here and simplify initialization
		// order code.
		switch obj := obj.(type) {
		case *Const:
			if check.cycle(obj) || obj.typ == nil {
				obj.typ = Typ[Invalid]
			}

		case *Var:
			if check.cycle(obj) || obj.typ == nil {
				obj.typ = Typ[Invalid]
			}

		case *TypeName:
			if check.cycle(obj) {
				// break cycle
				// (without this, calling underlying()
				// below may lead to an endless loop
				// if we have a cycle for a defined
				// (*Named) type)
				obj.typ = Typ[Invalid]
			}

		case *Func:
			if check.cycle(obj) {
				// Don't set obj.typ to Typ[Invalid] here
				// because plenty of code type-asserts that
				// functions have a *Signature type. Grey
				// functions have their type set to an empty
				// signature which makes it impossible to
				// initialize a variable with the function.
			}

		case *Contract:
			// TODO(gri) is there anything else we need to do here?
			if check.cycle(obj) {
				obj.typ = Typ[Invalid]
			}

		default:
			unreachable()
		}
		assert(obj.Type() != nil)
		return
	}

	d := check.objMap[obj]
	if d == nil {
		check.dump("%v: %s should have been declared", obj.Pos(), obj)
		unreachable()
	}

	// save/restore current context and setup object context
	defer func(ctxt context) {
		check.context = ctxt
	}(check.context)
	check.context = context{
		scope: d.file,
	}

	// Const and var declarations must not have initialization
	// cycles. We track them by remembering the current declaration
	// in check.decl. Initialization expressions depending on other
	// consts, vars, or functions, add dependencies to the current
	// check.decl.
	switch obj := obj.(type) {
	case *Const:
		check.decl = d // new package-level const decl
		check.constDecl(obj, d.vtyp, d.init)
	case *Var:
		check.decl = d // new package-level var decl
		check.varDecl(obj, d.lhs, d.vtyp, d.init)
	case *TypeName:
		// invalid recursive types are detected via path
		check.typeDecl(obj, d.tdecl, def)
		check.collectMethods(obj) // methods can only be added to top-level types
	case *Func:
		// functions may be recursive - no need to track dependencies
		check.funcDecl(obj, d)
	// case *Contract:
	// 	if !AcceptContracts {
	// 		check.errorf(obj.pos, "contracts are not accepted")
	// 		obj.typ = Typ[Invalid]
	// 		break
	// 	}
	// 	check.contractDecl(obj, d.cdecl)
	default:
		unreachable()
	}
}

// cycle checks if the cycle starting with obj is valid and
// reports an error if it is not.
func (check *Checker) cycle(obj Object) (isCycle bool) {
	// The object map contains the package scope objects and the non-interface methods.
	if debug {
		info := check.objMap[obj]
		inObjMap := info != nil && (info.fdecl == nil || info.fdecl.Recv == nil) // exclude methods
		isPkgObj := obj.Parent() == check.pkg.scope
		if isPkgObj != inObjMap {
			check.dump("%v: inconsistent object map for %s (isPkgObj = %v, inObjMap = %v)", obj.Pos(), obj, isPkgObj, inObjMap)
			unreachable()
		}
	}

	// Count cycle objects.
	assert(obj.color() >= grey)
	start := obj.color() - grey // index of obj in objPath
	cycle := check.objPath[start:]
	nval := 0 // number of (constant or variable) values in the cycle
	ndef := 0 // number of type definitions in the cycle
	for _, obj := range cycle {
		switch obj := obj.(type) {
		case *Const, *Var:
			nval++
		case *TypeName:
			// Determine if the type name is an alias or not. For
			// package-level objects, use the object map which
			// provides syntactic information (which doesn't rely
			// on the order in which the objects are set up). For
			// local objects, we can rely on the order, so use
			// the object's predicate.
			// TODO(gri) It would be less fragile to always access
			// the syntactic information. We should consider storing
			// this information explicitly in the object.
			var alias bool
			if d := check.objMap[obj]; d != nil {
				alias = d.tdecl.Alias // package-level object
			} else {
				alias = obj.IsAlias() // function local object
			}
			if !alias {
				ndef++
			}
		case *Func:
			// ignored for now
		case *Contract:
			// TODO(gri) what do we need to do here, if anything?
		default:
			unreachable()
		}
	}

	if check.conf.Trace {
		check.trace(obj.Pos(), "## cycle detected: objPath = %s->%s (len = %d)", pathString(cycle), obj.Name(), len(cycle))
		check.trace(obj.Pos(), "## cycle contains: %d values, %d type definitions", nval, ndef)
		defer func() {
			if isCycle {
				check.trace(obj.Pos(), "=> error: cycle is invalid")
			}
		}()
	}

	// A cycle involving only constants and variables is invalid but we
	// ignore them here because they are reported via the initialization
	// cycle check.
	if nval == len(cycle) {
		return false
	}

	// A cycle involving only types (and possibly functions) must have at least
	// one type definition to be permitted: If there is no type definition, we
	// have a sequence of alias type names which will expand ad infinitum.
	if nval == 0 && ndef > 0 {
		return false // cycle is permitted
	}

	check.cycleError(cycle)

	return true
}

type typeInfo uint

// validType verifies that the given type does not "expand" infinitely
// producing a cycle in the type graph. Cycles are detected by marking
// defined types.
// (Cycles involving alias types, as in "type A = [10]A" are detected
// earlier, via the objDecl cycle detection mechanism.)
func (check *Checker) validType(typ Type, path []Object) typeInfo {
	const (
		unknown typeInfo = iota
		marked
		valid
		invalid
	)

	switch t := typ.(type) {
	case *Array:
		return check.validType(t.elem, path)

	case *Struct:
		for _, f := range t.fields {
			if check.validType(f.typ, path) == invalid {
				return invalid
			}
		}

	case *Interface:
		for _, etyp := range t.embeddeds {
			if check.validType(etyp, path) == invalid {
				return invalid
			}
		}

	case *Named:
		// don't touch the type if it is from a different package or the Universe scope
		// (doing so would lead to a race condition - was issue #35049)
		if t.obj.pkg != check.pkg {
			return valid
		}

		// don't report a 2nd error if we already know the type is invalid
		// (e.g., if a cycle was detected earlier, via Checker.underlying).
		if t.underlying == Typ[Invalid] {
			t.info = invalid
			return invalid
		}

		switch t.info {
		case unknown:
			t.info = marked
			t.info = check.validType(t.orig, append(path, t.obj)) // only types of current package added to path
		case marked:
			// cycle detected
			for i, tn := range path {
				if t.obj.pkg != check.pkg {
					panic("internal error: type cycle via package-external type")
				}
				if tn == t.obj {
					check.cycleError(path[i:])
					t.info = invalid
					t.underlying = Typ[Invalid]
					return t.info
				}
			}
			panic("internal error: cycle start not found")
		}
		return t.info
	}

	return valid
}

// cycleError reports a declaration cycle starting with
// the object in cycle that is "first" in the source.
func (check *Checker) cycleError(cycle []Object) {
	// TODO(gri) Should we start with the last (rather than the first) object in the cycle
	//           since that is the earliest point in the source where we start seeing the
	//           cycle? That would be more consistent with other error messages.
	i := firstInSrc(cycle)
	obj := cycle[i]
	check.errorf(obj.Pos(), "illegal cycle in declaration of %s", obj.Name())
	for range cycle {
		check.errorf(obj.Pos(), "\t%s refers to", obj.Name()) // secondary error, \t indented
		i++
		if i >= len(cycle) {
			i = 0
		}
		obj = cycle[i]
	}
	check.errorf(obj.Pos(), "\t%s", obj.Name())
}

// TODO(gri) This functionality should probably be with the Pos implementation.
func cmpPos(p, q syntax.Pos) int {
	// TODO(gri) is RelFilename correct here?
	pname := p.RelFilename()
	qname := q.RelFilename()
	switch {
	case pname < qname:
		return -1
	case pname > qname:
		return +1
	}

	pline := p.Line()
	qline := q.Line()
	switch {
	case pline < qline:
		return -1
	case pline > qline:
		return +1
	}

	pcol := p.Col()
	qcol := q.Col()
	switch {
	case pcol < qcol:
		return -1
	case pcol > qcol:
		return +1
	}

	return 0
}

// firstInSrc reports the index of the object with the "smallest"
// source position in path. path must not be empty.
func firstInSrc(path []Object) int {
	fst, pos := 0, path[0].Pos()
	for i, t := range path[1:] {
		if cmpPos(t.Pos(), pos) < 0 {
			fst, pos = i+1, t.Pos()
		}
	}
	return fst
}

func (check *Checker) constDecl(obj *Const, typ, init syntax.Expr) {
	assert(obj.typ == nil)

	// use the correct value of iota
	defer func(iota constant.Value) { check.iota = iota }(check.iota)
	check.iota = obj.val

	// provide valid constant value under all circumstances
	obj.val = constant.MakeUnknown()

	// determine type, if any
	if typ != nil {
		t := check.typ(typ)
		if !isConstType(t) {
			// don't report an error if the type is an invalid C (defined) type
			// (issue #22090)
			if t.Under() != Typ[Invalid] {
				check.errorf(typ.Pos(), "invalid constant type %s", t)
			}
			obj.typ = Typ[Invalid]
			return
		}
		obj.typ = t
	}

	// check initialization
	var x operand
	if init != nil {
		check.expr(&x, init)
	}
	check.initConst(obj, &x)
}

func (check *Checker) varDecl(obj *Var, lhs []*Var, typ, init syntax.Expr) {
	assert(obj.typ == nil)

	// determine type, if any
	if typ != nil {
		obj.typ = check.typ(typ)
		// We cannot spread the type to all lhs variables if there
		// are more than one since that would mark them as checked
		// (see Checker.objDecl) and the assignment of init exprs,
		// if any, would not be checked.
		//
		// TODO(gri) If we have no init expr, we should distribute
		// a given type otherwise we need to re-evalate the type
		// expr for each lhs variable, leading to duplicate work.
	}

	// check initialization
	if init == nil {
		if typ == nil {
			// error reported before by arityMatch
			obj.typ = Typ[Invalid]
		}
		return
	}

	if lhs == nil || len(lhs) == 1 {
		assert(lhs == nil || lhs[0] == obj)
		var x operand
		check.expr(&x, init)
		check.initVar(obj, &x, "variable declaration")
		return
	}

	if debug {
		// obj must be one of lhs
		found := false
		for _, lhs := range lhs {
			if obj == lhs {
				found = true
				break
			}
		}
		if !found {
			panic("inconsistent lhs")
		}
	}

	// We have multiple variables on the lhs and one init expr.
	// Make sure all variables have been given the same type if
	// one was specified, otherwise they assume the type of the
	// init expression values (was issue #15755).
	if typ != nil {
		for _, lhs := range lhs {
			lhs.typ = obj.typ
		}
	}

	check.initVars(lhs, []syntax.Expr{init}, nopos)
}

// Under returns the expanded underlying type of n0; possibly by following
// forward chains of named types. If an underlying type is found, resolve
// the chain by setting the underlying type for each defined type in the
// chain before returning it. If no underlying type is found or a cycle
// is detected, the result is Typ[Invalid]. If a cycle is detected and
// n0.check != nil, the cycle is reported.
func (n0 *Named) Under() Type {
	u := n0.underlying
	if u == nil {
		return Typ[Invalid]
	}

	// If the underlying type of a defined type is not a defined
	// type, then that is the desired underlying type.
	n := u.Named()
	if n == nil {
		return u // common case
	}

	// Otherwise, follow the forward chain.
	seen := map[*Named]int{n0: 0}
	path := []Object{n0.obj}
	for {
		u = n.underlying
		if u == nil {
			u = Typ[Invalid]
			break
		}
		n1 := u.Named()
		if n1 == nil {
			break // end of chain
		}

		seen[n] = len(seen)
		path = append(path, n.obj)
		n = n1

		if i, ok := seen[n]; ok {
			// cycle
			if n0.check != nil {
				n0.check.cycleError(path[i:])
			}
			u = Typ[Invalid]
			break
		}
	}

	for n := range seen {
		// We should never have to update the underlying type of an imported type;
		// those underlying types should have been resolved during the import.
		// Also, doing so would lead to a race condition (was issue #31749).
		// Do this check always, not just in debug more (it's cheap).
		if n0.check != nil && n.obj.pkg != n0.check.pkg {
			panic("internal error: imported type with unresolved underlying type")
		}
		n.underlying = u
	}

	return u
}

func (n *Named) setUnderlying(typ Type) {
	if n != nil {
		n.underlying = typ
	}
}

func (check *Checker) typeDecl(obj *TypeName, tdecl *syntax.TypeDecl, def *Named) {
	assert(obj.typ == nil)

	check.later(func() {
		check.validType(obj.typ, nil)
	})

	if tdecl.Alias {
		// type alias declaration

		if tdecl.TParamList != nil {
			// check.errorf(tdecl.TParamList.Pos(), "type alias cannot be parameterized")
			check.errorf(tdecl.Pos(), "type alias cannot be parameterized")
			// continue but ignore type parameters
		}

		obj.typ = Typ[Invalid]
		obj.typ = check.typ(tdecl.Type)

	} else {
		// defined type declaration

		named := &Named{check: check, obj: obj}
		def.setUnderlying(named)
		obj.typ = named // make sure recursive type declarations terminate

		if tdecl.TParamList != nil {
			check.openScope(tdecl, "type parameters")
			defer check.closeScope()
			named.tparams = check.collectTypeParams(tdecl.TParamList)
		}

		// determine underlying type of named
		named.orig = check.definedType(tdecl.Type, named)

		// The underlying type of named may be itself a named type that is
		// incomplete:
		//
		//	type (
		//		A B
		//		B *C
		//		C A
		//	)
		//
		// The type of C is the (named) type of A which is incomplete,
		// and which has as its underlying type the named type B.
		// Determine the (final, unnamed) underlying type by resolving
		// any forward chain.
		// TODO(gri) Investigate if we can just use named.origin here
		//           and rely on lazy computation of the underlying type.
		named.underlying = named.Under()
	}

}

func (check *Checker) collectTypeParams(list []*syntax.Field) (tparams []*TypeName) {
	// Declare type parameters up-front, with empty interface as type bound.
	// If we use interfaces as type bounds, the scope of type parameters starts at
	// the beginning of the type parameter list (so we can have mutually recursive
	// parameterized interfaces).
	for _, f := range list {
		tparams = check.declareTypeParam(tparams, f.Name)
	}

	for i, j := 0, 0; i < len(list); i = j {
		f := list[i]
		ftype := f.Type

		// determine the range of type parameters list[i:j] with identical type bound
		// (declared as in (type a, b, c B))
		j = i + 1
		for j < len(list) && list[j].Type == ftype {
			j++
		}

		// this should never be the case, but be careful
		if ftype == nil {
			continue
		}

		// TODO(gri) We should try to delay the IsInterface check
		//           as it may expand a possibly incomplete type.
		if bound := check.anyType(ftype); IsInterface(bound) {
			// If we have exactly one type parameter and the type bound expects exactly
			// one type argument, permit leaving away the type argument for the type
			// bound. This allows us to write (type T B(T)) as (type T B) instead.
			if isGeneric(bound) {
				// bound has not been instantiated yet
				base := bound.(*Named) // only a *Named type can be generic
				if j-i != 1 || len(base.tparams) != 1 {
					// TODO(gri) make this error message better
					check.errorf(ftype.Pos(), "cannot use generic type %s without instantiation (more than one type parameter)", bound)
					bound = Typ[Invalid]
					continue
				}
				// We have and expect exactly one type parameter.
				// "Manually" instantiate the bound with the parameter.
				// TODO(gri) this code (in more general form) is also in
				// checker.typInternal for the *syntax.CallExpr case. Factor?
				typ := new(instance)
				typ.check = check
				typ.pos = ftype.Pos()
				typ.base = base
				typ.targs = []Type{tparams[i].typ}
				typ.poslist = []syntax.Pos{f.Name.Pos()}
				// make sure we check instantiation works at least once
				// and that the resulting type is valid
				check.atEnd(func() {
					t := typ.expand()
					check.validType(t, nil)
				})
				// update bound and recorded type
				bound = typ
				check.recordTypeAndValue(ftype, typexpr, typ, nil)
			}
			// set the type bounds
			for i < j {
				tparams[i].typ.(*TypeParam).bound = bound
				i++
			}
		} else if bound != Typ[Invalid] {
			check.errorf(f.Type.Pos(), "%s is not an interface or contract", bound)
		}
	}

	return
}

// contractExpr returns the contract obj of a contract name x = C or
// the contract obj and type arguments targs of an instantiated contract
// expression x = C(T1, T2, ...), and whether the expression is valid.
// The set unused contains all (outer, incoming) type parameters that
// have not yet been used in a contract expression. It must be set prior
// to calling contractExpr and is updated by contractExpr.
//
// If x denotes a contract, the result obj is that contract; otherwise
// obj == nil and the remaining results are undefined. If the contract
// exists but the contract or the type arguments (if any) have errors
// valid is false.
// If x is a valid instantiated contract expression, targs is the list
// of (incomming) type parameters used as arguments for the contract,
// with their type bounds set according to the contract.
func (check *Checker) contractExpr(x syntax.Expr, unused map[*TypeParam]bool) (obj *Contract, targs []Type, valid bool) {
	// permit any parenthesized expression
	x = unparen(x)

	// a call expression might be an instantiated contract => unpack
	var call *syntax.CallExpr
	if call, _ = x.(*syntax.CallExpr); call != nil {
		x = call.Fun
	}

	// determine contract obj
	switch x := x.(type) {
	case *syntax.Name:
		// local contract
		if obj, _ = check.lookup(x.Value).(*Contract); obj != nil {
			// set up contract if not yet done
			if obj.typ == nil {
				check.objDecl(obj, nil)
			}
		}

	case *syntax.SelectorExpr:
		// imported contract
		// TODO(gri) use a shared function between this and check.selector
		if ident, _ := x.X.(*syntax.Name); ident != nil {
			identObj := check.lookup(ident.Value)
			if pname, _ := identObj.(*PkgName); pname != nil {
				assert(pname.pkg == check.pkg)
				check.recordUse(ident, pname)
				pname.used = true
				pkg := pname.imported
				exp := pkg.scope.Lookup(x.Sel.Value)
				if exp == nil {
					if !pkg.fake {
						check.errorf(x.Pos(), "%s not declared by package %s", ident.Value, pkg.name)
						return
					}
				} else if !exp.Exported() {
					check.errorf(x.Pos(), "%s not exported by packge %s", ident.Value, pkg.name)
					return
				} else {
					obj, _ = exp.(*Contract)
				}
			}
		}
	}

	if obj == nil {
		return // not a contract
	}

	assert(obj.typ != nil)
	if obj.typ == Typ[Invalid] {
		if call != nil {
			check.use(call.ArgList...)
		}
		return // we have a contract but it's broken
	}

	if call != nil {
		// collect type arguments
		if len(call.ArgList) != len(obj.TParams) {
			check.errorf(call.Pos(), "%d type parameters but contract expects %d", len(call.ArgList), len(obj.TParams))
			check.use(call.ArgList...)
			return
		}
		// For now, a contract type argument must be one of the (incoming)
		// type parameters, and each of these type parameters may be used
		// at most once.
		for _, arg := range call.ArgList {
			targ := check.typ(arg)
			if tparam, _ := targ.(*TypeParam); tparam != nil {
				if ok, found := unused[tparam]; ok {
					unused[tparam] = false
					targs = append(targs, targ)
				} else if found {
					check.errorf(arg.Pos(), "%s used multiple times (not supported due to implementation restriction)", arg)
				} else {
					check.errorf(arg.Pos(), "%s is not an incoming type parameter (not supported due to implementation restriction)", arg)
				}
			} else if targ != Typ[Invalid] {
				check.errorf(arg.Pos(), "%s is not a type parameter (not supported due to implementation restriction)", arg)
			}
		}
		if len(targs) != len(call.ArgList) {
			return // some arguments are invalid
		}
		// Use contract's matching type parameter bound, instantiate
		// it with the actual type arguments targs, and set the bound
		// for the type parameter.
		for i, bound := range obj.Bounds {
			targs[i].(*TypeParam).bound = check.instantiate(call.ArgList[i].Pos(), bound, targs, nil).(*Named)
		}
	}

	valid = true
	return
}

func (check *Checker) declareTypeParam(tparams []*TypeName, name *syntax.Name) []*TypeName {
	tpar := NewTypeName(name.Pos(), check.pkg, name.Value, nil)
	check.NewTypeParam(tpar, len(tparams), &emptyInterface) // assigns type to tpar as a side-effect
	check.declare(check.scope, name, tpar, check.scope.pos) // TODO(gri) check scope position
	tparams = append(tparams, tpar)

	if check.conf.Trace {
		check.trace(name.Pos(), "type param = %v", tparams[len(tparams)-1])
	}

	return tparams
}

func (check *Checker) collectMethods(obj *TypeName) {
	// get associated methods
	// (Checker.collectObjects only collects methods with non-blank names;
	// Checker.resolveBaseTypeName ensures that obj is not an alias name
	// if it has attached methods.)
	methods := check.methods[obj]
	if methods == nil {
		return
	}
	delete(check.methods, obj)
	assert(!check.objMap[obj].tdecl.Alias) // don't use TypeName.IsAlias (requires fully set up object)

	// use an objset to check for name conflicts
	var mset objset

	// spec: "If the base type is a struct type, the non-blank method
	// and field names must be distinct."
	base := obj.typ.Named() // shouldn't fail but be conservative
	if base != nil {
		if t, _ := base.underlying.(*Struct); t != nil {
			for _, fld := range t.fields {
				if fld.name != "_" {
					assert(mset.insert(fld) == nil)
				}
			}
		}

		// Checker.Files may be called multiple times; additional package files
		// may add methods to already type-checked types. Add pre-existing methods
		// so that we can detect redeclarations.
		for _, m := range base.methods {
			assert(m.name != "_")
			assert(mset.insert(m) == nil)
		}
	}

	// add valid methods
	for _, m := range methods {
		// spec: "For a base type, the non-blank names of methods bound
		// to it must be unique."
		assert(m.name != "_")
		if alt := mset.insert(m); alt != nil {
			switch alt.(type) {
			case *Var:
				check.errorf(m.pos, "field and method with the same name %s", m.name)
			case *Func:
				check.errorf(m.pos, "method %s already declared for %s", m.name, obj)
			default:
				unreachable()
			}
			check.reportAltDecl(alt)
			continue
		}

		if base != nil {
			base.methods = append(base.methods, m)
		}
	}
}

func (check *Checker) funcDecl(obj *Func, decl *declInfo) {
	assert(obj.typ == nil)

	// func declarations cannot use iota
	assert(check.iota == nil)

	sig := new(Signature)
	obj.typ = sig // guard against cycles

	// Avoid cycle error when referring to method while type-checking the signature.
	// This avoids a nuisance in the best case (non-parameterized receiver type) and
	// since the method is not a type, we get an error. If we have a parameterized
	// receiver type, instantiating the receiver type leads to the instantiation of
	// its methods, and we don't want a cycle error in that case.
	// TODO(gri) review if this is correct and/or whether we still need this?
	saved := obj.color_
	obj.color_ = black
	fdecl := decl.fdecl
	check.funcType(sig, fdecl.Recv, fdecl.TParamList, fdecl.Type)
	obj.color_ = saved

	// function body must be type-checked after global declarations
	// (functions implemented elsewhere have no body)
	if !check.conf.IgnoreFuncBodies && fdecl.Body != nil {
		check.later(func() {
			check.funcBody(decl, obj.name, sig, fdecl.Body, nil)
		})
	}
}

func (check *Checker) declStmt(list []syntax.Decl) {
	pkg := check.pkg

	var iota int64             // valid if last != nil
	var last *syntax.ConstDecl // last ConstDecl with type or init exprs seen
	for _, decl := range list {
		if last != nil {
			if cdecl, _ := decl.(*syntax.ConstDecl); cdecl != nil && cdecl.Group == last.Group {
				iota++
			} else {
				iota = 0
				last = nil
			}
		}

		switch s := decl.(type) {
		case *syntax.ConstDecl:
			top := len(check.delayed)

			// determine which init exprs to use
			values := unpack(s.Values)
			switch {
			case s.Type != nil || len(values) > 0:
				last = s
			case last == nil:
				last = new(syntax.ConstDecl) // make sure last exists
			}

			// declare all constants
			lhs := make([]*Const, len(s.NameList))
			lastValues := unpack(last.Values)
			for i, name := range s.NameList {
				obj := NewConst(name.Pos(), pkg, name.Value, nil, constant.MakeInt64(int64(iota)))
				lhs[i] = obj

				var init syntax.Expr
				if i < len(lastValues) {
					init = lastValues[i]
				}

				check.constDecl(obj, last.Type, init)
			}

			//check.arityMatch(s, last)
			check.arityMatch(s.Pos(), s.NameList, s.Type, unpack(s.Values), lastValues)

			// process function literals in init expressions before scope changes
			check.processDelayed(top)

			// spec: "The scope of a constant or variable identifier declared
			// inside a function begins at the end of the ConstSpec or VarSpec
			// (ShortVarDecl for short variable declarations) and ends at the
			// end of the innermost containing block."
			scopePos := endPos("s.End()")
			for i, name := range s.NameList {
				check.declare(check.scope, name, lhs[i], scopePos)
			}

		case *syntax.VarDecl:
			top := len(check.delayed)

			lhs0 := make([]*Var, len(s.NameList))
			for i, name := range s.NameList {
				lhs0[i] = NewVar(name.Pos(), pkg, name.Value, nil)
			}

			// initialize all variables
			values := unpack(s.Values)
			for i, obj := range lhs0 {
				var lhs []*Var
				var init syntax.Expr
				switch len(values) {
				case len(s.NameList):
					// lhs and rhs match
					init = values[i]
				case 1:
					// rhs is expected to be a multi-valued expression
					lhs = lhs0
					init = values[0]
				default:
					if i < len(values) {
						init = values[i]
					}
				}
				check.varDecl(obj, lhs, s.Type, init)
				if len(values) == 1 {
					// If we have a single lhs variable we are done either way.
					// If we have a single rhs expression, it must be a multi-
					// valued expression, in which case handling the first lhs
					// variable will cause all lhs variables to have a type
					// assigned, and we are done as well.
					if debug {
						for _, obj := range lhs0 {
							assert(obj.typ != nil)
						}
					}
					break
				}
			}

			//check.arityMatch(s, nil)
			check.arityMatch(s.Pos(), s.NameList, s.Type, values, nil)

			// process function literals in init expressions before scope changes
			check.processDelayed(top)

			// declare all variables
			// (only at this point are the variable scopes (parents) set)
			scopePos := endPos("s.End()") // see constant declarations
			for i, name := range s.NameList {
				// see constant declarations
				check.declare(check.scope, name, lhs0[i], scopePos)
			}

		case *syntax.TypeDecl:
			obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Value, nil)
			// spec: "The scope of a type identifier declared inside a function
			// begins at the identifier in the TypeSpec and ends at the end of
			// the innermost containing block."
			scopePos := s.Name.Pos()
			check.declare(check.scope, s.Name, obj, scopePos)
			// mark and unmark type before calling typeDecl; its type is still nil (see Checker.objDecl)
			obj.setColor(grey + color(check.push(obj)))
			check.typeDecl(obj, s, nil)
			check.pop().setColor(black)

		default:
			check.invalidAST(s.Pos(), "unknown syntax.Decl node %T", s)
		}
	}
}
