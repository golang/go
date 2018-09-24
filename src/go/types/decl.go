// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/constant"
	"go/token"
)

func (check *Checker) reportAltDecl(obj Object) {
	if pos := obj.Pos(); pos.IsValid() {
		// We use "other" rather than "previous" here because
		// the first declaration seen may not be textually
		// earlier in the source.
		check.errorf(pos, "\tother declaration of %s", obj.Name()) // secondary error, \t indented
	}
}

func (check *Checker) declare(scope *Scope, id *ast.Ident, obj Object, pos token.Pos) {
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
// TODO(gri) remove once we don't need the old cycle detection (explicitly passed
//           []*TypeName path) anymore
func pathString(path []*TypeName) string {
	var s string
	for i, p := range path {
		if i > 0 {
			s += "->"
		}
		s += p.Name()
	}
	return s
}

// objPathString returns a string of the form a->b-> ... ->g for a path [a, b, ... g].
// TODO(gri) s/objPathString/pathString/ once we got rid of pathString above
func objPathString(path []Object) string {
	var s string
	for i, p := range path {
		if i > 0 {
			s += "->"
		}
		s += p.Name()
	}
	return s
}

// useCycleMarking enables the new coloring-based cycle marking scheme
// for package-level objects. Set this flag to false to disable this
// code quickly and revert to the existing mechanism (and comment out
// some of the new tests in cycles5.src that will fail again).
// TODO(gri) remove this for Go 1.12
const useCycleMarking = true

// objDecl type-checks the declaration of obj in its respective (file) context.
// See check.typ for the details on def and path.
func (check *Checker) objDecl(obj Object, def *Named, path []*TypeName) {
	if trace {
		check.trace(obj.Pos(), "-- checking %s %s (path = %s, objPath = %s)", obj.color(), obj, pathString(path), check.pathString())
		check.indent++
		defer func() {
			check.indent--
			check.trace(obj.Pos(), "=> %s", obj)
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
			if useCycleMarking && check.typeCycle(obj) {
				obj.typ = Typ[Invalid]
				break
			}
			if obj.typ == nil {
				obj.typ = Typ[Invalid]
			}

		case *Var:
			if useCycleMarking && check.typeCycle(obj) {
				obj.typ = Typ[Invalid]
				break
			}
			if obj.typ == nil {
				obj.typ = Typ[Invalid]
			}

		case *TypeName:
			// fixFor26390 enables a temporary work-around to handle alias type names
			// that have not been given a type yet even though the underlying type
			// is already known. See testdata/issue26390.src for a simple example.
			// Set this flag to false to disable this code quickly (and comment
			// out the new test in decls4.src that will fail again).
			// TODO(gri) remove this for Go 1.12 in favor of a more comprehensive fix
			const fixFor26390 = true
			if fixFor26390 {
				// If we have a package-level alias type name that has not been
				// given a type yet but the underlying type is a type name that
				// has been given a type already, don't report a cycle but use
				// the underlying type name's type instead. The cycle shouldn't
				// exist in the first place in this case and is due to the way
				// methods are type-checked at the moment. See also the comment
				// at the end of Checker.typeDecl below.
				if d := check.objMap[obj]; d != nil && d.alias && obj.typ == Typ[Invalid] {
					// If we can find the underlying type name syntactically
					// and it has a type, use that type.
					if tname := check.resolveBaseTypeName(ast.NewIdent(obj.name)); tname != nil && tname.typ != nil {
						obj.typ = tname.typ
						break
					}
				}
			}

			if useCycleMarking && check.typeCycle(obj) {
				// break cycle
				// (without this, calling underlying()
				// below may lead to an endless loop
				// if we have a cycle for a defined
				// (*Named) type)
				obj.typ = Typ[Invalid]
			}

		case *Func:
			if useCycleMarking && check.typeCycle(obj) {
				// Don't set obj.typ to Typ[Invalid] here
				// because plenty of code type-asserts that
				// functions have a *Signature type. Grey
				// functions have their type set to an empty
				// signature which makes it impossible to
				// initialize a variable with the function.
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
		check.constDecl(obj, d.typ, d.init)
	case *Var:
		check.decl = d // new package-level var decl
		check.varDecl(obj, d.lhs, d.typ, d.init)
	case *TypeName:
		// invalid recursive types are detected via path
		check.typeDecl(obj, d.typ, def, path, d.alias)
	case *Func:
		// functions may be recursive - no need to track dependencies
		check.funcDecl(obj, d)
	default:
		unreachable()
	}
}

// indir is a sentinel type name that is pushed onto the object path
// to indicate an "indirection" in the dependency from one type name
// to the next. For instance, for "type p *p" the object path contains
// p followed by indir, indicating that there's an indirection *p.
// Indirections are used to break type cycles.
var indir = NewTypeName(token.NoPos, nil, "*", nil)

// cutCycle is a sentinel type name that is pushed onto the object path
// to indicate that a cycle doesn't actually exist. This is currently
// needed to break cycles formed via method declarations because they
// are type-checked together with their receiver base types. Once methods
// are type-checked separately (see also TODO in Checker.typeDecl), we
// can get rid of this.
var cutCycle = NewTypeName(token.NoPos, nil, "!", nil)

// typeCycle checks if the cycle starting with obj is valid and
// reports an error if it is not.
// TODO(gri) rename s/typeCycle/cycle/ once we don't need the other
// cycle method anymore.
func (check *Checker) typeCycle(obj Object) (isCycle bool) {
	d := check.objMap[obj]
	if d == nil {
		check.dump("%v: %s should have been declared", obj.Pos(), obj)
		unreachable()
	}

	// Given the number of constants and variables (nval) in the cycle
	// and the cycle length (ncycle = number of named objects in the cycle),
	// we distinguish between cycles involving only constants and variables
	// (nval = ncycle), cycles involving types (and functions) only
	// (nval == 0), and mixed cycles (nval != 0 && nval != ncycle).
	// We ignore functions at the moment (taking them into account correctly
	// is complicated and it doesn't improve error reporting significantly).
	//
	// A cycle must have at least one indirection and one type definition
	// to be permitted: If there is no indirection, the size of the type
	// cannot be computed (it's either infinite or 0); if there is no type
	// definition, we have a sequence of alias type names which will expand
	// ad infinitum.
	var nval, ncycle int
	var hasIndir, hasTDef bool
	assert(obj.color() >= grey)
	start := obj.color() - grey // index of obj in objPath
	cycle := check.objPath[start:]
	ncycle = len(cycle) // including indirections
	for _, obj := range cycle {
		switch obj := obj.(type) {
		case *Const, *Var:
			nval++
		case *TypeName:
			switch {
			case obj == indir:
				ncycle-- // don't count (indirections are not objects)
				hasIndir = true
			case obj == cutCycle:
				// The cycle is not real and only caused by the fact
				// that we type-check methods when we type-check their
				// receiver base types.
				return false
			case !check.objMap[obj].alias:
				hasTDef = true
			}
		case *Func:
			// ignored for now
		default:
			unreachable()
		}
	}

	if trace {
		check.trace(obj.Pos(), "## cycle detected: objPath = %s->%s (len = %d)", objPathString(cycle), obj.Name(), ncycle)
		check.trace(obj.Pos(), "## cycle contains: %d values, has indirection = %v, has type definition = %v", nval, hasIndir, hasTDef)
		defer func() {
			if isCycle {
				check.trace(obj.Pos(), "=> error: cycle is invalid")
			}
		}()
	}

	// A cycle involving only constants and variables is invalid but we
	// ignore them here because they are reported via the initialization
	// cycle check.
	if nval == ncycle {
		return false
	}

	// A cycle involving only types (and possibly functions) must have at
	// least one indirection and one type definition to be permitted: If
	// there is no indirection, the size of the type cannot be computed
	// (it's either infinite or 0); if there is no type definition, we
	// have a sequence of alias type names which will expand ad infinitum.
	if nval == 0 && hasIndir && hasTDef {
		return false // cycle is permitted
	}

	// report cycle
	check.errorf(obj.Pos(), "illegal cycle in declaration of %s", obj.Name())
	for _, obj := range cycle {
		if obj == indir {
			continue // don't print indir sentinels
		}
		check.errorf(obj.Pos(), "\t%s refers to", obj.Name()) // secondary error, \t indented
	}
	check.errorf(obj.Pos(), "\t%s", obj.Name())

	return true
}

func (check *Checker) constDecl(obj *Const, typ, init ast.Expr) {
	assert(obj.typ == nil)

	// use the correct value of iota
	check.iota = obj.val
	defer func() { check.iota = nil }()

	// provide valid constant value under all circumstances
	obj.val = constant.MakeUnknown()

	// determine type, if any
	if typ != nil {
		t := check.typExpr(typ, nil, nil)
		if !isConstType(t) {
			// don't report an error if the type is an invalid C (defined) type
			// (issue #22090)
			if t.Underlying() != Typ[Invalid] {
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

func (check *Checker) varDecl(obj *Var, lhs []*Var, typ, init ast.Expr) {
	assert(obj.typ == nil)

	// determine type, if any
	if typ != nil {
		obj.typ = check.typExpr(typ, nil, nil)
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

	check.initVars(lhs, []ast.Expr{init}, token.NoPos)
}

// underlying returns the underlying type of typ; possibly by following
// forward chains of named types. Such chains only exist while named types
// are incomplete.
func underlying(typ Type) Type {
	for {
		n, _ := typ.(*Named)
		if n == nil {
			break
		}
		typ = n.underlying
	}
	return typ
}

func (n *Named) setUnderlying(typ Type) {
	if n != nil {
		n.underlying = typ
	}
}

func (check *Checker) typeDecl(obj *TypeName, typ ast.Expr, def *Named, path []*TypeName, alias bool) {
	assert(obj.typ == nil)

	if alias {

		obj.typ = Typ[Invalid]
		obj.typ = check.typExpr(typ, nil, append(path, obj))

	} else {

		named := &Named{obj: obj}
		def.setUnderlying(named)
		obj.typ = named // make sure recursive type declarations terminate

		// determine underlying type of named
		check.typExpr(typ, named, append(path, obj))

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
		// any forward chain (they always end in an unnamed type).
		named.underlying = underlying(named.underlying)

	}

	// check and add associated methods
	// TODO(gri) It's easy to create pathological cases where the
	// current approach is incorrect: In general we need to know
	// and add all methods _before_ type-checking the type.
	// See https://play.golang.org/p/WMpE0q2wK8
	check.addMethodDecls(obj)
}

func (check *Checker) addMethodDecls(obj *TypeName) {
	// get associated methods
	// (Checker.collectObjects only collects methods with non-blank names;
	// Checker.resolveBaseTypeName ensures that obj is not an alias name
	// if it has attached methods.)
	methods := check.methods[obj]
	if methods == nil {
		return
	}
	delete(check.methods, obj)
	assert(!check.objMap[obj].alias) // don't use TypeName.IsAlias (requires fully set up object)

	// use an objset to check for name conflicts
	var mset objset

	// spec: "If the base type is a struct type, the non-blank method
	// and field names must be distinct."
	base, _ := obj.typ.(*Named) // shouldn't fail but be conservative
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

	if useCycleMarking {
		// Suppress detection of type cycles occurring through method
		// declarations - they wouldn't exist if methods were type-
		// checked separately from their receiver base types. See also
		// comment at the end of Checker.typeDecl.
		// TODO(gri) Remove this once methods are type-checked separately.
		check.push(cutCycle)
		defer check.pop()
	}

	// type-check methods
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

		// type-check
		check.objDecl(m, nil, nil)

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
	fdecl := decl.fdecl
	check.funcType(sig, fdecl.Recv, fdecl.Type)
	if sig.recv == nil && obj.name == "init" && (sig.params.Len() > 0 || sig.results.Len() > 0) {
		check.errorf(fdecl.Pos(), "func init must have no arguments and no return values")
		// ok to continue
	}

	// function body must be type-checked after global declarations
	// (functions implemented elsewhere have no body)
	if !check.conf.IgnoreFuncBodies && fdecl.Body != nil {
		check.later(func() {
			check.funcBody(decl, obj.name, sig, fdecl.Body, nil)
		})
	}
}

func (check *Checker) declStmt(decl ast.Decl) {
	pkg := check.pkg

	switch d := decl.(type) {
	case *ast.BadDecl:
		// ignore

	case *ast.GenDecl:
		var last *ast.ValueSpec // last ValueSpec with type or init exprs seen
		for iota, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.ValueSpec:
				switch d.Tok {
				case token.CONST:
					top := len(check.delayed)

					// determine which init exprs to use
					switch {
					case s.Type != nil || len(s.Values) > 0:
						last = s
					case last == nil:
						last = new(ast.ValueSpec) // make sure last exists
					}

					// declare all constants
					lhs := make([]*Const, len(s.Names))
					for i, name := range s.Names {
						obj := NewConst(name.Pos(), pkg, name.Name, nil, constant.MakeInt64(int64(iota)))
						lhs[i] = obj

						var init ast.Expr
						if i < len(last.Values) {
							init = last.Values[i]
						}

						check.constDecl(obj, last.Type, init)
					}

					check.arityMatch(s, last)

					// process function literals in init expressions before scope changes
					check.processDelayed(top)

					// spec: "The scope of a constant or variable identifier declared
					// inside a function begins at the end of the ConstSpec or VarSpec
					// (ShortVarDecl for short variable declarations) and ends at the
					// end of the innermost containing block."
					scopePos := s.End()
					for i, name := range s.Names {
						check.declare(check.scope, name, lhs[i], scopePos)
					}

				case token.VAR:
					top := len(check.delayed)

					lhs0 := make([]*Var, len(s.Names))
					for i, name := range s.Names {
						lhs0[i] = NewVar(name.Pos(), pkg, name.Name, nil)
					}

					// initialize all variables
					for i, obj := range lhs0 {
						var lhs []*Var
						var init ast.Expr
						switch len(s.Values) {
						case len(s.Names):
							// lhs and rhs match
							init = s.Values[i]
						case 1:
							// rhs is expected to be a multi-valued expression
							lhs = lhs0
							init = s.Values[0]
						default:
							if i < len(s.Values) {
								init = s.Values[i]
							}
						}
						check.varDecl(obj, lhs, s.Type, init)
						if len(s.Values) == 1 {
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

					check.arityMatch(s, nil)

					// process function literals in init expressions before scope changes
					check.processDelayed(top)

					// declare all variables
					// (only at this point are the variable scopes (parents) set)
					scopePos := s.End() // see constant declarations
					for i, name := range s.Names {
						// see constant declarations
						check.declare(check.scope, name, lhs0[i], scopePos)
					}

				default:
					check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
				}

			case *ast.TypeSpec:
				obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
				// spec: "The scope of a type identifier declared inside a function
				// begins at the identifier in the TypeSpec and ends at the end of
				// the innermost containing block."
				scopePos := s.Name.Pos()
				check.declare(check.scope, s.Name, obj, scopePos)
				check.typeDecl(obj, s.Type, nil, nil, s.Assign.IsValid())

			default:
				check.invalidAST(s.Pos(), "const, type, or var declaration expected")
			}
		}

	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}
