// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

func (check *checker) reportAltDecl(obj Object) {
	if pos := obj.Pos(); pos.IsValid() {
		// We use "other" rather than "previous" here because
		// the first declaration seen may not be textually
		// earlier in the source.
		check.errorf(pos, "\tother declaration of %s", obj.Name()) // secondary error, \t indented
	}
}

func (check *checker) declare(scope *Scope, id *ast.Ident, obj Object) {
	if alt := scope.Insert(obj); alt != nil {
		check.errorf(obj.Pos(), "%s redeclared in this block", obj.Name())
		check.reportAltDecl(alt)
		return
	}
	if id != nil {
		check.recordObject(id, obj)
	}
}

// objDecl type-checks the declaration of obj in its respective file scope.
// See typeDecl for the details on def and cycleOk.
func (check *checker) objDecl(obj Object, def *Named, cycle []*TypeName) {
	if obj.Type() != nil {
		return // already checked - nothing to do
	}

	if trace {
		check.trace(obj.Pos(), "-- resolving %s", obj.Name())
	}

	d := check.objMap[obj]
	if debug && d == nil {
		if check.objMap == nil {
			check.dump("%s: %s should have been declared (we are inside a function)", obj.Pos(), obj)
			unreachable()
		}
		check.dump("%s: %s should have been forward-declared", obj.Pos(), obj)
		unreachable()
	}

	// adjust file scope for current object
	oldScope := check.topScope
	check.topScope = d.file // for lookup

	// save current iota
	oldIota := check.iota
	check.iota = nil

	// save current decl
	oldDecl := check.decl
	check.decl = nil

	switch obj := obj.(type) {
	case *Const:
		check.constDecl(obj, d.typ, d.init)
	case *Var:
		check.decl = d // new package-level var decl
		check.varDecl(obj, d.lhs, d.typ, d.init)
	case *TypeName:
		check.typeDecl(obj, d.typ, def, cycle)
	case *Func:
		check.funcDecl(obj, d)
	default:
		unreachable()
	}

	check.decl = oldDecl
	check.iota = oldIota
	check.topScope = oldScope
}

func (check *checker) constDecl(obj *Const, typ, init ast.Expr) {
	assert(obj.typ == nil)

	// TODO(gri) consider using the same cycle detection as for types
	// so that we can print the actual cycle in case of an error
	if obj.visited {
		check.errorf(obj.Pos(), "illegal cycle in initialization of constant %s", obj.name)
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true

	// use the correct value of iota
	assert(check.iota == nil)
	check.iota = obj.val

	// determine type, if any
	if typ != nil {
		t := check.typ(typ, nil, nil)
		if !isConstType(t) {
			check.errorf(typ.Pos(), "invalid constant type %s", t)
			obj.typ = Typ[Invalid]
			check.iota = nil
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

	check.iota = nil
}

// TODO(gri) document arguments
func (check *checker) varDecl(obj *Var, lhs []*Var, typ, init ast.Expr) {
	assert(obj.typ == nil)

	// TODO(gri) consider using the same cycle detection as for types
	// so that we can print the actual cycle in case of an error
	if obj.visited {
		check.errorf(obj.Pos(), "illegal cycle in initialization of variable %s", obj.name)
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true

	// var declarations cannot use iota
	assert(check.iota == nil)

	// determine type, if any
	if typ != nil {
		obj.typ = check.typ(typ, nil, nil)
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
		check.initVar(obj, &x)
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
	check.initVars(lhs, []ast.Expr{init}, token.NoPos)
}

// underlying returns the underlying type of typ; possibly by following
// forward chains of named types. Such chains only exist while names types
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

func (check *checker) typeDecl(obj *TypeName, typ ast.Expr, def *Named, cycle []*TypeName) {
	assert(obj.typ == nil)

	// type declarations cannot use iota
	assert(check.iota == nil)

	named := &Named{obj: obj}
	def.setUnderlying(named)
	obj.typ = named // make sure recursive type declarations terminate

	// determine underlying type of named
	check.typ(typ, named, append(cycle, obj))

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

	// type-check signatures of associated methods
	methods := check.methods[obj.name]
	if len(methods) == 0 {
		return // no methods
	}

	// spec: "For a base type, the non-blank names of methods bound
	// to it must be unique."
	// => use an objset to determine redeclarations
	var mset objset

	// spec: "If the base type is a struct type, the non-blank method
	// and field names must be distinct."
	// => pre-populate the objset to find conflicts
	// TODO(gri) consider keeping the objset with the struct instead
	if t, _ := named.underlying.(*Struct); t != nil {
		for _, fld := range t.fields {
			if fld.name != "_" {
				assert(mset.insert(fld) == nil)
			}
		}
	}

	// check each method
	for _, m := range methods {
		if m.name != "_" {
			if alt := mset.insert(m); alt != nil {
				switch alt.(type) {
				case *Var:
					check.errorf(m.pos, "field and method with the same name %s", m.name)
				case *Func:
					check.errorf(m.pos, "method %s already declared for %s", m.name, named)
				default:
					unreachable()
				}
				check.reportAltDecl(alt)
				continue
			}
		}
		check.recordObject(check.objMap[m].fdecl.Name, m)
		check.objDecl(m, nil, nil)
		// Methods with blank _ names cannot be found.
		// Don't add them to the method list.
		if m.name != "_" {
			named.methods = append(named.methods, m)
		}
	}

	delete(check.methods, obj.name) // we don't need them anymore
}

type funcInfo struct {
	name string    // for tracing only
	info *declInfo // for cycle detection
	sig  *Signature
	body *ast.BlockStmt
}

func (check *checker) funcDecl(obj *Func, info *declInfo) {
	assert(obj.typ == nil)

	// func declarations cannot use iota
	assert(check.iota == nil)

	sig := new(Signature)
	obj.typ = sig // guard against cycles
	fdecl := info.fdecl
	check.funcType(sig, fdecl.Recv, fdecl.Type)
	if sig.recv == nil && obj.name == "init" && (sig.params.Len() > 0 || sig.results.Len() > 0) {
		check.errorf(fdecl.Pos(), "func init must have no arguments and no return values")
		// ok to continue
	}

	// function body must be type-checked after global declarations
	// (functions implemented elsewhere have no body)
	if !check.conf.IgnoreFuncBodies && fdecl.Body != nil {
		check.funcList = append(check.funcList, funcInfo{obj.name, info, sig, fdecl.Body})
	}
}

func (check *checker) declStmt(decl ast.Decl) {
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
						obj := NewConst(name.Pos(), pkg, name.Name, nil, exact.MakeInt64(int64(iota)))
						lhs[i] = obj

						var init ast.Expr
						if i < len(last.Values) {
							init = last.Values[i]
						}

						check.constDecl(obj, last.Type, init)
					}

					check.arityMatch(s, last)

					for i, name := range s.Names {
						check.declare(check.topScope, name, lhs[i])
					}

				case token.VAR:
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

					// declare all variables
					// (only at this point are the variable scopes (parents) set)
					for i, name := range s.Names {
						check.declare(check.topScope, name, lhs0[i])
					}

				default:
					check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
				}

			case *ast.TypeSpec:
				obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
				check.declare(check.topScope, s.Name, obj)
				check.typeDecl(obj, s.Type, nil, nil)

			default:
				check.invalidAST(s.Pos(), "const, type, or var declaration expected")
			}
		}

	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}
