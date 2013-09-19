// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Check function, which drives type-checking.

package types

import (
	"fmt"
	"go/ast"
	"go/token"
	"path"

	"code.google.com/p/go.tools/go/exact"
)

// debugging/development support
const (
	debug = true  // leave on during development
	trace = false // turn on for detailed type resolution traces
)

// exprInfo stores type and constant value for an untyped expression.
type exprInfo struct {
	isLhs bool // expression is lhs operand of a shift with delayed type-check
	typ   *Basic
	val   exact.Value // constant value; or nil (if not a constant)
}

// A checker is an instance of the type-checker.
type checker struct {
	conf *Config
	fset *token.FileSet
	pkg  *Package

	methods     map[string][]*Func     // maps type names to associated methods
	conversions map[*ast.CallExpr]bool // set of type-checked conversions (to distinguish from calls)
	untyped     map[ast.Expr]exprInfo  // map of expressions without final type
	lhsVarsList [][]*Var               // type switch lhs variable sets, for 'declared but not used' errors

	firstErr error // first error encountered
	Info           // collected type info

	objMap   map[Object]declInfo // if set we are in the package-level declaration phase (otherwise all objects seen must be declared)
	topScope *Scope              // current topScope for lookups
	iota     exact.Value         // value of iota in a constant declaration; nil otherwise

	// functions
	funcList []funcInfo // list of functions/methods with correct signatures and non-empty bodies
	funcSig  *Signature // signature of currently type-checked function
	labels   *Scope     // label scope of currently type-checked function; lazily allocated

	// debugging
	indent int // indentation for tracing
}

func newChecker(conf *Config, fset *token.FileSet, pkg *Package) *checker {
	return &checker{
		conf:        conf,
		fset:        fset,
		pkg:         pkg,
		methods:     make(map[string][]*Func),
		conversions: make(map[*ast.CallExpr]bool),
		untyped:     make(map[ast.Expr]exprInfo),
	}
}

func (check *checker) recordTypeAndValue(x ast.Expr, typ Type, val exact.Value) {
	assert(x != nil && typ != nil)
	if m := check.Types; m != nil {
		m[x] = typ
	}
	if val != nil {
		if m := check.Values; m != nil {
			m[x] = val
		}
	}
}

func (check *checker) recordCommaOkTypes(x ast.Expr, t1, t2 Type) {
	assert(x != nil && !isUntyped(t1) && !isUntyped(t2) && isBoolean(t2))
	if m := check.Types; m != nil {
		assert(m[x] != nil) // should have been recorded already
		pos := x.Pos()
		m[x] = NewTuple(
			NewVar(pos, check.pkg, "", t1),
			NewVar(pos, check.pkg, "", t2),
		)
	}
}

func (check *checker) recordObject(id *ast.Ident, obj Object) {
	assert(id != nil)
	if m := check.Objects; m != nil {
		m[id] = obj
	}
}

func (check *checker) recordImplicit(node ast.Node, obj Object) {
	assert(node != nil && obj != nil)
	if m := check.Implicits; m != nil {
		m[node] = obj
	}
}

func (check *checker) recordSelection(x *ast.SelectorExpr, kind SelectionKind, recv Type, obj Object, index []int, indirect bool) {
	assert(obj != nil && (recv == nil || len(index) > 0))
	check.recordObject(x.Sel, obj)
	// TODO(gri) Should we also call recordTypeAndValue?
	if m := check.Selections; m != nil {
		m[x] = &Selection{kind, recv, obj, index, indirect}
	}
}

func (check *checker) recordScope(node ast.Node, scope *Scope) {
	assert(node != nil && scope != nil)
	if m := check.Scopes; m != nil {
		m[node] = scope
	}
}

// A bailout panic is raised to indicate early termination.
type bailout struct{}

func (check *checker) handleBailout(err *error) {
	switch p := recover().(type) {
	case nil, bailout:
		// normal return or early exit
		*err = check.firstErr
	default:
		// unexpected panic: don't crash clients
		// TODO(gri) add a test case for this scenario
		*err = fmt.Errorf("types internal error: %v", p)
		if debug {
			check.dump("INTERNAL PANIC: %v", p)
			panic(p)
		}
	}
}

func (conf *Config) check(pkgPath string, fset *token.FileSet, files []*ast.File, info *Info) (pkg *Package, err error) {
	// make sure we have a package canonicalization map
	if conf.Packages == nil {
		conf.Packages = make(map[string]*Package)
	}

	pkg = NewPackage(pkgPath, "", NewScope(Universe)) // package name is set below
	check := newChecker(conf, fset, pkg)
	defer check.handleBailout(&err)

	// we need a reasonable package path to continue
	if path.Clean(pkgPath) == "." {
		check.errorf(token.NoPos, "invalid package path provided: %q", pkgPath)
		return
	}

	// determine package name and files
	i := 0
	for _, file := range files {
		switch name := file.Name.Name; pkg.name {
		case "":
			pkg.name = name
			fallthrough
		case name:
			files[i] = file
			i++
		default:
			check.errorf(file.Package, "package %s; expected %s", name, pkg.name)
			// ignore this file
		}
	}

	// install optional info
	if info != nil {
		check.Info = *info
	}

	check.resolveFiles(files[:i])

	// remaining untyped expressions must indeed be untyped
	if debug {
		for x, info := range check.untyped {
			if !isUntyped(info.typ) {
				check.dump("%s: %s (type %s) is not untyped", x.Pos(), x, info.typ)
				panic(0)
			}
		}
	}

	// notify client of any untyped types left
	// TODO(gri) Consider doing this before and
	// after function body checking for smaller
	// map size and more immediate feedback.
	if check.Types != nil || check.Values != nil {
		for x, info := range check.untyped {
			check.recordTypeAndValue(x, info.typ, info.val)
		}
	}

	return
}
