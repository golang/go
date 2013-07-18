// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Check function, which typechecks a package.

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

// If retainASTLinks is set, scopes maintain a link to the node
// responsible for it.
// TODO(gri) Decide if this should be a permanent (always present)
//           or optional feature (enabled with a mode flag).
const retainASTLinks = true

// exprInfo stores type and constant value for an untyped expression.
type exprInfo struct {
	isLhs bool // expression is lhs operand of a shift with delayed type check
	typ   *Basic
	val   exact.Value // constant value; or nil (if not a constant)
}

// A checker is an instance of the type checker.
type checker struct {
	ctxt *Context
	fset *token.FileSet
	Info

	// lazily initialized
	pkg         *Package               // current package
	firsterr    error                  // first error encountered
	methods     map[*TypeName]*Scope   // maps type names to associated methods
	conversions map[*ast.CallExpr]bool // set of type-checked conversions (to distinguish from calls)
	untyped     map[ast.Expr]exprInfo  // map of expressions without final type

	objMap   map[Object]*decl // if set we are in the package-global declaration phase (otherwise all objects seen must be declared)
	topScope *Scope           // topScope for lookups, non-global declarations
	iota     exact.Value      // value of iota in a constant declaration; nil otherwise

	// functions
	funclist []function // list of functions/methods with correct signatures and non-empty bodies
	funcsig  *Signature // signature of currently typechecked function

	// debugging
	indent int // indentation for tracing
}

func newChecker(ctxt *Context, fset *token.FileSet, pkg *Package) *checker {
	return &checker{
		ctxt:        ctxt,
		fset:        fset,
		pkg:         pkg,
		methods:     make(map[*TypeName]*Scope),
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

type function struct {
	file *Scope // only valid if resolve is set
	obj  *Func  // for debugging/tracing only
	sig  *Signature
	body *ast.BlockStmt
}

// later adds a function with non-empty body to the list of functions
// that need to be processed after all package-level declarations
// are typechecked.
//
func (check *checker) later(f *Func, sig *Signature, body *ast.BlockStmt) {
	// functions implemented elsewhere (say in assembly) have no body
	if body != nil {
		check.funclist = append(check.funclist, function{check.topScope, f, sig, body})
	}
}

// A bailout panic is raised to indicate early termination.
type bailout struct{}

func (check *checker) handleBailout(err *error) {
	switch p := recover().(type) {
	case nil, bailout:
		// normal return or early exit
		*err = check.firsterr
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

func (ctxt *Context) check(pkgPath string, fset *token.FileSet, files []*ast.File, info *Info) (pkg *Package, err error) {
	pkg = &Package{
		path:    pkgPath,
		scope:   NewScope(Universe),
		imports: make(map[string]*Package),
	}

	check := newChecker(ctxt, fset, pkg)
	defer check.handleBailout(&err)

	// we need a reasonable path to continue
	if path.Clean(pkgPath) == "." {
		check.errorf(token.NoPos, "invalid package path provided: %q", pkgPath)
		return
	}

	// install optional info
	if info != nil {
		check.Info = *info
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

	// TODO(gri) resolveFiles needs to be split up and renamed (cleanup)
	check.resolveFiles(files[:i])

	// typecheck all function/method bodies
	// (funclist may grow when checking statements - do not use range clause!)
	for i := 0; i < len(check.funclist); i++ {
		f := check.funclist[i]
		if trace {
			s := "<function literal>"
			if f.obj != nil {
				s = f.obj.name
			}
			fmt.Println("---", s)
		}
		check.topScope = f.sig.scope // open the function scope
		check.funcsig = f.sig
		check.stmtList(f.body.List)
		if f.sig.results.Len() > 0 && f.body != nil && !check.isTerminating(f.body, "") {
			check.errorf(f.body.Rbrace, "missing return")
		}
	}

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
