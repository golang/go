// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Check function, which drives type-checking.

package types

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

// debugging/development support
const (
	debug = false // leave on during development
	trace = false // turn on for detailed type resolution traces
)

// exprInfo stores type and constant value for an untyped expression.
type exprInfo struct {
	isLhs bool // expression is lhs operand of a shift with delayed type-check
	typ   *Basic
	val   exact.Value // constant value; or nil (if not a constant)
}

// A context represents the context within which an object is type-checked.
type context struct {
	decl          *declInfo   // package-level declaration whose init expression/function body is checked
	scope         *Scope      // top-most scope for lookups
	iota          exact.Value // value of iota in a constant declaration; nil otherwise
	sig           *Signature  // function signature if inside a function; nil otherwise
	hasLabel      bool        // set if a function makes use of labels (only ~1% of functions); unused outside functions
	hasCallOrRecv bool        // set if an expression contains a function call or channel receive operation
}

// A checker maintains the state of the type checker.
// It must be created with newChecker.
type checker struct {
	// package information
	// (set by newChecker)
	conf *Config
	fset *token.FileSet
	pkg  *Package
	*Info

	// information collected during type-checking of an entire package
	// (maps are allocated lazily)
	firstErr error                 // first error encountered
	methods  map[string][]*Func    // maps type names to associated methods
	untyped  map[ast.Expr]exprInfo // map of expressions without final type
	funcs    []funcInfo            // list of functions/methods with correct signatures and non-empty bodies
	delayed  []func()              // delayed checks that require fully setup types

	objMap  map[Object]*declInfo // if set we are in the package-level declaration phase (otherwise all objects seen must be declared)
	initMap map[Object]*declInfo // map of variables/functions with init expressions/bodies

	// context within which the current object is type-checked
	// (valid only for the duration of type-checking a specific object)
	context

	// debugging
	indent int // indentation for tracing
}

func (check *checker) assocMethod(tname string, meth *Func) {
	m := check.methods
	if m == nil {
		m = make(map[string][]*Func)
		check.methods = m
	}
	m[tname] = append(m[tname], meth)
}

func (check *checker) rememberUntyped(e ast.Expr, lhs bool, typ *Basic, val exact.Value) {
	m := check.untyped
	if m == nil {
		m = make(map[ast.Expr]exprInfo)
		check.untyped = m
	}
	m[e] = exprInfo{lhs, typ, val}
}

func (check *checker) delay(f func()) {
	check.delayed = append(check.delayed, f)
}

// newChecker returns a new Checker instance.
func newChecker(conf *Config, fset *token.FileSet, pkg *Package, info *Info) *checker {
	// make sure we have a configuration
	if conf == nil {
		conf = new(Config)
	}

	// make sure we have a package canonicalization map
	if conf.Packages == nil {
		conf.Packages = make(map[string]*Package)
	}

	// make sure we have an info struct
	if info == nil {
		info = new(Info)
	}

	return &checker{
		conf: conf,
		fset: fset,
		pkg:  pkg,
		Info: info,
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
		// re-panic
		panic(p)
	}
}

func (check *checker) files(files []*ast.File) (err error) {
	defer check.handleBailout(&err)

	pkg := check.pkg

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

	check.resolveFiles(files[:i])

	// perform delayed checks
	for _, f := range check.delayed {
		f()
	}
	check.delayed = nil // not needed anymore

	// remaining untyped expressions must indeed be untyped
	if debug {
		for x, info := range check.untyped {
			if isTyped(info.typ) {
				check.dump("%s: %s (type %s) is typed", x.Pos(), x, info.typ)
				panic(0)
			}
		}
	}

	// notify client of any untyped types left
	// TODO(gri) Consider doing this before and
	// after function body checking for smaller
	// map size and more immediate feedback.
	if check.Types != nil {
		for x, info := range check.untyped {
			check.recordTypeAndValue(x, info.typ, info.val)
		}
	}

	// copy check.InitOrder back to incoming *info if necessary
	// (In case of early (error) bailout, this is not done, but we don't care in that case.)
	// if info != nil {
	// 	info.InitOrder = check.InitOrder
	// }

	pkg.complete = true
	return
}

// addDeclDep adds the dependency edge (check.decl -> to)
// if check.decl exists and to has an init expression.
func (check *checker) addDeclDep(to Object) {
	from := check.decl
	if from == nil {
		return // not in a package-level init expression
	}
	init := check.initMap[to]
	if init == nil {
		return // to does not have a package-level init expression
	}
	m := from.deps
	if m == nil {
		m = make(map[Object]*declInfo)
		from.deps = m
	}
	m[to] = init
}

func (check *checker) recordTypeAndValue(x ast.Expr, typ Type, val exact.Value) {
	assert(x != nil && typ != nil)
	if val != nil {
		assert(isConstType(typ))
	}
	if m := check.Types; m != nil {
		m[x] = TypeAndValue{typ, val}
	}
}

func (check *checker) recordBuiltinType(f ast.Expr, sig *Signature) {
	// f must be a (possibly parenthesized) identifier denoting a built-in
	// (built-ins in package unsafe always produce a constant result and
	// we don't record their signatures, so we don't see qualified idents
	// here): record the signature for f and possible children.
	for {
		check.recordTypeAndValue(f, sig, nil)
		switch p := f.(type) {
		case *ast.Ident:
			return // we're done
		case *ast.ParenExpr:
			f = p.X
		default:
			unreachable()
		}
	}
}

func (check *checker) recordCommaOkTypes(x ast.Expr, a [2]Type) {
	assert(x != nil)
	if a[0] == nil || a[1] == nil {
		return
	}
	assert(isTyped(a[0]) && isTyped(a[1]) && isBoolean(a[1]))
	if m := check.Types; m != nil {
		for {
			tv := m[x]
			assert(tv.Type != nil) // should have been recorded already
			pos := x.Pos()
			tv.Type = NewTuple(
				NewVar(pos, check.pkg, "", a[0]),
				NewVar(pos, check.pkg, "", a[1]),
			)
			m[x] = tv
			// if x is a parenthesized expression (p.X), update p.X
			p, _ := x.(*ast.ParenExpr)
			if p == nil {
				break
			}
			x = p.X
		}
	}
}

func (check *checker) recordDef(id *ast.Ident, obj Object) {
	assert(id != nil)
	if m := check.Defs; m != nil {
		m[id] = obj
	}
}

func (check *checker) recordUse(id *ast.Ident, obj Object) {
	assert(id != nil)
	assert(obj != nil)
	if m := check.Uses; m != nil {
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
	check.recordUse(x.Sel, obj)
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
