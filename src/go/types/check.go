// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Check function, which drives type-checking.

package types

import (
	"errors"
	"go/ast"
	"go/constant"
	"go/token"
)

// debugging/development support
const (
	debug = false // leave on during development
	trace = false // turn on for detailed type resolution traces
)

// If Strict is set, the type-checker enforces additional
// rules not specified by the Go 1 spec, but which will
// catch guaranteed run-time errors if the respective
// code is executed. In other words, programs passing in
// Strict mode are Go 1 compliant, but not all Go 1 programs
// will pass in Strict mode. The additional rules are:
//
// - A type assertion x.(T) where T is an interface type
//   is invalid if any (statically known) method that exists
//   for both x and T have different signatures.
//
const strict = false

// exprInfo stores information about an untyped expression.
type exprInfo struct {
	isLhs bool // expression is lhs operand of a shift with delayed type-check
	mode  operandMode
	typ   *Basic
	val   constant.Value // constant value; or nil (if not a constant)
}

// A context represents the context within which an object is type-checked.
type context struct {
	decl          *declInfo              // package-level declaration whose init expression/function body is checked
	scope         *Scope                 // top-most scope for lookups
	pos           token.Pos              // if valid, identifiers are looked up as if at position pos (used by Eval)
	iota          constant.Value         // value of iota in a constant declaration; nil otherwise
	sig           *Signature             // function signature if inside a function; nil otherwise
	isPanic       map[*ast.CallExpr]bool // set of panic call expressions (used for termination check)
	hasLabel      bool                   // set if a function makes use of labels (only ~1% of functions); unused outside functions
	hasCallOrRecv bool                   // set if an expression contains a function call or channel receive operation
}

// lookup looks up name in the current context and returns the matching object, or nil.
func (ctxt *context) lookup(name string) Object {
	_, obj := ctxt.scope.LookupParent(name, ctxt.pos)
	return obj
}

// An importKey identifies an imported package by import path and source directory
// (directory containing the file containing the import). In practice, the directory
// may always be the same, or may not matter. Given an (import path, directory), an
// importer must always return the same package (but given two different import paths,
// an importer may still return the same package by mapping them to the same package
// paths).
type importKey struct {
	path, dir string
}

// A Checker maintains the state of the type checker.
// It must be created with NewChecker.
type Checker struct {
	// package information
	// (initialized by NewChecker, valid for the life-time of checker)
	conf *Config
	fset *token.FileSet
	pkg  *Package
	*Info
	objMap map[Object]*declInfo       // maps package-level objects and (non-interface) methods to declaration info
	impMap map[importKey]*Package     // maps (import path, source directory) to (complete or fake) package
	posMap map[*Interface][]token.Pos // maps interface types to lists of embedded interface positions
	pkgCnt map[string]int             // counts number of imported packages with a given name (for better error messages)

	// information collected during type-checking of a set of package files
	// (initialized by Files, valid only for the duration of check.Files;
	// maps and lists are allocated on demand)
	files            []*ast.File                       // package files
	unusedDotImports map[*Scope]map[*Package]token.Pos // positions of unused dot-imported packages for each file scope

	firstErr error                 // first error encountered
	methods  map[*TypeName][]*Func // maps package scope type names to associated non-blank (non-interface) methods
	untyped  map[ast.Expr]exprInfo // map of expressions without final type
	delayed  []func()              // stack of delayed action segments; segments are processed in FIFO order
	finals   []func()              // list of final actions; processed at the end of type-checking the current set of files
	objPath  []Object              // path of object dependencies during type inference (for cycle reporting)

	// context within which the current object is type-checked
	// (valid only for the duration of type-checking a specific object)
	context

	// debugging
	indent int // indentation for tracing
}

// addUnusedImport adds the position of a dot-imported package
// pkg to the map of dot imports for the given file scope.
func (check *Checker) addUnusedDotImport(scope *Scope, pkg *Package, pos token.Pos) {
	mm := check.unusedDotImports
	if mm == nil {
		mm = make(map[*Scope]map[*Package]token.Pos)
		check.unusedDotImports = mm
	}
	m := mm[scope]
	if m == nil {
		m = make(map[*Package]token.Pos)
		mm[scope] = m
	}
	m[pkg] = pos
}

// addDeclDep adds the dependency edge (check.decl -> to) if check.decl exists
func (check *Checker) addDeclDep(to Object) {
	from := check.decl
	if from == nil {
		return // not in a package-level init expression
	}
	if _, found := check.objMap[to]; !found {
		return // to is not a package-level object
	}
	from.addDep(to)
}

func (check *Checker) rememberUntyped(e ast.Expr, lhs bool, mode operandMode, typ *Basic, val constant.Value) {
	m := check.untyped
	if m == nil {
		m = make(map[ast.Expr]exprInfo)
		check.untyped = m
	}
	m[e] = exprInfo{lhs, mode, typ, val}
}

// later pushes f on to the stack of actions that will be processed later;
// either at the end of the current statement, or in case of a local constant
// or variable declaration, before the constant or variable is in scope
// (so that f still sees the scope before any new declarations).
func (check *Checker) later(f func()) {
	check.delayed = append(check.delayed, f)
}

// atEnd adds f to the list of actions processed at the end
// of type-checking, before initialization order computation.
// Actions added by atEnd are processed after any actions
// added by later.
func (check *Checker) atEnd(f func()) {
	check.finals = append(check.finals, f)
}

// push pushes obj onto the object path and returns its index in the path.
func (check *Checker) push(obj Object) int {
	check.objPath = append(check.objPath, obj)
	return len(check.objPath) - 1
}

// pop pops and returns the topmost object from the object path.
func (check *Checker) pop() Object {
	i := len(check.objPath) - 1
	obj := check.objPath[i]
	check.objPath[i] = nil
	check.objPath = check.objPath[:i]
	return obj
}

// NewChecker returns a new Checker instance for a given package.
// Package files may be added incrementally via checker.Files.
func NewChecker(conf *Config, fset *token.FileSet, pkg *Package, info *Info) *Checker {
	// make sure we have a configuration
	if conf == nil {
		conf = new(Config)
	}

	// make sure we have an info struct
	if info == nil {
		info = new(Info)
	}

	return &Checker{
		conf:   conf,
		fset:   fset,
		pkg:    pkg,
		Info:   info,
		objMap: make(map[Object]*declInfo),
		impMap: make(map[importKey]*Package),
		posMap: make(map[*Interface][]token.Pos),
		pkgCnt: make(map[string]int),
	}
}

// initFiles initializes the files-specific portion of checker.
// The provided files must all belong to the same package.
func (check *Checker) initFiles(files []*ast.File) {
	// start with a clean slate (check.Files may be called multiple times)
	check.files = nil
	check.unusedDotImports = nil

	check.firstErr = nil
	check.methods = nil
	check.untyped = nil
	check.delayed = nil
	check.finals = nil

	// determine package name and collect valid files
	pkg := check.pkg
	for _, file := range files {
		switch name := file.Name.Name; pkg.name {
		case "":
			if name != "_" {
				pkg.name = name
			} else {
				check.errorf(file.Name.Pos(), "invalid package name _")
			}
			fallthrough

		case name:
			check.files = append(check.files, file)

		default:
			check.errorf(file.Package, "package %s; expected %s", name, pkg.name)
			// ignore this file
		}
	}
}

// A bailout panic is used for early termination.
type bailout struct{}

func (check *Checker) handleBailout(err *error) {
	switch p := recover().(type) {
	case nil, bailout:
		// normal return or early exit
		*err = check.firstErr
	default:
		// re-panic
		panic(p)
	}
}

// Files checks the provided files as part of the checker's package.
func (check *Checker) Files(files []*ast.File) error { return check.checkFiles(files) }

var errBadCgo = errors.New("cannot use FakeImportC and go115UsesCgo together")

func (check *Checker) checkFiles(files []*ast.File) (err error) {
	if check.conf.FakeImportC && check.conf.go115UsesCgo {
		return errBadCgo
	}

	defer check.handleBailout(&err)

	check.initFiles(files)

	check.collectObjects()

	check.packageObjects()

	check.processDelayed(0) // incl. all functions
	check.processFinals()

	check.initOrder()

	if !check.conf.DisableUnusedImportCheck {
		check.unusedImports()
	}

	check.recordUntyped()

	check.pkg.complete = true
	return
}

// processDelayed processes all delayed actions pushed after top.
func (check *Checker) processDelayed(top int) {
	// If each delayed action pushes a new action, the
	// stack will continue to grow during this loop.
	// However, it is only processing functions (which
	// are processed in a delayed fashion) that may
	// add more actions (such as nested functions), so
	// this is a sufficiently bounded process.
	for i := top; i < len(check.delayed); i++ {
		check.delayed[i]() // may append to check.delayed
	}
	assert(top <= len(check.delayed)) // stack must not have shrunk
	check.delayed = check.delayed[:top]
}

func (check *Checker) processFinals() {
	n := len(check.finals)
	for _, f := range check.finals {
		f() // must not append to check.finals
	}
	if len(check.finals) != n {
		panic("internal error: final action list grew")
	}
}

func (check *Checker) recordUntyped() {
	if !debug && check.Types == nil {
		return // nothing to do
	}

	for x, info := range check.untyped {
		if debug && isTyped(info.typ) {
			check.dump("%v: %s (type %s) is typed", x.Pos(), x, info.typ)
			unreachable()
		}
		check.recordTypeAndValue(x, info.mode, info.typ, info.val)
	}
}

func (check *Checker) recordTypeAndValue(x ast.Expr, mode operandMode, typ Type, val constant.Value) {
	assert(x != nil)
	assert(typ != nil)
	if mode == invalid {
		return // omit
	}
	if mode == constant_ {
		assert(val != nil)
		assert(typ == Typ[Invalid] || isConstType(typ))
	}
	if m := check.Types; m != nil {
		m[x] = TypeAndValue{mode, typ, val}
	}
}

func (check *Checker) recordBuiltinType(f ast.Expr, sig *Signature) {
	// f must be a (possibly parenthesized) identifier denoting a built-in
	// (built-ins in package unsafe always produce a constant result and
	// we don't record their signatures, so we don't see qualified idents
	// here): record the signature for f and possible children.
	for {
		check.recordTypeAndValue(f, builtin, sig, nil)
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

func (check *Checker) recordCommaOkTypes(x ast.Expr, a [2]Type) {
	assert(x != nil)
	if a[0] == nil || a[1] == nil {
		return
	}
	assert(isTyped(a[0]) && isTyped(a[1]) && (isBoolean(a[1]) || a[1] == universeError))
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

func (check *Checker) recordDef(id *ast.Ident, obj Object) {
	assert(id != nil)
	if m := check.Defs; m != nil {
		m[id] = obj
	}
}

func (check *Checker) recordUse(id *ast.Ident, obj Object) {
	assert(id != nil)
	assert(obj != nil)
	if m := check.Uses; m != nil {
		m[id] = obj
	}
}

func (check *Checker) recordImplicit(node ast.Node, obj Object) {
	assert(node != nil)
	assert(obj != nil)
	if m := check.Implicits; m != nil {
		m[node] = obj
	}
}

func (check *Checker) recordSelection(x *ast.SelectorExpr, kind SelectionKind, recv Type, obj Object, index []int, indirect bool) {
	assert(obj != nil && (recv == nil || len(index) > 0))
	check.recordUse(x.Sel, obj)
	if m := check.Selections; m != nil {
		m[x] = &Selection{kind, recv, obj, index, indirect}
	}
}

func (check *Checker) recordScope(node ast.Node, scope *Scope) {
	assert(node != nil)
	assert(scope != nil)
	if m := check.Scopes; m != nil {
		m[node] = scope
	}
}
