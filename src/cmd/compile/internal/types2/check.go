// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Check function, which drives type-checking.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"go/constant"
	. "internal/types/errors"
	"sync/atomic"
)

// nopos indicates an unknown position
var nopos syntax.Pos

// debugging/development support
const debug = false // leave on during development

// _aliasAny changes the behavior of [Scope.Lookup] for "any" in the
// [Universe] scope.
//
// This is necessary because while Alias creation is controlled by
// [Config.EnableAlias], the representation of "any" is a global. In
// [Scope.Lookup], we select this global representation based on the result of
// [aliasAny], but as a result need to guard against this behavior changing
// during the type checking pass. Therefore we implement the following rule:
// any number of goroutines can type check concurrently with the same
// EnableAlias value, but if any goroutine tries to type check concurrently
// with a different EnableAlias value, we panic.
//
// To achieve this, _aliasAny is a state machine:
//
//	0:        no type checking is occurring
//	negative: type checking is occurring without EnableAlias set
//	positive: type checking is occurring with EnableAlias set
var _aliasAny int32

func aliasAny() bool {
	return atomic.LoadInt32(&_aliasAny) >= 0 // default true
}

// exprInfo stores information about an untyped expression.
type exprInfo struct {
	isLhs bool // expression is lhs operand of a shift with delayed type-check
	mode  operandMode
	typ   *Basic
	val   constant.Value // constant value; or nil (if not a constant)
}

// An environment represents the environment within which an object is
// type-checked.
type environment struct {
	decl          *declInfo                 // package-level declaration whose init expression/function body is checked
	scope         *Scope                    // top-most scope for lookups
	pos           syntax.Pos                // if valid, identifiers are looked up as if at position pos (used by Eval)
	iota          constant.Value            // value of iota in a constant declaration; nil otherwise
	errpos        syntax.Pos                // if valid, identifier position of a constant with inherited initializer
	inTParamList  bool                      // set if inside a type parameter list
	sig           *Signature                // function signature if inside a function; nil otherwise
	isPanic       map[*syntax.CallExpr]bool // set of panic call expressions (used for termination check)
	hasLabel      bool                      // set if a function makes use of labels (only ~1% of functions); unused outside functions
	hasCallOrRecv bool                      // set if an expression contains a function call or channel receive operation
}

// lookup looks up name in the current environment and returns the matching object, or nil.
func (env *environment) lookup(name string) Object {
	_, obj := env.scope.LookupParent(name, env.pos)
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

// A dotImportKey describes a dot-imported object in the given scope.
type dotImportKey struct {
	scope *Scope
	name  string
}

// An action describes a (delayed) action.
type action struct {
	f    func()      // action to be executed
	desc *actionDesc // action description; may be nil, requires debug to be set
}

// If debug is set, describef sets a printf-formatted description for action a.
// Otherwise, it is a no-op.
func (a *action) describef(pos poser, format string, args ...interface{}) {
	if debug {
		a.desc = &actionDesc{pos, format, args}
	}
}

// An actionDesc provides information on an action.
// For debugging only.
type actionDesc struct {
	pos    poser
	format string
	args   []interface{}
}

// A Checker maintains the state of the type checker.
// It must be created with NewChecker.
type Checker struct {
	// package information
	// (initialized by NewChecker, valid for the life-time of checker)
	conf *Config
	ctxt *Context // context for de-duplicating instances
	pkg  *Package
	*Info
	version goVersion              // accepted language version
	nextID  uint64                 // unique Id for type parameters (first valid Id is 1)
	objMap  map[Object]*declInfo   // maps package-level objects and (non-interface) methods to declaration info
	impMap  map[importKey]*Package // maps (import path, source directory) to (complete or fake) package
	// see TODO in validtype.go
	// valids  instanceLookup      // valid *Named (incl. instantiated) types per the validType check

	// pkgPathMap maps package names to the set of distinct import paths we've
	// seen for that name, anywhere in the import graph. It is used for
	// disambiguating package names in error messages.
	//
	// pkgPathMap is allocated lazily, so that we don't pay the price of building
	// it on the happy path. seenPkgMap tracks the packages that we've already
	// walked.
	pkgPathMap map[string]map[string]bool
	seenPkgMap map[*Package]bool

	// information collected during type-checking of a set of package files
	// (initialized by Files, valid only for the duration of check.Files;
	// maps and lists are allocated on demand)
	files         []*syntax.File              // list of package files
	versions      map[*syntax.PosBase]string  // maps files to version strings (each file has an entry); shared with Info.FileVersions if present
	imports       []*PkgName                  // list of imported packages
	dotImportMap  map[dotImportKey]*PkgName   // maps dot-imported objects to the package they were dot-imported through
	recvTParamMap map[*syntax.Name]*TypeParam // maps blank receiver type parameters to their type
	brokenAliases map[*TypeName]bool          // set of aliases with broken (not yet determined) types
	unionTypeSets map[*Union]*_TypeSet        // computed type sets for union types
	mono          monoGraph                   // graph for detecting non-monomorphizable instantiation loops

	firstErr error                    // first error encountered
	methods  map[*TypeName][]*Func    // maps package scope type names to associated non-blank (non-interface) methods
	untyped  map[syntax.Expr]exprInfo // map of expressions without final type
	delayed  []action                 // stack of delayed action segments; segments are processed in FIFO order
	objPath  []Object                 // path of object dependencies during type inference (for cycle reporting)
	cleaners []cleaner                // list of types that may need a final cleanup at the end of type-checking

	// environment within which the current object is type-checked (valid only
	// for the duration of type-checking a specific object)
	environment

	// debugging
	indent int // indentation for tracing
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

// Note: The following three alias-related functions are only used
//       when Alias types are not enabled.

// brokenAlias records that alias doesn't have a determined type yet.
// It also sets alias.typ to Typ[Invalid].
// Not used if check.conf.EnableAlias is set.
func (check *Checker) brokenAlias(alias *TypeName) {
	assert(!check.conf.EnableAlias)
	if check.brokenAliases == nil {
		check.brokenAliases = make(map[*TypeName]bool)
	}
	check.brokenAliases[alias] = true
	alias.typ = Typ[Invalid]
}

// validAlias records that alias has the valid type typ (possibly Typ[Invalid]).
func (check *Checker) validAlias(alias *TypeName, typ Type) {
	assert(!check.conf.EnableAlias)
	delete(check.brokenAliases, alias)
	alias.typ = typ
}

// isBrokenAlias reports whether alias doesn't have a determined type yet.
func (check *Checker) isBrokenAlias(alias *TypeName) bool {
	assert(!check.conf.EnableAlias)
	return check.brokenAliases[alias]
}

func (check *Checker) rememberUntyped(e syntax.Expr, lhs bool, mode operandMode, typ *Basic, val constant.Value) {
	m := check.untyped
	if m == nil {
		m = make(map[syntax.Expr]exprInfo)
		check.untyped = m
	}
	m[e] = exprInfo{lhs, mode, typ, val}
}

// later pushes f on to the stack of actions that will be processed later;
// either at the end of the current statement, or in case of a local constant
// or variable declaration, before the constant or variable is in scope
// (so that f still sees the scope before any new declarations).
// later returns the pushed action so one can provide a description
// via action.describef for debugging, if desired.
func (check *Checker) later(f func()) *action {
	i := len(check.delayed)
	check.delayed = append(check.delayed, action{f: f})
	return &check.delayed[i]
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

type cleaner interface {
	cleanup()
}

// needsCleanup records objects/types that implement the cleanup method
// which will be called at the end of type-checking.
func (check *Checker) needsCleanup(c cleaner) {
	check.cleaners = append(check.cleaners, c)
}

// NewChecker returns a new Checker instance for a given package.
// Package files may be added incrementally via checker.Files.
func NewChecker(conf *Config, pkg *Package, info *Info) *Checker {
	// make sure we have a configuration
	if conf == nil {
		conf = new(Config)
	}

	// make sure we have an info struct
	if info == nil {
		info = new(Info)
	}

	// Note: clients may call NewChecker with the Unsafe package, which is
	// globally shared and must not be mutated. Therefore NewChecker must not
	// mutate *pkg.
	//
	// (previously, pkg.goVersion was mutated here: go.dev/issue/61212)

	return &Checker{
		conf:    conf,
		ctxt:    conf.Context,
		pkg:     pkg,
		Info:    info,
		version: asGoVersion(conf.GoVersion),
		objMap:  make(map[Object]*declInfo),
		impMap:  make(map[importKey]*Package),
	}
}

// initFiles initializes the files-specific portion of checker.
// The provided files must all belong to the same package.
func (check *Checker) initFiles(files []*syntax.File) {
	// start with a clean slate (check.Files may be called multiple times)
	check.files = nil
	check.imports = nil
	check.dotImportMap = nil

	check.firstErr = nil
	check.methods = nil
	check.untyped = nil
	check.delayed = nil
	check.objPath = nil
	check.cleaners = nil

	// determine package name and collect valid files
	pkg := check.pkg
	for _, file := range files {
		switch name := file.PkgName.Value; pkg.name {
		case "":
			if name != "_" {
				pkg.name = name
			} else {
				check.error(file.PkgName, BlankPkgName, "invalid package name _")
			}
			fallthrough

		case name:
			check.files = append(check.files, file)

		default:
			check.errorf(file, MismatchedPkgName, "package %s; expected %s", quote(name), quote(pkg.name))
			// ignore this file
		}
	}

	// reuse Info.FileVersions if provided
	versions := check.Info.FileVersions
	if versions == nil {
		versions = make(map[*syntax.PosBase]string)
	}
	check.versions = versions

	pkgVersionOk := check.version.isValid()
	if pkgVersionOk && len(files) > 0 && check.version.cmp(go_current) > 0 {
		check.errorf(files[0], TooNew, "package requires newer Go version %v (application built with %v)",
			check.version, go_current)
	}
	downgradeOk := check.version.cmp(go1_21) >= 0

	// determine Go version for each file
	for _, file := range check.files {
		// use unaltered Config.GoVersion by default
		// (This version string may contain dot-release numbers as in go1.20.1,
		// unlike file versions which are Go language versions only, if valid.)
		v := check.conf.GoVersion

		fileVersion := asGoVersion(file.GoVersion)
		if fileVersion.isValid() {
			// use the file version, if applicable
			// (file versions are either the empty string or of the form go1.dd)
			if pkgVersionOk {
				cmp := fileVersion.cmp(check.version)
				// Go 1.21 introduced the feature of setting the go.mod
				// go line to an early version of Go and allowing //go:build lines
				// to “upgrade” (cmp > 0) the Go version in a given file.
				// We can do that backwards compatibly.
				//
				// Go 1.21 also introduced the feature of allowing //go:build lines
				// to “downgrade” (cmp < 0) the Go version in a given file.
				// That can't be done compatibly in general, since before the
				// build lines were ignored and code got the module's Go version.
				// To work around this, downgrades are only allowed when the
				// module's Go version is Go 1.21 or later.
				//
				// If there is no valid check.version, then we don't really know what
				// Go version to apply.
				// Legacy tools may do this, and they historically have accepted everything.
				// Preserve that behavior by ignoring //go:build constraints entirely in that
				// case (!pkgVersionOk).
				if cmp > 0 || cmp < 0 && downgradeOk {
					v = file.GoVersion
				}
			}

			// Report a specific error for each tagged file that's too new.
			// (Normally the build system will have filtered files by version,
			// but clients can present arbitrary files to the type checker.)
			if fileVersion.cmp(go_current) > 0 {
				// Use position of 'package [p]' for types/types2 consistency.
				// (Ideally we would use the //build tag itself.)
				check.errorf(file.PkgName, TooNew, "file requires newer Go version %v", fileVersion)
			}
		}
		versions[base(file.Pos())] = v // base(file.Pos()) may be nil for tests
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
func (check *Checker) Files(files []*syntax.File) (err error) {
	if check.pkg == Unsafe {
		// Defensive handling for Unsafe, which cannot be type checked, and must
		// not be mutated. See https://go.dev/issue/61212 for an example of where
		// Unsafe is passed to NewChecker.
		return nil
	}

	// Avoid early returns here! Nearly all errors can be
	// localized to a piece of syntax and needn't prevent
	// type-checking of the rest of the package.

	defer check.handleBailout(&err)
	check.checkFiles(files)
	return
}

// checkFiles type-checks the specified files. Errors are reported as
// a side effect, not by returning early, to ensure that well-formed
// syntax is properly type annotated even in a package containing
// errors.
func (check *Checker) checkFiles(files []*syntax.File) {
	// Ensure that EnableAlias is consistent among concurrent type checking
	// operations. See the documentation of [_aliasAny] for details.
	if check.conf.EnableAlias {
		if atomic.AddInt32(&_aliasAny, 1) <= 0 {
			panic("EnableAlias set while !EnableAlias type checking is ongoing")
		}
		defer atomic.AddInt32(&_aliasAny, -1)
	} else {
		if atomic.AddInt32(&_aliasAny, -1) >= 0 {
			panic("!EnableAlias set while EnableAlias type checking is ongoing")
		}
		defer atomic.AddInt32(&_aliasAny, 1)
	}

	print := func(msg string) {
		if check.conf.Trace {
			fmt.Println()
			fmt.Println(msg)
		}
	}

	print("== initFiles ==")
	check.initFiles(files)

	print("== collectObjects ==")
	check.collectObjects()

	print("== packageObjects ==")
	check.packageObjects()

	print("== processDelayed ==")
	check.processDelayed(0) // incl. all functions

	print("== cleanup ==")
	check.cleanup()

	print("== initOrder ==")
	check.initOrder()

	if !check.conf.DisableUnusedImportCheck {
		print("== unusedImports ==")
		check.unusedImports()
	}

	print("== recordUntyped ==")
	check.recordUntyped()

	if check.firstErr == nil {
		// TODO(mdempsky): Ensure monomorph is safe when errors exist.
		check.monomorph()
	}

	check.pkg.goVersion = check.conf.GoVersion
	check.pkg.complete = true

	// no longer needed - release memory
	check.imports = nil
	check.dotImportMap = nil
	check.pkgPathMap = nil
	check.seenPkgMap = nil
	check.recvTParamMap = nil
	check.brokenAliases = nil
	check.unionTypeSets = nil
	check.ctxt = nil

	// TODO(gri) There's more memory we should release at this point.
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
		a := &check.delayed[i]
		if check.conf.Trace {
			if a.desc != nil {
				check.trace(a.desc.pos.Pos(), "-- "+a.desc.format, a.desc.args...)
			} else {
				check.trace(nopos, "-- delayed %p", a.f)
			}
		}
		a.f() // may append to check.delayed
		if check.conf.Trace {
			fmt.Println()
		}
	}
	assert(top <= len(check.delayed)) // stack must not have shrunk
	check.delayed = check.delayed[:top]
}

// cleanup runs cleanup for all collected cleaners.
func (check *Checker) cleanup() {
	// Don't use a range clause since Named.cleanup may add more cleaners.
	for i := 0; i < len(check.cleaners); i++ {
		check.cleaners[i].cleanup()
	}
	check.cleaners = nil
}

func (check *Checker) record(x *operand) {
	// convert x into a user-friendly set of values
	// TODO(gri) this code can be simplified
	var typ Type
	var val constant.Value
	switch x.mode {
	case invalid:
		typ = Typ[Invalid]
	case novalue:
		typ = (*Tuple)(nil)
	case constant_:
		typ = x.typ
		val = x.val
	default:
		typ = x.typ
	}
	assert(x.expr != nil && typ != nil)

	if isUntyped(typ) {
		// delay type and value recording until we know the type
		// or until the end of type checking
		check.rememberUntyped(x.expr, false, x.mode, typ.(*Basic), val)
	} else {
		check.recordTypeAndValue(x.expr, x.mode, typ, val)
	}
}

func (check *Checker) recordUntyped() {
	if !debug && !check.recordTypes() {
		return // nothing to do
	}

	for x, info := range check.untyped {
		if debug && isTyped(info.typ) {
			check.dump("%v: %s (type %s) is typed", atPos(x), x, info.typ)
			panic("unreachable")
		}
		check.recordTypeAndValue(x, info.mode, info.typ, info.val)
	}
}

func (check *Checker) recordTypeAndValue(x syntax.Expr, mode operandMode, typ Type, val constant.Value) {
	assert(x != nil)
	assert(typ != nil)
	if mode == invalid {
		return // omit
	}
	if mode == constant_ {
		assert(val != nil)
		// We check allBasic(typ, IsConstType) here as constant expressions may be
		// recorded as type parameters.
		assert(!isValid(typ) || allBasic(typ, IsConstType))
	}
	if m := check.Types; m != nil {
		m[x] = TypeAndValue{mode, typ, val}
	}
	if check.StoreTypesInSyntax {
		tv := TypeAndValue{mode, typ, val}
		stv := syntax.TypeAndValue{Type: typ, Value: val}
		if tv.IsVoid() {
			stv.SetIsVoid()
		}
		if tv.IsType() {
			stv.SetIsType()
		}
		if tv.IsBuiltin() {
			stv.SetIsBuiltin()
		}
		if tv.IsValue() {
			stv.SetIsValue()
		}
		if tv.IsNil() {
			stv.SetIsNil()
		}
		if tv.Addressable() {
			stv.SetAddressable()
		}
		if tv.Assignable() {
			stv.SetAssignable()
		}
		if tv.HasOk() {
			stv.SetHasOk()
		}
		x.SetTypeInfo(stv)
	}
}

func (check *Checker) recordBuiltinType(f syntax.Expr, sig *Signature) {
	// f must be a (possibly parenthesized, possibly qualified)
	// identifier denoting a built-in (including unsafe's non-constant
	// functions Add and Slice): record the signature for f and possible
	// children.
	for {
		check.recordTypeAndValue(f, builtin, sig, nil)
		switch p := f.(type) {
		case *syntax.Name, *syntax.SelectorExpr:
			return // we're done
		case *syntax.ParenExpr:
			f = p.X
		default:
			panic("unreachable")
		}
	}
}

// recordCommaOkTypes updates recorded types to reflect that x is used in a commaOk context
// (and therefore has tuple type).
func (check *Checker) recordCommaOkTypes(x syntax.Expr, a []*operand) {
	assert(x != nil)
	assert(len(a) == 2)
	if a[0].mode == invalid {
		return
	}
	t0, t1 := a[0].typ, a[1].typ
	assert(isTyped(t0) && isTyped(t1) && (allBoolean(t1) || t1 == universeError))
	if m := check.Types; m != nil {
		for {
			tv := m[x]
			assert(tv.Type != nil) // should have been recorded already
			pos := x.Pos()
			tv.Type = NewTuple(
				NewVar(pos, check.pkg, "", t0),
				NewVar(pos, check.pkg, "", t1),
			)
			m[x] = tv
			// if x is a parenthesized expression (p.X), update p.X
			p, _ := x.(*syntax.ParenExpr)
			if p == nil {
				break
			}
			x = p.X
		}
	}
	if check.StoreTypesInSyntax {
		// Note: this loop is duplicated because the type of tv is different.
		// Above it is types2.TypeAndValue, here it is syntax.TypeAndValue.
		for {
			tv := x.GetTypeInfo()
			assert(tv.Type != nil) // should have been recorded already
			pos := x.Pos()
			tv.Type = NewTuple(
				NewVar(pos, check.pkg, "", t0),
				NewVar(pos, check.pkg, "", t1),
			)
			x.SetTypeInfo(tv)
			p, _ := x.(*syntax.ParenExpr)
			if p == nil {
				break
			}
			x = p.X
		}
	}
}

// recordInstance records instantiation information into check.Info, if the
// Instances map is non-nil. The given expr must be an ident, selector, or
// index (list) expr with ident or selector operand.
//
// TODO(rfindley): the expr parameter is fragile. See if we can access the
// instantiated identifier in some other way.
func (check *Checker) recordInstance(expr syntax.Expr, targs []Type, typ Type) {
	ident := instantiatedIdent(expr)
	assert(ident != nil)
	assert(typ != nil)
	if m := check.Instances; m != nil {
		m[ident] = Instance{newTypeList(targs), typ}
	}
}

func instantiatedIdent(expr syntax.Expr) *syntax.Name {
	var selOrIdent syntax.Expr
	switch e := expr.(type) {
	case *syntax.IndexExpr:
		selOrIdent = e.X
	case *syntax.SelectorExpr, *syntax.Name:
		selOrIdent = e
	}
	switch x := selOrIdent.(type) {
	case *syntax.Name:
		return x
	case *syntax.SelectorExpr:
		return x.Sel
	}
	panic("instantiated ident not found")
}

func (check *Checker) recordDef(id *syntax.Name, obj Object) {
	assert(id != nil)
	if m := check.Defs; m != nil {
		m[id] = obj
	}
}

func (check *Checker) recordUse(id *syntax.Name, obj Object) {
	assert(id != nil)
	assert(obj != nil)
	if m := check.Uses; m != nil {
		m[id] = obj
	}
}

func (check *Checker) recordImplicit(node syntax.Node, obj Object) {
	assert(node != nil)
	assert(obj != nil)
	if m := check.Implicits; m != nil {
		m[node] = obj
	}
}

func (check *Checker) recordSelection(x *syntax.SelectorExpr, kind SelectionKind, recv Type, obj Object, index []int, indirect bool) {
	assert(obj != nil && (recv == nil || len(index) > 0))
	check.recordUse(x.Sel, obj)
	if m := check.Selections; m != nil {
		m[x] = &Selection{kind, recv, obj, index, indirect}
	}
}

func (check *Checker) recordScope(node syntax.Node, scope *Scope) {
	assert(node != nil)
	assert(scope != nil)
	if m := check.Scopes; m != nil {
		m[node] = scope
	}
}
