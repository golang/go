// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package types2 declares the data types and implements
// the algorithms for type-checking of Go packages. Use
// Config.Check to invoke the type checker for a package.
// Alternatively, create a new type checker with NewChecker
// and invoke it incrementally by calling Checker.Files.
//
// Type-checking consists of several interdependent phases:
//
// Name resolution maps each identifier (syntax.Name) in the program to the
// language object (Object) it denotes.
// Use Info.{Defs,Uses,Implicits} for the results of name resolution.
//
// Constant folding computes the exact constant value (constant.Value)
// for every expression (syntax.Expr) that is a compile-time constant.
// Use Info.Types[expr].Value for the results of constant folding.
//
// Type inference computes the type (Type) of every expression (syntax.Expr)
// and checks for compliance with the language specification.
// Use Info.Types[expr].Type for the results of type inference.
package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"go/constant"
	. "internal/types/errors"
	"strings"
)

// An Error describes a type-checking error; it implements the error interface.
// A "soft" error is an error that still permits a valid interpretation of a
// package (such as "unused variable"); "hard" errors may lead to unpredictable
// behavior if ignored.
type Error struct {
	Pos  syntax.Pos // error position
	Msg  string     // default error message, user-friendly
	Full string     // full error message, for debugging (may contain internal details)
	Soft bool       // if set, error is "soft"
	Code Code       // error code
}

// Error returns an error string formatted as follows:
// filename:line:column: message
func (err Error) Error() string {
	return fmt.Sprintf("%s: %s", err.Pos, err.Msg)
}

// FullError returns an error string like Error, buy it may contain
// type-checker internal details such as subscript indices for type
// parameters and more. Useful for debugging.
func (err Error) FullError() string {
	return fmt.Sprintf("%s: %s", err.Pos, err.Full)
}

// An ArgumentError holds an error associated with an argument index.
type ArgumentError struct {
	Index int
	Err   error
}

func (e *ArgumentError) Error() string { return e.Err.Error() }
func (e *ArgumentError) Unwrap() error { return e.Err }

// An Importer resolves import paths to Packages.
//
// CAUTION: This interface does not support the import of locally
// vendored packages. See https://golang.org/s/go15vendor.
// If possible, external implementations should implement ImporterFrom.
type Importer interface {
	// Import returns the imported package for the given import path.
	// The semantics is like for ImporterFrom.ImportFrom except that
	// dir and mode are ignored (since they are not present).
	Import(path string) (*Package, error)
}

// ImportMode is reserved for future use.
type ImportMode int

// An ImporterFrom resolves import paths to packages; it
// supports vendoring per https://golang.org/s/go15vendor.
// Use go/importer to obtain an ImporterFrom implementation.
type ImporterFrom interface {
	// Importer is present for backward-compatibility. Calling
	// Import(path) is the same as calling ImportFrom(path, "", 0);
	// i.e., locally vendored packages may not be found.
	// The types package does not call Import if an ImporterFrom
	// is present.
	Importer

	// ImportFrom returns the imported package for the given import
	// path when imported by a package file located in dir.
	// If the import failed, besides returning an error, ImportFrom
	// is encouraged to cache and return a package anyway, if one
	// was created. This will reduce package inconsistencies and
	// follow-on type checker errors due to the missing package.
	// The mode value must be 0; it is reserved for future use.
	// Two calls to ImportFrom with the same path and dir must
	// return the same package.
	ImportFrom(path, dir string, mode ImportMode) (*Package, error)
}

// A Config specifies the configuration for type checking.
// The zero value for Config is a ready-to-use default configuration.
type Config struct {
	// Context is the context used for resolving global identifiers. If nil, the
	// type checker will initialize this field with a newly created context.
	Context *Context

	// GoVersion describes the accepted Go language version. The string must
	// start with a prefix of the form "go%d.%d" (e.g. "go1.20", "go1.21rc1", or
	// "go1.21.0") or it must be empty; an empty string disables Go language
	// version checks. If the format is invalid, invoking the type checker will
	// result in an error.
	GoVersion string

	// If IgnoreFuncBodies is set, function bodies are not
	// type-checked.
	IgnoreFuncBodies bool

	// If FakeImportC is set, `import "C"` (for packages requiring Cgo)
	// declares an empty "C" package and errors are omitted for qualified
	// identifiers referring to package C (which won't find an object).
	// This feature is intended for the standard library cmd/api tool.
	//
	// Caution: Effects may be unpredictable due to follow-on errors.
	//          Do not use casually!
	FakeImportC bool

	// If IgnoreBranchErrors is set, branch/label errors are ignored.
	IgnoreBranchErrors bool

	// If go115UsesCgo is set, the type checker expects the
	// _cgo_gotypes.go file generated by running cmd/cgo to be
	// provided as a package source file. Qualified identifiers
	// referring to package C will be resolved to cgo-provided
	// declarations within _cgo_gotypes.go.
	//
	// It is an error to set both FakeImportC and go115UsesCgo.
	go115UsesCgo bool

	// If Trace is set, a debug trace is printed to stdout.
	Trace bool

	// If Error != nil, it is called with each error found
	// during type checking; err has dynamic type Error.
	// Secondary errors (for instance, to enumerate all types
	// involved in an invalid recursive type declaration) have
	// error strings that start with a '\t' character.
	// If Error == nil, type-checking stops with the first
	// error found.
	Error func(err error)

	// An importer is used to import packages referred to from
	// import declarations.
	// If the installed importer implements ImporterFrom, the type
	// checker calls ImportFrom instead of Import.
	// The type checker reports an error if an importer is needed
	// but none was installed.
	Importer Importer

	// If Sizes != nil, it provides the sizing functions for package unsafe.
	// Otherwise SizesFor("gc", "amd64") is used instead.
	Sizes Sizes

	// If DisableUnusedImportCheck is set, packages are not checked
	// for unused imports.
	DisableUnusedImportCheck bool

	// If a non-empty ErrorURL format string is provided, it is used
	// to format an error URL link that is appended to the first line
	// of an error message. ErrorURL must be a format string containing
	// exactly one "%s" format, e.g. "[go.dev/e/%s]".
	ErrorURL string

	// If EnableAlias is set, alias declarations produce an Alias type. Otherwise
	// the alias information is only in the type name, which points directly to
	// the actual (aliased) type.
	//
	// This setting must not differ among concurrent type-checking operations,
	// since it affects the behavior of Universe.Lookup("any").
	//
	// This flag will eventually be removed (with Go 1.24 at the earliest).
	EnableAlias bool
}

func srcimporter_setUsesCgo(conf *Config) {
	conf.go115UsesCgo = true
}

// Info holds result type information for a type-checked package.
// Only the information for which a map is provided is collected.
// If the package has type errors, the collected information may
// be incomplete.
type Info struct {
	// Types maps expressions to their types, and for constant
	// expressions, also their values. Invalid expressions are
	// omitted.
	//
	// For (possibly parenthesized) identifiers denoting built-in
	// functions, the recorded signatures are call-site specific:
	// if the call result is not a constant, the recorded type is
	// an argument-specific signature. Otherwise, the recorded type
	// is invalid.
	//
	// The Types map does not record the type of every identifier,
	// only those that appear where an arbitrary expression is
	// permitted. For instance:
	// - an identifier f in a selector expression x.f is found
	//   only in the Selections map;
	// - an identifier z in a variable declaration 'var z int'
	//   is found only in the Defs map;
	// - an identifier p denoting a package in a qualified
	//   identifier p.X is found only in the Uses map.
	//
	// Similarly, no type is recorded for the (synthetic) FuncType
	// node in a FuncDecl.Type field, since there is no corresponding
	// syntactic function type expression in the source in this case
	// Instead, the function type is found in the Defs.map entry for
	// the corresponding function declaration.
	Types map[syntax.Expr]TypeAndValue

	// If StoreTypesInSyntax is set, type information identical to
	// that which would be put in the Types map, will be set in
	// syntax.Expr.TypeAndValue (independently of whether Types
	// is nil or not).
	StoreTypesInSyntax bool

	// Instances maps identifiers denoting generic types or functions to their
	// type arguments and instantiated type.
	//
	// For example, Instances will map the identifier for 'T' in the type
	// instantiation T[int, string] to the type arguments [int, string] and
	// resulting instantiated *Named type. Given a generic function
	// func F[A any](A), Instances will map the identifier for 'F' in the call
	// expression F(int(1)) to the inferred type arguments [int], and resulting
	// instantiated *Signature.
	//
	// Invariant: Instantiating Uses[id].Type() with Instances[id].TypeArgs
	// results in an equivalent of Instances[id].Type.
	Instances map[*syntax.Name]Instance

	// Defs maps identifiers to the objects they define (including
	// package names, dots "." of dot-imports, and blank "_" identifiers).
	// For identifiers that do not denote objects (e.g., the package name
	// in package clauses, or symbolic variables t in t := x.(type) of
	// type switch headers), the corresponding objects are nil.
	//
	// For an embedded field, Defs returns the field *Var it defines.
	//
	// Invariant: Defs[id] == nil || Defs[id].Pos() == id.Pos()
	Defs map[*syntax.Name]Object

	// Uses maps identifiers to the objects they denote.
	//
	// For an embedded field, Uses returns the *TypeName it denotes.
	//
	// Invariant: Uses[id].Pos() != id.Pos()
	Uses map[*syntax.Name]Object

	// Implicits maps nodes to their implicitly declared objects, if any.
	// The following node and object types may appear:
	//
	//     node               declared object
	//
	//     *syntax.ImportDecl    *PkgName for imports without renames
	//     *syntax.CaseClause    type-specific *Var for each type switch case clause (incl. default)
	//     *syntax.Field         anonymous parameter *Var (incl. unnamed results)
	//
	Implicits map[syntax.Node]Object

	// Selections maps selector expressions (excluding qualified identifiers)
	// to their corresponding selections.
	Selections map[*syntax.SelectorExpr]*Selection

	// OptionalChainSelections maps optional chain expressions (x?.field)
	// to their corresponding selections. This is used for nil-safe field access.
	OptionalChainSelections map[*syntax.OptionalChainExpr]*Selection

	// Scopes maps syntax.Nodes to the scopes they define. Package scopes are not
	// associated with a specific node but with all files belonging to a package.
	// Thus, the package scope can be found in the type-checked Package object.
	// Scopes nest, with the Universe scope being the outermost scope, enclosing
	// the package scope, which contains (one or more) files scopes, which enclose
	// function scopes which in turn enclose statement and function literal scopes.
	// Note that even though package-level functions are declared in the package
	// scope, the function scopes are embedded in the file scope of the file
	// containing the function declaration.
	//
	// The Scope of a function contains the declarations of any
	// type parameters, parameters, and named results, plus any
	// local declarations in the body block.
	// It is coextensive with the complete extent of the
	// function's syntax ([*ast.FuncDecl] or [*ast.FuncLit]).
	// The Scopes mapping does not contain an entry for the
	// function body ([*ast.BlockStmt]); the function's scope is
	// associated with the [*ast.FuncType].
	//
	// The following node types may appear in Scopes:
	//
	//     *syntax.File
	//     *syntax.FuncType
	//     *syntax.TypeDecl
	//     *syntax.BlockStmt
	//     *syntax.IfStmt
	//     *syntax.SwitchStmt
	//     *syntax.CaseClause
	//     *syntax.CommClause
	//     *syntax.ForStmt
	//
	Scopes map[syntax.Node]*Scope

	// InitOrder is the list of package-level initializers in the order in which
	// they must be executed. Initializers referring to variables related by an
	// initialization dependency appear in topological order, the others appear
	// in source order. Variables without an initialization expression do not
	// appear in this list.
	InitOrder []*Initializer

	// FileVersions maps a file to its Go version string.
	// If the file doesn't specify a version, the reported
	// string is Config.GoVersion.
	// Version strings begin with “go”, like “go1.21”, and
	// are suitable for use with the [go/version] package.
	FileVersions map[*syntax.PosBase]string
}

func (info *Info) recordTypes() bool {
	return info.Types != nil || info.StoreTypesInSyntax
}

// TypeOf returns the type of expression e, or nil if not found.
// Precondition 1: the Types map is populated or StoreTypesInSyntax is set.
// Precondition 2: Uses and Defs maps are populated.
func (info *Info) TypeOf(e syntax.Expr) Type {
	if info.Types != nil {
		if t, ok := info.Types[e]; ok {
			return t.Type
		}
	} else if info.StoreTypesInSyntax {
		if tv := e.GetTypeInfo(); tv.Type != nil {
			return tv.Type
		}
	}

	if id, _ := e.(*syntax.Name); id != nil {
		if obj := info.ObjectOf(id); obj != nil {
			return obj.Type()
		}
	}
	return nil
}

// ObjectOf returns the object denoted by the specified id,
// or nil if not found.
//
// If id is an embedded struct field, ObjectOf returns the field (*Var)
// it defines, not the type (*TypeName) it uses.
//
// Precondition: the Uses and Defs maps are populated.
func (info *Info) ObjectOf(id *syntax.Name) Object {
	if obj := info.Defs[id]; obj != nil {
		return obj
	}
	return info.Uses[id]
}

// PkgNameOf returns the local package name defined by the import,
// or nil if not found.
//
// For dot-imports, the package name is ".".
//
// Precondition: the Defs and Implicts maps are populated.
func (info *Info) PkgNameOf(imp *syntax.ImportDecl) *PkgName {
	var obj Object
	if imp.LocalPkgName != nil {
		obj = info.Defs[imp.LocalPkgName]
	} else {
		obj = info.Implicits[imp]
	}
	pkgname, _ := obj.(*PkgName)
	return pkgname
}

// TypeAndValue reports the type and value (for constants)
// of the corresponding expression.
type TypeAndValue struct {
	mode  operandMode
	Type  Type
	Value constant.Value
}

// IsVoid reports whether the corresponding expression
// is a function call without results.
func (tv TypeAndValue) IsVoid() bool {
	return tv.mode == novalue
}

// IsType reports whether the corresponding expression specifies a type.
func (tv TypeAndValue) IsType() bool {
	return tv.mode == typexpr
}

// IsBuiltin reports whether the corresponding expression denotes
// a (possibly parenthesized) built-in function.
func (tv TypeAndValue) IsBuiltin() bool {
	return tv.mode == builtin
}

// IsValue reports whether the corresponding expression is a value.
// Builtins are not considered values. Constant values have a non-
// nil Value.
func (tv TypeAndValue) IsValue() bool {
	switch tv.mode {
	case constant_, variable, mapindex, value, nilvalue, commaok, commaerr:
		return true
	}
	return false
}

// IsNil reports whether the corresponding expression denotes the
// predeclared value nil. Depending on context, it may have been
// given a type different from UntypedNil.
func (tv TypeAndValue) IsNil() bool {
	return tv.mode == nilvalue
}

// Addressable reports whether the corresponding expression
// is addressable (https://golang.org/ref/spec#Address_operators).
func (tv TypeAndValue) Addressable() bool {
	return tv.mode == variable
}

// Assignable reports whether the corresponding expression
// is assignable to (provided a value of the right type).
func (tv TypeAndValue) Assignable() bool {
	return tv.mode == variable || tv.mode == mapindex
}

// HasOk reports whether the corresponding expression may be
// used on the rhs of a comma-ok assignment.
func (tv TypeAndValue) HasOk() bool {
	return tv.mode == commaok || tv.mode == mapindex
}

// Instance reports the type arguments and instantiated type for type and
// function instantiations. For type instantiations, Type will be of dynamic
// type *Named. For function instantiations, Type will be of dynamic type
// *Signature.
type Instance struct {
	TypeArgs *TypeList
	Type     Type
}

// An Initializer describes a package-level variable, or a list of variables in case
// of a multi-valued initialization expression, and the corresponding initialization
// expression.
type Initializer struct {
	Lhs []*Var // var Lhs = Rhs
	Rhs syntax.Expr
}

func (init *Initializer) String() string {
	var buf strings.Builder
	for i, lhs := range init.Lhs {
		if i > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(lhs.Name())
	}
	buf.WriteString(" = ")
	syntax.Fprint(&buf, init.Rhs, syntax.ShortForm)
	return buf.String()
}

// Check type-checks a package and returns the resulting package object and
// the first error if any. Additionally, if info != nil, Check populates each
// of the non-nil maps in the Info struct.
//
// The package is marked as complete if no errors occurred, otherwise it is
// incomplete. See Config.Error for controlling behavior in the presence of
// errors.
//
// The package is specified by a list of *syntax.Files and corresponding
// file set, and the package path the package is identified with.
// The clean path must not be empty or dot (".").
func (conf *Config) Check(path string, files []*syntax.File, info *Info) (*Package, error) {
	pkg := NewPackage(path, "")
	return pkg, NewChecker(conf, pkg, info).Files(files)
}
