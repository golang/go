// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"sync"
)

// LoGo/s M7-S2/M7-S3 virtual `comptime` package (SPEC Sec 2.5, T-205).
// The package has no source: types2 provides it in Logos mode, the
// cmd/go loader skips it like "C", and eject drops the import. Values
// of these types exist only at compile time; uses outside their
// compile-time positions report [comptime::runtime_use].
//
// API:
//	type StructTag string
//	func (t StructTag) Get(k string) string
//	type Type struct{}
//	type Field struct{ Name string; Index int; Type Type; Tag StructTag; Exported bool }
//	func Fields[T any]() []Field
//	func IsStruct[T any]() bool
//	func TypeOf[X any]() Type
//	func Error(msg string)
// ViewAs, declared by M7-S5 (viewas_logos.go), is part of the package
// whichever provider creates it.

var (
	logosComptimeOnce sync.Once
	logosComptimePkg  *Package
	logosComptimeAny  Type
)

func logosComptimePackage() *Package {
	logosComptimeOnce.Do(func() {
		pkg := NewPackage("comptime", "comptime")
		logosComptimeAny = universeAny.Type()

		tagName := NewTypeName(nopos, pkg, "StructTag", nil)
		tagNamed := NewNamed(tagName, Typ[String], nil)
		tagNamed.AddMethod(NewFunc(nopos, pkg, "Get", NewSignatureType(
			NewVar(nopos, pkg, "", tagNamed),
			nil, nil,
			NewTuple(NewVar(nopos, pkg, "k", Typ[String])),
			NewTuple(NewVar(nopos, pkg, "", Typ[String])),
			false,
		)))
		pkg.Scope().Insert(tagName)

		typeName := NewTypeName(nopos, pkg, "Type", nil)
		typeNamed := NewNamed(typeName, NewStruct(nil, nil), nil)
		pkg.Scope().Insert(typeName)

		fieldName := NewTypeName(nopos, pkg, "Field", nil)
		fieldNamed := NewNamed(fieldName, NewStruct([]*Var{
			NewField(nopos, pkg, "Name", Typ[String], false),
			NewField(nopos, pkg, "Index", Typ[Int], false),
			NewField(nopos, pkg, "Type", typeNamed, false),
			NewField(nopos, pkg, "Tag", tagNamed, false),
			NewField(nopos, pkg, "Exported", Typ[Bool], false),
		}, nil), nil)
		pkg.Scope().Insert(fieldName)

		addGeneric := func(name string, tparam string, params, results *Tuple) {
			tname := NewTypeName(nopos, nil, tparam, nil)
			tp := NewTypeParam(tname, logosComptimeAny)
			fn := NewFunc(nopos, pkg, name, NewSignatureType(nil, nil, []*TypeParam{tp}, params, results, false))
			pkg.Scope().Insert(fn)
		}
		fieldSlice := NewSlice(fieldNamed)
		addGeneric("Fields", "T", NewTuple(), NewTuple(NewVar(nopos, pkg, "", fieldSlice)))
		addGeneric("IsStruct", "T", NewTuple(), NewTuple(NewVar(nopos, pkg, "", Typ[Bool])))
		addGeneric("TypeOf", "X", NewTuple(), NewTuple(NewVar(nopos, pkg, "", typeNamed)))
		pkg.Scope().Insert(NewFunc(nopos, pkg, "Error", NewSignatureType(
			nil, nil, nil,
			NewTuple(NewVar(nopos, pkg, "msg", Typ[String])),
			NewTuple(),
			false,
		)))

		logosEnsureComptimeViewAs(pkg)
		pkg.MarkComplete()
		logosComptimePkg = pkg
	})
	return logosComptimePkg
}

// logosComptimeEnsure inserts this slice's declarations into imp when a
// sibling hook created the shell first. It reports whether all of them
// are present afterwards.
func logosComptimeEnsure(imp *Package) {
	for _, name := range []string{"StructTag", "Type", "Field", "Fields", "IsStruct", "TypeOf", "Error"} {
		if imp.Scope().Lookup(name) == nil {
			full := logosComptimePackage()
			if obj := full.Scope().Lookup(name); obj != nil {
				imp.Scope().Insert(obj)
			}
		}
	}
	logosEnsureComptimeViewAs(imp)
}

// logosComptimeImport serves the virtual `comptime` package in Logos
// mode, storing it through pimp. It mirrors the importer's finalization
// (complete package cached in impMap), so the real importer never sees
// the path.
func (check *Checker) logosComptimeImport(pimp **Package, path, dir string) {
	if path != "comptime" || !check.conf.Logos {
		return
	}
	key := importKey{path, dir}
	if imp := check.impMap[key]; imp != nil {
		logosComptimeEnsure(imp)
		*pimp = imp
		return
	}
	imp := logosComptimePackage()
	logosComptimeEnsure(imp)
	check.impMap[key] = imp
	if check.pkgPathMap != nil {
		check.markImports(imp)
	}
	*pimp = imp
}

// logosComptimeFunc reports whether obj is one of the virtual comptime
// package's functions.
func logosComptimeFunc(obj Object) (string, bool) {
	fn, ok := obj.(*Func)
	if !ok || fn.Pkg() != logosComptimePkg {
		return "", false
	}
	return fn.Name(), true
}

// logosComptimeQualifier resolves a package-qualified comptime call's
// function: Fun unwrapped through instantiation syntax must select from
// an imported name bound to the virtual package.
func (check *Checker) logosComptimeQualifier(e syntax.Expr) (*syntax.SelectorExpr, string, bool) {
	call, ok := e.(*syntax.CallExpr)
	if !ok {
		return nil, "", false
	}
	fun := call.Fun
	if idx, ok := fun.(*syntax.IndexExpr); ok {
		fun = idx.X
	}
	sel, ok := fun.(*syntax.SelectorExpr)
	if !ok {
		return nil, "", false
	}
	name, ok := sel.X.(*syntax.Name)
	if !ok {
		return nil, "", false
	}
	_, obj := check.lookupScope(name.Value)
	pn, ok := obj.(*PkgName)
	if !ok || pn.Imported() != logosComptimePkg {
		return nil, "", false
	}
	return sel, sel.Sel.Value, true
}
