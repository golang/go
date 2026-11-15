// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"sync"
)

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
		addGeneric("SizeOf", "T", NewTuple(), NewTuple(NewVar(nopos, pkg, "", Typ[Int])))
		addGeneric("AlignOf", "T", NewTuple(), NewTuple(NewVar(nopos, pkg, "", Typ[Int])))
		addGeneric("HasPointers", "T", NewTuple(), NewTuple(NewVar(nopos, pkg, "", Typ[Bool])))
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

func logosComptimeEnsure(imp *Package) {
	for _, name := range []string{"StructTag", "Type", "Field", "Fields", "IsStruct", "SizeOf", "AlignOf", "HasPointers", "TypeOf", "Error"} {
		if imp.Scope().Lookup(name) == nil {
			full := logosComptimePackage()
			if obj := full.Scope().Lookup(name); obj != nil {
				imp.Scope().Insert(obj)
			}
		}
	}
	logosEnsureComptimeViewAs(imp)
}

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

func logosComptimeFunc(obj Object) (string, bool) {
	fn, ok := obj.(*Func)
	if !ok || fn.Pkg() != logosComptimePkg {
		return "", false
	}
	return fn.Name(), true
}

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
