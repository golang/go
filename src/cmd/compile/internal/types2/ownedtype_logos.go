// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

type LogosOwnedStepKind uint8

const (
	LogosOwnedUnderlying LogosOwnedStepKind = iota + 1
	LogosOwnedMapKey
	LogosOwnedMapElem
	LogosOwnedSliceElem
	LogosOwnedArrayElem
	LogosOwnedPointerElem
	LogosOwnedStructField
	LogosOwnedSigResult
	LogosOwnedSigParam
)

type LogosOwnedStep struct {
	Kind  LogosOwnedStepKind
	Index int
}

type LogosOwnedType struct {
	FactoryPath string
	FactoryName string
	Witness     syntax.Expr
	Path        []LogosOwnedStep
}

func LogosOwnedFactory(pkgPath, object string, path ...LogosOwnedStep) LogosOwnedType {
	return LogosOwnedType{FactoryPath: pkgPath, FactoryName: object, Path: path}
}

func LogosOwnedWitness(witness syntax.Expr, path ...LogosOwnedStep) LogosOwnedType {
	return LogosOwnedType{Witness: witness, Path: path}
}

func (check *Checker) logosOwnedRegistered(e syntax.Expr) bool {
	if check == nil || check.conf == nil || e == nil {
		return false
	}
	_, ok := check.conf.LogosOwnedTypes[e]
	return ok
}

func (d LogosOwnedType) logosOwnedIsZero() bool {
	return d.FactoryPath == "" && d.FactoryName == "" && d.Witness == nil && len(d.Path) == 0
}

func (check *Checker) logosOwnedTypeOf(e syntax.Expr) (Type, bool) {
	if check == nil || check.conf == nil || e == nil {
		return nil, false
	}
	d, ok := check.conf.LogosOwnedTypes[e]
	if !ok {
		return nil, false
	}
	return check.logosOwnedResolve(e, d), true
}

func (check *Checker) logosOwnedResolve(e syntax.Expr, d LogosOwnedType) Type {
	root, ok := check.logosOwnedRoot(d, e)
	if !ok {
		return Typ[Invalid]
	}
	t, ok := check.logosOwnedWalk(root, d.Path, e)
	if !ok {
		return Typ[Invalid]
	}
	return t
}

func (check *Checker) logosOwnedRoot(d LogosOwnedType, at poser) (Type, bool) {
	factory := d.FactoryPath != "" || d.FactoryName != ""
	witness := d.Witness != nil
	if factory == witness {
		check.error(at, BadDecl, "logos: owned type needs exactly one of factory or witness")
		return nil, false
	}
	if factory {
		if d.FactoryPath == "" || d.FactoryName == "" {
			check.error(at, BadDecl, "logos: owned factory needs both package and object")
			return nil, false
		}
		if !isExported(d.FactoryName) {
			check.errorf(at, UnexportedName, "logos: owned factory object %s not exported by package %s", d.FactoryName, d.FactoryPath)
			return nil, false
		}
		if len(d.Path) == 0 {
			check.error(at, BadDecl, "logos: owned factory needs a structural path from the function result")
			return nil, false
		}
		pkg, ok := check.logosOwnedImport(d.FactoryPath, at)
		if !ok {
			return nil, false
		}
		return check.logosOwnedFuncRoot(pkg, d, at)
	}
	if check.Info == nil {
		check.error(at, BadDecl, "logos: owned witness needs type information")
		return nil, false
	}
	if t := check.Info.TypeOf(d.Witness); t != nil && isValid(t) {
		return t, true
	}
	var x operand
	check.genericExpr(&x, d.Witness, nil)
	if x.mode() == invalid || !isValid(x.typ()) {
		return nil, false
	}
	return x.typ(), true
}

func (check *Checker) logosOwnedFuncRoot(pkg *Package, d LogosOwnedType, at poser) (Type, bool) {
	obj := pkg.Scope().Lookup(d.FactoryName)
	if obj == nil {
		check.errorf(at, UndeclaredImportedName, "logos: %s has no such object %s", d.FactoryPath, d.FactoryName)
		return nil, false
	}
	if !obj.Exported() {
		check.errorf(at, UnexportedName, "logos: owned factory object %s not exported by package %s", d.FactoryName, d.FactoryPath)
		return nil, false
	}
	fn, ok := obj.(*Func)
	if !ok {
		check.errorf(at, BadDecl, "logos: owned factory object %s.%s is not a function", d.FactoryPath, d.FactoryName)
		return nil, false
	}
	if typ := fn.Type(); isValid(typ) {
		return typ, true
	}
	check.errorf(at, BadDecl, "logos: owned factory function %s.%s has no signature", d.FactoryPath, d.FactoryName)
	return nil, false
}

func (check *Checker) logosOwnedImport(path string, at poser) (*Package, bool) {
	if check.pkg != nil && path == check.pkg.Path() {
		return check.pkg, true
	}
	for _, pn := range check.imports {
		if imp := pn.Imported(); imp != nil && imp.Path() == path {
			return imp, true
		}
	}
	for _, imp := range check.impMap {
		if imp != nil && imp.Path() == path {
			return imp, true
		}
	}
	imp := check.conf.Importer
	if imp == nil {
		check.errorf(at, BrokenImport, "logos: cannot import %s for owned type: no importer installed", path)
		return nil, false
	}
	pkg, err := imp.Import(path)
	if err != nil || pkg == nil {
		check.errorf(at, BrokenImport, "logos: cannot import %s for owned type: %v", path, err)
		return nil, false
	}
	return pkg, true
}

func (check *Checker) logosOwnedWalk(root Type, path []LogosOwnedStep, at poser) (Type, bool) {
	if root == nil || !isValid(root) {
		check.error(at, BadDecl, "logos: owned path has no valid root")
		return nil, false
	}
	t := root
	for i, s := range path {
		switch s.Kind {
		case LogosOwnedUnderlying:
			switch u := t.(type) {
			case *Named:
				t = u.Underlying()
			case *Alias:
				t = unalias(u)
			default:
				check.errorf(at, MismatchedTypes, "logos: owned path step %d needs a named type for underlying, have %s", i, t)
				return nil, false
			}
		case LogosOwnedMapKey:
			m, ok := t.(*Map)
			if !ok {
				check.errorf(at, MismatchedTypes, "logos: owned path step %d needs a map for key, have %s", i, t)
				return nil, false
			}
			t = m.Key()
		case LogosOwnedMapElem:
			m, ok := t.(*Map)
			if !ok {
				check.errorf(at, MismatchedTypes, "logos: owned path step %d needs a map for element, have %s", i, t)
				return nil, false
			}
			t = m.Elem()
		case LogosOwnedSliceElem:
			s, ok := t.(*Slice)
			if !ok {
				check.errorf(at, MismatchedTypes, "logos: owned path step %d needs a slice for element, have %s", i, t)
				return nil, false
			}
			t = s.Elem()
		case LogosOwnedArrayElem:
			a, ok := t.(*Array)
			if !ok {
				check.errorf(at, MismatchedTypes, "logos: owned path step %d needs an array for element, have %s", i, t)
				return nil, false
			}
			t = a.Elem()
		case LogosOwnedPointerElem:
			p, ok := t.(*Pointer)
			if !ok {
				check.errorf(at, MismatchedTypes, "logos: owned path step %d needs a pointer for element, have %s", i, t)
				return nil, false
			}
			t = p.Elem()
		case LogosOwnedStructField:
			st, ok := t.(*Struct)
			if !ok {
				check.errorf(at, MismatchedTypes, "logos: owned path step %d needs a struct for field, have %s", i, t)
				return nil, false
			}
			if s.Index < 0 || s.Index >= st.NumFields() {
				check.errorf(at, InvalidIndex, "logos: owned path step %d field index %d out of range (%d fields)", i, s.Index, st.NumFields())
				return nil, false
			}
			t = st.Field(s.Index).Type()
		case LogosOwnedSigResult:
			sig, ok := t.(*Signature)
			if !ok {
				check.errorf(at, MismatchedTypes, "logos: owned path step %d needs a signature for result, have %s", i, t)
				return nil, false
			}
			res := sig.Results()
			if res == nil || s.Index < 0 || s.Index >= res.Len() {
				n := 0
				if res != nil {
					n = res.Len()
				}
				check.errorf(at, InvalidIndex, "logos: owned path step %d result index %d out of range (%d results)", i, s.Index, n)
				return nil, false
			}
			t = res.At(s.Index).Type()
		case LogosOwnedSigParam:
			sig, ok := t.(*Signature)
			if !ok {
				check.errorf(at, MismatchedTypes, "logos: owned path step %d needs a signature for parameter, have %s", i, t)
				return nil, false
			}
			params := sig.Params()
			if params == nil || s.Index < 0 || s.Index >= params.Len() {
				n := 0
				if params != nil {
					n = params.Len()
				}
				check.errorf(at, InvalidIndex, "logos: owned path step %d parameter index %d out of range (%d parameters)", i, s.Index, n)
				return nil, false
			}
			t = params.At(s.Index).Type()
		default:
			check.errorf(at, BadDecl, "logos: owned path step %d has unknown kind %d", i, uint8(s.Kind))
			return nil, false
		}
		if t == nil || !isValid(t) {
			check.errorf(at, MismatchedTypes, "logos: owned path step %d resolved to an invalid type", i)
			return nil, false
		}
	}
	return t, true
}

func (check *Checker) logosOwnedSelector(x *operand, e *syntax.SelectorExpr) bool {
	if !check.logosOwnedRegistered(e) {
		return false
	}
	sel := e.Sel.Value
	if sel == "_" {
		return false
	}
	fail := func(code Code, format string, args ...any) {
		check.errorf(e.Sel, code, format, args...)
		x.invalidate()
		x.typ_ = Typ[Invalid]
		x.expr = e
	}
	if d := check.conf.LogosOwnedTypes[e]; !d.logosOwnedIsZero() {
		fail(BadDecl, "logos: owned selector %s must carry the zero descriptor", sel)
		return true
	}
	base := x.typ()
	if !isValid(base) {
		fail(MissingFieldOrMethod, "logos: owned selector %s has no valid base type", sel)
		return true
	}
	typ, isPtr := deref(base)
	st, ok := typ.Underlying().(*Struct)
	if !ok {
		fail(MissingFieldOrMethod, "logos: owned selector %s needs a struct base, have %s", sel, base)
		return true
	}
	var fld *Var
	index := -1
	for i := range st.NumFields() {
		if f := st.Field(i); f.Name() == sel {
			fld, index = f, i
			break
		}
	}
	if fld == nil {
		fail(MissingFieldOrMethod, "logos: %s has no owned field %s", base, sel)
		return true
	}
	check.recordSelection(e, FieldVal, x.typ(), fld, []int{index}, isPtr)
	if x.mode() == variable || isPtr {
		x.mode_ = variable
	} else {
		x.mode_ = value
	}
	x.typ_ = fld.Type()
	x.expr = e
	return true
}

func (check *Checker) logosOwnedLitSkip(lit *syntax.CompositeLit, utyp *Struct, name string, obj *Object, index *[]int, indirect *bool) bool {
	fld, idx, ok := check.logosOwnedLitField(lit, utyp, name, *obj)
	if !ok {
		return false
	}
	if fld == nil {
		return true
	}
	*obj, *index, *indirect = fld, idx, false
	return false
}

func (check *Checker) logosOwnedLitField(lit *syntax.CompositeLit, utyp *Struct, name string, found Object) (*Var, []int, bool) {
	if name == "_" {
		return nil, nil, false
	}
	if check == nil || check.conf == nil {
		return nil, nil, false
	}
	d, ok := check.conf.LogosOwnedTypes[lit]
	if !ok {
		return nil, nil, false
	}
	if !d.logosOwnedIsZero() {
		check.error(lit, BadDecl, "logos: owned literal must carry the zero descriptor")
		return nil, nil, true
	}
	if found != nil || utyp == nil {
		return nil, nil, false
	}
	for i := range utyp.NumFields() {
		if f := utyp.Field(i); f.Name() == name {
			return f, []int{i}, true
		}
	}
	return nil, nil, false
}

func (check *Checker) logosOwnedPosLitField(lit *syntax.CompositeLit) bool {
	if check == nil || check.conf == nil || lit == nil {
		return false
	}
	d, ok := check.conf.LogosOwnedTypes[lit]
	if !ok {
		return false
	}
	if !d.logosOwnedIsZero() {
		check.error(lit, BadDecl, "logos: owned literal must carry the zero descriptor")
	}
	return true
}
