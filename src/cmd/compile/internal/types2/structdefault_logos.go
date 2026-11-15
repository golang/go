// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"slices"
	"strings"
	"sync"
)

const (
	logosDefaultNotConstant = "[correctness::default_not_constant]"
	logosUnappliedDefault   = "[suspicious::unapplied_struct_default]"
)

type LogosFieldDefault struct {
	Field   string
	Default syntax.Expr
}

func (check *Checker) logosStructFieldType(e syntax.Expr) Type {
	if fd, ok := e.(*syntax.FieldDefault); ok && check.conf.Logos {
		return check.varType(fd.Type)
	}
	return check.varType(e)
}

func (check *Checker) logosStructDefaults(e *syntax.StructType, styp *Struct) {
	if !check.conf.Logos || check.Info == nil {
		return
	}
	seen := make(map[*syntax.FieldDefault][]string)
	var order []*syntax.FieldDefault
	for _, f := range e.FieldList {
		fd, ok := f.Type.(*syntax.FieldDefault)
		if !ok || f.Name == nil {
			continue
		}
		if _, dup := seen[fd]; !dup {
			order = append(order, fd)
		}
		seen[fd] = append(seen[fd], f.Name.Value)
	}
	for _, fd := range order {
		check.logosRecordFieldDefault(styp, fd, seen[fd])
	}
}

func (check *Checker) logosRecordFieldDefault(styp *Struct, fd *syntax.FieldDefault, names []string) {
	if check.Info.LogosFieldDefaults == nil {
		check.Info.LogosFieldDefaults = make(map[*Struct][]LogosFieldDefault)
	}
	cur := check.Info.LogosFieldDefaults[styp]
	known := make(map[string]bool, len(cur))
	for _, d := range cur {
		known[d.Field] = true
	}
	for _, name := range names {
		if !known[name] {
			cur = append(cur, LogosFieldDefault{Field: name, Default: fd.Default})
			known[name] = true
		}
	}
	check.Info.LogosFieldDefaults[styp] = cur
	check.later(func() {
		outer := check.scope
		check.scope = check.pkg.Scope()
		defer func() { check.scope = outer }()
		check.logosCheckFieldDefault(fd, names[0])
	}).describef(fd.Default, "check field default %s", names[0])
}

func (check *Checker) logosCheckFieldDefault(fd *syntax.FieldDefault, name string) {
	var x operand
	check.expr(nil, &x, fd.Default)
	if !x.isValid() {
		return
	}
	if x.mode() == constant_ {
		return
	}
	if lit, ok := syntax.Unparen(fd.Default).(*syntax.CompositeLit); ok && check.logosCompositeConst(lit) {
		return
	}
	check.errorf(fd.Default, InvalidConstInit, "%s non-constant default for field %s", logosDefaultNotConstant, name)
}

func (check *Checker) logosCompositeConst(lit *syntax.CompositeLit) bool {
	for _, e := range lit.ElemList {
		if kv, ok := syntax.Unparen(e).(*syntax.KeyValueExpr); ok {
			e = kv.Value
		}
		if inner, ok := syntax.Unparen(e).(*syntax.CompositeLit); ok {
			if !check.logosCompositeConst(inner) {
				return false
			}
			continue
		}
		var x operand
		check.expr(nil, &x, e)
		if !x.isValid() || x.mode() != constant_ {
			return false
		}
	}
	return true
}

func (check *Checker) logosWarnZeroDefault(obj *Var, typexpr syntax.Expr) {
	if !check.conf.Logos || obj.name == "_" || !isValid(obj.typ) {
		return
	}
	typ := obj.typ
	name := obj.name
	check.later(func() {
		typeName, ok := logosDefaultedStructName(check, typ)
		if !ok {
			return
		}
		check.softErrorf(typexpr, BadDecl, "%s zero declaration of %s with field defaults", logosUnappliedDefault, typeName)
	}).describef(typexpr, "check zero declaration of %s", name)
}

func (check *Checker) logosWarnNewDefault(x *operand, call *syntax.CallExpr, id builtinId) {
	if !check.conf.Logos || id != _New || !x.isValid() {
		return
	}
	ptr, ok := Unalias(x.typ()).(*Pointer)
	if !ok {
		return
	}
	base := ptr.base
	pos := syntax.StartPos(call.Fun)
	check.later(func() {
		typeName, ok := logosDefaultedStructName(check, base)
		if !ok {
			return
		}
		check.softErrorf(pos, BadDecl, "%s zero declaration of %s with field defaults", logosUnappliedDefault, typeName)
	}).describef(pos, "check new of defaulted struct")
}

func logosDefaultedStructName(check *Checker, t Type) (string, bool) {
	u := Unalias(t)
	var st *Struct
	if n, ok := u.(*Named); ok {
		if s, ok := n.Underlying().(*Struct); ok {
			st = s
		}
	} else {
		st, _ = u.(*Struct)
	}
	if st != nil && check.Info != nil && len(check.Info.LogosFieldDefaults[st]) > 0 {
		if n, ok := u.(*Named); ok {
			return n.Obj().Name(), true
		}
		return st.String(), true
	}
	if n, ok := u.(*Named); ok {
		if obj := n.Obj(); obj != nil && obj.Pkg() != nil && obj.Pkg() != check.pkg {
			if len(logosLookupImportedDefaults(obj.Pkg().Path(), obj.Name())) > 0 {
				return obj.Name(), true
			}
		}
	}
	return "", false
}

func (check *Checker) logosWarnOnly(e Error) bool {
	return check.conf.Logos && LogosWarnOnly(e)
}

func LogosWarnOnly(err error) bool {
	e, ok := err.(Error)
	return ok && e.Soft && (strings.HasPrefix(e.Msg, logosUnappliedDefault) ||
		strings.HasPrefix(e.Msg, logosSideEffectBeforeSort))
}

type LogosImportedDefault struct {
	Field string
	Text  string
}

var logosImportedDefaults = struct {
	sync.RWMutex
	m map[string]map[string][]LogosImportedDefault
}{m: make(map[string]map[string][]LogosImportedDefault)}

func LogosNoteImportedDefaults(pkgPath, typeName string, defs []LogosImportedDefault) {
	logosImportedDefaults.Lock()
	defer logosImportedDefaults.Unlock()
	byName := logosImportedDefaults.m[pkgPath]
	if byName == nil {
		byName = make(map[string][]LogosImportedDefault)
		logosImportedDefaults.m[pkgPath] = byName
	}
	byName[typeName] = slices.Clone(defs)
}

func logosLookupImportedDefaults(pkgPath, typeName string) []LogosImportedDefault {
	logosImportedDefaults.RLock()
	defer logosImportedDefaults.RUnlock()
	return logosImportedDefaults.m[pkgPath][typeName]
}
