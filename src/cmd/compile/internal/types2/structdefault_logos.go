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
	seen := make(map[*syntax.FieldDefault][]*syntax.Name)
	var order []*syntax.FieldDefault
	for _, f := range e.FieldList {
		fd, ok := f.Type.(*syntax.FieldDefault)
		if !ok || f.Name == nil {
			continue
		}
		if _, dup := seen[fd]; !dup {
			order = append(order, fd)
		}
		seen[fd] = append(seen[fd], f.Name)
	}
	for _, fd := range order {
		check.logosRecordFieldDefault(styp, fd, seen[fd])
	}
}

func (check *Checker) logosRecordFieldDefault(styp *Struct, fd *syntax.FieldDefault, names []*syntax.Name) {
	if check.Info.LogosFieldDefaults == nil {
		check.Info.LogosFieldDefaults = make(map[*Struct][]LogosFieldDefault)
	}
	cur := check.Info.LogosFieldDefaults[styp]
	known := make(map[string]bool, len(cur))
	for _, d := range cur {
		known[d.Field] = true
	}
	for _, name := range names {
		if !known[name.Value] {
			cur = append(cur, LogosFieldDefault{Field: name.Value, Default: fd.Default})
			known[name.Value] = true
		}
	}
	check.Info.LogosFieldDefaults[styp] = cur
	var typ Type
	for _, field := range styp.fields {
		if field.Pos() == names[0].Pos() {
			typ = field.typ
			break
		}
	}
	if typ == nil {
		return // a duplicate non-blank field already has a diagnostic
	}
	scope := check.scope
	check.later(func() {
		outer := check.scope
		check.scope = scope
		defer func() { check.scope = outer }()
		check.logosCheckFieldDefault(fd, names[0].Value, typ)
	}).describef(fd.Default, "check field default %s", names[0].Value)
}

func (check *Checker) logosCheckFieldDefault(fd *syntax.FieldDefault, name string, typ Type) {
	// Types is optional for callers of Config.Check. The constant-composite
	// proof needs the checked type of each literal to distinguish map keys
	// from struct field names, even when the caller did not request it.
	if check.Types == nil {
		check.Types = make(map[syntax.Expr]TypeAndValue)
		defer func() { check.Types = nil }()
	}
	var x operand
	check.expr(nil, &x, fd.Default)
	if !x.isValid() {
		return
	}
	if !check.logosConstantDefault(fd.Default) {
		check.errorf(fd.Default, InvalidConstInit, "%s non-constant default for field %s", logosDefaultNotConstant, name)
		return
	}
	check.assignment(&x, typ, "default for field "+name)
}

func (check *Checker) logosConstantDefault(e syntax.Expr) bool {
	e = syntax.Unparen(e)
	if check.Types[e].Value != nil {
		return true
	}
	if info, ok := check.untyped[e]; ok && info.mode == constant_ {
		return true
	}
	lit, ok := e.(*syntax.CompositeLit)
	if !ok {
		return false
	}
	typ := check.Types[lit].Type
	if typ == nil || !isValid(typ) {
		return false // an invalid nested literal already has a diagnostic
	}
	u, _ := commonUnder(typ, nil)
	if lit.Type == nil {
		if ptr, ok := u.(*Pointer); ok {
			u, _ = commonUnder(ptr.base, nil)
		}
	}
	_, mapLiteral := u.(*Map)
	for _, e := range lit.ElemList {
		if kv, ok := syntax.Unparen(e).(*syntax.KeyValueExpr); ok {
			if mapLiteral && !check.logosConstantDefault(kv.Key) {
				return false
			}
			e = kv.Value
		}
		if !check.logosConstantDefault(e) {
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
