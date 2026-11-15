// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"fmt"
	"reflect"
	"slices"
)

// This file erases M5-S1 const views during lowering (SPEC Sec 2.1,
// T-201): every `const T` node becomes its element type, and recorded
// slice expressions over const views gain a runtime check so a bound
// past the length panics at run time. The parent wires one call into
// lower.go, ahead of the other passes:
//
//	fl.eraseConstViews() // M5-S1: erase const views and bound recorded slices.
//
// A two-index slice over a plain name gains a max bound of len(base),
// which changes cap() of the derived slice to len(base) and keeps every
// derived slice inside its view; a bound the checker already rejected
// never reaches here. Every other recorded slice (a three-index slice,
// where overwriting the max would drop the user's bound, or a compound
// base, where a textual len(base) would evaluate the base twice)
// becomes a closure call binding the base once, so the trapping check
// runs after the base, low, high, and max operands have all evaluated.
// Comments between `const` and its element do not survive the splice;
// comment-bearing programs still lower.

// eraseConstViews rewrites every const view type of one file to its
// element and bounds the recorded slice expressions. Every const view
// sits in a type-holding field below, so direct assignment keeps every
// replacement addressable. The readonly `const` qualifier of interface
// methods (Field.Const) is cleared so later lowering passes that print
// and re-parse types (cloneType/respellType, mode 0) never see the
// keyword, while the eject text drops the exact five-byte keyword at the
// field position, preserving following comments and positions; ordinary
// methods stay unqualified. Cleared fields are recorded by identity for a
// scoped restore around the lowered recheck only (see
// LogosRestoreErasedAnonReadonly/LogosClearErasedAnonReadonly), so the
// recheck still sees the qualified obligation without ever letting the
// flag leak into printed/cloned types. Property elements
// (PropertyType.Const) keep their own accessor lowering and are never
// touched here.
//
// erasedAnonReadonly records every interface-method Field whose Const was
// cleared above, by node identity (never by name), for the recheck-window
// restore. It lives only from Lower through the lowered rechecks of the
// same checkFiles invocation.
var erasedAnonReadonly []*syntax.Field

func constAsView(t types2.Type) *types2.ConstView {
	if t == nil {
		return nil
	}
	if c, ok := types2.Unalias(t).(*types2.ConstView); ok {
		return c
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if u := n.Underlying(); u != nil {
			if c, ok := u.(*types2.ConstView); ok {
				return c
			}
		}
	}
	return nil
}

var constViewDecls []syntax.Pos

func (fl *fileLower) recordConstViewDecl(pos syntax.Pos) {
	if !pos.IsKnown() {
		return
	}
	if !slices.Contains(constViewDecls, pos) {
		constViewDecls = append(constViewDecls, pos)
	}
}

func ConstViewDecls() []syntax.Pos {
	decls := constViewDecls
	constViewDecls = nil
	return decls
}

func (fl *fileLower) recordConstViewDecls() {
	info := fl.pkg.info
	if info == nil || info.Defs == nil {
		return
	}
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.VarDecl:
			for _, name := range n.NameList {
				fl.recordIfConstView(name)
			}
		case *syntax.FuncType:
			for _, field := range n.ParamList {
				fl.recordIfConstView(field.Name)
			}
		}
		return true
	})
}

func (fl *fileLower) recordIfConstView(name *syntax.Name) {
	if name == nil {
		return
	}
	obj, ok := fl.pkg.info.Defs[name].(*types2.Var)
	if !ok || constAsView(obj.Type()) == nil {
		return
	}
	fl.recordConstViewDecl(name.Pos())
}

// LogosRestoreErasedAnonReadonly re-arms every recorded readonly method
// flag for the lowered recheck. Call after Lower has finished all
// print/clone passes, before the recheck.
func LogosRestoreErasedAnonReadonly() {
	for _, f := range erasedAnonReadonly {
		if f != nil {
			f.Const = true
		}
	}
}

// LogosClearErasedAnonReadonly clears every restored flag after the last
// lowered recheck and drops the record, so later backend prints see plain
// methods. It is also the discard path when no recheck follows.
func LogosClearErasedAnonReadonly() {
	for _, f := range erasedAnonReadonly {
		if f != nil {
			f.Const = false
		}
	}
	erasedAnonReadonly = nil
}
func (fl *fileLower) eraseConstViews() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		replaceFieldExprs(n, anyInterfaceFields, func(e syntax.Expr) syntax.Expr {
			fl.eraseConstIn(&e)
			return e
		})
		if f, ok := n.(*syntax.Field); ok && f.Const {
			if _, ok := f.Type.(*syntax.FuncType); ok {
				start := fl.off(f.Pos())
				fl.add(start, start+len("const"), "")
				erasedAnonReadonly = append(erasedAnonReadonly, f)
				f.Const = false
			}
		}
		return true
	})
	for e := range fl.pkg.info.LogosConstSlices {
		fl.boundConstSlice(e)
	}
	fl.wrapConstSlices()
	fl.recordConstViewDecls()
}

// eraseConstIn replaces ConstType wrappers in *e with their elements,
// flattening chains.
func (fl *fileLower) eraseConstIn(e *syntax.Expr) {
	for {
		c, ok := (*e).(*syntax.ConstType)
		if !ok {
			return
		}
		fl.eraseConstNode(c)
		*e = c.Elem
	}
}

// eraseConstNode records the eject splice deleting the `const `
// prefix of one const view type.
func (fl *fileLower) eraseConstNode(c *syntax.ConstType) {
	if fl.erasedConst[c] {
		return
	}
	if fl.erasedConst == nil {
		fl.erasedConst = make(map[*syntax.ConstType]bool)
	}
	fl.erasedConst[c] = true
	start := fl.off(c.Pos())
	end := start + len("const")
	for end < len(fl.src) && isSpace(fl.src[end]) {
		end++
	}
	fl.add(start, end, "")
}

// boundConstSlice gives one recorded slice expression a max bound of
// len(base), so a bound past the const length panics through the
// ordinary out-of-range failure. It runs only when the high bound
// needs a check, no max is present, and the base is a plain name.
// Handled entries leave the recorded set; three-index slices and
// compound bases stay recorded for wrapConstSlices.
func (fl *fileLower) boundConstSlice(e syntax.Expr) {
	s, ok := e.(*syntax.SliceExpr)
	if !ok || !fl.constSliceIsView(s) {
		delete(fl.pkg.info.LogosConstSlices, e)
		return
	}
	if s.Index[2] != nil {
		return
	}
	base, ok := syntax.Unparen(s.X).(*syntax.Name)
	if !ok || s.Index[1] == nil {
		return
	}
	if !fl.constBoundNeedsCheck(s, s.Index[1]) {
		delete(fl.pkg.info.LogosConstSlices, e)
		return
	}
	lenCall := &syntax.CallExpr{
		Fun:     nameOf(s.Pos(), "len"),
		ArgList: []syntax.Expr{nameOf(base.Pos(), base.Value)},
	}
	lenCall.SetPos(s.Pos())
	s.Index[2] = lenCall
	s.Full = true
	end := fl.exprEndOf(s.Index[1])
	fl.add(end, end, ":len("+base.Value+")")
	delete(fl.pkg.info.LogosConstSlices, e)
}

// constSliceIsView reports whether the base of s has a const view
// type. The recorder only records const view bases, so a missing type
// keeps the fail-safe direction toward trapping.
func (fl *fileLower) constSliceIsView(s *syntax.SliceExpr) bool {
	t := fl.pkg.info.TypeOf(syntax.Unparen(s.X))
	return t == nil || constAsView(t) != nil
}

// constClosureParam captures a wrapped const slice base. Each wrapper
// is a fresh function literal, so one spelling cannot capture.
const constClosureParam = "_logos_cb"

type constSliceMeta struct {
	baseType   types2.Type
	boundTypes [3]types2.Type
	resType    types2.Type
}

// wrapConstSlices wraps every slice the max splice cannot cover:
// three-index slices and compound bases. Original checked types are
// snapshotted before the walk so nested slice mutations never leave an
// enclosing target with synthetic, unrecorded child types. A single
// post-order walk wraps inner slices before outer ones so nested
// splices compose without overlap.
func (fl *fileLower) wrapConstSlices() {
	if len(fl.pkg.info.LogosConstSlices) == 0 {
		return
	}
	meta := make(map[*syntax.SliceExpr]constSliceMeta, len(fl.pkg.info.LogosConstSlices))
	for e := range fl.pkg.info.LogosConstSlices {
		s, ok := e.(*syntax.SliceExpr)
		if !ok || !fl.constSliceIsView(s) || !fl.constSliceIsWrappable(s) {
			continue
		}
		var m constSliceMeta
		m.baseType = fl.pkg.info.TypeOf(syntax.Unparen(s.X))
		m.resType = fl.pkg.info.TypeOf(s)
		for i, ix := range s.Index {
			if ix != nil {
				m.boundTypes[i] = fl.pkg.info.TypeOf(syntax.Unparen(ix))
			}
		}
		meta[s] = m
	}
	var stack []syntax.Node
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if n != nil {
			stack = append(stack, n)
			return true
		}
		top := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		fl.replaceConstSlice(top, meta)
		return true
	})
	clear(fl.pkg.info.LogosConstSlices)
}

// replaceConstSlice checks whether n holds a recorded const slice in
// an exported field, and replaces it with its closure wrapper.
func (fl *fileLower) replaceConstSlice(n syntax.Node, meta map[*syntax.SliceExpr]constSliceMeta) {
	v := reflect.ValueOf(n)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	typ := v.Type()
	for i := range v.NumField() {
		if typ.Field(i).IsExported() {
			fl.replaceConstSliceField(v.Field(i), meta)
		}
	}
}

// replaceConstSliceField replaces a recorded const slice held by the
// interface value f, or by an element of the slice or array f.
func (fl *fileLower) replaceConstSliceField(f reflect.Value, meta map[*syntax.SliceExpr]constSliceMeta) {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() || !f.CanSet() {
			return
		}
		if e, ok := f.Interface().(syntax.Expr); ok {
			if s, ok := syntax.Unparen(e).(*syntax.SliceExpr); ok && fl.pkg.info.LogosConstSlices[s] {
				delete(fl.pkg.info.LogosConstSlices, s)
				m, ok := meta[s]
				if !ok {
					return
				}
				f.Set(reflect.ValueOf(fl.wrapConstSlice(s, m)))
			}
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			fl.replaceConstSliceField(f.Index(i), meta)
		}
	}
}

// constSliceIsWrappable reports whether s needs the closure check.
// Only slice views qualify: arrays and strings carry no spare
// capacity, and a max splice on a string would not parse.
func (fl *fileLower) constSliceIsWrappable(s *syntax.SliceExpr) bool {
	needsCheck := false
	for _, b := range []syntax.Expr{s.Index[1], s.Index[2]} {
		if b != nil && fl.constBoundNeedsCheck(s, b) {
			needsCheck = true
			break
		}
	}
	if !needsCheck {
		return false
	}
	t := fl.pkg.info.TypeOf(syntax.Unparen(s.X))
	if t == nil {
		return true
	}
	c := constAsView(t)
	if c == nil {
		return false
	}
	elem := c.Elem
	if n, ok := types2.Unalias(elem).(*types2.Named); ok {
		elem = n.Underlying()
	}
	_, ok := types2.Unalias(elem).(*types2.Slice)
	return ok
}

type constBoundArg struct {
	index int
	name  string
	typ   string
	expr  syntax.Expr
	arg   otext
}

// wrapConstSlice builds the closure call replacing s and records its
// eject text. The base and all present bound expressions evaluate in
// their original caller scope as arguments to the closure call, in
// left-to-right source order. The closure body then references the
// captured parameters, preserving index types, evaluation order, and
// caller-frame semantics for operations like recover().
func (fl *fileLower) wrapConstSlice(s *syntax.SliceExpr, m constSliceMeta) syntax.Expr {
	pos := s.Pos()
	baseStart := fl.off(leftmostOf(s))
	kStart := -1
	for _, ix := range s.Index {
		if ix != nil {
			kStart = fl.off(leftmostOf(ix))
			break
		}
	}
	lbrack := fl.findLbrack(baseStart, kStart)
	rbrack := fl.findRbrack(kStart)
	baseText := fl.textOf(span{start: baseStart, end: lbrack})
	three := s.Index[2] != nil

	var bounds []constBoundArg
	for i := range s.Index {
		if s.Index[i] == nil {
			continue
		}
		t := m.boundTypes[i]
		typ := "int"
		if t != nil {
			spelled, ok := fl.spellLowered(types2.Default(t))
			if !ok {
				base.Fatalf("logos: cannot spell type %s of the const slice bound at %s", t, s.Index[i].Pos().String())
			}
			typ = spelled
		}
		from := fl.off(leftmostOf(s.Index[i]))
		bounds = append(bounds, constBoundArg{
			index: i,
			name:  fmt.Sprintf("%s%d", constClosureParam, i),
			typ:   typ,
			expr:  s.Index[i],
			arg:   fl.textOf(span{start: from, end: fl.exprEnd(from)}),
		})
	}

	paramText, ok := fl.spellLowered(m.baseType)
	if !ok {
		base.Fatalf("logos: cannot spell const slice base type at %s", pos.String())
	}
	resText, ok := fl.spellLowered(m.resType)
	if !ok {
		base.Fatalf("logos: cannot spell const slice result type at %s", pos.String())
	}

	paramType, ok := typeExprAt(paramText, pos)
	if !ok {
		base.Fatalf("logos: cannot respell type %q of the const slice wrapper at %s", paramText, pos.String())
	}
	resType, ok := typeExprAt(resText, pos)
	if !ok {
		base.Fatalf("logos: cannot respell type %q of the const slice wrapper at %s", resText, pos.String())
	}

	fields := []*syntax.Field{at(&syntax.Field{Name: nameOf(pos, constClosureParam), Type: paramType}, pos)}
	for _, b := range bounds {
		bt, ok := typeExprAt(b.typ, pos)
		if !ok {
			base.Fatalf("logos: cannot respell type %q of the const slice bound at %s", b.typ, pos.String())
		}
		fields = append(fields, at(&syntax.Field{Name: nameOf(pos, b.name), Type: bt}, pos))
	}
	signature := at(&syntax.FuncType{
		ParamList:  fields,
		ResultList: []*syntax.Field{at(&syntax.Field{Type: resType}, pos)},
	}, pos)

	idx, idxTexts := constSliceIndices(s, pos)
	body := at(&syntax.SliceExpr{
		X:     constSliceTarget(s, pos),
		Index: idx,
		Full:  true,
	}, pos)
	lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, []syntax.Stmt{returnOf(pos, []syntax.Expr{body})})}, pos)

	callArgs := []syntax.Expr{s.X}
	for _, b := range bounds {
		callArgs = append(callArgs, b.expr)
	}
	out := callOf(pos, lit, callArgs)

	targetText := constClosureParam
	if three {
		targetText = "(" + constClosureParam + "[:len(" + constClosureParam + "):len(" + constClosureParam + ")])"
	}
	bracketText := "[" + idxTexts[0] + ":" + idxTexts[1] + ":" + idxTexts[2] + "]"

	var parts []otext
	parts = append(parts, gen("func("+constClosureParam+" "+paramText))
	for _, b := range bounds {
		parts = append(parts, gen(", "+b.name+" "+b.typ))
	}
	parts = append(parts, gen(") "+resText+" { return "+targetText+bracketText+" }("), baseText)
	for _, b := range bounds {
		parts = append(parts, gen(", "), b.arg)
	}
	parts = append(parts, gen(")"))

	fl.addT(baseStart, rbrack+1, cat(parts...))
	return out
}

// constLenOf builds len(capture) at pos.
func constLenOf(pos syntax.Pos) syntax.Expr {
	return callOf(pos, nameOf(pos, "len"), []syntax.Expr{nameOf(pos, constClosureParam)})
}

// constSliceTarget builds the base expression inside the closure:
// three-index slices narrow the capture to its own length so the
// user's maximum faces the view length, while two-index slices use
// the capture directly.
func constSliceTarget(s *syntax.SliceExpr, pos syntax.Pos) syntax.Expr {
	base := nameOf(pos, constClosureParam)
	if s.Index[2] != nil {
		return at(&syntax.SliceExpr{
			X:     base,
			Index: [3]syntax.Expr{nil, constLenOf(pos), constLenOf(pos)},
			Full:  true,
		}, pos)
	}
	return base
}

// constSliceIndices builds the lowered indices and their text spellings.
func constSliceIndices(s *syntax.SliceExpr, pos syntax.Pos) (idx [3]syntax.Expr, texts [3]string) {
	for i := range 2 {
		if s.Index[i] != nil {
			name := fmt.Sprintf("%s%d", constClosureParam, i)
			idx[i] = nameOf(pos, name)
			texts[i] = name
		}
	}
	if s.Index[2] != nil {
		name := fmt.Sprintf("%s2", constClosureParam)
		idx[2] = nameOf(pos, name)
		texts[2] = name
	} else {
		idx[2] = constLenOf(pos)
		texts[2] = "len(" + constClosureParam + ")"
	}
	return idx, texts
}

// constBoundNeedsCheck reports whether a recorded high bound still
// needs its runtime check: cap calls over the base already failed in
// the checker, len calls over the base and constant in-range array
// bounds are provably fine.
func (fl *fileLower) constBoundNeedsCheck(s *syntax.SliceExpr, hi syntax.Expr) bool {
	if fl.constIsLenOrCapOf(hi, s.X, "cap") {
		return false
	}
	if fl.constIsLenOrCapOf(hi, s.X, "len") {
		return false
	}
	if fl.constBoundValue(hi) {
		if c := constAsView(fl.pkg.info.TypeOf(syntax.Unparen(s.X))); c != nil {
			if _, ok := types2.Unalias(c.Elem).(*types2.Array); ok {
				return false
			}
		}
	}
	return true
}

// constIsLenOrCapOf reports whether bound is a call to the named
// built-in over the same variable as base.
func (fl *fileLower) constIsLenOrCapOf(bound, base syntax.Expr, name string) bool {
	call, ok := syntax.Unparen(bound).(*syntax.CallExpr)
	if !ok || len(call.ArgList) != 1 {
		return false
	}
	fn, ok := syntax.Unparen(call.Fun).(*syntax.Name)
	if !ok {
		return false
	}
	obj, ok := fl.pkg.info.Uses[fn].(*types2.Builtin)
	if !ok || obj == nil || obj.Name() != name {
		return false
	}
	bVar, ok := syntax.Unparen(base).(*syntax.Name)
	if !ok {
		return false
	}
	aVar, ok := syntax.Unparen(call.ArgList[0]).(*syntax.Name)
	if !ok {
		return false
	}
	bObj, bOk := fl.pkg.info.Uses[bVar]
	aObj, aOk := fl.pkg.info.Uses[aVar]
	return bOk && aOk && bObj == aObj
}

// constBoundValue reports whether bound is a constant.
func (fl *fileLower) constBoundValue(bound syntax.Expr) bool {
	un := syntax.Unparen(bound)
	if tv, ok := fl.pkg.info.Types[un]; ok && tv.Value != nil {
		return true
	}
	return un.GetTypeInfo().Value != nil
}

// infoHasConstView reports whether the first-check info holds a const
// view in a declared variable, an expression type, or an instantiation
// argument (SPEC T-201, M5-S6): the rt.Freeze call result is a const
// with no ConstType syntax node, so a file that uses only Freeze
// reports no rewrite and would hand const types to the export writer.
// A true result forces the plain recheck, which re-derives Go-facing
// types for the writer and the backend. files covers the production
// check, which stores types in the syntax tree and leaves info.Types
// nil, so an inline-only Freeze is visible only there.
func infoHasConstView(info *types2.Info, files []*syntax.File) bool {
	if info == nil {
		return false
	}
	for _, f := range files {
		found := false
		syntax.Inspect(f, func(n syntax.Node) bool {
			if e, ok := n.(syntax.Expr); ok && constAsView(info.TypeOf(e)) != nil {
				found = true
				return false
			}
			return true
		})
		if found {
			return true
		}
	}
	for _, obj := range info.Defs {
		if v, ok := obj.(*types2.Var); ok && constAsView(v.Type()) != nil {
			return true
		}
	}
	for _, obj := range info.Uses {
		if v, ok := obj.(*types2.Var); ok && constAsView(v.Type()) != nil {
			return true
		}
	}
	for _, tv := range info.Types {
		if constAsView(tv.Type) != nil {
			return true
		}
	}
	for _, inst := range info.Instances {
		if targs := inst.TypeArgs; targs != nil {
			for i := range targs.Len() {
				if constAsView(targs.At(i)) != nil {
					return true
				}
			}
		}
	}
	return false
}
