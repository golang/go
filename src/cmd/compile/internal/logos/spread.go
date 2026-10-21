// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"reflect"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers the M6-S4 spread elements `..src` of struct
// composite literals (SPEC T-104.5). After this pass the backend sees
// no *syntax.SpreadExpr: every spread literal becomes either one tuple
// assignment (the self-update form) or an immediately invoked function
// literal that binds each source once and fills the copied fields.
//
// The pass runs after lowerOptions, so the eject text built here
// through textOf already holds every inner lowering. Every rewrite
// builds the native tree and the eject twin together through the twin
// markers of rewrite.go, so the two agree by construction (A2). The
// walk is bottom-up: nested spreads inside explicit values and sources
// lower before the enclosing literal reads their text.

// lowerSpreads rewrites every spread composite literal of one file.
func (fl *fileLower) lowerSpreads() {
	for _, decl := range fl.file.DeclList {
		switch d := decl.(type) {
		case *syntax.FuncDecl:
			if d.Body != nil {
				d.Body.List = fl.lowerSpreadList(d.Body.List)
			}
		case *syntax.VarDecl:
			if d.Values != nil {
				d.Values = fl.spreadValue(d.Values)
			}
		case *syntax.ConstDecl:
			if d.Values != nil {
				d.Values = fl.spreadValue(d.Values)
			}
		}
	}
}

// hasSpread reports whether c is a spread composite literal the checker
// recorded field copies for.
func (fl *fileLower) hasSpread(c *syntax.CompositeLit) bool {
	return len(fl.pkg.info.LogosSpreads[c]) > 0
}

// lowerSpreadList lowers a statement list, splaying self-update tuple
// assignments.
func (fl *fileLower) lowerSpreadList(list []syntax.Stmt) []syntax.Stmt {
	var out []syntax.Stmt
	for _, s := range list {
		out = append(out, fl.lowerSpreadStmt(s)...)
	}
	return out
}

// lowerSpreadStmt lowers one statement. An assignment matched by the
// self-update rule becomes one tuple assignment; every other statement
// keeps its shape with spreads lowered inside.
func (fl *fileLower) lowerSpreadStmt(s syntax.Stmt) []syntax.Stmt {
	if as, ok := s.(*syntax.AssignStmt); ok {
		if tup, ok := fl.spreadSelfUpdate(as); ok {
			return []syntax.Stmt{tup}
		}
		as.Lhs = fl.spreadValue(as.Lhs)
		as.Rhs = fl.spreadValue(as.Rhs)
		return []syntax.Stmt{s}
	}
	switch s := s.(type) {
	case *syntax.BlockStmt:
		s.List = fl.lowerSpreadList(s.List)
	case *syntax.IfStmt:
		s.Init = fl.spreadSimple(s.Init)
		s.Cond = fl.spreadValue(s.Cond)
		s.Then.List = fl.lowerSpreadList(s.Then.List)
		fl.lowerSpreadElse(s)
	case *syntax.ForStmt:
		s.Init = fl.spreadSimple(s.Init)
		s.Cond = fl.spreadValue(s.Cond)
		s.Post = fl.spreadSimple(s.Post)
		s.Body.List = fl.lowerSpreadList(s.Body.List)
	case *syntax.SwitchStmt:
		s.Init = fl.spreadSimple(s.Init)
		s.Tag = fl.spreadValue(s.Tag)
		for _, cc := range s.Body {
			cc.Cases = fl.spreadValue(cc.Cases)
			cc.Body = fl.lowerSpreadList(cc.Body)
		}
	case *syntax.SelectStmt:
		for _, cc := range s.Body {
			cc.Comm = fl.spreadSimple(cc.Comm)
			cc.Body = fl.lowerSpreadList(cc.Body)
		}
	case *syntax.LabeledStmt:
		inner := fl.lowerSpreadStmt(s.Stmt)
		s.Stmt = inner[0]
		return append([]syntax.Stmt{s}, inner[1:]...)
	case *syntax.ExprStmt:
		s.X = fl.spreadValue(s.X)
	case *syntax.SendStmt:
		s.Chan = fl.spreadValue(s.Chan)
		s.Value = fl.spreadValue(s.Value)
	case *syntax.CallStmt:
		s.Call = fl.spreadValue(s.Call)
	case *syntax.ReturnStmt:
		s.Results = fl.spreadValue(s.Results)
	case *syntax.DeclStmt:
		for _, decl := range s.DeclList {
			switch d := decl.(type) {
			case *syntax.VarDecl:
				if d.Values != nil {
					d.Values = fl.spreadValue(d.Values)
				}
			case *syntax.ConstDecl:
				if d.Values != nil {
					d.Values = fl.spreadValue(d.Values)
				}
			}
		}
	case *syntax.RangeClause:
		s.X = fl.spreadValue(s.X)
	}
	return []syntax.Stmt{s}
}

// lowerSpreadElse lowers the else branch of an if statement.
func (fl *fileLower) lowerSpreadElse(s *syntax.IfStmt) {
	switch els := s.Else.(type) {
	case nil:
	case *syntax.BlockStmt:
		els.List = fl.lowerSpreadList(els.List)
	case *syntax.IfStmt:
		s.Else = fl.lowerSpreadStmt(els)[0]
	}
}

// spreadSimple lowers a simple statement: assignments lower their
// sides, anything else lowers its expressions in place.
func (fl *fileLower) spreadSimple(s syntax.SimpleStmt) syntax.SimpleStmt {
	if s == nil {
		return nil
	}
	switch s := s.(type) {
	case *syntax.AssignStmt:
		// An if/for init holding a self-update keeps tuple form: the
		// single replacement carries the same meaning.
		if tup, ok := fl.spreadSelfUpdate(s); ok {
			return tup
		}
		s.Lhs = fl.spreadValue(s.Lhs)
		s.Rhs = fl.spreadValue(s.Rhs)
	case *syntax.RangeClause:
		s.X = fl.spreadValue(s.X)
	case *syntax.SendStmt:
		s.Chan = fl.spreadValue(s.Chan)
		s.Value = fl.spreadValue(s.Value)
	case *syntax.ExprStmt:
		s.X = fl.spreadValue(s.X)
	}
	return s
}

// spreadValue rewrites the spreads of e bottom-up: children first
// through reflection, then a spread composite itself. A function
// literal body lowers as a statement list. A spread composite directly
// under `&` takes the address path, which returns a pointer and drops
// the `&`; every other position takes the value path.
func (fl *fileLower) spreadValue(e syntax.Expr) syntax.Expr {
	if e == nil {
		return nil
	}
	if lit, ok := e.(*syntax.FuncLit); ok {
		lit.Body.List = fl.lowerSpreadList(lit.Body.List)
		return e
	}
	if op, ok := e.(*syntax.Operation); ok && op.Op == syntax.And && op.Y == nil {
		if c, ok := syntax.Unparen(op.X).(*syntax.CompositeLit); ok && fl.hasSpread(c) {
			fl.spreadFields(c)
			return fl.spreadAddrOp(op, c)
		}
		op.X = fl.spreadValue(op.X)
		return e
	}
	fl.spreadFields(e)
	if c, ok := e.(*syntax.CompositeLit); ok && fl.hasSpread(c) {
		return fl.spreadValueComposite(c)
	}
	return e
}

// spreadFields rewrites the spreads held directly by n: every field of
// static type syntax.Expr, []syntax.Expr, or [N]syntax.Expr goes
// through spreadValue in a value position. Function literal bodies are
// handled by the caller, never here.
func (fl *fileLower) spreadFields(n syntax.Node) {
	v := reflect.ValueOf(n)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	t := v.Type()
	for i := range v.NumField() {
		if !t.Field(i).IsExported() {
			continue
		}
		fl.spreadField(v.Field(i), t.Field(i).Type)
	}
}

// spreadField rewrites one field of static type ft through spreadValue
// when it holds expressions.
func (fl *fileLower) spreadField(v reflect.Value, ft reflect.Type) {
	if ft == exprType {
		if v.IsNil() {
			return
		}
		v.Set(reflect.ValueOf(fl.spreadValue(v.Interface().(syntax.Expr))))
		return
	}
	k := ft.Kind()
	if (k == reflect.Slice || k == reflect.Array) && ft.Elem() == exprType {
		for i := range v.Len() {
			ev := v.Index(i)
			if ev.IsNil() {
				continue
			}
			ev.Set(reflect.ValueOf(fl.spreadValue(ev.Interface().(syntax.Expr))))
		}
	}
}

// spreadBuilder carries one spread literal lowering: the file state,
// the literal, its position, the result temporary spelling, the
// checked field copies, the twin markers, and the native and text
// bodies built so far.
type spreadBuilder struct {
	fl        *fileLower
	c         *syntax.CompositeLit
	pos       syntax.Pos
	vSpelling string
	infos     []types2.LogosSpread
	tw        *twin
	native    []syntax.Stmt
	text      []syntax.Stmt
}

// spreadValueComposite lowers `T{..src}` to an immediately invoked
// function literal returning T.
func (fl *fileLower) spreadValueComposite(c *syntax.CompositeLit) syntax.Expr {
	b := fl.spreadOpen(c)
	b.body()
	v := nameOf(b.pos, b.vSpelling)
	b.native = append(b.native, returnOf(b.pos, []syntax.Expr{v}))
	b.text = append(b.text, returnOf(b.pos, []syntax.Expr{nameOf(b.pos, b.vSpelling)}))
	native := callOf(b.pos, at(&syntax.FuncLit{
		Type: at(&syntax.FuncType{
			ResultList: []*syntax.Field{at(&syntax.Field{Type: b.nativeType()}, b.pos)},
		}, b.pos),
		Body: blockOf(b.pos, b.native),
	}, b.pos), nil)
	text := "func() " + b.typeText() + " {\n" + renderText(b.text, b.tw.subs) + "}()"
	fl.add(fl.off(syntax.StartPos(c)), fl.off(c.Rbrace)+1, text)
	return native
}

// spreadAddrComposite lowers a spread composite directly under `&` to
// an immediately invoked function literal returning *T.
func (fl *fileLower) spreadAddrComposite(c *syntax.CompositeLit) syntax.Expr {
	b := fl.spreadOpen(c)
	b.body()
	b.native = append(b.native, returnOf(b.pos, []syntax.Expr{
		at(&syntax.Operation{Op: syntax.And, X: nameOf(b.pos, b.vSpelling)}, b.pos)}))
	b.text = append(b.text, returnOf(b.pos, []syntax.Expr{
		at(&syntax.Operation{Op: syntax.And, X: nameOf(b.pos, b.vSpelling)}, b.pos)}))
	native := callOf(b.pos, at(&syntax.FuncLit{
		Type: at(&syntax.FuncType{
			ResultList: []*syntax.Field{at(&syntax.Field{
				Type: at(&syntax.Operation{Op: syntax.Mul, X: b.nativeType()}, b.pos),
			}, b.pos)},
		}, b.pos),
		Body: blockOf(b.pos, b.native),
	}, b.pos), nil)
	text := "func() *" + b.typeText() + " {\n" + renderText(b.text, b.tw.subs) + "}()"
	fl.add(fl.off(syntax.StartPos(c)), fl.off(c.Rbrace)+1, text)
	return native
}

// spreadAddrOp lowers `&T{..src}` to the pointer function literal,
// dropping the `&`: the span covers the operator so no stray `&`
// survives in the ejected text.
func (fl *fileLower) spreadAddrOp(op *syntax.Operation, c *syntax.CompositeLit) syntax.Expr {
	start := fl.off(op.Pos())
	native := fl.spreadAddrComposite(c)
	// spreadAddrComposite spliced the composite span; widen it to the
	// operator, dropping the `&` itself: the IIFE already returns *T,
	// so keeping the operator would take the address of a pointer.
	// The drained span starts just after the operator, so a comment in
	// that gap stays while only the `&` itself is dropped.
	inner := fl.textOf(span{start: start + len("&"), end: fl.off(c.Rbrace) + 1})
	fl.add(start, skipCloseParens(fl.src, fl.off(c.Rbrace)+1), inner)
	return native
}

// spreadOpen numbers one spread literal and snapshots its checked
// field copies.
func (fl *fileLower) spreadOpen(c *syntax.CompositeLit) *spreadBuilder {
	vSpelling := fmt.Sprintf("_logos_v%d", fl.spreadSeq)
	fl.spreadSeq++
	return &spreadBuilder{
		fl:        fl,
		c:         c,
		pos:       c.Pos(),
		vSpelling: vSpelling,
		infos:     fl.pkg.info.LogosSpreads[c],
		tw:        &twin{},
	}
}

// typeText spells the literal type for the eject text, substituting
// every inner lowering. A type elided in source (`[]T{{..s}}`
// elements) spells from the checked type.
func (b *spreadBuilder) typeText() string {
	c := b.c
	if c.Type == nil {
		text, ok := b.fl.spellType(b.fl.pkg.info.TypeOf(c))
		if !ok {
			base.Fatalf("logos: cannot spell spread literal type %s", b.fl.pkg.info.TypeOf(c))
		}
		return text
	}
	return b.fl.textOf(span{start: b.fl.off(syntax.StartPos(c.Type)), end: b.fl.exprEndOf(c.Type)})
}

// nativeType spells the literal type for the native tree: the original
// type node moves into `_v := T{}`, so the result clones it. A type
// elided in source re-parses the spelled text.
func (b *spreadBuilder) nativeType() syntax.Expr {
	c := b.c
	if c.Type == nil {
		text, ok := b.fl.spellType(b.fl.pkg.info.TypeOf(c))
		if !ok {
			base.Fatalf("logos: cannot spell spread literal type %s", b.fl.pkg.info.TypeOf(c))
		}
		reparsed, ok := reparseType(text)
		if !ok {
			base.Fatalf("logos: cannot reparse spread literal type %s", text)
		}
		syntax.Inspect(reparsed, func(n syntax.Node) bool {
			if n != nil {
				n.SetPos(b.pos)
			}
			return true
		})
		return reparsed
	}
	return cloneType(c.Type, b.pos)
}

// body builds `_v := T{}` followed by one assignment per explicit field
// and per spread-copied field, in source order.
func (b *spreadBuilder) body() {
	c := b.c
	pos := b.pos
	v := func() *syntax.Name { return nameOf(pos, b.vSpelling) }
	b.native = append(b.native, defineOf(pos,
		[]syntax.Expr{v()},
		[]syntax.Expr{at(&syntax.CompositeLit{Type: b.nativeType(), Rbrace: c.Rbrace}, pos)}))
	b.text = append(b.text, defineOf(pos,
		[]syntax.Expr{v()},
		[]syntax.Expr{at(&syntax.CompositeLit{Type: b.tw.mark(pos, b.typeText()), Rbrace: c.Rbrace}, pos)}))
	for _, el := range c.ElemList {
		if se, ok := el.(*syntax.SpreadExpr); ok {
			b.copy(se)
			continue
		}
		kv, ok := el.(*syntax.KeyValueExpr)
		if !ok {
			continue
		}
		key, ok := kv.Key.(*syntax.Name)
		if !ok {
			continue
		}
		valText := b.fl.textOf(span{start: b.fl.spreadStartOf(kv.Value), end: b.fl.exprEndOf(kv.Value)})
		b.native = append(b.native, assignOf(pos,
			[]syntax.Expr{selOf(pos, v(), key.Value)},
			[]syntax.Expr{kv.Value}))
		b.text = append(b.text, assignOf(pos,
			[]syntax.Expr{selOf(pos, v(), key.Value)},
			[]syntax.Expr{b.tw.mark(pos, valText)}))
	}
}

// copy binds one spread source to a fresh temporary and fills its
// copied fields from it. With no copied fields the source still
// evaluates once (SPEC T-104.5), so it assigns to the blank instead of
// an unread temporary, which the compiler would reject as unused.
func (b *spreadBuilder) copy(se *syntax.SpreadExpr) {
	var fields []string
	for _, info := range b.infos {
		if info.Src == se.X {
			fields = info.Fields
			break
		}
	}
	pos := b.pos
	srcText := b.fl.textOf(span{start: b.fl.spreadStartOf(se.X), end: b.fl.exprEndOf(se.X)})
	if len(fields) == 0 {
		b.native = append(b.native, assignOf(pos,
			[]syntax.Expr{nameOf(pos, "_")},
			[]syntax.Expr{se.X}))
		b.text = append(b.text, assignOf(pos,
			[]syntax.Expr{nameOf(pos, "_")},
			[]syntax.Expr{b.tw.mark(pos, srcText)}))
		return
	}
	tmpSpelling := fmt.Sprintf("_logos_spread%d", b.fl.spreadSeq)
	b.fl.spreadSeq++
	b.native = append(b.native, defineOf(pos,
		[]syntax.Expr{nameOf(pos, tmpSpelling)},
		[]syntax.Expr{se.X}))
	b.text = append(b.text, defineOf(pos,
		[]syntax.Expr{nameOf(pos, tmpSpelling)},
		[]syntax.Expr{b.tw.mark(pos, srcText)}))
	for _, name := range fields {
		b.native = append(b.native, assignOf(pos,
			[]syntax.Expr{selOf(pos, nameOf(pos, b.vSpelling), name)},
			[]syntax.Expr{selOf(pos, nameOf(pos, tmpSpelling), name)}))
		b.text = append(b.text, assignOf(pos,
			[]syntax.Expr{selOf(pos, nameOf(pos, b.vSpelling), name)},
			[]syntax.Expr{selOf(pos, nameOf(pos, tmpSpelling), name)}))
	}
}

// spreadSelfUpdate matches `x = T{f: e, ..x}` where every spread source
// is the assigned variable itself: the update lowers to one tuple
// assignment of the explicit fields (SPEC T-104.5), so a swap like
// `s = State{A: s.B, B: s.A, ..s}` reads every right-hand side before
// writing. The target is a plain name; anything else takes the general
// path, which stays correct through a temporary.
func (fl *fileLower) spreadSelfUpdate(s *syntax.AssignStmt) (*syntax.AssignStmt, bool) {
	if s.Op != 0 {
		return nil, false
	}
	lhs := syntax.UnpackListExpr(s.Lhs)
	rhs := syntax.UnpackListExpr(s.Rhs)
	if len(lhs) != 1 || len(rhs) != 1 {
		return nil, false
	}
	target, ok := lhs[0].(*syntax.Name)
	if !ok || target.Value == "_" {
		return nil, false
	}
	c, ok := syntax.Unparen(rhs[0]).(*syntax.CompositeLit)
	if !ok || !fl.hasSpread(c) {
		return nil, false
	}
	obj := fl.pkg.info.Uses[target]
	if obj == nil {
		return nil, false
	}
	for _, info := range fl.pkg.info.LogosSpreads[c] {
		src, ok := syntax.Unparen(info.Src).(*syntax.Name)
		if !ok || fl.pkg.info.Uses[src] != obj {
			return nil, false
		}
	}
	pos := c.Pos()
	var fields []string
	var values []syntax.Expr
	tw := &twin{}
	var ltexts []string
	var vtexts []string
	for _, el := range c.ElemList {
		kv, ok := el.(*syntax.KeyValueExpr)
		if !ok {
			continue
		}
		key, ok := kv.Key.(*syntax.Name)
		if !ok {
			continue
		}
		val := fl.spreadValue(kv.Value)
		fields = append(fields, key.Value)
		values = append(values, val)
		ltexts = append(ltexts, target.Value+"."+key.Value)
		vtexts = append(vtexts, fl.textOf(span{start: fl.spreadStartOf(val), end: fl.exprEndOf(val)}))
	}
	if len(fields) == 0 {
		native := assignOf(pos, []syntax.Expr{nameOf(pos, target.Value)}, []syntax.Expr{nameOf(pos, target.Value)})
		fl.add(fl.stmtSpanStart(s, lhs[0]), fl.spanEndOf(rhs[0]), target.Value+" = "+target.Value)
		return native, true
	}
	nativeLhs := make([]syntax.Expr, len(fields))
	textLhs := make([]syntax.Expr, len(fields))
	textVals := make([]syntax.Expr, len(values))
	for i, name := range fields {
		nativeLhs[i] = selOf(pos, nameOf(pos, target.Value), name)
		textLhs[i] = tw.mark(pos, ltexts[i])
		textVals[i] = tw.mark(pos, vtexts[i])
	}
	native := assignOf(pos, nativeLhs, values)
	text := renderText([]syntax.Stmt{assignOf(pos, textLhs, textVals)}, tw.subs)
	fl.add(fl.stmtSpanStart(s, lhs[0]), fl.spanEndOf(rhs[0]), text)
	return native, true
}

// stmtSpanStart returns the eject offset where the statement holding
// lhs begins, extended backward past parentheses the parser dropped.
func (fl *fileLower) stmtSpanStart(s *syntax.AssignStmt, lhs syntax.Expr) int {
	return spanStartBack(fl.src, fl.off(leftmostOf(lhs)))
}

// spreadStartOf returns the eject offset where the source text of e
// begins: the type of a composite literal (a bare brace position would
// drop it), the operand of any other expression, extended backward
// past parentheses the parser dropped.
func (fl *fileLower) spreadStartOf(e syntax.Expr) int {
	if c, ok := syntax.Unparen(e).(*syntax.CompositeLit); ok && c.Type != nil {
		return spanStartBack(fl.src, fl.off(syntax.StartPos(c)))
	}
	return spanStartBack(fl.src, fl.off(leftmostOf(e)))
}

// spanEndOf returns the eject offset just past e, extended past
// parentheses the parser dropped.
func (fl *fileLower) spanEndOf(e syntax.Expr) int {
	if c, ok := syntax.Unparen(e).(*syntax.CompositeLit); ok {
		return skipCloseParens(fl.src, fl.off(c.Rbrace)+1)
	}
	return fl.exprEndOf(e)
}

// exprEndOf returns the eject offset just past the expression e: the
// end of a composite literal is its closing brace, anything else ends
// where the generic expression scanner stops.
func (fl *fileLower) exprEndOf(e syntax.Expr) int {
	if c, ok := syntax.Unparen(e).(*syntax.CompositeLit); ok {
		return skipCloseParens(fl.src, fl.off(c.Rbrace)+1)
	}
	return fl.exprEnd(fl.off(leftmostOf(e)))
}
