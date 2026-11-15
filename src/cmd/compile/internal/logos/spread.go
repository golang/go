// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"reflect"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func (fl *fileLower) lowerSpreads() {
	if len(fl.pkg.info.LogosSpreads) == 0 {
		return
	}
	for _, decl := range fl.file.DeclList {
		fl.spreadChildren(decl)
	}
}

func (fl *fileLower) hasSpread(c *syntax.CompositeLit) bool {
	return len(fl.pkg.info.LogosSpreads[c]) > 0
}

func (fl *fileLower) spreadChildren(n syntax.Node) {
	v := reflect.ValueOf(n).Elem()
	typ := v.Type()
	for i := range v.NumField() {
		if typ.Field(i).IsExported() {
			fl.spreadField(v.Field(i))
		}
	}
}

func (fl *fileLower) spreadField(f reflect.Value) {
	switch f.Kind() {
	case reflect.Slice:
		for i := range f.Len() {
			fl.spreadField(f.Index(i))
		}
	case reflect.Interface, reflect.Pointer:
		if f.IsNil() {
			return
		}
		if n, ok := f.Interface().(syntax.Node); ok {
			fl.spreadNode(f, n)
		}
	}
}

func (fl *fileLower) spreadNode(slot reflect.Value, n syntax.Node) {
	switch n := n.(type) {
	case *syntax.BranchStmt:
	case *syntax.AssignStmt:
		fl.spreadAssign(n)
	case syntax.Expr:
		if r := fl.spreadValue(n); r != n && slot.Kind() == reflect.Interface {
			slot.Set(reflect.ValueOf(r))
		}
	default:
		fl.spreadChildren(n)
	}
}

func (fl *fileLower) spreadAssign(as *syntax.AssignStmt) {
	if tup, ok := fl.spreadSelfUpdate(as); ok {
		*as = *tup
		return
	}
	fl.spreadChildren(as)
}

func (fl *fileLower) spreadValue(e syntax.Expr) syntax.Expr {
	if op, ok := e.(*syntax.Operation); ok && op.Op == syntax.And && op.Y == nil {
		if c, ok := syntax.Unparen(op.X).(*syntax.CompositeLit); ok && fl.hasSpread(c) {
			fl.spreadChildren(c)
			return fl.retyped(e, fl.spreadAddrOp(op, c))
		}
	}
	fl.spreadChildren(e)
	if c, ok := e.(*syntax.CompositeLit); ok && fl.hasSpread(c) {
		return fl.retyped(e, fl.spreadValueComposite(c))
	}
	return e
}

func (fl *fileLower) retyped(old, native syntax.Expr) syntax.Expr {
	fl.stashBodyType(fl.pkg.info.TypeOf(old), old, native)
	return native
}

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

func (fl *fileLower) spreadValueComposite(c *syntax.CompositeLit) syntax.Expr {
	return fl.spreadComposite(c, false)
}

func (fl *fileLower) spreadAddrComposite(c *syntax.CompositeLit) syntax.Expr {
	return fl.spreadComposite(c, true)
}

func (fl *fileLower) spreadComposite(c *syntax.CompositeLit, addr bool) syntax.Expr {
	b := fl.spreadOpen(c)
	b.body()
	ret := func() syntax.Expr {
		v := nameOf(b.pos, b.vSpelling)
		if addr {
			return at(&syntax.Operation{Op: syntax.And, X: v}, b.pos)
		}
		return v
	}
	b.native = append(b.native, returnOf(b.pos, []syntax.Expr{ret()}))
	b.text = append(b.text, returnOf(b.pos, []syntax.Expr{ret()}))
	nat := b.nativeType()
	if addr {
		nat = at(&syntax.Operation{Op: syntax.Mul, X: nat}, b.pos)
	}
	native := callOf(b.pos, at(&syntax.FuncLit{
		Type: at(&syntax.FuncType{
			ResultList: []*syntax.Field{at(&syntax.Field{Type: nat}, b.pos)},
		}, b.pos),
		Body: blockOf(b.pos, b.native),
	}, b.pos), nil)
	ptr := ""
	if addr {
		ptr = "*"
	}
	text := cat(gen("func() "+ptr), b.typeText(), gen(" {\n"), renderText(b.text, b.tw.subs), gen("}()"))
	fl.addT(fl.off(syntax.StartPos(c)), fl.off(c.Rbrace)+1, text)
	fl.wrapSpan[native] = span{start: fl.off(syntax.StartPos(c)), end: fl.off(c.Rbrace) + 1}
	return native
}

func (fl *fileLower) spreadAddrOp(op *syntax.Operation, c *syntax.CompositeLit) syntax.Expr {
	start := fl.off(op.Pos())
	native := fl.spreadAddrComposite(c)
	fl.add(start, start+len("&"), "")
	fl.wrapSpan[native] = span{start: start, end: fl.off(c.Rbrace) + 1}
	return native
}

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

func (b *spreadBuilder) typeText() otext {
	c := b.c
	if c.Type == nil {
		text, ok := b.fl.spellType(b.fl.pkg.info.TypeOf(c))
		if !ok {
			base.Fatalf("logos: cannot spell spread literal type %s", b.fl.pkg.info.TypeOf(c))
		}
		return gen(text)
	}
	return b.fl.textOf(span{start: b.fl.off(syntax.StartPos(c.Type)), end: b.fl.exprEndOf(c.Type)})
}

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
	spreads := 0
	for _, el := range c.ElemList {
		if se, ok := el.(*syntax.SpreadExpr); ok {
			b.copy(se, b.infos[spreads].Fields)
			spreads++
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
		valText := b.fl.lazyText(b.fl.spreadSpan(kv.Value))
		b.native = append(b.native, assignOf(pos,
			[]syntax.Expr{selOf(pos, v(), key.Value)},
			[]syntax.Expr{kv.Value}))
		b.text = append(b.text, assignOf(pos,
			[]syntax.Expr{selOf(pos, v(), key.Value)},
			[]syntax.Expr{b.tw.mark(pos, valText)}))
	}
}

func (b *spreadBuilder) copy(se *syntax.SpreadExpr, fields []string) {
	pos := b.pos
	srcText := b.fl.lazyText(b.fl.spreadSpan(se.X))
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
	var vtexts []otext
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
		vtexts = append(vtexts, fl.lazyText(fl.spreadSpan(val)))
	}
	if len(fields) == 0 {
		native := assignOf(pos, []syntax.Expr{nameOf(pos, target.Value)}, []syntax.Expr{nameOf(pos, target.Value)})
		fl.add(fl.stmtSpanStart(s, lhs[0]), fl.exprEndOf(rhs[0]), target.Value+" = "+target.Value)
		return native, true
	}
	nativeLhs := make([]syntax.Expr, len(fields))
	textLhs := make([]syntax.Expr, len(fields))
	textVals := make([]syntax.Expr, len(values))
	for i, name := range fields {
		nativeLhs[i] = selOf(pos, nameOf(pos, target.Value), name)
		textLhs[i] = tw.mark(pos, gen(ltexts[i]))
		textVals[i] = tw.mark(pos, vtexts[i])
	}
	native := assignOf(pos, nativeLhs, values)
	text := renderText([]syntax.Stmt{assignOf(pos, textLhs, textVals)}, tw.subs).trimRight("\n")
	fl.addT(fl.stmtSpanStart(s, lhs[0]), fl.exprEndOf(rhs[0]), text)
	return native, true
}

func (fl *fileLower) stmtSpanStart(s *syntax.AssignStmt, lhs syntax.Expr) int {
	return spanStartBack(fl.src, fl.off(leftmostOf(lhs)))
}

func (fl *fileLower) spreadStartOf(e syntax.Expr) int {
	if c, ok := syntax.Unparen(e).(*syntax.CompositeLit); ok && c.Type != nil {
		return spanStartBack(fl.src, fl.off(syntax.StartPos(c)))
	}
	return spanStartBack(fl.src, fl.off(leftmostOf(e)))
}

func (fl *fileLower) spreadSpan(e syntax.Expr) span {
	if _, ok := syntax.Unparen(e).(*syntax.CompositeLit); ok {
		return span{fl.spreadStartOf(e), fl.exprEndOf(e)}
	}
	return fl.exprSpan(e)
}

func (fl *fileLower) exprEndOf(e syntax.Expr) int {
	if n, ok := e.(*syntax.Name); ok {
		if end, hosted := fl.stubEnd[n]; hosted {
			return end
		}
	}
	if c, ok := syntax.Unparen(e).(*syntax.CompositeLit); ok {
		return simdFwdEnd(fl.src, fl.off(syntax.StartPos(e)), fl.off(c.Rbrace)+1)
	}
	if lam, ok := syntax.Unparen(e).(*syntax.LambdaExpr); ok {
		end := fl.lambdaEnd(lam)
		for p, ok := e.(*syntax.ParenExpr); ok; p, ok = p.X.(*syntax.ParenExpr) {
			end = closeParenEnd(fl.src, end)
		}
		return end
	}
	from := fl.exprStart(e)
	through := from
	if rbrace, ok := lastBraceOf(e); ok {
		through = fl.off(rbrace) + 1
	}
	return fl.exprEndThrough(from, through)
}

func closeParenEnd(src []byte, from int) int {
	end := from
	scanSegment(src, from, len(src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if t.kind == token.RPAREN {
			end = t.end
		}
		return false
	})
	return end
}
