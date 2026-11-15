// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"reflect"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type cswKind int

const (
	cswExact cswKind = iota
	cswTilde
	cswIface
)

type cswCase struct {
	syn  *syntax.ConstSwitchCase
	kind cswKind
	typ  types2.Type

	spell otext

	lo, hi int

	close syntax.Pos

	sub types2.LogosConstCase
}

type cswLower struct {
	fc    *funcLower
	fl    *fileLower
	info  *types2.Info
	stmt  *syntax.ConstSwitchStmt
	start int
	end   int
	tag   types2.Type
	cases []cswCase
}

type cswRebind struct {
	name  string
	spell string
}

func (fc *funcLower) lowerConstSwitchStmt(s *syntax.ConstSwitchStmt) []syntax.Stmt {
	c := newCswLower(fc, s)
	if _, ok := c.tag.(*types2.TypeParam); ok {
		return c.lowerRuntime()
	}
	return c.lowerFold()
}

func newCswLower(fc *funcLower, s *syntax.ConstSwitchStmt) *cswLower {
	fl := fc.fl
	c := &cswLower{
		fc:    fc,
		fl:    fl,
		info:  fl.pkg.info,
		stmt:  s,
		start: fl.off(s.Pos()),
		end:   fl.off(s.Rbrace) + 1,
		tag:   checkedTypeOf(fl.pkg.info, s.Targ),
	}
	for i, syn := range s.Cases {
		typ := checkedTypeOf(fl.pkg.info, syn.Type)
		cs := cswCase{
			syn:   syn,
			kind:  cswExact,
			typ:   typ,
			spell: c.spell(fl.off(syn.Case), fl.off(syn.Colon)),
			lo:    fl.off(syn.Colon) + 1,
			hi:    fl.off(s.Rbrace),
			sub:   fl.pkg.info.LogosConstCases[syn],
		}
		cs.close = s.Rbrace
		if i+1 < len(s.Cases) {
			cs.hi = fl.off(s.Cases[i+1].Case)
			cs.close = s.Cases[i+1].Case
		}
		switch {
		case syn.Tilde:
			cs.kind = cswTilde
		default:
			if _, ok := types2.Unalias(typ).Underlying().(*types2.Interface); ok {
				cs.kind = cswIface
			}
		}
		c.cases = append(c.cases, cs)
	}
	return c
}

func checkedTypeOf(info *types2.Info, e syntax.Expr) types2.Type {
	var name *syntax.Name
	switch e := e.(type) {
	case *syntax.Name:
		name = e
	case *syntax.SelectorExpr:
		name = e.Sel
	}
	if name != nil {
		if tn, ok := info.Uses[name].(*types2.TypeName); ok {
			return tn.Type()
		}
	}
	if typ := info.TypeOf(e); typ != nil {
		return typ
	}
	base.Fatalf("logos: const switch type at %s has no checked type", e.Pos().String())
	return nil
}

func (c *cswLower) spell(from, bound int) otext {
	start := typeStart(c.fl.src, from, bound)
	return c.fl.textOf(span{start, typeEnd(c.fl.src, start, bound)})
}

func typeStart(src []byte, from, bound int) int {
	start := bound
	scanSegment(src, from, bound, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.CASE, token.CONST, token.SWITCH, token.TILDE:
			return true
		}
		start = t.start
		return false
	})
	return start
}

func typeEnd(src []byte, start, bound int) int {
	end := start
	depth := 0
	scanSegment(src, start, bound, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				return false
			}
			depth--
		}
		end = t.end
		return true
	})
	return end
}

func (c *cswLower) lowerFold() []syntax.Stmt {
	sel, static := types2.LogosConstSelectedCase(c.info, c.stmt)
	if !static {
		base.Fatalf("logos: const switch at %s has no compile-time selection", c.stmt.Pos().String())
	}
	var match *cswCase
	for i := range c.cases {
		if c.cases[i].syn == sel {
			match = &c.cases[i]
			break
		}
	}
	if match == nil {
		c.fc.deleteConstSpan(c.start, c.end)
		return nil
	}
	stmts, text := c.caseBody(match, "")
	block := closeAt(c.stmt.Pos(), stmts, c.stmt.Rbrace)
	kept := span{match.lo, match.hi}
	c.fl.addT(c.start, c.end, cat(leftoverComments(c.fl, span{c.start, c.end}, []span{kept}), gen("{"), text, gen("}")))
	return []syntax.Stmt{block}
}

func (c *cswLower) runs() [][]*cswCase {
	var out [][]*cswCase
	for i := range c.cases {
		cs := &c.cases[i]
		n := len(out)
		if n > 0 && (out[n-1][0].kind == cswExact) == (cs.kind == cswExact) {
			out[n-1] = append(out[n-1], cs)
			continue
		}
		out = append(out, []*cswCase{cs})
	}
	return out
}

func (c *cswLower) lowerRuntime() []syntax.Stmt {
	runs := c.runs()
	if len(runs) <= 1 || (len(runs) == 2 && runs[0][0].kind == cswExact) {
		var exacts, rest []*cswCase
		for _, run := range runs {
			if run[0].kind == cswExact {
				exacts = run
			} else {
				rest = run
			}
		}
		stmt, text := c.typeSwitch(exacts, rest, "")
		c.fl.addT(c.start, c.end, text)
		return []syntax.Stmt{stmt}
	}
	pos := c.stmt.Pos()
	done := fmt.Sprintf("_logos_csw%d", c.start)
	block := at(&syntax.BlockStmt{Rbrace: c.stmt.Rbrace}, pos)
	block.List = append(block.List, defineOf(pos, []syntax.Expr{nameOf(pos, done)}, []syntax.Expr{nameOf(pos, "false")}))
	text := gen("{" + done + " := false\n")
	for _, run := range runs {
		first := run[0].syn.Case
		close := run[len(run)-1].close
		var inner syntax.Stmt
		var innerText otext
		if run[0].kind == cswExact {
			inner, innerText = c.typeSwitch(run, nil, done)
			if sw, ok := inner.(*syntax.SwitchStmt); ok {
				sw.Rbrace = close
			}
		} else {
			inner, innerText = c.chainIf(run, done)
		}
		guard := at(&syntax.Operation{Op: syntax.Not, X: nameOf(first, done)}, first)
		block.List = append(block.List, at(&syntax.IfStmt{
			Cond: guard,
			Then: closeAt(first, []syntax.Stmt{inner}, close),
		}, first))
		text = cat(text, gen("if !"+done+" {"), innerText, gen("}\n"))
	}
	text = cat(text, gen("}"))
	c.fl.addT(c.start, c.end, text)
	return []syntax.Stmt{block}
}

func (c *cswLower) typeSwitch(exacts, rest []*cswCase, done string) (syntax.Stmt, otext) {
	pos := c.stmt.Pos()
	if len(exacts) > 0 {
		pos = exacts[0].syn.Case
	} else if len(rest) > 0 {
		pos = rest[0].syn.Case
	}
	tag := c.spell(c.fl.off(c.stmt.Pos()), c.fl.off(c.stmt.Lbrace))
	guard, ok := parseExprAt("any((*"+tag.s+")(nil))", pos)
	if !ok {
		base.Fatalf("logos: const switch tag %s at %s cannot be respelled", tag.s, pos.String())
	}
	sw := at(&syntax.SwitchStmt{
		Tag:    at(&syntax.TypeSwitchGuard{X: guard}, pos),
		Rbrace: c.stmt.Rbrace,
	}, pos)
	text := cat(gen("switch any((*"), tag, gen(")(nil)).(type) {"))
	for _, cs := range exacts {
		stmts, bodyText := c.caseBody(cs, done)
		ptr, ok := parseExprAt("*"+cs.spell.s, cs.syn.Case)
		if !ok {
			base.Fatalf("logos: const switch case %s at %s cannot be respelled", cs.spell.s, cs.syn.Case.String())
		}
		sw.Body = append(sw.Body, at(&syntax.CaseClause{Cases: ptr, Body: stmts}, cs.syn.Case))
		text = cat(text, gen("\ncase *"), cs.spell, gen(":"), bodyText)
	}
	if len(rest) > 0 {
		chain, chainText := c.chainIf(rest, done)
		sw.Body = append(sw.Body, at(&syntax.CaseClause{Body: []syntax.Stmt{chain}}, rest[0].syn.Case))
		text = cat(text, gen("\ndefault:\n"), chainText)
	}
	text = cat(text, gen("\n}"))
	return sw, text
}

func (c *cswLower) chainIf(run []*cswCase, done string) (syntax.Stmt, otext) {
	var next syntax.Stmt
	var nextText otext
	for i := len(run) - 1; i >= 0; i-- {
		cs := run[i]
		cond, condText := c.cond(cs)
		stmts, bodyText := c.caseBody(cs, done)
		link := at(&syntax.IfStmt{
			Cond: cond,
			Then: closeAt(cs.syn.Case, stmts, cs.close),
			Else: next,
		}, cs.syn.Case)
		text := cat(gen("if "), condText, gen(" {"), bodyText, gen("}"))
		if next != nil {
			text = cat(text, gen(" else "), nextText)
		}
		next, nextText = link, text
	}
	return next, nextText
}

func closeAt(pos syntax.Pos, list []syntax.Stmt, close syntax.Pos) *syntax.BlockStmt {
	return at(&syntax.BlockStmt{List: list, Rbrace: close}, pos)
}

func (c *cswLower) cond(cs *cswCase) (syntax.Expr, otext) {
	c.fl.needRT = true
	tag := c.spell(c.fl.off(c.stmt.Pos()), c.fl.off(c.stmt.Lbrace))
	helper := "Implements"
	if cs.kind == cswTilde {
		helper = "TildeIs"
	}
	text := cat(gen(rtAlias+"."+helper+"["), tag, gen(", "), cs.spell, gen("]()"))
	expr, ok := parseExprAt(text.s, cs.syn.Case)
	if !ok {
		base.Fatalf("logos: const switch case %s at %s cannot be respelled", cs.spell.s, cs.syn.Case.String())
	}
	return expr, text
}

func (c *cswLower) caseBody(cs *cswCase, done string) ([]syntax.Stmt, otext) {
	if cs.kind == cswIface {
		c.wrapSelectors(cs.syn.Body, cs)
	} else if cs.sub.Target != nil {
		c.replaceTargetNames(cs.syn.Body, cs)
	}
	lowered := c.fc.lowerList(cs.syn.Body)
	var head []syntax.Stmt
	var headText otext
	if done != "" {
		head = append(head, at(&syntax.AssignStmt{Lhs: nameOf(cs.syn.Case, done), Rhs: nameOf(cs.syn.Case, "true")}, cs.syn.Case))
		headText = gen(done + " = true\n")
	}
	decls, declText := c.rebindings(cs)
	body := c.fl.textOf(span{cs.lo, cs.hi})
	stmts := append(head, decls...)
	return append(stmts, lowered...), cat(headText, declText, body)
}

func (c *cswLower) rebindings(cs *cswCase) ([]syntax.Stmt, otext) {
	var decls []syntax.Stmt
	var text otext
	for _, r := range c.rebound(cs) {
		c.fl.needRT = true
		rhs := rtAlias + ".Reinterpret[" + r.spell + "](&" + r.name + ")"
		expr, ok := parseExprAt(rhs, cs.syn.Case)
		if !ok {
			base.Fatalf("logos: const switch rebinding of %s at %s cannot be respelled", r.name, cs.syn.Case.String())
		}
		decls = append(decls, defineOf(cs.syn.Case, []syntax.Expr{nameOf(cs.syn.Case, r.name)}, []syntax.Expr{expr}))
		decls = append(decls, at(&syntax.AssignStmt{Lhs: nameOf(cs.syn.Case, "_"), Rhs: nameOf(cs.syn.Case, r.name)}, cs.syn.Case))
		text = cat(text, gen(r.name+" := "+rhs+"\n_ = "+r.name+"\n"))
	}
	return decls, text
}

func (c *cswLower) rebound(cs *cswCase) []cswRebind {
	var out []cswRebind
	seen := make(map[*types2.Var]bool)
	for _, s := range cs.syn.Body {
		syntax.Inspect(s, func(n syntax.Node) bool {
			name, ok := n.(*syntax.Name)
			if !ok {
				return true
			}
			v, ok := c.info.Uses[name].(*types2.Var)
			if !ok || seen[v] {
				return true
			}
			sub, ok := cs.sub.Vars[v]
			if !ok {
				return true
			}
			seen[v] = true
			out = append(out, cswRebind{name: v.Name(), spell: c.cswSpell(sub)})
			return true
		})
	}
	return out
}

func (c *cswLower) cswSpell(typ types2.Type) string {
	names := c.fl.importNames()
	return types2.TypeString(typ, func(p *types2.Package) string {
		if p == c.fl.pkg.tpkg {
			return ""
		}
		q, ok := names[p]
		if !ok {
			base.Fatalf("logos: const switch rebinding of type %s needs package %s, which the file does not import", types2.TypeString(typ, nil), p.Path())
		}
		return q
	})
}

func (c *cswLower) replaceTargetNames(body []syntax.Stmt, cs *cswCase) {
	spell := cs.spell
	w := cswWalker{
		info:   c.info,
		target: cs.sub.Target,
		repl: func(pos syntax.Pos) syntax.Expr {
			e, ok := parseExprAt(spell.s, pos)
			if !ok {
				base.Fatalf("logos: const switch case %s at %s cannot be respelled", spell.s, pos.String())
			}
			return e
		},
		off:    c.fl.off,
		splice: func(start, end int) { c.fl.addT(start, end, spell) },
	}
	w.list(body)
}

type cswWalker struct {
	info   *types2.Info
	target types2.Object
	repl   func(syntax.Pos) syntax.Expr
	off    func(syntax.Pos) int
	splice func(start, end int)
}

func (w cswWalker) list(stmts []syntax.Stmt) {
	for _, s := range stmts {
		w.node(s)
	}
}

func (w cswWalker) node(root syntax.Node) {
	syntax.Inspect(root, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		if sw, ok := n.(*syntax.ConstSwitchStmt); ok {
			for _, cc := range sw.Cases {
				w.list(cc.Body)
			}
			return false
		}
		w.replaceChildren(n)
		return true
	})
}

func (w cswWalker) replaceChildren(n syntax.Node) {
	v := reflect.ValueOf(n)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	for i := range v.NumField() {
		if v.Type().Field(i).IsExported() {
			w.replaceNames(v.Field(i))
		}
	}
}

func (w cswWalker) replaceNames(f reflect.Value) {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() || !f.CanSet() {
			return
		}
		if x, ok := f.Elem().Interface().(*syntax.Name); ok {
			w.replaceName(f, x)
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			w.replaceNames(f.Index(i))
		}
	}
}

func (w cswWalker) replaceName(f reflect.Value, x *syntax.Name) {
	if w.info.Uses[x] != w.target {
		return
	}
	start := w.off(x.Pos())
	w.splice(start, start+len(x.Value))
	f.Set(reflect.ValueOf(w.repl(x.Pos())))
}

func (c *cswLower) wrapSelectors(body []syntax.Stmt, cs *cswCase) {
	for _, s := range body {
		syntax.Inspect(s, func(n syntax.Node) bool {
			sel, ok := n.(*syntax.SelectorExpr)
			if !ok || c.info.LogosConstSel[sel] != cs.syn {
				return true
			}
			xpos := leftmostOf(sel.X)
			iface, ok := typeExprAt(cs.spell.s, xpos)
			if !ok {
				base.Fatalf("logos: const switch interface %s at %s cannot be respelled", cs.spell.s, xpos.String())
			}
			xstart := c.fl.off(xpos)
			xend := typeEnd(c.fl.src, xstart, c.fl.off(sel.Sel.Pos()))
			if xend > xstart && c.fl.src[xend-1] == '.' {
				xend--
			}
			xspan := span{xstart, xend}
			xtext := c.fl.textOf(xspan)
			sel.X = at(&syntax.AssertExpr{
				X: at(&syntax.CallExpr{
					Fun:     at(nameOf(xpos, "any"), xpos),
					ArgList: []syntax.Expr{sel.X},
				}, xpos),
				Type: iface,
			}, xpos)
			c.fl.addT(xspan.start, xspan.end, cat(gen("any("), xtext, gen(").("), cs.spell, gen(")")))
			return true
		})
	}
}

func parseExprAt(text string, pos syntax.Pos) (syntax.Expr, bool) {
	file, err := syntax.Parse(syntax.NewFileBase("logos-clone"), strings.NewReader("package logosclone\nvar _ = "+text), nil, nil, 0)
	if err != nil || file == nil || len(file.DeclList) != 1 {
		return nil, false
	}
	decl, ok := file.DeclList[0].(*syntax.VarDecl)
	if !ok || decl.Values == nil {
		return nil, false
	}
	syntax.Inspect(decl.Values, func(n syntax.Node) bool {
		if n != nil {
			n.SetPos(pos)
		}
		return true
	})
	return decl.Values, true
}
