// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"reflect"
	"slices"
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

	alias string // case-body alias for spell, immune to inner shadowing
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
	obj   *types2.Var
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
	var head []syntax.Stmt
	var headText otext
	if cs.kind == cswIface || cs.sub.Target != nil {
		// A type alias spelled at the top of the case body names the case
		// type: every respelled name resolves to it even inside blocks
		// that redeclare the case type's own name.
		cs.alias = c.fl.cswName("caseT")
		typ, ok := parseExprAt(cs.spell.s, cs.syn.Case)
		if !ok {
			base.Fatalf("logos: const switch case %s at %s cannot be respelled", cs.spell.s, cs.syn.Case.String())
		}
		head = append(head, at(&syntax.DeclStmt{DeclList: []syntax.Decl{
			at(&syntax.TypeDecl{Name: nameOf(cs.syn.Case, cs.alias), Alias: true, Type: typ}, cs.syn.Case),
		}}, cs.syn.Case))
		headText = gen("type " + cs.alias + " = " + cs.spell.s + "\n")
	}
	if cs.kind == cswIface {
		selWraps := c.wrapSelectors(cs.syn.Body, cs)
		decls, dtext := c.wrapConvs(cs.syn.Body, cs, selWraps)
		head = append(head, decls...)
		headText = cat(headText, dtext)
	} else if cs.sub.Target != nil {
		decls, dtext := c.rewriteRebound(cs)
		head = append(head, decls...)
		headText = cat(headText, dtext)
	}
	lowered := c.fc.lowerList(cs.syn.Body)
	if done != "" {
		head = append(head, at(&syntax.AssignStmt{Lhs: nameOf(cs.syn.Case, done), Rhs: nameOf(cs.syn.Case, "true")}, cs.syn.Case))
		headText = cat(headText, gen(done+" = true\n"))
	}
	body := c.fl.textOf(span{cs.lo, cs.hi})
	return append(head, lowered...), cat(headText, body)
}

// rewriteRebound respells the switched type's name to the case type and
// rewrites every use of a variable bound to the switched type as a
// dereference of a type-correct pointer into the variable's own storage,
// taken at the use site so the address is always the variable visible there:
// reads see the current value and writes land directly, on every exit path.
func (c *cswLower) rewriteRebound(cs *cswCase) ([]syntax.Stmt, otext) {
	var decls []syntax.Stmt
	var dtext otext
	targets := make(map[types2.Object]cswTarget)
	spell := gen(cs.alias) // the case-body alias declared by caseBody
	targets[cs.sub.Target] = cswTarget{
		repl: func(pos syntax.Pos) syntax.Expr {
			e, ok := parseExprAt(spell.s, pos)
			if !ok {
				base.Fatalf("logos: const switch case %s at %s cannot be respelled", spell.s, pos.String())
			}
			return e
		},
		splice: func(start, end int) { c.fl.addT(start, end, spell) },
	}
	// aliasFor emits a case-body type alias for a substituted spell that
	// differs from the case type: the spell may name a shadowable identifier.
	varAlias := map[string]string{cs.spell.s: cs.alias}
	aliasFor := func(spellText string) string {
		if a, ok := varAlias[spellText]; ok {
			return a
		}
		a := c.fl.cswName("caseV")
		varAlias[spellText] = a
		typ, ok := parseExprAt(spellText, cs.syn.Case)
		if !ok {
			base.Fatalf("logos: const switch rebind type %s at %s cannot be respelled", spellText, cs.syn.Case.String())
		}
		decls = append(decls, at(&syntax.DeclStmt{DeclList: []syntax.Decl{
			at(&syntax.TypeDecl{Name: nameOf(cs.syn.Case, a), Alias: true, Type: typ}, cs.syn.Case),
		}}, cs.syn.Case))
		dtext = cat(dtext, gen("type "+a+" = "+spellText+"\n"))
		return a
	}
	// len/cap of a rebound array variable is a constant expression; a
	// pointer into its storage is not. Rewrite such arguments to a nil
	// pointer dereference, which len/cap accept without evaluation. The
	// dereference spells the variable's own substituted type: it may carry
	// a different length than the case type.
	for _, s := range cs.syn.Body {
		syntax.Inspect(s, func(n syntax.Node) bool {
			call, ok := n.(*syntax.CallExpr)
			if !ok {
				return true
			}
			fn, ok := call.Fun.(*syntax.Name)
			if !ok || (fn.Value != "len" && fn.Value != "cap") {
				return true
			}
			if _, ok := c.info.Uses[fn].(*types2.Builtin); !ok {
				return true
			}
			if len(call.ArgList) != 1 {
				return true
			}
			arg, ok := call.ArgList[0].(*syntax.Name)
			if !ok {
				return true
			}
			v, ok := c.info.Uses[arg].(*types2.Var)
			if !ok {
				return true
			}
			sub, ok := cs.sub.Vars[v]
			if !ok {
				return true
			}
			if _, ok := types2.Unalias(sub).Underlying().(*types2.Array); !ok {
				return true
			}
			text := "(*(*" + aliasFor(c.cswSpell(sub)) + ")(nil))"
			e, ok := parseExprAt(text, arg.Pos())
			if !ok {
				base.Fatalf("logos: const switch len/cap argument at %s cannot be respelled", arg.Pos().String())
			}
			call.ArgList[0] = e
			c.fl.addT(c.fl.off(arg.Pos()), c.fl.off(arg.Pos())+len(arg.Value), gen(text))
			return true
		})
	}
	pun := "PtrAs"
	if cs.kind == cswTilde {
		// The rebound type keeps the layout but not the identity of the
		// type argument (Port vs int), so an identity assert cannot see
		// through it; the tilde match guarantees the layouts agree.
		pun = "UnderPtrAs"
	}
	for _, r := range c.rebound(cs) {
		c.fl.needRT = true
		// A derived substitution (*F -> *int) gets its own alias:
		// the name may spell a shadowable identifier.
		spell := aliasFor(r.spell)
		replText := "(*(" + rtAlias + "." + pun + "[" + spell + "](&" + r.name + ")))"
		targets[r.obj] = cswTarget{
			repl: func(pos syntax.Pos) syntax.Expr {
				e, ok := parseExprAt(replText, pos)
				if !ok {
					base.Fatalf("logos: const switch rebinding of %s at %s cannot be respelled", r.name, pos.String())
				}
				return e
			},
			splice: func(start, end int) { c.fl.addT(start, end, gen(replText)) },
		}
	}
	c.fixReboundDef(cs, targets)
	w := cswWalker{info: c.info, off: c.fl.off, targets: targets}
	w.list(cs.syn.Body)
	return decls, dtext
}

// fixReboundDef rewrites `x, y := ...` statements whose left side mixes new
// names with names that reuse a rebound variable. types2 records such a name
// as a use of the existing variable, so the rebind pass would respell it into
// a dereference — which := cannot carry. Each such statement becomes an
// ordinary assignment (a dereference is a legal = operand) and the genuinely
// new names are declared by var statements hoisted before the statement that
// heads the enclosing clause (the if/for/switch/select holding the :=).
func (c *cswLower) fixReboundDef(cs *cswCase, targets map[types2.Object]cswTarget) {
	parents := map[syntax.Node]*[]syntax.Stmt{}
	owner := map[syntax.Node]syntax.Node{}
	for i := range cs.syn.Body {
		parents[cs.syn.Body[i]] = &cs.syn.Body
	}
	var assigns []*syntax.AssignStmt
	var ranges []*syntax.RangeClause
	for _, s := range cs.syn.Body {
		syntax.Inspect(s, func(n syntax.Node) bool {
			switch t := n.(type) {
			case *syntax.BlockStmt:
				for i := range t.List {
					parents[t.List[i]] = &t.List
				}
			case *syntax.CaseClause:
				for i := range t.Body {
					parents[t.Body[i]] = &t.Body
				}
			case *syntax.CommClause:
				for i := range t.Body {
					parents[t.Body[i]] = &t.Body
				}
				if t.Comm != nil {
					owner[t.Comm] = t
				}
			case *syntax.IfStmt:
				if t.Init != nil {
					owner[t.Init] = t
				}
				if t.Then != nil {
					owner[t.Then] = t
				}
				if t.Else != nil {
					owner[t.Else] = t
				}
			case *syntax.ForStmt:
				if t.Init != nil {
					owner[t.Init] = t
				}
				if t.Post != nil {
					owner[t.Post] = t
				}
				if t.Body != nil {
					owner[t.Body] = t
				}
			case *syntax.SwitchStmt:
				if t.Init != nil {
					owner[t.Init] = t
				}
				for _, cc := range t.Body {
					owner[cc] = t
				}
			case *syntax.SelectStmt:
				for _, cc := range t.Body {
					owner[cc] = t
				}
			case *syntax.LabeledStmt:
				if t.Stmt != nil {
					owner[t.Stmt] = t
				}
			case *syntax.FuncLit:
				if t.Body != nil {
					owner[t.Body] = t
				}
			case *syntax.AssignStmt:
				if t.Op == syntax.Def && c.reboundLhs(t.Lhs, cs) {
					assigns = append(assigns, t)
				}
			case *syntax.RangeClause:
				if t.Def && t.Lhs != nil && c.reboundLhs(t.Lhs, cs) {
					ranges = append(ranges, t)
				}
			}
			return true
		})
	}
	if len(assigns) == 0 && len(ranges) == 0 {
		return
	}

	hoist := func(head syntax.Node, n *syntax.Name) {
		for owner[head] != nil {
			head = owner[head]
		}
		pl, ok := parents[head]
		if !ok {
			return
		}
		nv, ok := c.info.Defs[n].(*types2.Var)
		if !ok {
			return
		}
		// The declaration always hoists under a fresh temp: a same-named var
		// placed before the statement would shadow an outer variable that the
		// initializer itself may read.
		orig := n.Value
		name := c.fl.cswName("rv")
		pos := n.Pos()
		text := gen(name)
		targets[nv] = cswTarget{
			repl: func(atp syntax.Pos) syntax.Expr {
				e, ok := parseExprAt(text.s, atp)
				if !ok {
					base.Fatalf("logos: const switch temp name at %s cannot be respelled", atp.String())
				}
				return e
			},
			splice: func(start, end int) { c.fl.addT(start, end, text) },
		}
		n.Value = name
		c.fl.addT(c.fl.off(pos), c.fl.off(pos)+len(orig), text)
		spell := c.cswSpell(nv.Type())
		typ, ok := typeExprAt(spell, n.Pos())
		if !ok {
			base.Fatalf("logos: const switch hoisted type %s at %s cannot be respelled", spell, n.Pos().String())
		}
		decl := at(&syntax.DeclStmt{DeclList: []syntax.Decl{
			at(&syntax.VarDecl{NameList: []*syntax.Name{nameOf(n.Pos(), name)}, Type: typ}, n.Pos()),
		}}, n.Pos())
		idx := -1
		for i, s := range *pl {
			if s == head {
				idx = i
				break
			}
		}
		if idx < 0 {
			return
		}
		*pl = append(*pl, nil)
		copy((*pl)[idx+1:], (*pl)[idx:])
		(*pl)[idx] = decl
		// An AssignStmt's Pos is its operator token, not its left side:
		// splice before the statement's first token instead.
		start := head.Pos()
		if as, ok := head.(*syntax.AssignStmt); ok {
			start = as.Lhs.Pos()
		}
		c.fl.addT(c.fl.off(start), c.fl.off(start), gen("var "+name+" "+spell+"\n"))
	}

	defToAssign := func(lo, hi int) {
		i := strings.IndexByte(string(c.fl.src[lo:hi]), ':')
		if i >= 0 {
			c.fl.addT(lo+i, lo+i+1, gen(""))
		}
	}
	for _, as := range assigns {
		// Take the left side's end and the initializer's start before hoist
		// renames any declared name: EndPos follows the mutated Value length.
		lo := c.fl.off(syntax.EndPos(as.Lhs))
		hi := c.fl.off(as.Rhs.Pos())
		for _, n := range lhsNames(as.Lhs) {
			if c.info.Defs[n] != nil {
				hoist(as, n)
			}
		}
		as.Op = 0
		defToAssign(lo, hi)
	}
	for _, rc := range ranges {
		lo := c.fl.off(syntax.EndPos(rc.Lhs))
		hi := c.fl.off(rc.X.Pos())
		for _, n := range lhsNames(rc.Lhs) {
			if c.info.Defs[n] != nil {
				hoist(rc, n)
			}
		}
		rc.Def = false
		defToAssign(lo, hi)
	}
}

// reboundLhs reports whether a := left side reuses at least one variable the
// case rebinds: the checker put it in Uses, not Defs.
func (c *cswLower) reboundLhs(lhs syntax.Expr, cs *cswCase) bool {
	for _, n := range lhsNames(lhs) {
		if c.info.Defs[n] != nil {
			continue
		}
		if v, ok := c.info.Uses[n].(*types2.Var); ok {
			if _, ok := cs.sub.Vars[v]; ok {
				return true
			}
		}
	}
	return false
}

func lhsNames(e syntax.Expr) []*syntax.Name {
	if l, ok := e.(*syntax.ListExpr); ok {
		var out []*syntax.Name
		for _, x := range l.ElemList {
			if n, ok := x.(*syntax.Name); ok {
				out = append(out, n)
			}
		}
		return out
	}
	if n, ok := e.(*syntax.Name); ok {
		return []*syntax.Name{n}
	}
	return nil
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
			out = append(out, cswRebind{obj: v, name: v.Name(), spell: c.cswSpell(sub)})
			return true
		})
	}
	return out
}

func (c *cswLower) cswSpell(typ types2.Type) string {
	return c.fl.typeSpell(typ)
}

// cswTarget is one name the walker rewrites: the AST replacement for the
// node and the source-text splice for the emitted file.
type cswTarget struct {
	repl   func(syntax.Pos) syntax.Expr
	splice func(start, end int)
}

type cswWalker struct {
	info    *types2.Info
	off     func(syntax.Pos) int
	targets map[types2.Object]cswTarget
	conv    map[syntax.Expr]syntax.Expr
	wrap    func(x, r syntax.Expr)
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
		x, ok := f.Elem().Interface().(syntax.Expr)
		if !ok {
			return
		}
		if r, ok := w.conv[x]; ok {
			delete(w.conv, x) // the replacement embeds x; consume the entry so it is not wrapped again
			if w.wrap != nil {
				w.wrap(x, r)
			}
			f.Set(reflect.ValueOf(r))
			return
		}
		if name, ok := x.(*syntax.Name); ok {
			w.replaceName(f, name)
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			w.replaceNames(f.Index(i))
		}
	}
}

func (w cswWalker) replaceName(f reflect.Value, x *syntax.Name) {
	if w.info.Defs[x] != nil {
		return // a declaration site (e.g. `v, n := f()`) declares a new name
	}
	t, ok := w.targets[w.info.Uses[x]]
	if !ok {
		return
	}
	t.splice(w.off(x.Pos()), w.off(x.Pos())+len(x.Value))
	f.Set(reflect.ValueOf(t.repl(x.Pos())))
}

// wrapConvs boxes each operand the checker recorded under the interface case
// as rt.Box[I](e): the value keeps its concrete layout at storage sites and
// widens to the case interface where the checker admitted it (SPEC Sec 2.5).
// A recorded expression nested inside another one is composed inside the
// outer replacement so the splices never overlap.
func (c *cswLower) wrapConvs(body []syntax.Stmt, cs *cswCase, selWraps []span) ([]syntax.Stmt, otext) {
	var decls []syntax.Stmt
	var dtext otext
	type rec struct {
		e          syntax.Expr
		start, end int
	}
	var recs []rec // Inspect is a preorder walk, so recs arrive in source order.
	// A recorded operand in a receive clause cannot be boxed in place — a
	// comm clause admits only a receive, never a call — so those operands
	// are respelled by the clause rewrite at the end instead. A send's
	// value is an ordinary operand and boxes in place.
	clauseOf := make(map[syntax.Expr]*syntax.CommClause)
	for _, s := range body {
		syntax.Inspect(s, func(n syntax.Node) bool {
			if cc, ok := n.(*syntax.CommClause); ok && cc.Comm != nil {
				pinned := []syntax.Node{cc.Comm}
				if send, ok := cc.Comm.(*syntax.SendStmt); ok {
					pinned = []syntax.Node{send.Chan}
				}
				for _, root := range pinned {
					syntax.Inspect(root, func(m syntax.Node) bool {
						if e, ok := m.(syntax.Expr); ok {
							clauseOf[e] = cc
						}
						return true
					})
				}
			}
			e, ok := n.(syntax.Expr)
			if !ok || c.info.LogosConstConv[e] != cs.syn {
				return true
			}
			start := c.fl.off(leftmostOf(e))
			recs = append(recs, rec{e: e, start: start, end: c.fl.exprEndNested(start)})
			return true
		})
	}
	if len(recs) == 0 {
		return decls, dtext
	}
	c.fl.needRT = true
	recOf := make(map[syntax.Expr]rec, len(recs))
	for _, r := range recs {
		recOf[r.e] = r
	}
	// localDecls maps the name position of each type declared inside the
	// case body to the offset just past its declaration and to the list
	// holding it. A spell that resolves to such a type can't go in a
	// case-top alias — the alias would precede the declaration — and a
	// bare spell at the use site could re-resolve to a nested block's
	// shadow of the name. Its alias is instead spliced right after the
	// declaration it spells: after the name is in scope, before every
	// use that the declaration permits, and past any shadow of either
	// name.
	type localTypeDecl struct {
		end  int
		list *[]syntax.Stmt
		stmt *syntax.DeclStmt
	}
	localDecls := map[int]localTypeDecl{}
	var scanTypeDecls func(list *[]syntax.Stmt)
	scanTypeDecls = func(list *[]syntax.Stmt) {
		for _, s := range *list {
			if d, ok := s.(*syntax.DeclStmt); ok {
				// stmtEnd scans past balanced brackets to the declaration's
				// line end: EndPos stops at a grouped declaration's closing
				// token or a generic alias's last type argument on some
				// shapes, and exprEndNested stops at an alias `=` — the later
				// offset is the one past the whole declaration.
				end := max(stmtEnd(c.fl.src, c.fl.off(syntax.StartPos(d))), c.fl.off(syntax.EndPos(d)))
				for _, decl := range d.DeclList {
					if td, ok := decl.(*syntax.TypeDecl); ok && td.Name != nil {
						localDecls[c.fl.off(td.Name.Pos())] = localTypeDecl{end, list, d}
					}
				}
			}
			syntax.Inspect(s, func(n syntax.Node) bool {
				switch x := n.(type) {
				case *syntax.BlockStmt:
					scanTypeDecls(&x.List)
				case *syntax.CaseClause:
					scanTypeDecls(&x.Body)
				case *syntax.CommClause:
					scanTypeDecls(&x.Body)
				default:
					return true
				}
				return false
			})
		}
	}
	scanTypeDecls(&cs.syn.Body)
	// localTypeOf reports the last case-body declaration that any object
	// inside t names, by object identity — a same-named declaration of
	// another scope never matches, so a package type shadowed inside the
	// case body still gets the ordinary case-top alias.
	localTypeOf := func(t types2.Type) (localTypeDecl, bool) {
		var out localTypeDecl
		found := false
		seen := map[types2.Type]bool{}
		var walk func(t types2.Type)
		walk = func(t types2.Type) {
			if t == nil || seen[t] {
				return
			}
			seen[t] = true
			var obj *types2.TypeName
			switch x := t.(type) {
			case *types2.Alias:
				obj = x.Obj()
				if args := x.TypeArgs(); args != nil {
					for i := 0; i < args.Len(); i++ {
						walk(args.At(i))
					}
				}
			case *types2.Named:
				obj = x.Obj()
				if args := x.TypeArgs(); args != nil {
					for i := 0; i < args.Len(); i++ {
						walk(args.At(i))
					}
				}
			case *types2.Pointer:
				walk(x.Elem())
			case *types2.Slice:
				walk(x.Elem())
			case *types2.Array:
				walk(x.Elem())
			case *types2.Map:
				walk(x.Key())
				walk(x.Elem())
			case *types2.Chan:
				walk(x.Elem())
			case *types2.Struct:
				for i := 0; i < x.NumFields(); i++ {
					walk(x.Field(i).Type())
				}
			case *types2.Tuple:
				for i := 0; i < x.Len(); i++ {
					walk(x.At(i).Type())
				}
			case *types2.Signature:
				walk(x.Params())
				walk(x.Results())
			case *types2.Interface:
				for i := 0; i < x.NumEmbeddeds(); i++ {
					walk(x.EmbeddedType(i))
				}
				for i := 0; i < x.NumMethods(); i++ {
					walk(x.Method(i).Type())
				}
			}
			if obj != nil && obj.Pos().IsKnown() &&
				obj.Pos().FileBase() == c.fl.file.Pos().FileBase() {
				if lt, ok := localDecls[c.fl.off(obj.Pos())]; ok && (!found || lt.end > out.end) {
					out, found = lt, true
				}
			}
		}
		walk(t)
		return out, found
	}
	// spellFor names each destination type through a case-body alias: a
	// declared or inferred type may spell an identifier a nested block
	// shadows, and aliases mint lazily so assignments needing no rewrite
	// never spell an unimported type. A spell naming a type declared
	// inside the case body gets its alias right after that declaration
	// instead of at the case top.
	// Pinned aliases are emitted after the body walk: inserting a node
	// into a statement list mid-walk shifts the slice the walk iterates.
	type pinnedAlias struct {
		d    *syntax.DeclStmt
		list *[]syntax.Stmt
		stmt *syntax.DeclStmt
		end  int
		text string
	}
	var pinned []pinnedAlias
	// secAlias dedupes by type, not spell: two types can spell the same
	// name when a nested declaration shadows a package name.
	secAlias := map[types2.Type]string{}
	spellFor := func(t types2.Type) string {
		s := c.cswSpell(t)
		if a, ok := secAlias[t]; ok {
			return a
		}
		a := c.fl.cswName("caseV")
		secAlias[t] = a
		typ, ok := parseExprAt(s, cs.syn.Case)
		if !ok {
			base.Fatalf("logos: const switch conv type %s at %s cannot be respelled", s, cs.syn.Case.String())
		}
		d := at(&syntax.DeclStmt{DeclList: []syntax.Decl{
			at(&syntax.TypeDecl{Name: nameOf(cs.syn.Case, a), Alias: true, Type: typ}, cs.syn.Case),
		}}, cs.syn.Case)
		if lt, ok := localTypeOf(t); ok {
			pinned = append(pinned, pinnedAlias{d, lt.list, lt.stmt, lt.end, "\ntype " + a + " = " + s})
			return a
		}
		decls = append(decls, d)
		dtext = cat(dtext, gen("type "+a+" = "+s+"\n"))
		return a
	}
	box := func(text string) string {
		return rtAlias + ".Box[" + cs.alias + "](" + text + ")"
	}
	secondOf := func(e syntax.Expr) (types2.Type, bool) {
		switch x := syntax.Unparen(e).(type) {
		case *syntax.IndexExpr, *syntax.AssertExpr:
			return types2.Typ[types2.Bool], true
		case *syntax.Operation:
			if x.Op == syntax.Recv {
				return types2.Typ[types2.Bool], true
			}
		}
		if tup, ok := c.info.TypeOf(e).(*types2.Tuple); ok && tup.Len() == 2 {
			return tup.At(1).Type(), true
		}
		return nil, false
	}
	// `return e` is multi-result only when the innermost enclosing
	// function's signature takes two results: a literal inside a
	// two-result declaration has its own arity, and vice versa.
	type fnSpan struct {
		start, end int
		ft         *syntax.FuncType
	}
	var spans []fnSpan
	for _, s := range body {
		syntax.Inspect(s, func(n syntax.Node) bool {
			if lit, ok := n.(*syntax.FuncLit); ok {
				spans = append(spans, fnSpan{c.fl.off(lit.Pos()), c.fl.off(syntax.EndPos(lit)), lit.Type})
			}
			return true
		})
	}
	enclosing := func(pos int) *syntax.FuncType {
		best := c.fc.ft
		bestSize := int(^uint(0) >> 1)
		for _, s := range spans {
			if s.start <= pos && pos < s.end && s.end-s.start < bestSize {
				best, bestSize = s.ft, s.end-s.start
			}
		}
		return best
	}
	// A recorded operand in multi-result position — the sole right side of
	// a two-result assignment, declaration, or return — keeps its arity
	// through a two-result closure that boxes the first value and converts
	// the second to the left side's declared type: `p, ok = m[k]` becomes
	// `p, ok = func() (I, bool) { v, ok := m[k]; return Box[I](v), ok }()`.
	// The second type stays a types2.Type until render actually needs it:
	// an inferred type that spells a package the file does not import
	// aborts cswSpell, and assignments needing no rewrite must never ask.
	multi := make(map[syntax.Expr]types2.Type)
	for _, s := range body {
		syntax.Inspect(s, func(n syntax.Node) bool {
			var rhs syntax.Expr
			var secondT types2.Type
			two := false
			switch st := n.(type) {
			case *syntax.AssignStmt:
				rhs = st.Rhs
				if lhs, ok := st.Lhs.(*syntax.ListExpr); ok && len(lhs.ElemList) == 2 {
					if t := c.info.TypeOf(lhs.ElemList[1]); t != nil {
						two = true
						secondT = t
					}
				}
			case *syntax.VarDecl:
				rhs = st.Values
				if len(st.NameList) == 2 {
					if t := c.info.TypeOf(st.NameList[1]); t != nil {
						two = true
						secondT = t
					}
				}
			case *syntax.ReturnStmt:
				rhs = st.Results
				if ft := enclosing(c.fl.off(st.Pos())); ft != nil && len(ft.ResultList) == 2 {
					if t := c.info.TypeOf(ft.ResultList[1].Type); t != nil {
						two = true
						secondT = t
					}
				}
			}
			if two && rhs != nil && c.info.LogosConstConv[rhs] == cs.syn {
				if secondT == nil {
					secondT, two = secondOf(rhs)
				}
				if !two {
					base.Fatalf("logos: const switch boxing of a multi-result at %s cannot name its second type", rhs.Pos().String())
				}
				multi[rhs] = secondT
			}
			return true
		})
	}
	// expand renders the span's source text with every recorded expression
	// inside it boxed; nested recordings compose inside their outer boxes.
	var expand func(start, end int) string
	// render emits the operand's source text boxed, or wrapped in a
	// two-result closure when the operand keeps a comma-ok arity.
	render := func(r rec) string {
		if secondT, ok := multi[r.e]; ok {
			second := spellFor(secondT)
			return "func() (" + cs.alias + ", " + second + ") { _logos_v, _logos_s := " + expand(r.start, r.end) + "; return " + box("_logos_v") + ", " + second + "(_logos_s) }()"
		}
		return box(expand(r.start, r.end))
	}
	expand = func(start, end int) string {
		var inner []rec
		for _, r := range recs {
			if r.start >= start && r.end <= end && (r.start != start || r.end != end) {
				inner = append(inner, r)
			}
		}
		var out strings.Builder
		pos := start
		for _, r := range inner {
			nested := false
			for _, o := range inner {
				if o != r && o.start <= r.start && r.end <= o.end {
					nested = true
					break
				}
			}
			if nested {
				continue
			}
			if clauseOf[r.e] != nil {
				base.Fatalf("logos: const switch select receive inside a boxed operand at %s is not supported", r.e.Pos().String())
			}
			if r.start > pos {
				out.WriteString(c.fl.textOf(span{pos, r.start}).s)
			}
			out.WriteString(render(r))
			pos = r.end
		}
		if pos < end {
			out.WriteString(c.fl.textOf(span{pos, end}).s)
		}
		return out.String()
	}
	conv := make(map[syntax.Expr]syntax.Expr)
	texts := make(map[syntax.Expr]otext)
	var clauses []*syntax.CommClause
	clauseSeen := make(map[*syntax.CommClause]bool)
	for _, r := range recs {
		if cc := clauseOf[r.e]; cc != nil {
			if !clauseSeen[cc] {
				clauseSeen[cc] = true
				clauses = append(clauses, cc)
			}
			continue
		}
		contained := false
		for _, o := range recs {
			if o != r && clauseOf[o.e] == nil && o.start <= r.start && r.end <= o.end {
				contained = true
				break
			}
		}
		if contained {
			continue // emitted inside the outer box's composed text
		}
		text := render(r)
		node, ok := parseExprAt(text, r.e.Pos())
		if !ok {
			base.Fatalf("logos: const switch interface boxing at %s cannot be respelled", r.e.Pos().String())
		}
		conv[r.e] = node
		texts[r.e] = gen(text)
	}
	// `p, ok = <-ch` inside a select cannot name the box call in the comm
	// clause: receive into fresh variables and assign the boxed value in
	// the clause body.
	for _, cc := range clauses {
		as, ok := cc.Comm.(*syntax.AssignStmt)
		if !ok || as.Op != 0 {
			continue // a bare `case <-ch` discards the value; `:=` needs no box
		}
		var elems []syntax.Expr
		if le, ok := as.Lhs.(*syntax.ListExpr); ok {
			elems = le.ElemList
		} else if as.Lhs != nil {
			elems = []syntax.Expr{as.Lhs}
		}
		var temps []syntax.Expr
		var tempNames []string
		svbase := c.fl.cswName("sv")
		for i, l := range elems {
			t := fmt.Sprintf("%s_%d", svbase, i)
			temps = append(temps, nameOf(l.Pos(), t))
			tempNames = append(tempNames, t)
		}
		if len(temps) == 1 {
			as.Lhs = temps[0]
		} else {
			as.Lhs = at(&syntax.ListExpr{ElemList: temps}, as.Lhs.Pos())
		}
		as.Op = syntax.Def
		// A multi-target receive assigns in phases: index operands on the
		// left are evaluated before any store. The split assigns below run
		// in turn, so freeze those reads into temps first — an earlier
		// target may be the variable an index operand reads.
		var freezeStmts []syntax.Stmt
		var freezeText strings.Builder
		freezeN := 0
		// lhsSubs records the span of each frozen operand inside its left-hand
		// element so the emitted store text names the temp; the left side's own
		// span is replaced wholesale by the receive temps, so no source splice
		// may cover it.
		type lhsSub struct {
			start, end int
			name       string
		}
		lhsSubs := map[syntax.Expr][]lhsSub{}
		// freeze emits `t := <src of [start,end)>` early.
		freeze := func(e syntax.Expr, start, end int, t string) {
			var src string
			if r, ok := recOf[e]; ok {
				src = render(r)
			} else {
				src = expand(start, end)
			}
			rhs, ok := parseExprAt(src, e.Pos())
			if !ok {
				base.Fatalf("logos: const switch select target temp at %s cannot be respelled", e.Pos().String())
			}
			freezeStmts = append(freezeStmts, at(&syntax.AssignStmt{Op: syntax.Def, Lhs: nameOf(e.Pos(), t), Rhs: rhs}, e.Pos()))
			freezeText.WriteString("\n" + t + " := " + src)
		}
		if len(elems) > 1 {
			for _, l := range elems {
				// Candidates are collected as the walk goes so a frozen X's
				// replacement name prunes the reads nested inside it; the
				// statements are emitted in source order afterwards because
				// a left-hand operand evaluates left to right (m[i()][j()]
				// reads i() before j()).
				type cand struct {
					e          syntax.Expr
					start, end int
					name       string
				}
				var cands []cand
				var visit func(syntax.Node) bool
				visit = func(n syntax.Node) bool {
					ix, ok := n.(*syntax.IndexExpr)
					if !ok {
						return true
					}
					if ix.X != nil {
						frozen := false
						if _, ok := recOf[ix.X]; ok {
							// A recorded conversion inside a left-hand target must be
							// detached here: the clause's left side is replaced by
							// receive temps, so the use-site splice would overlap it.
							frozen = true
						} else if u := c.info.TypeOf(ix.X); u != nil {
							switch types2.Unalias(u).Underlying().(type) {
							case *types2.Slice, *types2.Map:
								// A slice or map read yields a header value that
								// must be taken early; an array or pointer base is
								// an address read and stays stable.
								frozen = true
							}
						}
						if frozen {
							// X ends where the parent's '[' opens; exprEndNested
							// would swallow the index brackets.
							st := c.fl.off(leftmostOf(ix.X))
							en := c.fl.off(ix.Pos())
							t := fmt.Sprintf("%s_lv%d", svbase, freezeN)
							freezeN++
							cands = append(cands, cand{ix.X, st, en, t})
							lhsSubs[l] = append(lhsSubs[l], lhsSub{st, en, t})
							// The temp takes X's leftmost position so the
							// element's recomputed start still reaches it —
							// a selector's own Pos sits on the field name.
							ix.X = nameOf(leftmostOf(ix.X), t)
						}
					}
					if ix.Index != nil {
						// Literals, constants, and type arguments are not
						// runtime reads: leaving them keeps the map's own
						// key conversion and never freezes a type as a value.
						idx := syntax.Unparen(ix.Index)
						_, lit := idx.(*syntax.BasicLit)
						tv := c.info.Types[idx]
						freezable := !lit && !tv.IsType() && tv.Value == nil
						if freezable {
							if name, ok := idx.(*syntax.Name); ok {
								switch c.info.ObjectOf(name).(type) {
								case *types2.TypeName, *types2.Const:
									freezable = false
								}
							}
						}
						if freezable {
							// The index operand ends where the parent's ']'
							// closes at nesting depth zero.
							st := c.fl.off(leftmostOf(ix.Index))
							en := c.fl.exprEndNested(st)
							t := fmt.Sprintf("%s_lv%d", svbase, freezeN)
							freezeN++
							cands = append(cands, cand{ix.Index, st, en, t})
							lhsSubs[l] = append(lhsSubs[l], lhsSub{st, en, t})
							ix.Index = nameOf(leftmostOf(ix.Index), t)
						}
					}
					// Keep descending into X: an unfrozen X on the address path
					// may hold another index whose operand still needs freezing,
					// and a frozen X was already replaced by its temp name. A
					// skipped index operand is a compile-time expression whose
					// calls never run — unsafe.Sizeof does not evaluate its
					// argument — so the walk must not descend into it.
					syntax.Inspect(ix.X, visit)
					return false
				}
				syntax.Inspect(l, visit)
				slices.SortFunc(cands, func(a, b cand) int { return a.start - b.start })
				for _, cd := range cands {
					freeze(cd.e, cd.start, cd.end, cd.name)
				}
			}
		}
		var assigns []syntax.Stmt
		var texts strings.Builder
		for i, l := range elems {
			// l.Pos() is not the element's leftmost byte for every shape
			// (IndexExpr.Pos() is the '['), and syntax.EndPos follows the
			// mutated children, so take the span from the source: leftmost
			// token through the expression's own nesting.
			lstart := c.fl.off(leftmostOf(l))
			lend := c.fl.exprEndNested(lstart)
			lhsText := c.fl.srcUser(lstart, lend)
			subs := lhsSubs[l]
			// A matched-selector rewrite inside a surviving part of the
			// target folds into the store text here: leaving it as a splice
			// would overlap the receive-target replacement. A rewrite
			// inside a frozen operand's span already reached the emitted
			// temp through expand's textOf.
			for _, w := range selWraps {
				if w.start < lstart || w.end > lend {
					continue
				}
				inside := false
				var inner []lhsSub
				for _, s := range subs {
					if w.start >= s.start && w.end <= s.end {
						inside = true
						break
					}
					if w.start <= s.start && s.end <= w.end {
						inner = append(inner, s)
					}
				}
				if inside {
					continue
				}
				if len(inner) == 0 {
					subs = append(subs, lhsSub{w.start, w.end, c.fl.textOf(w).s})
					continue
				}
				// The wrapped receiver holds frozen operands: drain the
				// pending splice and rebuild the cast around the text the
				// inner subs spell, so the temps — not the original operands
				// — appear inside any(...). Each gap between subs is read
				// through textOf so a splice recorded inside it — an option
				// fallback, a conversion — survives in its lowered form
				// instead of leaking logos-only syntax into the store text.
				c.fl.textOf(w)
				slices.SortFunc(inner, func(a, b lhsSub) int { return a.start - b.start })
				innerText := ""
				cur := w.start
				for _, s := range inner {
					innerText += c.fl.textOf(span{cur, s.start}).s + s.name
					cur = s.end
				}
				innerText += c.fl.textOf(span{cur, w.end}).s
				kept := subs[:0]
				for _, s := range subs {
					if !(w.start <= s.start && s.end <= w.end) {
						kept = append(kept, s)
					}
				}
				subs = append(kept, lhsSub{w.start, w.end, "any(" + innerText + ").(" + cs.alias + ")"})
			}
			// Other pending splices inside the target — an option
			// fallback's `??` lowering — would overlap the receive
			// replacement that covers it, so drain and fold them into
			// the store text. Inside a covering splice, a frozen
			// operand's span still spells its temp: the splice's text
			// holds the original expression, which would be evaluated a
			// second time.
			for _, sp := range c.fl.drainInner(span{lstart, lend}) {
				text := sp.text
				var kept []lhsSub
				for _, s := range subs {
					if sp.start <= s.start && s.end <= sp.end {
						text = text[:s.start-sp.start] + s.name + text[s.end-sp.start:]
						continue
					}
					kept = append(kept, s)
				}
				subs = append(kept, lhsSub{sp.start, sp.end, text})
			}
			slices.SortFunc(subs, func(a, b lhsSub) int { return b.start - a.start })
			for _, s := range subs {
				if s.start-lstart < 0 || s.end-lstart > len(lhsText.s) {
					base.Fatalf("logos: lhs sub [%d,%d) outside element %T span [%d,%d) text %q", s.start, s.end, l, lstart, lstart+len(lhsText.s), lhsText.s)
				}
				lhsText.s = lhsText.s[:s.start-lstart] + s.name + lhsText.s[s.end-lstart:]
			}
			rhsText := tempNames[i]
			var rhs syntax.Expr
			if i == 0 {
				rhsText = box(rhsText)
			} else if t := c.info.TypeOf(l); t != nil {
				rhsText = "(" + spellFor(t) + ")(" + rhsText + ")"
			}
			rhs, _ = parseExprAt(rhsText, l.Pos())
			assigns = append(assigns, at(&syntax.AssignStmt{Lhs: l, Rhs: rhs}, l.Pos()))
			texts.WriteString("\n" + lhsText.s + " = " + rhsText)
		}
		cc.Body = append(append(freezeStmts, assigns...), cc.Body...)
		lhsStart := c.fl.off(leftmostOf(as.Lhs))
		recvStart := c.fl.off(leftmostOf(as.Rhs))
		// The temps declaration replaces the whole receive target,
		// including the gaps between elements that no per-element
		// drain reaches (a wrap crossing an element boundary). Any
		// splice still inside the span is dead text.
		c.fl.dropInner(span{lhsStart, recvStart})
		var tempsText strings.Builder
		for i, t := range tempNames {
			if i > 0 {
				tempsText.WriteString(", ")
			}
			tempsText.WriteString(t)
		}
		tempsText.WriteString(" := ")
		c.fl.addT(lhsStart, recvStart, gen(tempsText.String()))
		texts.WriteString("\n")
		c.fl.addT(c.fl.off(cc.Colon)+1, c.fl.off(cc.Colon)+1, gen(freezeText.String()+texts.String()))
	}
	// Pinned aliases are emitted after the store walk: inserting a node
	// into a statement list mid-walk shifts the slice the walk iterates.
	emitPinned := func() {
		for _, p := range pinned {
			c.fl.addT(p.end, p.end, gen(p.text))
			if i := slices.Index(*p.list, syntax.Stmt(p.stmt)); i >= 0 {
				*p.list = slices.Insert(*p.list, i+1, syntax.Stmt(p.d))
			}
		}
	}
	if len(conv) == 0 {
		emitPinned()
		return decls, dtext
	}
	w := cswWalker{
		info: c.info,
		off:  c.fl.off,
		conv: conv,
		wrap: func(x, r syntax.Expr) {
			start := c.fl.off(leftmostOf(x))
			c.fl.addT(start, c.fl.exprEndNested(start), texts[x])
		},
	}
	w.list(body)
	emitPinned()
	return decls, dtext
}

// wrapSelectors rewrites a matched selector's receiver to
// any(x).(caseT) and returns each replaced receiver span so the
// select-receive rewrite can fold the same edit into a replaced target's
// store text instead of overlapping it.
func (c *cswLower) wrapSelectors(body []syntax.Stmt, cs *cswCase) []span {
	var wraps []span
	for _, s := range body {
		syntax.Inspect(s, func(n syntax.Node) bool {
			sel, ok := n.(*syntax.SelectorExpr)
			if !ok || c.info.LogosConstSel[sel] != cs.syn {
				return true
			}
			xpos := leftmostOf(sel.X)
			iface, ok := typeExprAt(cs.alias, xpos)
			if !ok {
				base.Fatalf("logos: const switch interface %s at %s cannot be respelled", cs.alias, xpos.String())
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
			c.fl.addT(xspan.start, xspan.end, cat(gen("any("), xtext, gen(").("), gen(cs.alias), gen(")")))
			wraps = append(wraps, xspan)
			return true
		})
	}
	return wraps
}

func parseExprAt(text string, pos syntax.Pos) (syntax.Expr, bool) {
	file, err := syntax.Parse(syntax.NewFileBase("logos-clone"), strings.NewReader("package logosclone\nvar _ = "+text), nil, nil, syntax.LogosMode)
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
