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

// This file lowers M7-S4 `const switch` statements (SPEC Sec 2.5, T-205).
// A switch over a concrete type folds to its first matching case, placed in
// a block. A switch over a type parameter lowers to the PLAN Sec 7.2
// run-time form: a type switch on any((*T)(nil)) whose clauses are the exact
// cases, with a default clause holding an if chain of the tilde and interface
// cases.
//
// The clauses keep source order. When every exact case precedes every other
// case, that is the single form above. Otherwise the cases split into maximal
// runs: each run of exact cases becomes its own type switch, each run of the
// other cases becomes its own if chain, and a flag records the matched case so
// that later runs are skipped. Source order is kept in every form, because the
// lowered tree's scope positions must not decrease (DWARF scope markers).
//
// Each statement is replaced by one splice whose text is printed from the
// same clause list as the lowered tree, so the ejected text and the lowered
// tree agree (A2). A case body's own lowering records its splices first; the
// case-scope rewrites run before that lowering, and the body text is read
// last, with every splice applied once.

// cswKind classifies a case of a `const switch`.
type cswKind int

const (
	cswExact cswKind = iota
	cswTilde
	cswIface
)

// cswCase is one case of the statement being lowered.
type cswCase struct {
	syn  *syntax.ConstSwitchCase
	kind cswKind
	typ  types2.Type

	// spell is the source spelling of the case type.
	spell string

	// lo and hi are the byte offsets where the clause body starts and ends.
	lo, hi int

	// close is where the clause's scope closes: the next case, or the
	// statement's brace.
	close syntax.Pos

	sub types2.LogosConstCase
}

// cswLower carries the state of lowering one `const switch` statement.
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

// cswRebind is one variable rebound to a case type at the start of a body.
type cswRebind struct {
	name  string
	spell string
}

// lowerConstSwitchStmt lowers one `const switch` statement, returning the
// statements that replace it.
func (fc *funcLower) lowerConstSwitchStmt(s *syntax.ConstSwitchStmt) []syntax.Stmt {
	c := newCswLower(fc, s)
	if _, ok := c.tag.(*types2.TypeParam); ok {
		return c.lowerRuntime()
	}
	return c.lowerFold()
}

// newCswLower reads the checked tag and case types and the byte ranges of
// the clause bodies.
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
		case isIfaceLower(typ):
			cs.kind = cswIface
		}
		c.cases = append(c.cases, cs)
	}
	return c
}

// checkedTypeOf returns the checked type denoted by the type expression e. A
// name resolves through its declared object, because the checker records the
// object and not the type of a type name.
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

// isIfaceLower reports whether typ is an interface type.
func isIfaceLower(typ types2.Type) bool {
	_, ok := types2.Unalias(typ).Underlying().(*types2.Interface)
	return ok
}

// spell returns the source text of the switched type between the header at
// from and bound, with any splices already recorded inside it applied.
// from is the header keyword (the case, or the const of the switch) and
// bound is the token that ends the type (the case colon, or the switch
// brace for the tag), because EndPos stops early on composite types. The
// type starts at the first token after the header keywords and ends at the
// last token before bound.
func (c *cswLower) spell(from, bound int) string {
	start := typeStart(c.fl.src, from, bound)
	return c.fl.textOf(span{start, typeEnd(c.fl.src, start, bound)})
}

// typeStart returns the offset of the first token of the switched type
// after its header: it skips the case, const, switch, and ~ keywords,
// plus comments and whitespace. A function type starts at func, which a
// node position cannot supply, so the start comes from the header.
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

// typeEnd returns the end of the type starting at start: the last token
// before bound, stopping at a bracket that closes before start. A closer
// at depth zero matches a parenthesis around the type, not the type
// itself, so the spelling ends before it.
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

// lowerFold replaces a concrete-tag switch with a block holding its first
// matching case. No case matching removes the statement. Selection is the
// shared checker operation, so lowering and lint agree on every case. The
// tag here is concrete (lowerConstSwitchStmt routes a type parameter to
// lowerRuntime) and newCswLower already resolved every type, so an unknown
// selection fails loudly like the nil types newCswLower rejects.
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
	c.fl.add(c.start, c.end, leftoverComments(c.fl, span{c.start, c.end}, []span{kept})+"{"+text+"}")
	return []syntax.Stmt{block}
}

// runs splits the cases into maximal runs of exact cases and runs of the
// other cases, keeping source order.
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

// lowerRuntime lowers a switch over a type parameter. When the runs are
// exact then the rest, one type switch holds the exact clauses and its default
// clause holds the if chain of the rest. Any other order splits into runs with
// a matched flag, as the file comment describes.
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
		c.fl.add(c.start, c.end, text)
		return []syntax.Stmt{stmt}
	}
	pos := c.stmt.Pos()
	done := fmt.Sprintf("_logos_csw%d", c.start)
	block := at(&syntax.BlockStmt{Rbrace: c.stmt.Rbrace}, pos)
	block.List = append(block.List, defineOf(pos, []syntax.Expr{nameOf(pos, done)}, []syntax.Expr{nameOf(pos, "false")}))
	text := "{" + done + " := false\n"
	for _, run := range runs {
		first := run[0].syn.Case
		close := run[len(run)-1].close
		var inner syntax.Stmt
		var innerText string
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
		text += "if !" + done + " {" + innerText + "}\n"
	}
	text += "}"
	c.fl.add(c.start, c.end, text)
	return []syntax.Stmt{block}
}

// typeSwitch lowers the exact clauses of one run into a type switch on
// any((*T)(nil)), with rest as its default clause when given. done names the
// matched flag of a multi-run statement, or is empty.
func (c *cswLower) typeSwitch(exacts, rest []*cswCase, done string) (syntax.Stmt, string) {
	pos := c.stmt.Pos()
	if len(exacts) > 0 {
		pos = exacts[0].syn.Case
	} else if len(rest) > 0 {
		pos = rest[0].syn.Case
	}
	tag := c.spell(c.fl.off(c.stmt.Pos()), c.fl.off(c.stmt.Lbrace))
	guard, ok := parseExprAt("any((*"+tag+")(nil))", pos)
	if !ok {
		base.Fatalf("logos: const switch tag %s at %s cannot be respelled", tag, pos.String())
	}
	sw := at(&syntax.SwitchStmt{
		Tag:    at(&syntax.TypeSwitchGuard{X: guard}, pos),
		Rbrace: c.stmt.Rbrace,
	}, pos)
	text := "switch any((*" + tag + ")(nil)).(type) {"
	for _, cs := range exacts {
		stmts, bodyText := c.caseBody(cs, done)
		ptr, ok := parseExprAt("*"+cs.spell, cs.syn.Case)
		if !ok {
			base.Fatalf("logos: const switch case %s at %s cannot be respelled", cs.spell, cs.syn.Case.String())
		}
		sw.Body = append(sw.Body, at(&syntax.CaseClause{Cases: ptr, Body: stmts}, cs.syn.Case))
		text += "\ncase *" + cs.spell + ":" + bodyText
	}
	if len(rest) > 0 {
		chain, chainText := c.chainIf(rest, done)
		sw.Body = append(sw.Body, at(&syntax.CaseClause{Body: []syntax.Stmt{chain}}, rest[0].syn.Case))
		text += "\ndefault:\n" + chainText
	}
	text += "\n}"
	return sw, text
}

// chainIf lowers a run of tilde and interface cases into one if chain, in
// source order. The run is never empty.
func (c *cswLower) chainIf(run []*cswCase, done string) (syntax.Stmt, string) {
	var next syntax.Stmt
	var nextText string
	for i := len(run) - 1; i >= 0; i-- {
		cs := run[i]
		cond, condText := c.cond(cs)
		stmts, bodyText := c.caseBody(cs, done)
		link := at(&syntax.IfStmt{
			Cond: cond,
			Then: closeAt(cs.syn.Case, stmts, cs.close),
			Else: next,
		}, cs.syn.Case)
		text := "if " + condText + " {" + bodyText + "}"
		if next != nil {
			text += " else " + nextText
		}
		next, nextText = link, text
	}
	return next, nextText
}

// closeAt returns a block over list that closes at close. A scope closes
// after every position its body emits, and before the next clause opens, so
// the DWARF scope marks stay monotonic (see dwarfgen/marker.go).
func closeAt(pos syntax.Pos, list []syntax.Stmt, close syntax.Pos) *syntax.BlockStmt {
	return at(&syntax.BlockStmt{List: list, Rbrace: close}, pos)
}

// cond returns the run-time test of a tilde or interface case, as native
// syntax and as eject text. Both use the support helpers of logos/rt.
func (c *cswLower) cond(cs *cswCase) (syntax.Expr, string) {
	c.fl.needRT = true
	tag := c.spell(c.fl.off(c.stmt.Pos()), c.fl.off(c.stmt.Lbrace))
	helper := "Implements"
	if cs.kind == cswTilde {
		helper = "TildeIs"
	}
	text := rtAlias + "." + helper + "[" + tag + ", " + cs.spell + "]()"
	expr, ok := parseExprAt(text, cs.syn.Case)
	if !ok {
		base.Fatalf("logos: const switch case %s at %s cannot be respelled", cs.spell, cs.syn.Case.String())
	}
	return expr, text
}

// caseBody lowers one case body and returns its native statements and its
// eject text. The order is fixed here: the case-scope rewrites (switched type
// name, interface selectors) run on the source body first, so the body's own
// lowering (`?`, nested constructs) sees them and composes its splices with
// them; the rebinding declarations and the matched flag come first; and the
// body text is read once, with every splice applied. done, when set, is the
// matched flag that the body raises before it runs.
func (c *cswLower) caseBody(cs *cswCase, done string) ([]syntax.Stmt, string) {
	if cs.kind == cswIface {
		c.wrapSelectors(cs.syn.Body, cs)
	} else if cs.sub.Target != nil {
		c.replaceTargetNames(cs.syn.Body, cs)
	}
	lowered := c.fc.lowerList(cs.syn.Body)
	var head []syntax.Stmt
	var headText string
	if done != "" {
		head = append(head, at(&syntax.AssignStmt{Lhs: nameOf(cs.syn.Case, done), Rhs: nameOf(cs.syn.Case, "true")}, cs.syn.Case))
		headText = done + " = true\n"
	}
	decls, declText := c.rebindings(cs)
	body := c.fl.textOf(span{cs.lo, cs.hi})
	stmts := append(head, decls...)
	return append(stmts, lowered...), headText + declText + body
}

// rebindings declares each variable of cs whose type names the switched type
// with its case type, at the start of the body. It returns the declarations
// as native statements and as eject text.
func (c *cswLower) rebindings(cs *cswCase) ([]syntax.Stmt, string) {
	var decls []syntax.Stmt
	var text string
	for _, r := range c.rebound(cs) {
		c.fl.needRT = true
		rhs := rtAlias + ".Reinterpret[" + r.spell + "](&" + r.name + ")"
		expr, ok := parseExprAt(rhs, cs.syn.Case)
		if !ok {
			base.Fatalf("logos: const switch rebinding of %s at %s cannot be respelled", r.name, cs.syn.Case.String())
		}
		decls = append(decls, defineOf(cs.syn.Case, []syntax.Expr{nameOf(cs.syn.Case, r.name)}, []syntax.Expr{expr}))
		text += r.name + " := " + rhs + "\n"
	}
	return decls, text
}

// rebound lists the variables of cs used in its body that the checker
// rebound, in first-use order. Only used variables are rebound, so the
// rebinding never introduces an unused declaration.
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

// cswSpell renders a checked type as Go source in this file. The current
// package stands bare, and every other package uses the name the file imports
// it under.
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

// replaceTargetNames replaces each body name that denotes the case-scope name
// of the switched type with the case spelling, recording one eject splice per
// name.
func (c *cswLower) replaceTargetNames(body []syntax.Stmt, cs *cswCase) {
	spell := cs.spell
	w := cswWalker{
		info:   c.info,
		target: cs.sub.Target,
		repl: func(pos syntax.Pos) syntax.Expr {
			e, ok := parseExprAt(spell, pos)
			if !ok {
				base.Fatalf("logos: const switch case %s at %s cannot be respelled", spell, pos.String())
			}
			return e
		},
		off:    c.fl.off,
		splice: func(start, end int) { c.fl.add(start, end, spell) },
	}
	w.list(body)
}

// cswWalker rewrites the uses of one type name in a body. It walks the syntax
// tree by reflection, because a rewrite must replace a node held in its
// parent's field.
type cswWalker struct {
	info   *types2.Info
	target types2.Object
	repl   func(syntax.Pos) syntax.Expr
	off    func(syntax.Pos) int
	splice func(start, end int)
}

// list rewrites every statement of a body.
func (w cswWalker) list(stmts []syntax.Stmt) {
	for _, s := range stmts {
		w.node(s)
	}
}

// node rewrites the uses inside one node. A nested `const switch` keeps its
// header: the nested statement lowers itself, and its header is replaced with
// the rest of the statement, so no splice may reach it. Its case bodies are
// still rewritten, because they belong to this body.
func (w cswWalker) node(n syntax.Node) {
	if sw, ok := n.(*syntax.ConstSwitchStmt); ok {
		for _, cc := range sw.Cases {
			w.list(cc.Body)
		}
		return
	}
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
			w.field(v.Field(i))
		}
	}
}

// field rewrites the uses held by one field.
func (w cswWalker) field(f reflect.Value) {
	switch f.Kind() {
	case reflect.Interface:
		w.interfaceField(f)
	case reflect.Pointer:
		w.pointerField(f)
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			w.field(f.Index(i))
		}
	}
}

// interfaceField rewrites an interface-typed field: a name that denotes the
// target is replaced, and any other node is walked.
func (w cswWalker) interfaceField(f reflect.Value) {
	if f.IsNil() || !f.CanSet() {
		return
	}
	switch x := f.Elem().Interface().(type) {
	case *syntax.Name:
		w.replaceName(f, x)
	case syntax.Node:
		w.node(x)
	}
}

// replaceName stores the case spelling in f, in place of x, when x denotes
// the target, and records the eject splice for it.
func (w cswWalker) replaceName(f reflect.Value, x *syntax.Name) {
	if w.info.Uses[x] != w.target {
		return
	}
	start := w.off(x.Pos())
	w.splice(start, start+len(x.Value))
	f.Set(reflect.ValueOf(w.repl(x.Pos())))
}

// pointerField walks a pointer-typed field that holds a node.
func (w cswWalker) pointerField(f reflect.Value) {
	if f.IsNil() {
		return
	}
	if n, ok := f.Interface().(syntax.Node); ok {
		w.node(n)
	}
}

// wrapSelectors rewrites each selector on the switched value in an interface
// case to assert the case interface, so the interface's methods resolve on the
// value at run time. The wrapper sits at the receiver's leftmost token, so a
// later splice over the same statement starts where the source does.
func (c *cswLower) wrapSelectors(body []syntax.Stmt, cs *cswCase) {
	for _, s := range body {
		syntax.Inspect(s, func(n syntax.Node) bool {
			sel, ok := n.(*syntax.SelectorExpr)
			if !ok || c.info.LogosConstSel[sel] != cs.syn {
				return true
			}
			xpos := leftmostOf(sel.X)
			iface, ok := typeExprAt(cs.spell, xpos)
			if !ok {
				base.Fatalf("logos: const switch interface %s at %s cannot be respelled", cs.spell, xpos.String())
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
			c.fl.add(xspan.start, xspan.end, "any("+xtext+").("+cs.spell+")")
			return true
		})
	}
}

// parseExprAt parses text as an expression and moves every node of the result
// to pos. It reports false when text is not an expression.
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
