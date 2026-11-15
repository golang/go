// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type tryHoist struct {
	fl  *fileLower
	seq int
}

type hoistOperand struct {
	e    syntax.Expr
	set  func(syntax.Expr)
	area span
}

type hoistHost struct {
	h       *tryHoist
	pos     syntax.Pos
	pending []hoistOperand
	native  []syntax.Stmt
	text    []otext
}

func isTryInit(e *syntax.SwitchExpr) bool {
	switch s := e.Init.(type) {
	case *syntax.AssignStmt:
		_, _, ok := tryAssignOf(s)
		return ok
	case *syntax.ExprStmt:
		_, _, ok := tryExprOf(s)
		return ok
	}
	return false
}

func (fl *fileLower) hoistTryInits() {
	h := &tryHoist{fl: fl}
	var bodies []*syntax.BlockStmt
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncDecl:
			if n.Body != nil {
				bodies = append(bodies, n.Body)
			}
		case *syntax.FuncLit:
			bodies = append(bodies, n.Body)
		case *syntax.LambdaExpr:
			if b, ok := n.Body.(*syntax.BlockStmt); ok {
				bodies = append(bodies, b)
			}
		}
		return true
	})
	for i := len(bodies) - 1; i >= 0; i-- {
		bodies[i].List = h.list(bodies[i].List)
	}
}

func (h *tryHoist) list(list []syntax.Stmt) []syntax.Stmt {
	out := make([]syntax.Stmt, 0, len(list))
	for _, s := range list {
		out = append(out, h.stmt(s)...)
	}
	return out
}

func (h *tryHoist) stmt(s syntax.Stmt) []syntax.Stmt {
	switch s := s.(type) {
	case *syntax.BlockStmt:
		s.List = h.list(s.List)
	case *syntax.LabeledStmt:
		switch inner := s.Stmt.(type) {
		case *syntax.ForStmt:
			return []syntax.Stmt{h.forStmt(inner, s)}
		case *syntax.SwitchStmt:
			return []syntax.Stmt{h.switchStmt(inner, s)}
		}
		res := h.stmt(s.Stmt)
		s.Stmt = res[0]
		return append([]syntax.Stmt{s}, res[1:]...)
	case *syntax.IfStmt:
		s.Then.List = h.list(s.Then.List)
		h.elseChain(s)
		return []syntax.Stmt{h.wrapIf(s)}
	case *syntax.ForStmt:
		return []syntax.Stmt{h.forStmt(s, nil)}
	case *syntax.SwitchStmt:
		return []syntax.Stmt{h.switchStmt(s, nil)}
	case *syntax.SelectStmt:
		for _, c := range s.Body {
			c.Body = h.list(c.Body)
		}
	case *syntax.GroupStmt:
		s.Body.List = h.list(s.Body.List)
	case *syntax.GroupChild:
		s.Body.List = h.list(s.Body.List)
	case *syntax.AssignStmt, *syntax.ReturnStmt, *syntax.DeclStmt, *syntax.ExprStmt, *syntax.CallStmt, *syntax.SendStmt:
		return h.simple(s)
	}
	return []syntax.Stmt{s}
}

func (h *tryHoist) elseChain(s *syntax.IfStmt) {
	switch els := s.Else.(type) {
	case *syntax.BlockStmt:
		els.List = h.list(els.List)
	case *syntax.IfStmt:
		els.Then.List = h.list(els.Then.List)
		h.elseChain(els)
		s.Else = h.wrapIf(els)
	}
}

func (h *tryHoist) forStmt(s *syntax.ForStmt, label *syntax.LabeledStmt) syntax.Stmt {
	s.Body.List = h.list(s.Body.List)
	return h.rewriteFor(s, label)
}

func (h *tryHoist) switchStmt(s *syntax.SwitchStmt, label *syntax.LabeledStmt) syntax.Stmt {
	for _, c := range s.Body {
		c.Body = h.list(c.Body)
	}
	return h.rewriteSwitch(s, label)
}

func (h *tryHoist) wrap(inner syntax.Stmt, label *syntax.LabeledStmt, native []syntax.Stmt, head otext) syntax.Stmt {
	fl := h.fl
	rbrace := hostRbrace(inner)
	anchor := fl.off(syntax.StartPos(inner))
	var last syntax.Stmt = inner
	var scoped *syntax.LabeledStmt
	switch {
	case label == nil:
	case fl.gotoTarget(label):
		scoped = label
		name := "_logos_in_" + label.Label.Value
		if fl.renameBranchLabels(inner, label.Label.Value, name) > 0 {
			head = cat(head, gen(name+": "))
			last = at(&syntax.LabeledStmt{Label: nameOf(label.Pos(), name), Stmt: inner}, label.Pos())
		}
	default:
		anchor = fl.stmtStartOf(label)
		last = label
	}
	fl.addT(anchor, anchor, cat(gen("{ "), head))
	end := fl.off(rbrace) + 1
	fl.add(end, end, " }")
	wrapper := blockOf(inner.Pos(), append(native, last))
	wrapper.Rbrace = rbrace
	if scoped == nil {
		return wrapper
	}
	scoped.Stmt = wrapper
	return scoped
}

func (h *tryHoist) newHost(pos syntax.Pos) *hoistHost {
	return &hoistHost{h: h, pos: pos}
}

func (x *hoistHost) cat() otext {
	return cat(x.text...)
}

func (h *tryHoist) simple(s syntax.Stmt) []syntax.Stmt {
	if !h.hasSwitch(s) {
		return []syntax.Stmt{s}
	}
	fl := h.fl
	pos := syntax.StartPos(s)
	start := fl.off(pos)
	end := stmtEnd(fl.src, start)
	x := h.newHost(pos)
	define := false
	switch s := s.(type) {
	case *syntax.AssignStmt:
		opOff := fl.off(s.Pos())
		if s.Op == syntax.Def {
			define = true
		} else {
			x.lvalues(s.Lhs, fl.tokenSpan(start, opOff))
		}
		if s.Rhs != nil {
			x.items(s.Rhs, func(n syntax.Expr) { s.Rhs = n }, fl.tokenSpan(h.tokenEndAt(opOff), end))
		}
	case *syntax.ReturnStmt:
		x.items(s.Results, func(n syntax.Expr) { s.Results = n }, fl.tokenSpan(start+len("return"), end))
	case *syntax.DeclStmt:
		define = true
		vd, ok := s.DeclList[0].(*syntax.VarDecl)
		if !ok || len(s.DeclList) != 1 || vd.Group != nil {
			base.Fatalf("logos: switch expression with a try init in a declaration group at %s", pos.String())
		}
		x.items(vd.Values, func(n syntax.Expr) { vd.Values = n }, fl.tokenSpan(h.afterToken(start, end, token.ASSIGN), end))
	case *syntax.ExprStmt:
		x.child(s.X, func(n syntax.Expr) { s.X = n }, fl.tokenSpan(start, end))
	case *syntax.CallStmt:
		x.child(s.Call, func(n syntax.Expr) { s.Call = n }, fl.tokenSpan(start+len(s.Tok.String()), end))
	case *syntax.SendStmt:
		opOff := fl.off(s.Pos())
		x.child(s.Chan, func(n syntax.Expr) { s.Chan = n }, fl.tokenSpan(start, opOff))
		x.child(s.Value, func(n syntax.Expr) { s.Value = n }, fl.tokenSpan(h.tokenEndAt(opOff), end))
	}
	if len(x.native) == 0 {
		base.Fatalf("logos: switch expression with a try init was not hoisted at %s", pos.String())
	}
	if define {
		fl.addT(start, start, x.cat())
		return append(x.native, s)
	}
	fl.addT(start, start, cat(gen("{ "), x.cat()))
	fl.add(end, end, " }")
	return []syntax.Stmt{blockOf(pos, append(x.native, s))}
}

func (h *tryHoist) afterToken(from, to int, kind token.Token) int {
	at := -1
	depth := 0
	scanSegment(h.fl.src, from, to, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		case kind:
			if depth == 0 {
				at = t.end
				return false
			}
		}
		return true
	})
	if at < 0 {
		base.Fatalf("logos: cannot locate %s at offset %d", kind, from)
	}
	return at
}

func (h *tryHoist) tokenEndAt(off int) int {
	end := off
	scanSegment(h.fl.src, off, len(h.fl.src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		end = t.end
		return false
	})
	return end
}

func (h *tryHoist) wrapIf(s *syntax.IfStmt) syntax.Stmt {
	fl := h.fl
	initHas := s.Init != nil && h.hasSwitch(s.Init)
	condHas := h.hasSwitch(s.Cond)
	if !initHas && !condHas {
		return s
	}
	pos := s.Pos()
	ifOff := fl.off(pos)
	thenOff := fl.off(s.Then.Pos())
	semi := -1
	if s.Init != nil {
		semis := h.headerSemis(ifOff+len("if"), thenOff, s.Init, s.Cond)
		if len(semis) != 1 {
			base.Fatalf("logos: cannot locate the if header at %s", pos.String())
		}
		semi = semis[0]
	}
	var native []syntax.Stmt
	var text []otext
	if initHas {
		x := h.newHost(pos)
		x.header(s.Init, fl.tokenSpan(ifOff+len("if"), semi))
		native, text = append(native, x.native...), append(text, x.text...)
	}
	extract := condHas && s.Init != nil
	if extract {
		native = append(native, s.Init)
		text = append(text, gen(fl.copyRegion(span{start: ifOff + len("if"), end: semi})), gen("; "))
		s.Init = nil
	}
	if condHas {
		x := h.newHost(pos)
		x.child(s.Cond, func(n syntax.Expr) { s.Cond = n }, fl.tokenSpan(h.condFrom(ifOff, semi), thenOff))
		native, text = append(native, x.native...), append(text, x.text...)
	}
	head := cat(append([]otext{gen("{ ")}, text...)...)
	if extract {
		fl.addT(ifOff, semi+1, cat(head, gen("if ")))
	} else {
		fl.addT(ifOff, ifOff, head)
	}
	end := ifEndOff(fl, s)
	fl.add(end, end, " }")
	wrapper := blockOf(pos, append(native, s))
	wrapper.Rbrace = hostRbrace(s)
	return wrapper
}

func (h *tryHoist) condFrom(ifOff, semi int) int {
	if semi >= 0 {
		return semi + 1
	}
	return ifOff + len("if")
}

func (h *tryHoist) rewriteSwitch(s *syntax.SwitchStmt, label *syntax.LabeledStmt) syntax.Stmt {
	fl := h.fl
	initHas := s.Init != nil && h.hasSwitch(s.Init)
	tagHas := s.Tag != nil && h.hasSwitch(s.Tag)
	if !initHas && !tagHas {
		return enclosing(s, label)
	}
	pos := s.Pos()
	if _, guard := s.Tag.(*syntax.TypeSwitchGuard); guard && tagHas {
		base.Fatalf("logos: switch expression with a try init in a type switch guard at %s", pos.String())
	}
	swOff := fl.off(pos)
	hdr := h.switchHeader(s)
	var native []syntax.Stmt
	var text []otext
	if initHas {
		x := h.newHost(pos)
		x.header(s.Init, hdr.init)
		native, text = append(native, x.native...), append(text, x.text...)
	}
	extract := tagHas && s.Init != nil
	if extract {
		native = append(native, s.Init)
		text = append(text, gen(fl.copyRegion(span{start: swOff + len("switch"), end: hdr.init.end})), gen("; "))
		s.Init = nil
		fl.add(swOff, hdr.tag.start, "switch ")
	}
	if tagHas {
		x := h.newHost(pos)
		x.child(s.Tag, func(n syntax.Expr) { s.Tag = n }, hdr.tag)
		native, text = append(native, x.native...), append(text, x.text...)
	}
	return h.wrap(s, label, native, cat(text...))
}

func (h *tryHoist) switchHeader(s *syntax.SwitchStmt) switchHeader {
	fl := h.fl
	from := fl.off(s.Pos()) + len("switch")
	first := fl.off(s.Rbrace)
	if len(s.Body) > 0 {
		first = fl.off(s.Body[0].Pos())
	}
	brace := -1
	scanSegment(fl.src, from, first, func(t tok) bool {
		if !t.comment && !t.auto {
			brace = t.start
		}
		return true
	})
	if brace < 0 || fl.src[brace] != '{' {
		base.Fatalf("logos: cannot locate the switch body at %s", s.Pos().String())
	}
	var hdr switchHeader
	semis := h.headerSemis(from, brace, s.Init, s.Tag)
	switch len(semis) {
	case 0:
		hdr.tag = fl.tokenSpan(from, brace)
	case 1:
		hdr.init = fl.tokenSpan(from, semis[0])
		hdr.tag = fl.tokenSpan(semis[0]+1, brace)
	default:
		base.Fatalf("logos: cannot locate the switch header at %s", s.Pos().String())
	}
	return hdr
}

func (h *tryHoist) rewriteFor(s *syntax.ForStmt, label *syntax.LabeledStmt) syntax.Stmt {
	initHas := s.Init != nil && h.hasSwitch(s.Init)
	condHas := s.Cond != nil && h.hasSwitch(s.Cond)
	if s.Post != nil && h.hasSwitch(s.Post) {
		base.Fatalf("logos: switch expression with a try init in a for post statement at %s", s.Pos().String())
	}
	if !initHas && !condHas {
		return enclosing(s, label)
	}
	parts := h.forParts(s)
	if condHas {
		h.hoistForCond(s, parts.cond)
	}
	if !initHas {
		return enclosing(s, label)
	}
	x := h.newHost(s.Pos())
	x.header(s.Init, parts.init)
	return h.wrap(s, label, x.native, x.cat())
}

func enclosing(s syntax.Stmt, label *syntax.LabeledStmt) syntax.Stmt {
	if label != nil {
		return label
	}
	return s
}

func (h *tryHoist) hoistForCond(s *syntax.ForStmt, area span) {
	fl := h.fl
	open := s.Body.Pos()
	x := h.newHost(open)
	setCond := func(n syntax.Expr) { s.Cond = n }
	var test string
	if bare, ok := s.Cond.(*syntax.SwitchExpr); ok && h.needsHoist(bare) {
		x.hoistSwitch(bare, setCond, "")
		test = s.Cond.(*syntax.Name).Value
	} else {
		x.child(s.Cond, setCond, area)
		test = fl.moveRegion(area, "")
	}
	cond := s.Cond
	s.Cond = nil
	brk := at(&syntax.BranchStmt{Tok: syntax.Break}, open)
	negated := at(&syntax.Operation{Op: syntax.Not, X: at(&syntax.ParenExpr{X: cond}, open)}, open)
	guard := ifOf(open, nil, negated, []syntax.Stmt{brk})
	x.text = append(x.text, gen("if !("+test+") { break }; "))
	bodyOff := fl.off(open) + 1
	fl.addT(bodyOff, bodyOff, cat(append([]otext{gen(" ")}, x.text...)...))
	s.Body.List = append(append(x.native, guard), s.Body.List...)
}

type forHeaderParts struct {
	init, cond span
}

func (h *tryHoist) forParts(s *syntax.ForStmt) forHeaderParts {
	fl := h.fl
	from := fl.off(s.Pos()) + len("for")
	to := fl.off(s.Body.Pos())
	semis := h.headerSemis(from, to, s.Init, s.Cond, s.Post)
	switch len(semis) {
	case 0:
		if _, ok := s.Init.(*syntax.RangeClause); ok {
			return forHeaderParts{init: fl.tokenSpan(from, to)}
		}
		return forHeaderParts{cond: fl.tokenSpan(from, to)}
	case 2:
		return forHeaderParts{
			init: fl.tokenSpan(from, semis[0]),
			cond: fl.tokenSpan(semis[0]+1, semis[1]),
		}
	}
	base.Fatalf("logos: cannot locate the for header at %s", s.Pos().String())
	return forHeaderParts{}
}

func (h *tryHoist) headerSemis(from, to int, parts ...syntax.Node) []int {
	var skip []span
	for _, part := range parts {
		if part == nil {
			continue
		}
		syntax.Inspect(part, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.FuncLit, *syntax.LambdaExpr:
				return false
			case *syntax.SwitchExpr:
				skip = append(skip, span{start: h.fl.off(n.Pos()), end: h.fl.off(n.Rbrace) + 1})
				return false
			}
			return true
		})
	}
	var out []int
	depth := 0
	scanSegment(h.fl.src, from, to, func(t tok) bool {
		if t.comment || t.auto || insideAny(t.span, skip) {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		case token.SEMICOLON:
			if depth == 0 {
				out = append(out, t.start)
			}
		}
		return true
	})
	return out
}

func (x *hoistHost) header(init syntax.SimpleStmt, area span) {
	fl := x.h.fl
	switch init := init.(type) {
	case *syntax.AssignStmt:
		opOff := fl.off(init.Pos())
		if init.Op != syntax.Def {
			x.lvalues(init.Lhs, fl.tokenSpan(area.start, opOff))
		}
		if init.Rhs != nil {
			x.items(init.Rhs, func(n syntax.Expr) { init.Rhs = n }, fl.tokenSpan(x.h.tokenEndAt(opOff), area.end))
		}
	case *syntax.ExprStmt:
		x.child(init.X, func(n syntax.Expr) { init.X = n }, area)
	case *syntax.RangeClause:
		x.child(init.X, func(n syntax.Expr) { init.X = n }, fl.tokenSpan(x.h.afterToken(area.start, area.end, token.RANGE), area.end))
	default:
		base.Fatalf("logos: switch expression with a try init in a %T header statement at %s", init, init.Pos().String())
	}
}

func (h *tryHoist) hasSwitch(n syntax.Node) bool {
	if n == nil {
		return false
	}
	found := false
	syntax.Inspect(n, func(m syntax.Node) bool {
		if found {
			return false
		}
		switch m := m.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return false
		case *syntax.SwitchExpr:
			if isTryInit(m) {
				found = true
				return false
			}
		}
		return true
	})
	return found
}

func (h *tryHoist) impure(n syntax.Node) bool {
	info := h.fl.pkg.info
	found := false
	syntax.Inspect(n, func(m syntax.Node) bool {
		if found {
			return false
		}
		switch m := m.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return false
		case *syntax.SwitchExpr, *syntax.TryExpr:
			found = true
			return false
		case *syntax.Operation:
			if m.Y == nil && m.Op == syntax.Recv {
				found = true
				return false
			}
		case *syntax.CallExpr:
			tv := info.Types[m.Fun]
			if tv.IsType() {
				return true
			}
			if tv.IsBuiltin() && pureBuiltin(m.Fun) {
				return true
			}
			found = true
			return false
		}
		return true
	})
	return found
}

func pureBuiltin(fun syntax.Expr) bool {
	name, ok := syntax.Unparen(fun).(*syntax.Name)
	if !ok {
		return false
	}
	switch name.Value {
	case "len", "cap", "min", "max", "real", "imag", "complex", "new", "make":
		return true
	}
	return false
}

func (h *tryHoist) scan(area span) []tok {
	var toks []tok
	scanSegment(h.fl.src, area.start, area.end, func(t tok) bool {
		if !t.comment && !t.auto {
			toks = append(toks, t)
		}
		return true
	})
	return toks
}

func (h *tryHoist) peel(area span) span {
	for {
		toks := h.scan(area)
		if len(toks) < 2 || toks[0].kind != token.LPAREN || toks[len(toks)-1].kind != token.RPAREN {
			return area
		}
		depth := 0
		for i, t := range toks {
			switch t.kind {
			case token.LPAREN:
				depth++
			case token.RPAREN:
				depth--
				if depth == 0 && i < len(toks)-1 {
					return area
				}
			}
		}
		area = h.fl.tokenSpan(toks[0].end, toks[len(toks)-1].start)
	}
}

func (h *tryHoist) peelOnce(area span) span {
	toks := h.scan(area)
	if len(toks) < 2 || toks[0].kind != token.LPAREN || toks[len(toks)-1].kind != token.RPAREN {
		base.Fatalf("logos: cannot locate the parenthesized expression at offset %d", area.start)
	}
	return h.fl.tokenSpan(toks[0].end, toks[len(toks)-1].start)
}

func (h *tryHoist) lastOpen(area span, open, closing token.Token) (openOff, closeOff int) {
	toks := h.scan(area)
	if len(toks) == 0 || toks[len(toks)-1].kind != closing {
		base.Fatalf("logos: cannot locate the closing %s at offset %d", closing, area.end)
	}
	openOff = -1
	depth := 0
	for _, t := range toks {
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			if t.kind == open && depth == 0 {
				openOff = t.start
			}
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		}
	}
	if openOff < 0 {
		base.Fatalf("logos: cannot locate the opening %s at offset %d", open, area.start)
	}
	return openOff, toks[len(toks)-1].start
}

func (h *tryHoist) commas(area span) []int {
	var out []int
	depth := 0
	for _, t := range h.scan(area) {
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		case token.COMMA:
			if depth == 0 {
				out = append(out, t.start)
			}
		}
	}
	return out
}

func (h *tryHoist) colon(area span) int {
	depth := 0
	for _, t := range h.scan(area) {
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		case token.COLON:
			if depth == 0 {
				return t.start
			}
		}
	}
	base.Fatalf("logos: cannot locate the key colon at offset %d", area.start)
	return 0
}

func (h *tryHoist) split(area span, n int) []span {
	if toks := h.scan(area); len(toks) > 0 && toks[len(toks)-1].kind == token.COMMA {
		area = h.fl.tokenSpan(area.start, toks[len(toks)-1].start)
	}
	if n == 1 {
		return []span{area}
	}
	commas := h.commas(area)
	if len(commas) != n-1 {
		base.Fatalf("logos: cannot split %d list items at offset %d", n, area.start)
	}
	out := make([]span, n)
	from := area.start
	for i := range n {
		to := area.end
		if i < n-1 {
			to = commas[i]
		}
		out[i] = h.fl.tokenSpan(from, to)
		if i < n-1 {
			from = commas[i] + 1
		}
	}
	return out
}

func (x *hoistHost) items(e syntax.Expr, set func(syntax.Expr), area span) {
	l, ok := e.(*syntax.ListExpr)
	if !ok {
		x.child(e, set, area)
		return
	}
	for i, part := range x.h.split(area, len(l.ElemList)) {
		x.child(l.ElemList[i], func(n syntax.Expr) { l.ElemList[i] = n }, part)
	}
}

func (x *hoistHost) lvalues(e syntax.Expr, area span) {
	l, ok := e.(*syntax.ListExpr)
	if !ok {
		x.lvalue(e, area)
		return
	}
	for i, part := range x.h.split(area, len(l.ElemList)) {
		x.lvalue(l.ElemList[i], part)
	}
}

func (h *tryHoist) underlying(e syntax.Expr) types2.Type {
	if t := h.fl.pkg.info.TypeOf(e); t != nil {
		return t.Underlying()
	}
	return nil
}

func (x *hoistHost) lvalue(e syntax.Expr, area span) {
	h := x.h
	area = h.peel(area)
	switch n := e.(type) {
	case *syntax.IndexExpr:
		lbrack, rbrack := h.lastOpen(area, token.LBRACK, token.RBRACK)
		xArea := h.fl.tokenSpan(area.start, lbrack)
		if _, array := h.underlying(n.X).(*types2.Array); array {
			x.lvalue(n.X, xArea)
		} else {
			x.child(n.X, func(v syntax.Expr) { n.X = v }, xArea)
		}
		x.child(n.Index, func(v syntax.Expr) { n.Index = v }, h.fl.tokenSpan(lbrack+1, rbrack))
	case *syntax.SelectorExpr:
		if h.fl.pkg.info.Selections[n] == nil {
			return
		}
		xArea := h.fl.tokenSpan(area.start, h.fl.off(n.Pos()))
		if _, ptr := h.underlying(n.X).(*types2.Pointer); ptr {
			x.child(n.X, func(v syntax.Expr) { n.X = v }, xArea)
		} else {
			x.lvalue(n.X, xArea)
		}
	case *syntax.Operation:
		if n.Y == nil && n.Op == syntax.Mul {
			x.child(n.X, func(v syntax.Expr) { n.X = v }, h.fl.tokenSpan(h.tokenEndAt(area.start), area.end))
		}
	case *syntax.ParenExpr:
		x.lvalue(n.X, h.peelOnce(area))
	}
}

func (x *hoistHost) child(e syntax.Expr, set func(syntax.Expr), area span) {
	if x.h.hasSwitch(e) {
		x.expr(e, set, area)
		return
	}
	x.operand(e, set, area)
}

func (x *hoistHost) operand(e syntax.Expr, set func(syntax.Expr), area span) {
	if !x.h.impure(e) {
		return
	}
	if cl, ok := e.(*syntax.CompositeLit); ok && cl.Type == nil {
		x.complit(cl, x.h.peel(area))
		return
	}
	x.pending = append(x.pending, hoistOperand{e: e, set: set, area: area})
}

func (x *hoistHost) expr(e syntax.Expr, set func(syntax.Expr), area span) {
	h := x.h
	fl := h.fl
	switch n := e.(type) {
	case *syntax.SwitchExpr:
		x.switchExpr(n, set)
	case *syntax.ParenExpr:
		x.child(n.X, func(v syntax.Expr) { n.X = v }, h.peelOnce(area))
	case *syntax.Operation:
		inner := h.peel(area)
		if n.Y == nil {
			x.child(n.X, func(v syntax.Expr) { n.X = v }, fl.tokenSpan(h.tokenEndAt(inner.start), inner.end))
			return
		}
		opOff := fl.off(n.Pos())
		xArea := fl.tokenSpan(inner.start, opOff)
		x.child(n.X, func(v syntax.Expr) { n.X = v }, xArea)
		yArea := fl.tokenSpan(h.tokenEndAt(opOff), inner.end)
		if n.Op == syntax.AndAnd || n.Op == syntax.OrOr {
			x.shortCircuit(n, set, shortCircuitAreas{whole: area, x: xArea, y: yArea})
			return
		}
		x.child(n.Y, func(v syntax.Expr) { n.Y = v }, yArea)
	case *syntax.IndexExpr:
		inner := h.peel(area)
		lbrack, rbrack := h.lastOpen(inner, token.LBRACK, token.RBRACK)
		x.child(n.X, func(v syntax.Expr) { n.X = v }, fl.tokenSpan(inner.start, lbrack))
		x.child(n.Index, func(v syntax.Expr) { n.Index = v }, fl.tokenSpan(lbrack+1, rbrack))
	case *syntax.SelectorExpr:
		inner := h.peel(area)
		x.child(n.X, func(v syntax.Expr) { n.X = v }, fl.tokenSpan(inner.start, fl.off(n.Pos())))
	case *syntax.CompositeLit:
		x.complit(n, h.peel(area))
	case *syntax.CallExpr:
		inner := h.peel(area)
		lparen, rparen := h.lastOpen(inner, token.LPAREN, token.RPAREN)
		x.callee(n, fl.tokenSpan(inner.start, lparen))
		if len(n.ArgList) > 0 {
			for i, part := range h.argSpans(n, fl.tokenSpan(lparen+1, rparen)) {
				x.child(n.ArgList[i], func(v syntax.Expr) { n.ArgList[i] = v }, part)
			}
		}
	case *syntax.TryExpr:
		inner := h.peel(area)
		toks := h.scan(inner)
		x.child(n.X, func(v syntax.Expr) { n.X = v }, fl.tokenSpan(inner.start, toks[len(toks)-1].start))
	case *syntax.InterpExpr:
		for i, hole := range n.Exprs {
			x.child(hole, func(v syntax.Expr) { n.Exprs[i] = v }, fl.tokenSpan(fl.off(n.Opens[i])+1, fl.off(n.Closes[i])))
		}
	case *syntax.TupleLit:
		for i, part := range h.split(h.peel(area), len(n.Elems)) {
			x.child(n.Elems[i], func(v syntax.Expr) { n.Elems[i] = v }, part)
		}
	case *syntax.OptSelectExpr:
		inner := h.peel(area)
		x.child(n.X, func(v syntax.Expr) { n.X = v }, fl.tokenSpan(inner.start, fl.off(n.Sel.Pos())-len("?.")))
	case *syntax.CoalesceExpr:
		inner := h.peel(area)
		if h.hasSwitch(n.Y) {
			base.Fatalf("logos: switch expression with a try init on the right of ?? at %s", n.Pos().String())
		}
		x.child(n.X, func(v syntax.Expr) { n.X = v }, fl.tokenSpan(inner.start, fl.off(n.Pos())))
	default:
		base.Fatalf("logos: switch expression with a try init under %T at %s", e, e.Pos().String())
	}
}

func (h *tryHoist) argSpans(call *syntax.CallExpr, args span) []span {
	toks := h.scan(args)
	end := args.end
	for len(toks) > 0 && (toks[len(toks)-1].kind == token.COMMA || toks[len(toks)-1].kind == token.ELLIPSIS) {
		end = toks[len(toks)-1].start
		toks = toks[:len(toks)-1]
	}
	return h.split(h.fl.tokenSpan(args.start, end), len(call.ArgList))
}

func (x *hoistHost) callee(call *syntax.CallExpr, area span) {
	h := x.h
	set := func(v syntax.Expr) { call.Fun = v }
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || h.fl.pkg.info.Selections[sel] == nil {
		x.child(call.Fun, set, area)
		return
	}
	recv := h.fl.tokenSpan(h.peel(area).start, h.fl.off(sel.Pos()))
	if h.fl.pkg.info.Types[sel.X].Addressable() {
		x.lvalue(sel.X, recv)
		return
	}
	x.child(sel.X, func(v syntax.Expr) { sel.X = v }, recv)
}

type shortCircuitAreas struct {
	whole, x, y span
}

func (x *hoistHost) shortCircuit(n *syntax.Operation, set func(syntax.Expr), areas shortCircuitAreas) {
	h := x.h
	fl := h.fl
	if !h.hasSwitch(n.Y) {
		if h.impure(n.Y) {
			x.pending = append(x.pending, hoistOperand{e: n, set: set, area: areas.whole})
		}
		return
	}
	x.flush()
	pos := syntax.StartPos(n)
	name := fmt.Sprintf("_logos_cc_%d_%d", pos.Line(), pos.Col())
	xText := x.areaText(n.X, areas.x)
	y := h.newHost(x.pos)
	y.child(n.Y, func(v syntax.Expr) { n.Y = v }, areas.y)
	yText := x.areaText(n.Y, areas.y)
	fl.moveRegion(areas.whole, name)
	var test syntax.Expr = nameOf(x.pos, name)
	testText := name
	if n.Op == syntax.OrOr {
		test = at(&syntax.Operation{Op: syntax.Not, X: test}, x.pos)
		testText = "!" + name
	}
	define := defineOf(x.pos, []syntax.Expr{nameOf(x.pos, name)}, []syntax.Expr{n.X})
	assign := assignOf(x.pos, []syntax.Expr{nameOf(x.pos, name)}, []syntax.Expr{n.Y})
	guard := ifOf(x.pos, nil, test, append(y.native, assign))
	x.native = append(x.native, define, guard)
	x.text = append(x.text, cat(
		gen(name+" := "), gen(xText), gen("; if "+testText+" { "),
		cat(y.text...),
		gen(name+" = "), gen(yText), gen(" }; "),
	))
	set(x.stub(name, pos, n))
}

func (x *hoistHost) areaText(e syntax.Expr, area span) string {
	if name, ok := e.(*syntax.Name); ok && strings.HasPrefix(name.Value, "_logos_") {
		return name.Value
	}
	return x.h.fl.copyRegion(area)
}

func (x *hoistHost) complit(cl *syntax.CompositeLit, area span) {
	h := x.h
	lbrace, rbrace := h.lastOpen(area, token.LBRACE, token.RBRACE)
	body := h.fl.tokenSpan(lbrace+1, rbrace)
	if len(cl.ElemList) == 0 {
		return
	}
	for i, part := range h.split(body, len(cl.ElemList)) {
		el := cl.ElemList[i]
		if kv, ok := el.(*syntax.KeyValueExpr); ok {
			colon := h.colon(part)
			x.child(kv.Key, func(v syntax.Expr) { kv.Key = v }, h.fl.tokenSpan(part.start, colon))
			x.child(kv.Value, func(v syntax.Expr) { kv.Value = v }, h.fl.tokenSpan(colon+1, part.end))
			continue
		}
		x.child(el, func(v syntax.Expr) { cl.ElemList[i] = v }, part)
	}
}

func (x *hoistHost) stub(name string, pos syntax.Pos, like syntax.Expr) *syntax.Name {
	n := nameOf(pos, name)
	n.SetTypeInfo(like.GetTypeInfo())
	if info := x.h.fl.pkg.info; info.Types != nil {
		if tv, ok := info.Types[like]; ok {
			info.Types[n] = tv
		}
	}
	return n
}

func (x *hoistHost) flush() {
	if len(x.pending) == 0 {
		return
	}
	fl := x.h.fl
	var names, exprs []syntax.Expr
	var lhs, rhs []otext
	for _, op := range x.pending {
		x.h.seq++
		name := fmt.Sprintf("_logos_op_%d_%d_%d", x.pos.Line(), x.pos.Col(), x.h.seq)
		mark := fl.moveRegion(op.area, name)
		names = append(names, nameOf(x.pos, name))
		exprs = append(exprs, op.e)
		lhs = append(lhs, gen(name))
		rhs = append(rhs, gen(mark))
		op.set(x.stub(name, syntax.StartPos(op.e), op.e))
	}
	x.pending = nil
	parts := make([]otext, 0, 2*len(lhs)+4)
	for i := range lhs {
		if i > 0 {
			parts = append(parts, gen(", "))
		}
		parts = append(parts, lhs[i])
	}
	parts = append(parts, gen(" := "))
	for i := range rhs {
		if i > 0 {
			parts = append(parts, gen(", "))
		}
		parts = append(parts, rhs[i])
	}
	parts = append(parts, gen("; "))
	x.native = append(x.native, defineOf(x.pos, names, exprs))
	x.text = append(x.text, cat(parts...))
}

func switchName(e *syntax.SwitchExpr) string {
	pos := e.Pos()
	return fmt.Sprintf("_logos_sw_%d_%d", pos.Line(), pos.Col())
}

func (h *tryHoist) needsHoist(e *syntax.SwitchExpr) bool {
	return isTryInit(e) || h.armsHoist(e)
}

func (h *tryHoist) armsHoist(e *syntax.SwitchExpr) bool {
	for _, arm := range e.Arms {
		if h.hasSwitch(arm.Body) || h.hasSwitch(arm.Cases) {
			return true
		}
	}
	return false
}

type switchExprHeader struct {
	init, tag span
	semi      int
}

func (h *tryHoist) exprHeader(e *syntax.SwitchExpr) switchExprHeader {
	fl := h.fl
	from := fl.off(e.Pos()) + len("switch")
	bound := fl.off(e.Rbrace)
	if len(e.Arms) > 0 {
		bound = fl.off(e.Arms[0].Pos())
	}
	brace := -1
	scanSegment(fl.src, from, bound, func(t tok) bool {
		if !t.comment && !t.auto {
			brace = t.start
		}
		return true
	})
	hdr := switchExprHeader{semi: -1}
	semis := fl.topSemis(from, bound)
	switch len(semis) {
	case 0:
		hdr.tag = fl.tokenSpan(from, brace)
	case 1:
		hdr.init = span{start: from, end: semis[0]}
		hdr.semi = semis[0]
		hdr.tag = fl.tokenSpan(semis[0]+1, brace)
	default:
		base.Fatalf("logos: cannot locate the header of the switch expression at %s", e.Pos().String())
	}
	return hdr
}

func (h *tryHoist) armBody(e *syntax.SwitchExpr, i int) span {
	fl := h.fl
	next := fl.off(e.Rbrace)
	if i+1 < len(e.Arms) {
		next = fl.off(e.Arms[i+1].Pos())
	}
	body := fl.tokenSpan(fl.off(e.Arms[i].Arrow)+len("=>"), next)
	toks := h.scan(body)
	if n := len(toks); n > 0 && toks[n-1].kind == token.SEMICOLON {
		body = fl.tokenSpan(body.start, toks[n-1].start)
	}
	return body
}

func (x *hoistHost) switchExpr(e *syntax.SwitchExpr, set func(syntax.Expr)) {
	h := x.h
	if h.needsHoist(e) {
		x.hoistSwitch(e, set, switchName(e))
		return
	}
	hdr := h.exprHeader(e)
	if h.hasSwitch(e.Init) {
		x.header(e.Init, h.fl.tokenSpan(hdr.init.start, hdr.init.end))
	}
	if h.hasSwitch(e.Tag) {
		x.child(e.Tag, func(v syntax.Expr) { e.Tag = v }, hdr.tag)
	}
}

func (x *hoistHost) hoistSwitch(e *syntax.SwitchExpr, set func(syntax.Expr), hostText string) {
	h := x.h
	fl := h.fl
	x.flush()
	pos := e.Pos()
	name := switchName(e)
	typ := fl.pkg.info.TypeOf(e)
	if typ == nil {
		base.Fatalf("logos: switch expression at %s has no checked type", pos.String())
	}
	nativeT, text, ok := fl.spellLoweredBoth(typ, pos)
	if !ok {
		base.Fatalf("logos: switch expression at %s has type %s that cannot be spelled in this file", pos.String(), typ)
	}
	hdr := h.exprHeader(e)
	native, prefix := x.switchPrefix(e, hdr)
	swMark := fl.moveRegion(span{start: fl.off(pos), end: fl.off(e.Rbrace) + 1}, hostText)
	body := []syntax.Stmt{assignOf(x.pos, []syntax.Expr{x.stub(name, pos, e)}, []syntax.Expr{e})}
	result := cat(gen(name+" = "), gen(swMark))
	if h.armsHoist(e) {
		body, result = x.armChain(e, name, hdr)
	}
	block := blockOf(x.pos, append(native, body...))
	if fl.anchorBlocks == nil {
		fl.anchorBlocks = make(map[*syntax.BlockStmt]bool)
	}
	fl.anchorBlocks[block] = true
	x.native = append(x.native, typedVarDecl(x.pos, nameOf(x.pos, name), nativeT), block)
	x.text = append(x.text, cat(gen("var "+name+" "+text+"; { "), cat(prefix...), result, gen(" }; ")))
	stub := x.stub(name, pos, e)
	if fl.stubEnd == nil {
		fl.stubEnd = make(map[*syntax.Name]int)
	}
	fl.stubEnd[stub] = fl.off(e.Rbrace) + 1
	set(stub)
}

func (x *hoistHost) switchPrefix(e *syntax.SwitchExpr, hdr switchExprHeader) ([]syntax.Stmt, []otext) {
	h := x.h
	fl := h.fl
	var native []syntax.Stmt
	var text []otext
	if e.Init != nil {
		if h.hasSwitch(e.Init) {
			ix := h.newHost(x.pos)
			ix.header(e.Init, fl.tokenSpan(hdr.init.start, hdr.init.end))
			native, text = ix.native, ix.text
		}
		mark := fl.moveRegion(hdr.init, "")
		fl.add(hdr.semi, hdr.semi+1, "")
		native = append(native, e.Init)
		text = append(text, gen(mark), gen("; "))
		e.Init = nil
	}
	if h.hasSwitch(e.Tag) {
		tx := h.newHost(x.pos)
		tx.child(e.Tag, func(v syntax.Expr) { e.Tag = v }, hdr.tag)
		native, text = append(native, tx.native...), append(text, tx.text...)
	}
	return native, text
}

func (x *hoistHost) armChain(e *syntax.SwitchExpr, name string, hdr switchExprHeader) ([]syntax.Stmt, otext) {
	h := x.h
	fl := h.fl
	pos := e.Pos()
	if _, guard := e.Tag.(*syntax.TypeSwitchGuard); guard {
		base.Fatalf("logos: switch expression with a try init in a type switch expression at %s", pos.String())
	}
	names := armNames{tag: fmt.Sprintf("_logos_tg_%d_%d", pos.Line(), pos.Col()), done: fmt.Sprintf("_logos_dn_%d_%d", pos.Line(), pos.Col())}
	var native []syntax.Stmt
	var text []otext
	if e.Tag != nil {
		native = append(native, defineOf(pos, []syntax.Expr{nameOf(pos, names.tag)}, []syntax.Expr{e.Tag}))
		text = append(text, gen(names.tag+" := "), gen(x.areaText(e.Tag, hdr.tag)), gen("; "))
	}
	native = append(native, defineOf(pos, []syntax.Expr{nameOf(pos, names.done)}, []syntax.Expr{nameOf(pos, "false")}))
	text = append(text, gen(names.done+" := false; "))
	var fallback []syntax.Stmt
	var fallbackText otext
	for i, arm := range e.Arms {
		if arm.Cases != nil {
			stmt, armText := x.caseArm(e, i, name, names)
			native, text = append(native, stmt), append(text, armText)
			continue
		}
		fallback, fallbackText = x.armResult(e, i, name)
	}
	if fallback == nil {
		msg := fmt.Sprintf("%q", fmt.Sprintf("logos: switch expression at %s:%d matched no arm", fl.baseName, pos.Line()))
		fail := exprStmtOf(e.Rbrace, callOf(e.Rbrace, nameOf(e.Rbrace, "panic"), []syntax.Expr{litOf(e.Rbrace, msg, syntax.StringLit)}))
		fallback, fallbackText = []syntax.Stmt{fail}, gen("panic("+msg+")")
	}
	native = append(native, ifOf(pos, nil, notDone(pos, names.done), fallback))
	text = append(text, cat(gen("if !"+names.done+" { "), fallbackText, gen(" }; ")))
	return native, cat(text...)
}

type armNames struct {
	tag, done string
}

func notDone(pos syntax.Pos, done string) syntax.Expr {
	return at(&syntax.Operation{Op: syntax.Not, X: nameOf(pos, done)}, pos)
}

func (x *hoistHost) armResult(e *syntax.SwitchExpr, i int, name string) ([]syntax.Stmt, otext) {
	h := x.h
	arm := e.Arms[i]
	ax := h.newHost(arm.Arrow)
	area := h.armBody(e, i)
	if h.hasSwitch(arm.Body) {
		ax.child(arm.Body, func(v syntax.Expr) { arm.Body = v }, area)
	}
	assign := assignOf(arm.Arrow, []syntax.Expr{x.stub(name, arm.Arrow, e)}, []syntax.Expr{arm.Body})
	return append(ax.native, assign), cat(ax.cat(), gen(name+" = "), gen(ax.areaText(arm.Body, area)))
}

func (x *hoistHost) caseArm(e *syntax.SwitchExpr, i int, name string, names armNames) (syntax.Stmt, otext) {
	h := x.h
	fl := h.fl
	arm := e.Arms[i]
	pos := e.Pos()
	area := fl.tokenSpan(fl.off(arm.Pos())+len("case"), fl.off(arm.Arrow))
	gx := h.newHost(arm.Pos())
	if l, list := arm.Cases.(*syntax.ListExpr); list {
		for _, later := range l.ElemList[1:] {
			if h.hasSwitch(later) {
				base.Fatalf("logos: switch expression with a try init after the first case expression at %s", later.Pos().String())
			}
		}
	}
	if h.hasSwitch(arm.Cases) {
		gx.items(arm.Cases, func(v syntax.Expr) { arm.Cases = v }, area)
	}
	guards := gx.areaText(arm.Cases, area)
	body, bodyText := x.armResult(e, i, name)
	matched := assignOf(pos, []syntax.Expr{nameOf(pos, names.done)}, []syntax.Expr{nameOf(pos, "true")})
	clause := at(&syntax.CaseClause{Cases: arm.Cases, Body: append([]syntax.Stmt{matched}, body...), Colon: arm.Arrow}, arm.Pos())
	var tag syntax.Expr
	head := "switch { case "
	if e.Tag != nil {
		tag = x.stub(names.tag, pos, e.Tag)
		head = "switch " + names.tag + " { case "
	}
	chosen := at(&syntax.SwitchStmt{Tag: tag, Body: []*syntax.CaseClause{clause}, Rbrace: e.Rbrace}, pos)
	stmt := ifOf(pos, nil, notDone(pos, names.done), append(gx.native, chosen))
	return stmt, cat(
		gen("if !"+names.done+" { "), gx.cat(), gen(head), gen(guards), gen(": "+names.done+" = true; "), bodyText, gen(" } }; "),
	)
}
