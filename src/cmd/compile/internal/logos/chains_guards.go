// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers the M6-S7 chain calls that sit in positions the plain
// statement hoist would move across an evaluation boundary: `for`
// conditions and post statements (which run every iteration), `if`
// conditions behind an init statement (whose names they read), `switch`
// case guards (which run only when earlier guards miss), right operands of
// `&&` and `||` (which run only when the left operand lets them), and
// `go { }` group children (whose `?` returns to the group, not to the
// enclosing function).
//
// Header text is located by scanning the source between the delimiters
// the parser records, never from syntax end positions, which stop before
// the closing delimiter of a call or a parenthesized operand. The
// parser keeps every parenthesis as a ParenExpr node, so each
// parenthesis the eject text adds around a user expression is a
// ParenExpr in the native tree as well; the printed native tree and the
// re-parse of the eject text then agree.

// chainGroup is the group-body scope of one chain list: inGroup tells a
// group statement body from ordinary code, inChild tells a child body
// (whose `?` returns only the error) from a group body.
type chainGroup struct {
	inGroup bool
	inChild bool
}

// chainChildResults spells the results enclosing a group child body: a
// child returns only the error, so a hoisted `?` returns it directly.
func chainChildResults() []chainResult {
	return []chainResult{{typ: types2.Universe.Lookup("error").Type()}}
}

// lowerChainSpecial lowers the statements whose chains need more than a
// statement-start hoist: groups, loops, switches, selects, and an if
// whose init statement declares names its condition may read. It reports
// false for every other statement, which keeps the plain path.
func (fl *fileLower) lowerChainSpecial(s syntax.Stmt, enc []chainResult) ([]syntax.Stmt, bool) {
	inner := s
	for {
		lab, ok := inner.(*syntax.LabeledStmt)
		if !ok {
			break
		}
		inner = lab.Stmt
	}
	switch inner := inner.(type) {
	case *syntax.GroupStmt, *syntax.GroupChild:
		return fl.lowerChainGroup(s, inner, enc), true
	case *syntax.ForStmt:
		return fl.lowerChainFor(s, inner, enc), true
	case *syntax.IfStmt:
		if inner.Init != nil {
			return fl.lowerChainIf(s, inner, enc), true
		}
	case *syntax.SwitchStmt:
		return fl.lowerChainSwitch(s, inner, enc), true
	case *syntax.SelectStmt:
		return fl.lowerChainSelect(s, inner, enc), true
	}
	return nil, false
}

// recvAssign reports the receive assignment `targets = <-ch` of a select
// clause, or nil for every other clause.
func recvAssign(c *syntax.CommClause) *syntax.AssignStmt {
	as, ok := c.Comm.(*syntax.AssignStmt)
	if !ok || as.Op != 0 {
		return nil
	}
	if op, ok := syntax.Unparen(as.Rhs).(*syntax.Operation); ok && op.Op == syntax.Recv && op.Y == nil {
		return as
	}
	return nil
}

// lowerChainSelect lowers the chains of one select statement. Channel
// and send operands evaluate once on entry in source order, which the
// plain hoist already does. The targets of a receive assignment evaluate
// only when their clause wins, so a hoisting chain in one receives into
// temporaries and runs behind the receive, at the top of the clause body.
func (fl *fileLower) lowerChainSelect(s syntax.Stmt, ss *syntax.SelectStmt, enc []chainResult) []syntax.Stmt {
	targets := make([][]chainHoist, len(ss.Body))
	for i, c := range ss.Body {
		if as := recvAssign(c); as != nil {
			targets[i] = fl.lowerChainPiece(as.Lhs, enc)
		}
	}
	out := fl.lowerChainPlain(s, enc)
	for i, c := range ss.Body {
		if len(targets[i]) > 0 {
			fl.splitRecvTarget(c, recvAssign(c), targets[i])
		}
	}
	return out
}

// splitRecvTarget rewrites `case lhs = <-ch:` into
// `case t0, ... := <-ch:` with the hoists and `lhs = t0, ...` leading the
// body.
func (fl *fileLower) splitRecvTarget(c *syntax.CommClause, as *syntax.AssignStmt, hoists []chainHoist) {
	pos := c.Pos()
	from := fl.off(pos) + len("case")
	to := fl.off(c.Colon)
	eq, eqEnd, depth := -1, -1, 0
	scanSegment(fl.src, from, to, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		case token.ASSIGN:
			if depth == 0 && eq < 0 {
				eq, eqEnd = t.start, t.end
			}
		}
		return true
	})
	if eq < 0 {
		base.Fatalf("logos: cannot locate the receive assignment at %s", pos.String())
	}
	lhs, rhs := fl.tokenSpan(from, eq), fl.tokenSpan(eqEnd, to)
	pfx := fmt.Sprintf("_logos_rv_%d_%d", pos.Line(), pos.Col())
	var names []string
	var temps, vals []syntax.Expr
	for i := range syntax.UnpackListExpr(as.Lhs) {
		name := fmt.Sprintf("%s_%d", pfx, i)
		names = append(names, name)
		temps = append(temps, nameOf(pos, name))
		vals = append(vals, nameOf(pos, name))
	}
	list := strings.Join(names, ", ")
	region := span{start: fl.off(pos), end: to + 1}
	var text strings.Builder
	text.WriteString(leftoverComments(fl, region, []span{lhs, rhs}))
	text.WriteString("case " + list + " := " + fl.copyRegion(rhs) + ": ")
	var prefix []syntax.Stmt
	for _, h := range hoists {
		text.WriteString(h.text + "; ")
		prefix = append(prefix, h.stmts...)
	}
	anchorScopes(prefix, pos)
	text.WriteString(fl.copyRegion(lhs) + " = " + list + "; ")
	fl.add(region.start, region.end, text.String())
	prefix = append(prefix, assignOf(pos, syntax.UnpackListExpr(as.Lhs), vals))
	c.Comm = defineOf(pos, temps, []syntax.Expr{as.Rhs})
	c.Body = append(prefix, c.Body...)
}

// lowerChainGroup lowers the chains of one group statement or child
// block. Head expressions of a group statement run with the enclosing
// results. A child block always runs as a child, so `?` returns only
// the error; so does a group statement that sits in a group body,
// because the statement pass wraps a nested group as a child of that
// group.
func (fl *fileLower) lowerChainGroup(s syntax.Stmt, inner syntax.Stmt, enc []chainResult) []syntax.Stmt {
	saved := fl.chainGrp
	encBody := enc
	bodyGrp := chainGroup{inGroup: true}
	if _, isChild := inner.(*syntax.GroupChild); isChild || saved.inGroup || saved.inChild {
		encBody = chainChildResults()
		bodyGrp = chainGroup{inGroup: true, inChild: true}
	}
	var out []syntax.Stmt
	var body *syntax.BlockStmt
	if gs, ok := inner.(*syntax.GroupStmt); ok {
		body = gs.Body
		for _, e := range []syntax.Expr{gs.Ctx, gs.Limit} {
			if e == nil {
				continue
			}
			for _, h := range fl.lowerChainPiece(e, enc) {
				out = append(out, h.stmts...)
				fl.add(fl.stmtStartOf(s), fl.stmtStartOf(s), h.text+"; ")
			}
		}
	} else {
		body = inner.(*syntax.GroupChild).Body
	}
	fl.chainGrp = bodyGrp
	body.List = fl.lowerChainList(body.List, encBody)
	fl.chainGrp = saved
	return append(out, s)
}

// lowerChainHeadLits lowers the function literals under one header piece
// with their own results and no group scope. Transparent option wrappers
// belong to the enclosing function, so the walk descends through them.
func (fl *fileLower) lowerChainHeadLits(n syntax.Node) {
	saved := fl.chainGrp
	syntax.Inspect(n, func(x syntax.Node) bool {
		if lit, ok := x.(*syntax.FuncLit); ok {
			if fl.transparent[lit] {
				return true
			}
			savedExit := fl.chainExit
			fl.chainGrp = chainGroup{}
			fl.chainExit = nil
			lit.Body.List = fl.lowerChainList(lit.Body.List, fl.chainFuncResults(lit.Type))
			fl.chainGrp = saved
			fl.chainExit = savedExit
			return false
		}
		return true
	})
}

// chainsInNode collects the outermost chain calls under one expression
// or simple statement in source order, skipping nested function
// literals other than transparent option wrappers, whose chains belong
// to the enclosing function.
func (fl *fileLower) chainsInNode(n syntax.Node) []*syntax.CallExpr {
	var all []*syntax.CallExpr
	syntax.Inspect(n, func(x syntax.Node) bool {
		if lit, ok := x.(*syntax.FuncLit); ok && !fl.transparent[lit] {
			return false
		}
		if call, ok := x.(*syntax.CallExpr); ok && fl.isChainCall(call) {
			all = append(all, call)
		}
		return true
	})
	return fl.outermostChains(all)
}

// lowerChainPiece lowers the chains of one header piece with the
// enclosing results and returns their hoists in source order. Function
// literals lower first, so every splice inside the piece exists before
// any caller reads the piece's text.
func (fl *fileLower) lowerChainPiece(n syntax.Node, enc []chainResult) []chainHoist {
	fl.lowerChainHeadLits(n)
	return fl.lowerChainUnit(n, fl.chainsInNode(n), enc)
}

// lowerChainPieceExit is lowerChainPiece for a header piece whose hoists
// run inside a closure: exit spells how a failing hoist leaves it.
func (fl *fileLower) lowerChainPieceExit(n syntax.Node, enc []chainResult, exit func(err string) string) []chainHoist {
	fl.lowerChainHeadLits(n)
	saved := fl.chainExit
	fl.chainExit = exit
	out := fl.lowerChainUnit(n, fl.chainsInNode(n), enc)
	fl.chainExit = saved
	return out
}

// stmtStartPos reports the position of the first token of s. The syntax
// package places a labeled statement at its colon, so a labeled
// statement starts at its outermost label.
func stmtStartPos(s syntax.Stmt) syntax.Pos {
	if lab, ok := s.(*syntax.LabeledStmt); ok {
		return lab.Label.Pos()
	}
	return syntax.StartPos(s)
}

// stmtLabels reports every label carried by one labeled statement,
// outermost first: the wrapper of a lowered switch repeats them ahead of
// the switch, so `break` and `continue` naming them keep their target.
func stmtLabels(s syntax.Stmt) []string {
	var out []string
	for {
		lab, ok := s.(*syntax.LabeledStmt)
		if !ok || lab.Label == nil {
			return out
		}
		out = append(out, lab.Label.Value)
		s = lab.Stmt
	}
}

// topSemis reports the offsets of the semicolons of src[from:to] that
// sit outside every bracket pair: the delimiters of a for or switch
// header. Automatic semicolons never delimit a header.
func (fl *fileLower) topSemis(from, to int) []int {
	var out []int
	depth := 0
	scanSegment(fl.src, from, to, func(t tok) bool {
		if t.comment || t.auto {
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

// tokenSpan reports the span of the tokens of src[from:to], from the
// start of the first to the end of the last, comments and automatic
// semicolons aside. A range without tokens reports an empty span at from.
func (fl *fileLower) tokenSpan(from, to int) span {
	out := span{start: from, end: from}
	seen := false
	scanSegment(fl.src, from, to, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if !seen {
			out.start = t.start
			seen = true
		}
		out.end = t.end
		return true
	})
	return out
}

// forHeader locates the tokens of the clauses a for statement keeps in
// its header.
type forHeader struct {
	cond, post span
}

// forHeaderOf scans the header of fs, which carries a condition or a post
// statement: either a lone condition (`for cond {`) or three clauses
// (`for init; cond; post {`).
func (fl *fileLower) forHeaderOf(fs *syntax.ForStmt) forHeader {
	from := fl.off(fs.Pos()) + len("for")
	to := fl.off(fs.Body.Pos())
	semis := fl.topSemis(from, to)
	var h forHeader
	switch len(semis) {
	case 0:
		h.cond = fl.tokenSpan(from, to)
		h.post = span{start: to, end: to}
	case 2:
		h.cond = fl.tokenSpan(semis[0]+1, semis[1])
		h.post = fl.tokenSpan(semis[1]+1, to)
	default:
		base.Fatalf("logos: cannot locate the for header at %s", fs.Pos().String())
	}
	return h
}

// chainErrName mints the error variable of one rewritten statement from
// its position, so every rewritten statement owns a name no user code can
// spell.
func chainErrName(pos syntax.Pos) string {
	return fmt.Sprintf("_logos_e_%d_%d", pos.Line(), pos.Col())
}

// condFail spells the failing branch of a hoist inside a condition
// closure: the error goes to the statement's variable and the condition
// reports true, so the guarded body runs next and returns the error
// before any user code.
func condFail(errVar string) func(err string) string {
	return func(err string) string { return errVar + " = " + err + "; return true" }
}

// postFail spells the failing branch of a hoist inside a post closure:
// the error goes to the statement's variable, which the condition and the
// body test before they run.
func postFail(errVar string) func(err string) string {
	return func(err string) string { return errVar + " = " + err + "; return" }
}

// declareChainErr declares the error variable of the statement s ahead of
// it, as eject text and as a native statement.
func (fl *fileLower) declareChainErr(s syntax.Stmt, errName string) []syntax.Stmt {
	start := fl.stmtStartOf(s)
	fl.add(start, start, "var "+errName+" error; ")
	return parseChainStmtFrag(fl, "var "+errName+" error", stmtStartPos(s))
}

// chainErrCheck builds `if err != nil { return ... }`, the first
// statement of a body whose header closures may have failed, as native
// statements and as eject text.
func (fl *fileLower) chainErrCheck(enc []chainResult, pos syntax.Pos, errName string) ([]syntax.Stmt, string) {
	ret := (&chainGen{fl: fl, pos: pos, enc: enc}).returnText(errName)
	text := "if " + errName + " != nil { " + ret + " }"
	return parseChainStmtFrag(fl, text, pos), text
}

// condRewrite is one condition rewritten into a closure: the native call
// and the eject text that goes before and after the condition's own
// tokens, which stay where they are.
type condRewrite struct {
	call       *syntax.CallExpr
	head, tail string
}

// condClosure wraps cond in `func() bool { hoists; return cond }()`. The
// hoists run every time the condition is evaluated, which is where the
// source evaluates its chains, and the condition's tokens stay in place:
// only insertions go around them, so edits of later passes inside the
// condition land where the tokens are. The closure's scopes open at
// anchor.
func (fl *fileLower) condClosure(cond syntax.Expr, hoists []chainHoist, anchor syntax.Pos) condRewrite {
	call, ok := parseChainExprFrag(fl, "func() bool { return _logos_cond }()", anchor).(*syntax.CallExpr)
	if !ok {
		base.Fatalf("logos: cannot build the condition closure at %s", anchor.String())
	}
	lit := call.Fun.(*syntax.FuncLit)
	ret := lit.Body.List[0].(*syntax.ReturnStmt)
	ret.Results = cond
	var body []syntax.Stmt
	var head strings.Builder
	head.WriteString("func() bool { ")
	for _, h := range hoists {
		anchorScopes(h.stmts, anchor)
		body = append(body, h.stmts...)
		head.WriteString(h.text + "; ")
	}
	head.WriteString("return ")
	lit.Body.List = append(body, ret)
	return condRewrite{call: call, head: head.String(), tail: " }()"}
}

// postClosure wraps post in `func() { hoists; post }()`. The call stays
// the loop's post statement, so Go still makes the next iteration's
// variables before it runs, and closures of the body keep the values of
// their own iteration. The post tokens stay in place between insertions.
func (fl *fileLower) postClosure(post syntax.SimpleStmt, hoists []chainHoist, sp span, anchor syntax.Pos) syntax.SimpleStmt {
	call, ok := parseChainExprFrag(fl, "func() { }()", anchor).(*syntax.CallExpr)
	if !ok {
		base.Fatalf("logos: cannot build the post closure at %s", anchor.String())
	}
	lit := call.Fun.(*syntax.FuncLit)
	var body []syntax.Stmt
	var head strings.Builder
	head.WriteString("func() { ")
	for _, h := range hoists {
		anchorScopes(h.stmts, anchor)
		body = append(body, h.stmts...)
		head.WriteString(h.text + "; ")
	}
	lit.Body.List = append(body, post)
	fl.add(sp.start, sp.start, head.String())
	fl.add(sp.end, sp.end, " }()")
	return exprStmtOf(anchor, call)
}

// lowerChainFor lowers the chains of one for statement. Init chains run
// once before the loop, which the plain hoist already does. A condition
// chain runs every iteration after the post statement, and a post chain
// runs every iteration after the body, so each one hoists into a closure
// that the header keeps in the clause's place: `continue` still reaches
// the post statement, Go still makes the iteration's variables before the
// post statement runs, and the clauses run when the source runs them. A
// closure cannot return from the function, so a failing hoist stores its
// error in a variable of the loop and lets the clause report true (the
// condition) or end (the post); the condition tests the variable before
// it runs again, and the body tests it before any user code. Pure chains
// lower in place and need no rewrite.
func (fl *fileLower) lowerChainFor(s syntax.Stmt, fs *syntax.ForStmt, enc []chainResult) []syntax.Stmt {
	var out []syntax.Stmt
	if fs.Init != nil {
		for _, h := range fl.lowerChainPiece(fs.Init, enc) {
			out = append(out, h.stmts...)
			fl.add(fl.stmtStartOf(s), fl.stmtStartOf(s), h.text+"; ")
		}
	}
	errName := chainErrName(fs.Pos())
	var condHoists, postHoists []chainHoist
	if fs.Cond != nil {
		condHoists = fl.lowerChainPieceExit(fs.Cond, enc, condFail(errName))
	}
	if fs.Post != nil {
		postHoists = fl.lowerChainPieceExit(fs.Post, enc, postFail(errName))
	}
	if len(condHoists) == 0 && len(postHoists) == 0 {
		fl.lowerChainChildren(s, enc)
		return append(out, s)
	}
	hdr := fl.forHeaderOf(fs)
	anchor := fs.Body.Pos()
	out = append(out, fl.declareChainErr(s, errName)...)
	// The rewritten clauses install after the body lowers, so the walk of
	// function literals never meets the synthesized closures.
	cond, post := fs.Cond, fs.Post
	var head, tail string
	if len(condHoists) > 0 {
		cc := fl.condClosure(fs.Cond, condHoists, anchor)
		cond, head, tail = cc.call, cc.head, cc.tail
	}
	if len(postHoists) > 0 {
		post = fl.postClosure(fs.Post, postHoists, hdr.post, anchor)
		if fs.Cond != nil {
			head = errName + " != nil || (" + head
			tail += ")"
			cond = at(&syntax.Operation{Op: syntax.OrOr, X: notNil(anchor, nameOf(anchor, errName)), Y: at(&syntax.ParenExpr{X: cond}, anchor)}, anchor)
		}
	}
	if head != "" {
		fl.add(hdr.cond.start, hdr.cond.start, head)
		fl.add(hdr.cond.end, hdr.cond.end, tail)
	}
	check, checkText := fl.chainErrCheck(enc, anchor, errName)
	open := fl.off(anchor) + 1
	fl.add(open, open, " "+checkText+"; ")
	fl.lowerChainChildren(s, enc)
	fs.Cond, fs.Post = cond, post
	fs.Body.List = append(check, fs.Body.List...)
	return append(out, s)
}

// ifCondSpan reports the tokens of the condition of an if with an init
// statement.
func (fl *fileLower) ifCondSpan(ifs *syntax.IfStmt) span {
	from := fl.off(ifs.Pos()) + len("if")
	to := fl.off(ifs.Then.Pos())
	semis := fl.topSemis(from, to)
	if len(semis) != 1 {
		base.Fatalf("logos: cannot locate the if header at %s", ifs.Pos().String())
	}
	return fl.tokenSpan(semis[0]+1, to)
}

// lowerChainIf lowers the chains of one if statement with an init
// statement. Init chains run once before the statement, which the plain
// hoist already does. Condition chains run after the init statement, whose
// names they may read, so they hoist into a closure that the header keeps
// in the condition's place, and the then branch tests the error a failing
// hoist leaves, before any user code: the condition reports true on
// failure, so the else branch never runs.
func (fl *fileLower) lowerChainIf(s syntax.Stmt, ifs *syntax.IfStmt, enc []chainResult) []syntax.Stmt {
	var out []syntax.Stmt
	start := fl.stmtStartOf(s)
	for _, h := range fl.lowerChainPiece(ifs.Init, enc) {
		out = append(out, h.stmts...)
		fl.add(start, start, h.text+"; ")
	}
	errName := chainErrName(ifs.Pos())
	hoists := fl.lowerChainPieceExit(ifs.Cond, enc, condFail(errName))
	if len(hoists) == 0 {
		fl.lowerChainChildren(s, enc)
		return append(out, s)
	}
	out = append(out, fl.declareChainErr(s, errName)...)
	cond, check := fl.rewriteIfCond(ifs, hoists, enc, errName)
	fl.lowerChainChildren(s, enc)
	ifs.Cond = cond
	ifs.Then.List = append(check, ifs.Then.List...)
	return append(out, s)
}

// rewriteIfCond records the eject text that turns the condition of ifs,
// whose chains hoist, into a closure call, and builds the native closure
// call and the error test that leads the then branch. The caller installs
// both, and declares errName ahead of the if.
func (fl *fileLower) rewriteIfCond(ifs *syntax.IfStmt, hoists []chainHoist, enc []chainResult, errName string) (*syntax.CallExpr, []syntax.Stmt) {
	anchor := ifs.Then.Pos()
	sp := fl.ifCondSpan(ifs)
	cc := fl.condClosure(ifs.Cond, hoists, anchor)
	fl.add(sp.start, sp.start, cc.head)
	fl.add(sp.end, sp.end, cc.tail)
	check, checkText := fl.chainErrCheck(enc, anchor, errName)
	open := fl.off(anchor) + 1
	fl.add(open, open, " "+checkText+"; ")
	return cc.call, check
}

// chainPaths records where the chains of one evaluation unit sit relative
// to its short-circuit operations: ops lists, for every chain, the
// operations whose right operand holds it, outermost first; spans and
// types describe the left operand of every operation listed, read from
// the tree before any lowering rewrites it.
type chainPaths struct {
	ops   map[*syntax.CallExpr][]*syntax.Operation
	spans map[*syntax.Operation]span
	types map[*syntax.Operation]string
}

// shortCircuitOffset reports the source offset of the operator of op when
// op is a written `&&` or `||`. Generated fragments carry normalized
// positions that never spell the operator.
func (fl *fileLower) shortCircuitOffset(op *syntax.Operation) (int, bool) {
	if op.Y == nil || (op.Op != syntax.AndAnd && op.Op != syntax.OrOr) || !op.Pos().IsKnown() {
		return 0, false
	}
	if line := int(op.Pos().Line()); line > len(fl.start) {
		return 0, false
	}
	off := fl.off(op.Pos())
	want := "&&"
	if op.Op == syntax.OrOr {
		want = "||"
	}
	if off < 0 || off+len(want) > len(fl.src) || string(fl.src[off:off+len(want)]) != want {
		return 0, false
	}
	return off, true
}

// operandSpan reports the source span of the left operand of op: from its
// first token to the end of its last token before the operator. The
// operand of a binary operation starts at its own leftmost operand.
func (fl *fileLower) operandSpan(op *syntax.Operation, opOff int) span {
	x := op.X
	for {
		inner, ok := x.(*syntax.Operation)
		if !ok || inner.Y == nil {
			break
		}
		x = inner.X
	}
	start := fl.exprStart(x)
	return span{start: start, end: lastTokenEnd(fl.src, start, opOff)}
}

// shortOperandType spells the type of a short-circuit operand for the
// temporary that holds it: bool, or the named boolean type the operand
// has.
func (fl *fileLower) shortOperandType(x syntax.Expr) string {
	t := fl.pkg.info.TypeOf(x)
	if t == nil || isUntypedBasic(t) {
		return "bool"
	}
	if b, ok := types2.Unalias(t).(*types2.Basic); ok && b.Kind() == types2.Bool {
		return "bool"
	}
	if spelled, ok := fl.spellType(t); ok {
		return spelled
	}
	return "bool"
}

// chainPathsOf locates the calls of one evaluation unit under the
// short-circuit operations of root. Function literals other than
// transparent option wrappers and statements nested under root belong to
// other units, so the walk never enters them.
func (fl *fileLower) chainPathsOf(root syntax.Node, calls []*syntax.CallExpr) chainPaths {
	want := make(map[*syntax.CallExpr]bool, len(calls))
	for _, c := range calls {
		want[c] = true
	}
	rootStmt, _ := root.(syntax.Stmt)
	skip := make(map[syntax.Stmt]bool)
	if rootStmt != nil {
		nestedChainStmts(rootStmt, skip)
	}
	out := chainPaths{
		ops:   make(map[*syntax.CallExpr][]*syntax.Operation),
		spans: make(map[*syntax.Operation]span),
		types: make(map[*syntax.Operation]string),
	}
	var walk func(n syntax.Node, path []*syntax.Operation)
	walk = func(n syntax.Node, path []*syntax.Operation) {
		syntax.Inspect(n, func(x syntax.Node) bool {
			switch x := x.(type) {
			case *syntax.FuncLit:
				return fl.transparent[x]
			case *syntax.CallExpr:
				if want[x] {
					out.ops[x] = path
					for _, op := range path {
						if _, seen := out.spans[op]; !seen {
							off, _ := fl.shortCircuitOffset(op)
							out.spans[op] = fl.operandSpan(op, off)
							out.types[op] = fl.shortOperandType(op.X)
						}
					}
					return false
				}
			case *syntax.Operation:
				if _, ok := fl.shortCircuitOffset(x); ok {
					walk(x.X, path)
					walk(x.Y, append(slices.Clone(path), x))
					return false
				}
			case syntax.Stmt:
				if x != rootStmt && skip[x] {
					return false
				}
			}
			return true
		})
	}
	walk(root, nil)
	return out
}

// chainTries reports whether lowering call hoists a computation: the call,
// or a chain nested under it, carries a `?` lambda.
func (fl *fileLower) chainTries(call *syntax.CallExpr) bool {
	found := false
	syntax.Inspect(call, func(n syntax.Node) bool {
		if found {
			return false
		}
		if lit, ok := n.(*syntax.FuncLit); ok && !fl.transparent[lit] {
			return false
		}
		c, ok := n.(*syntax.CallExpr)
		if !ok || !fl.isChainCall(c) {
			return true
		}
		for _, arg := range c.ArgList {
			lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr)
			if !ok {
				continue
			}
			if body, ok := lam.Body.(syntax.Expr); ok {
				if _, ok := syntax.Unparen(body).(*syntax.TryExpr); ok {
					found = true
				}
			}
		}
		return true
	})
	return found
}

// lowerChainUnit lowers the outermost chains of one evaluation unit, a
// statement or a header piece, and returns their hoists in the order the
// source evaluates them. A hoist computes where the unit starts, so a
// chain that the source evaluates only when a short-circuit operand lets
// it run needs more than a place in that order: the operand evaluates
// first, and the hoist runs only while the operand admits it.
func (fl *fileLower) lowerChainUnit(root syntax.Node, calls []*syntax.CallExpr, enc []chainResult) []chainHoist {
	if len(calls) == 0 {
		return nil
	}
	paths := fl.chainPathsOf(root, calls)
	var hoisting []*syntax.CallExpr
	guarded := false
	for _, call := range calls {
		if fl.chainTries(call) {
			hoisting = append(hoisting, call)
			guarded = guarded || len(paths.ops[call]) > 0
		}
	}
	if !guarded {
		var out []chainHoist
		for _, call := range calls {
			out = append(out, fl.lowerChainTopLevel(call, enc)...)
		}
		return out
	}
	// Chains that do not hoist lower where they stand, ahead of every
	// operand text the guarded hoists copy.
	for _, call := range calls {
		if !fl.chainTries(call) && len(fl.lowerChainTopLevel(call, enc)) > 0 {
			base.Fatalf("logos: chain at %s hoists without a `?` lambda", call.Pos().String())
		}
	}
	return fl.lowerChainRegion(hoisting, paths, 0, "", enc)
}

// chainItem is one step of a region in source order: a chain of the
// region itself, or a short-circuit operation holding chains in its right
// operand.
type chainItem struct {
	key   int
	call  *syntax.CallExpr
	op    *syntax.Operation
	calls []*syntax.CallExpr
}

// lowerChainRegion lowers the hoisting chains of one region of a unit:
// the chains whose enclosing short-circuit operations are the first depth
// entries of their paths, all behind the same guard. A chain with no
// further operation hoists under the guard. An operation holding chains
// in its right operand evaluates its left operand into a temporary, then
// the chains of the right operand hoist under the guard extended by that
// temporary (negated for `||`), and the operation reads the temporary in
// the operand's place. Items run in source order: the operator sits after
// every chain of its left operand and before every chain of its right.
func (fl *fileLower) lowerChainRegion(calls []*syntax.CallExpr, paths chainPaths, depth int, guard string, enc []chainResult) []chainHoist {
	var items []chainItem
	byOp := make(map[*syntax.Operation]int)
	for _, call := range calls {
		path := paths.ops[call]
		if len(path) == depth {
			items = append(items, chainItem{key: fl.exprStart(call), call: call})
			continue
		}
		op := path[depth]
		i, seen := byOp[op]
		if !seen {
			off, _ := fl.shortCircuitOffset(op)
			i = len(items)
			byOp[op] = i
			items = append(items, chainItem{key: off, op: op})
		}
		items[i].calls = append(items[i].calls, call)
	}
	slices.SortStableFunc(items, func(a, b chainItem) int { return a.key - b.key })
	var out []chainHoist
	for _, it := range items {
		if it.op == nil {
			out = append(out, fl.lowerChainGuarded(it.call, enc, guard)...)
			continue
		}
		name := fmt.Sprintf("_logos_sc_%d_%d", it.op.Pos().Line(), it.op.Pos().Col())
		out = append(out, fl.moveShortOperand(it.op, paths.spans[it.op], paths.types[it.op], name, guard))
		cond := name
		if it.op.Op == syntax.OrOr {
			cond = "!" + name
		}
		if guard != "" {
			cond = guard + " && " + cond
		}
		out = append(out, fl.lowerChainRegion(it.calls, paths, depth+1, cond, enc)...)
	}
	return out
}

// moveShortOperand hoists the left operand of a short-circuit operation
// into the temporary name and makes the operation read the temporary. The
// operand evaluates once, behind guard when the operation itself is
// conditional; the right operand stays where it is, so the operation
// still short-circuits on the same value. The operand's source region is
// a moved region: its edits, including those of later passes, belong to
// the copy in the hoist text, and the region itself holds the name.
func (fl *fileLower) moveShortOperand(op *syntax.Operation, sp span, typ, name, guard string) chainHoist {
	x := op.X
	pos := op.Pos()
	fl.lowerChainHeadLits(x)
	// The native hoists hold the operand node itself. Only the eject text
	// carries the mark, which resolveChainMoves replaces with the composed
	// copy of the operand before any edit is emitted; no parser sees it.
	mark := fl.moveRegion(sp, name)
	op.X = nameOf(leftmostOf(x), name)
	if guard == "" {
		return chainHoist{
			stmts: []syntax.Stmt{defineOf(pos, []syntax.Expr{nameOf(pos, name)}, []syntax.Expr{x})},
			text:  name + " := " + mark,
		}
	}
	take := ifOf(pos, nil, parseChainExprFrag(fl, guard, pos), []syntax.Stmt{
		assignOf(pos, []syntax.Expr{nameOf(pos, name)}, []syntax.Expr{x}),
	})
	return chainHoist{
		stmts: append(parseChainStmtFrag(fl, "var "+name+" "+typ, pos), take),
		text:  "var " + name + " " + typ + "; if " + guard + " { " + name + " = " + mark + " }",
	}
}

// swGuard is one guard expression of a case clause: the hoists of its
// chains and, once the switch is known to rewrite, its source tokens.
type swGuard struct {
	sp     span
	hoists []chainHoist
}

// switchHeader locates the parts of a switch statement header: the
// region from the first token of the statement (its outermost label, if
// any) through the opening brace, and the tokens of the init statement
// and of the tag inside it.
type switchHeader struct {
	region    span
	init, tag span
}

// isTypeSwitchTag reports whether tag is a type switch guard.
func isTypeSwitchTag(tag syntax.Expr) bool {
	_, ok := syntax.Unparen(tag).(*syntax.TypeSwitchGuard)
	return ok
}

// switchHeaderOf scans the header of ss, which has at least one clause:
// the opening brace is the last token before the first clause, and a
// semicolon outside every bracket pair splits the init statement from
// the tag.
func (fl *fileLower) switchHeaderOf(s syntax.Stmt, ss *syntax.SwitchStmt) switchHeader {
	from := fl.off(ss.Pos()) + len("switch")
	first := fl.off(ss.Body[0].Pos())
	brace := -1
	scanSegment(fl.src, from, first, func(t tok) bool {
		if !t.comment && !t.auto {
			brace = t.start
		}
		return true
	})
	if brace < 0 || fl.src[brace] != '{' {
		base.Fatalf("logos: cannot locate the switch body at %s", ss.Pos().String())
	}
	h := switchHeader{region: span{start: fl.off(stmtStartPos(s)), end: brace + 1}}
	semis := fl.topSemis(from, brace)
	switch len(semis) {
	case 0:
		h.tag = fl.tokenSpan(from, brace)
	case 1:
		h.init = fl.tokenSpan(from, semis[0])
		h.tag = fl.tokenSpan(semis[0]+1, brace)
	default:
		base.Fatalf("logos: cannot locate the switch header at %s", ss.Pos().String())
	}
	return h
}

// guardSpans splits the source between a case keyword and its colon at
// the commas outside every bracket pair: one token span per guard.
func (fl *fileLower) guardSpans(c *syntax.CaseClause) []span {
	from := fl.off(c.Pos()) + len("case")
	to := fl.off(c.Colon)
	var out []span
	depth := 0
	start := from
	scanSegment(fl.src, from, to, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE:
			depth--
		case token.COMMA:
			if depth == 0 {
				out = append(out, fl.tokenSpan(start, t.start))
				start = t.end
			}
		}
		return true
	})
	return append(out, fl.tokenSpan(start, to))
}

// clauseGuards reports the guard expressions of one case clause.
func clauseGuards(c *syntax.CaseClause) []syntax.Expr {
	return syntax.UnpackListExpr(c.Cases)
}

// lowerChainSwitch lowers the chains of one switch statement. Init and
// tag chains run once before any guard, which the plain hoist already
// does. Guard chains run only when earlier guards miss, so when one
// hoists the switch splits in two: a selection phase evaluates the tag
// once and then each guard in source order, only while no guard has
// matched, recording the number of the matching clause, and the switch
// itself keeps its clauses and bodies but switches on that number. The
// bodies keep their break, fallthrough, and default meaning because the
// clauses keep their order; a default clause anywhere runs when the
// selection stays zero. Type switches and switches with only pure
// chains keep the plain path.
func (fl *fileLower) lowerChainSwitch(s syntax.Stmt, ss *syntax.SwitchStmt, enc []chainResult) []syntax.Stmt {
	if len(ss.Body) == 0 || isTypeSwitchTag(ss.Tag) {
		return fl.lowerChainPlain(s, enc)
	}
	start := fl.stmtStartOf(s)
	var pre []syntax.Stmt
	if ss.Init != nil {
		for _, h := range fl.lowerChainPiece(ss.Init, enc) {
			pre = append(pre, h.stmts...)
			fl.add(start, start, h.text+"; ")
		}
	}
	var tagHoists []chainHoist
	if ss.Tag != nil {
		tagHoists = fl.lowerChainPiece(ss.Tag, enc)
	}
	runs := make([][]swGuard, len(ss.Body))
	guardHoists := 0
	for i, c := range ss.Body {
		if c.Cases == nil {
			continue
		}
		for _, g := range clauseGuards(c) {
			run := swGuard{hoists: fl.lowerChainPiece(g, enc)}
			guardHoists += len(run.hoists)
			runs[i] = append(runs[i], run)
		}
	}
	// Tag chains evaluate after the init statement, whose names they may
	// read, so a tag that hoists behind an init statement takes the
	// wrapper too.
	if guardHoists == 0 && (ss.Init == nil || len(tagHoists) == 0) {
		out := pre
		for _, h := range tagHoists {
			out = append(out, h.stmts...)
			fl.add(start, start, h.text+"; ")
		}
		out = append(out, s)
		fl.lowerChainChildren(s, enc)
		return out
	}
	return append(pre, fl.emitChainSwitch(s, ss, tagHoists, runs, enc))
}

// emitChainSwitch assembles the selection phase and the clause switch of
// one switch whose guards hoist, as native nodes and as eject text. The
// wrapper block opens at the first token of the statement and closes at
// the switch's closing brace, and every synthesized scope opens at the
// wrapper's position, so scope positions never move backward.
func (fl *fileLower) emitChainSwitch(s syntax.Stmt, ss *syntax.SwitchStmt, tagHoists []chainHoist, runs [][]swGuard, enc []chainResult) *syntax.BlockStmt {
	hdr := fl.switchHeaderOf(s, ss)
	p0 := stmtStartPos(s)
	pfx := fmt.Sprintf("_logos_sw_%d_%d", ss.Pos().Line(), ss.Pos().Col())
	tagN, selN, matchN := pfx+"_tag", pfx+"_sel", pfx+"_m"
	hasTag := ss.Tag != nil
	var wrapper []syntax.Stmt
	var text strings.Builder
	text.WriteString("{ " + leftoverComments(fl, hdr.region, []span{hdr.init, hdr.tag}))
	if ss.Init != nil {
		text.WriteString(fl.copyRegion(hdr.init) + "; ")
		wrapper = append(wrapper, ss.Init)
	}
	for _, h := range tagHoists {
		text.WriteString(h.text + "; ")
		anchorScopes(h.stmts, p0)
		wrapper = append(wrapper, h.stmts...)
	}
	if hasTag {
		text.WriteString(tagN + " := " + fl.copyRegion(hdr.tag) + "; ")
		wrapper = append(wrapper, defineOf(p0, []syntax.Expr{nameOf(p0, tagN)}, []syntax.Expr{ss.Tag}))
	}
	text.WriteString(selN + " := 0; ")
	wrapper = append(wrapper, defineOf(p0, []syntax.Expr{nameOf(p0, selN)}, []syntax.Expr{litOf(p0, "0", syntax.IntLit)}))
	for i, c := range ss.Body {
		if c.Cases == nil {
			continue
		}
		k := fmt.Sprint(i + 1)
		guards := clauseGuards(c)
		spans := fl.guardSpans(c)
		if len(spans) != len(guards) {
			base.Fatalf("logos: cannot locate the case guards at %s", c.Pos().String())
		}
		for j := range runs[i] {
			runs[i][j].sp = spans[j]
		}
		var kept []span
		for j, g := range runs[i] {
			kept = append(kept, g.sp)
			var hoists strings.Builder
			var inner []syntax.Stmt
			for _, h := range g.hoists {
				hoists.WriteString(h.text + "; ")
				anchorScopes(h.stmts, p0)
				inner = append(inner, h.stmts...)
			}
			var rhs syntax.Expr = at(&syntax.ParenExpr{X: guards[j]}, p0)
			rhsText := "(" + fl.copyRegion(g.sp) + ")"
			if hasTag {
				rhs = at(&syntax.Operation{Op: syntax.Eql, X: nameOf(p0, tagN), Y: rhs}, p0)
				rhsText = tagN + " == " + rhsText
			}
			take := ifOf(p0, nil, nameOf(p0, matchN), []syntax.Stmt{
				assignOf(p0, []syntax.Expr{nameOf(p0, selN)}, []syntax.Expr{litOf(p0, k, syntax.IntLit)}),
			})
			inner = append(inner, defineOf(p0, []syntax.Expr{nameOf(p0, matchN)}, []syntax.Expr{rhs}), take)
			unmatched := at(&syntax.Operation{Op: syntax.Eql, X: nameOf(p0, selN), Y: litOf(p0, "0", syntax.IntLit)}, p0)
			wrapper = append(wrapper, ifOf(p0, nil, unmatched, inner))
			text.WriteString("if " + selN + " == 0 { " + hoists.String() + matchN + " := " + rhsText + "; if " + matchN + " { " + selN + " = " + k + " } }; ")
		}
		clause := span{start: fl.off(c.Pos()), end: fl.off(c.Colon) + 1}
		fl.add(clause.start, clause.end, leftoverComments(fl, clause, kept)+"case "+k+":")
		c.Cases = litOf(c.Pos(), k, syntax.IntLit)
	}
	for _, l := range stmtLabels(s) {
		text.WriteString(l + ": ")
	}
	text.WriteString("switch " + selN + " {")
	fl.add(hdr.region.start, hdr.region.end, text.String())
	closeOff := fl.off(ss.Rbrace) + 1
	fl.add(closeOff, closeOff, " }")
	ss.Init = nil
	ss.Tag = nameOf(ss.Pos(), selN)
	block := blockOf(p0, append(wrapper, s))
	block.Rbrace = ss.Rbrace
	fl.lowerChainChildren(s, enc)
	return block
}
