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

// This file lowers M8-S2 `go group` statements and `go { }` children
// (SPEC Sec 2.6, T-206) to logos/rt.Group: the head opens the group,
// each child becomes a Go call with an error closure, and the tail
// waits with the `?` exit shape, so the group error propagates like `?`.
//
// One frame stack on funcLower tracks the group under lowering: every
// frame belongs to the current run, so nested function literals (fresh
// runs with empty stacks) never inherit membership. A group statement
// in a group body wraps as a child of that group; anywhere else it
// lowers inline. Every closure the group builds lowers eagerly through
// a child funcLower (which owns the single-error specs `?` needs) and
// is marked done, so the later literal scan prunes the finished subtree.
//
// Splice discipline: bodies and calls keep their source in place and
// compose through insertions and keyword deletions, so later passes
// (tuples, lambdas, interpolation) keep splicing inside them. Only the
// group head covers a span, and only the `go group (ctx, limit)`
// keywords: the context and limit are almost always a name and a
// literal, and they drain through textOf exactly like every call
// template. A tuple, lambda, or interpolation inside a group context
// or limit would overlap that head; keep them in child bodies.

// groupFrame is one group-body or child-body scope under lowering: g,
// gctx, and j spell the group, context, and joined-flag temporaries,
// child tells a child body (where the M8-S3 cancel rewrites apply)
// from a group body, and body is the inline body a branch-leaving check
// tests against (nil for child frames, whose exits never leave).
type groupFrame struct {
	g     string
	gctx  string
	j     string
	child bool
	body  *syntax.BlockStmt
}

// groupExprEnd returns the offset just past the expression starting at
// start: tokens scan with bracket depth, stopping before a closer,
// comma, or semicolon at depth zero that belongs to the enclosing
// syntax. EndPos alone stops at the last argument of a call, and the
// paren skipper overshoots into enclosing lists, so neither fits a
// context, limit, or call span on its own.
func groupExprEnd(src []byte, start int) int {
	e := exprScan{end: start}
	scanSegment(src, start, len(src), e.step)
	return e.end
}

// exprScan is the token state of groupExprEnd: end is the offset past
// the last token taken, depth counts open brackets, opened records that
// any bracket opened, and pending marks a just-closed call or index
// list, which only a postfix token may continue.
type exprScan struct {
	end     int
	depth   int
	opened  bool
	pending bool
}

// step takes one token and reports whether the expression continues.
func (e *exprScan) step(t tok) bool {
	if t.auto && e.depth == 0 {
		return false
	}
	if t.comment || t.auto {
		return true
	}
	if e.pending {
		e.pending = false
		if !groupPostfix(t.kind) {
			return false
		}
	}
	if !e.bracket(t.kind) {
		return false
	}
	e.end = t.end
	return true
}

// groupPostfix reports whether a token of kind k continues an
// expression after a closed call: a call, an index, or a selector.
func groupPostfix(k token.Token) bool {
	return k == token.LPAREN || k == token.LBRACK || k == token.PERIOD
}

// bracket tracks the bracket depth for a token of kind k and reports
// false at a closer, comma, or semicolon of the enclosing syntax.
func (e *exprScan) bracket(k token.Token) bool {
	switch k {
	case token.LPAREN, token.LBRACK, token.LBRACE:
		e.depth++
		e.opened = true
	case token.RPAREN, token.RBRACK, token.RBRACE:
		if e.depth == 0 {
			return false
		}
		e.depth--
		e.pending = e.depth == 0 && e.opened && k == token.RPAREN
	case token.COMMA, token.SEMICOLON:
		return e.depth != 0
	}
	return true
}

// groupTop reports the innermost group frame under lowering.
func (fc *funcLower) groupTop() (groupFrame, bool) {
	if n := len(fc.groupStack); n > 0 {
		return fc.groupStack[n-1], true
	}
	return groupFrame{}, false
}

// groupOf reports the group owning the statement under lowering: every
// frame on the stack scopes one group, so the top frame owns the point.
func (fc *funcLower) groupOf() (groupFrame, bool) {
	return fc.groupTop()
}

// lowerGroupHook is the one entry statement lowering takes into group
// lowering. It routes the statements a group frame rewrites (group
// statements, child blocks, child calls, exits from a group body, and
// communication in a child) and reports false for every other
// statement, which keeps its ordinary lowering.
func (fc *funcLower) lowerGroupHook(s syntax.Stmt) ([]syntax.Stmt, bool) {
	switch s := s.(type) {
	case *syntax.GroupStmt:
		return fc.lowerGroupStmt(s), true
	case *syntax.GroupChild:
		return fc.lowerGroupChild(s), true
	case *syntax.CallStmt:
		return fc.groupCallStmt(s)
	case *syntax.ReturnStmt:
		return fc.groupReturnStmt(s)
	case *syntax.BranchStmt:
		return fc.groupBranchStmt(s)
	}
	return fc.groupChanStmt(s)
}

// groupCallStmt turns a `go` call in a group body into a child.
func (fc *funcLower) groupCallStmt(s *syntax.CallStmt) ([]syntax.Stmt, bool) {
	if s.Tok != syntax.Go {
		return nil, false
	}
	if _, ok := fc.groupOf(); !ok {
		return nil, false
	}
	return fc.lowerGroupGo(s), true
}

// groupReturnStmt joins the group before a return leaves its body.
func (fc *funcLower) groupReturnStmt(s *syntax.ReturnStmt) ([]syntax.Stmt, bool) {
	if top, ok := fc.groupTop(); ok && !top.child {
		return fc.lowerGroupReturn(s, top), true
	}
	return nil, false
}

// groupBranchStmt joins the group before a branch leaves its body.
func (fc *funcLower) groupBranchStmt(s *syntax.BranchStmt) ([]syntax.Stmt, bool) {
	if top, ok := fc.groupTop(); ok && !top.child && fc.groupBranchLeaves(top, s) {
		return fc.lowerGroupBranch(s, top), true
	}
	return nil, false
}

// scopeGroupStmt binds the scope defers of a group body inline.
func (fc *funcLower) scopeGroupStmt(s *syntax.GroupStmt, active []*scopeDeferCtx, stack []scopeFrame) []syntax.Stmt {
	s.Body.List = fc.scopeBlockList(s.Body.List, s.Body, fc.fl.off(s.Body.Rbrace), active, stack)
	return []syntax.Stmt{s}
}

// yieldGroupStmt keeps yields synchronous inside a group body.
func (fc *funcLower) yieldGroupStmt(s *syntax.GroupStmt) syntax.Stmt {
	s.Body.List = fc.rewriteYieldList(s.Body.List)
	return s
}

// groupChildFuncType builds the error-closure type of a group child:
// `func(gctx _logos_rt.Context) (cr error)`. The context parameter
// shadows the group variable with the same value, so cancel references
// resolve inside; the named error result keeps ensureNamed silent.
func groupChildFuncType(pos syntax.Pos, gctx, cr string) *syntax.FuncType {
	return at(&syntax.FuncType{
		ParamList: []*syntax.Field{
			at(&syntax.Field{
				Name: nameOf(pos, gctx),
				Type: selOf(pos, nameOf(pos, rtAlias), "Context"),
			}, pos),
		},
		ResultList: []*syntax.Field{
			at(&syntax.Field{
				Name: nameOf(pos, cr),
				Type: nameOf(pos, "error"),
			}, pos),
		},
	}, pos)
}

// groupMarkDone records a finished group closure so the later literal
// scan prunes its subtree instead of lowering it twice.
func (fc *funcLower) groupMarkDone(lit *syntax.FuncLit) {
	if fc.fl.groupDone == nil {
		fc.fl.groupDone = make(map[*syntax.FuncLit]bool)
	}
	fc.fl.groupDone[lit] = true
}

// lowerGroupStmt lowers a group statement: wrapped as a child when the
// statement itself sits in a group body, inline otherwise.
func (fc *funcLower) lowerGroupStmt(s *syntax.GroupStmt) []syntax.Stmt {
	if top, ok := fc.groupTop(); ok && !top.child {
		return fc.lowerGroupWrap(s, s.Pos(), top)
	}
	return fc.lowerGroupInline(s)
}

// lowerGroupChild lowers a child block by wrapping it as a child of the
// owning group.
func (fc *funcLower) lowerGroupChild(s *syntax.GroupChild) []syntax.Stmt {
	top, ok := fc.groupOf()
	if !ok {
		return []syntax.Stmt{s}
	}
	return fc.lowerGroupWrap(s, s.Pos(), top)
}

// lowerGroupWrap wraps a group statement or a child block as a child of
// top. A child body lowers in place between brace keyword splices; a
// nested group statement lowers into a synthetic container between
// zero-width edge insertions, which never overlap its inline splices.
// Either way the closure body ends with `return nil`, since success
// falls through to a nil error.
func (fc *funcLower) lowerGroupWrap(s syntax.Stmt, pos syntax.Pos, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	cr := fmt.Sprintf("_logos_cr%d", fc.groupSeq)
	fc.groupSeq++
	ft := groupChildFuncType(pos, top.gctx, cr)
	prefix := top.g + ".Go(func(" + top.gctx + " " + rtAlias + ".Context) (" + cr + " error) { "
	var body *syntax.BlockStmt
	if child, ok := s.(*syntax.GroupChild); ok {
		body = child.Body
		cfc := fc.groupChildFunc(top, body, ft)
		cfc.run()
		fc.groupTakeChild(cfc)
		fl.add(fl.off(pos), fl.off(body.Pos())+1, prefix)
		fl.add(fl.off(body.Rbrace), fl.off(body.Rbrace)+1, "return nil })")
	} else if nested, ok := s.(*syntax.GroupStmt); ok {
		body = blockOf(pos, nil)
		cfc := fc.groupChildFunc(top, body, ft)
		body.List = cfc.lowerGroupInline(nested)
		fc.groupTakeChild(cfc)
		start := fl.off(syntax.StartPos(s))
		fl.add(start, start, prefix)
		end := stmtEnd(fl.src, start)
		fl.add(end, end, "return nil })")
	} else {
		return []syntax.Stmt{s}
	}
	retNil := returnOf(pos, []syntax.Expr{nilOf(pos)})
	body.List = append(body.List, retNil)
	lit := at(&syntax.FuncLit{Type: ft, Body: body}, pos)
	fc.groupMarkDone(lit)
	native := exprStmtOf(pos, callOf(pos, selOf(pos, nameOf(pos, top.g), "Go"), []syntax.Expr{lit}))
	return []syntax.Stmt{native}
}

// groupChildFunc starts a child funcLower under a child frame with the
// temp counters carried across; groupTakeChild carries the progress
// back when the child run finishes.
func (fc *funcLower) groupChildFunc(top groupFrame, body *syntax.BlockStmt, ft *syntax.FuncType) *funcLower {
	cfc := newFuncLower(fc.fl, ft, body)
	cfc.errSeq, cfc.funSeq, cfc.argSeq = fc.errSeq, fc.funSeq, fc.argSeq
	cfc.tmpSeq, cfc.scopeSeq, cfc.groupSeq = fc.tmpSeq, fc.scopeSeq, fc.groupSeq
	cfc.groupStack = []groupFrame{{g: top.g, gctx: top.gctx, child: true}}
	cfc.scopeParents = fc.scopeParents
	return cfc
}

// groupTakeChild carries child temp progress back to the parent.
func (fc *funcLower) groupTakeChild(cfc *funcLower) {
	fc.errSeq, fc.funSeq, fc.argSeq = cfc.errSeq, cfc.funSeq, cfc.argSeq
	fc.tmpSeq, fc.scopeSeq, fc.groupSeq = cfc.tmpSeq, cfc.scopeSeq, cfc.groupSeq
}

// groupJoinCheck builds the join-once block shared by the group tail
// and every exit from a group body: on first arrival it marks joined
// and waits, propagating a group failure exactly like `?`. Later
// arrivals (a second return, the tail, the panic guard) skip it.
func (fc *funcLower) groupJoinCheck(pos syntax.Pos, top groupFrame) (native, text []syntax.Stmt, subs []textSub) {
	fl := fc.fl
	errSpelling := fc.nextErr()
	siteSpelling := fl.newSite(pos, pos.Line())
	exitNative, exitText, exitSubs := fc.exitPair(pos, errSpelling, siteSpelling)
	subs = exitSubs
	mkWait := func() *syntax.CallExpr {
		return callOf(pos, selOf(pos, nameOf(pos, top.g), "Wait"), nil)
	}
	mkJoin := func(jname, ename *syntax.Name, exit []syntax.Stmt) []syntax.Stmt {
		set := assignOf(pos, []syntax.Expr{jname}, []syntax.Expr{nameOf(pos, "true")})
		check := errCheckInitOf(pos, nil, mkWait(), ename, exit)
		cond := at(&syntax.Operation{Op: syntax.Not, X: jname}, pos)
		return []syntax.Stmt{ifOf(pos, nil, cond, []syntax.Stmt{set, check})}
	}
	native = mkJoin(nameOf(pos, top.j), nameOf(pos, errSpelling), exitNative)
	text = mkJoin(nameOf(pos, top.j), nameOf(pos, errSpelling), exitText)
	return native, text, subs
}

// lowerGroupInline lowers a group statement in place: the head becomes
// the NewGroup binding with a joined flag and a panic guard, the body
// lowers through the plain list path, and the tail waits with the `?`
// exit shape. The child context stays referenced through a blank use
// even when no child selects on it, so a group of pure call children
// still compiles.
func (fc *funcLower) lowerGroupInline(s *syntax.GroupStmt) []syntax.Stmt {
	fl := fc.fl
	n := fc.groupSeq
	fc.groupSeq++
	g := fmt.Sprintf("_logos_g%d", n)
	gctx := fmt.Sprintf("_logos_gctx%d", n)
	j := fmt.Sprintf("_logos_j%d", n)
	fl.needRT = true
	pos := s.Pos()
	ctxSpan := span{start: fl.off(leftmostOf(s.Ctx)), end: groupExprEnd(fl.src, fl.off(leftmostOf(s.Ctx)))}
	ctxText := fl.textOf(ctxSpan)
	pieces := []span{ctxSpan}
	limitText := "0"
	limitNode := syntax.Expr(litOf(pos, "0", syntax.IntLit))
	if s.Limit != nil {
		limitSpan := span{start: fl.off(leftmostOf(s.Limit)), end: groupExprEnd(fl.src, fl.off(leftmostOf(s.Limit)))}
		limitText = fl.textOf(limitSpan)
		pieces = append(pieces, limitSpan)
		limitNode = s.Limit
	}
	headStart := fl.off(pos)
	braceOff := fl.off(s.Body.Pos())
	kept := leftoverComments(fl, span{start: headStart, end: braceOff + 1}, pieces)
	guardText := renderText([]syntax.Stmt{
		defineOf(pos, []syntax.Expr{nameOf(pos, j)}, []syntax.Expr{nameOf(pos, "false")}),
		deferOf(pos, callOf(pos, deferClosure(pos, []syntax.Stmt{
			ifOf(pos, nil,
				at(&syntax.Operation{Op: syntax.Not, X: nameOf(pos, j)}, pos),
				[]syntax.Stmt{
					assignOf(pos, []syntax.Expr{nameOf(pos, j)}, []syntax.Expr{nameOf(pos, "true")}),
					exprStmtOf(pos, callOf(pos, selOf(pos, nameOf(pos, g), "Wait"), nil)),
				}),
		}), nil)),
	}, nil)
	fl.add(headStart, braceOff+1, kept+g+", "+gctx+" := "+rtAlias+".NewGroup("+ctxText+", "+limitText+")\n_ = "+gctx+"\n"+guardText)
	header := defineOf(pos,
		[]syntax.Expr{nameOf(pos, g), nameOf(pos, gctx)},
		[]syntax.Expr{callOf(pos, rtSel(pos, "NewGroup"), []syntax.Expr{s.Ctx, limitNode})})
	blankUse := assignOf(pos, []syntax.Expr{nameOf(pos, "_")}, []syntax.Expr{nameOf(pos, gctx)})
	flagDef := defineOf(pos, []syntax.Expr{nameOf(pos, j)}, []syntax.Expr{nameOf(pos, "false")})
	guardBody := ifOf(pos, nil,
		at(&syntax.Operation{Op: syntax.Not, X: nameOf(pos, j)}, pos),
		[]syntax.Stmt{
			assignOf(pos, []syntax.Expr{nameOf(pos, j)}, []syntax.Expr{nameOf(pos, "true")}),
			exprStmtOf(pos, callOf(pos, selOf(pos, nameOf(pos, g), "Wait"), nil)),
		})
	guardDef := deferOf(pos, callOf(pos, deferClosure(pos, []syntax.Stmt{guardBody}), nil))
	fc.groupStack = append(fc.groupStack, groupFrame{g: g, gctx: gctx, j: j, body: s.Body})
	s.Body.List = fc.lowerList(s.Body.List)
	fc.groupStack = fc.groupStack[:len(fc.groupStack)-1]
	top := groupFrame{g: g, gctx: gctx, j: j, body: s.Body}
	rbrace := s.Body.Rbrace
	joinNative, joinText, joinSubs := fc.groupJoinCheck(rbrace, top)
	rbraceOff := fl.off(rbrace)
	fl.add(rbraceOff, rbraceOff, renderText(joinText, joinSubs))
	fl.add(rbraceOff, rbraceOff+1, "")
	out := append([]syntax.Stmt{header, blankUse, flagDef, guardDef}, s.Body.List...)
	return append(out, joinNative...)
}

// groupPrefixJoin emits the join-once block as a zero-width insertion
// at off and returns its native prefix: exits in group bodies (returns,
// branches, `?`) join the children first through one shared shape,
// keeping their own splices untouched in separate records.
func (fc *funcLower) groupPrefixJoin(pos syntax.Pos, top groupFrame, off int) []syntax.Stmt {
	joinNative, joinText, joinSubs := fc.groupJoinCheck(pos, top)
	fc.fl.add(off, off, renderText(joinText, joinSubs))
	return joinNative
}

// groupPrependJoin prefixes native with the join-once block when the
// `?` under lowering sits in a group body, joining the children before
// the error path can leave.
func (fc *funcLower) groupPrependJoin(pos syntax.Pos, spanStart int, native []syntax.Stmt) []syntax.Stmt {
	if top, ok := fc.groupTop(); ok && !top.child {
		return append(fc.groupPrefixJoin(pos, top, spanStart), native...)
	}
	return native
}

// lowerGroupReturn prefixes a return from a group body with the
// join-once block: children join before the function leaves, and a
// group failure propagates like `?` instead of the written results.
func (fc *funcLower) lowerGroupReturn(s *syntax.ReturnStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	prefix := fc.groupPrefixJoin(pos, top, fl.off(pos))
	var rest []syntax.Stmt
	if fc.hasErrdefer {
		rest = fc.lowerReturn(s)
	} else {
		scanForLiterals(fl, s)
		rest = []syntax.Stmt{s}
	}
	return append(prefix, rest...)
}

// groupBranchTarget resolves the target of a leaving branch: the
// innermost loop, switch, select, or matching label, or nil when the
// walk leaves the function (invalid code, which never lowers).
func (fc *funcLower) groupBranchTarget(s *syntax.BranchStmt) syntax.Node {
	node := syntax.Node(s)
	for {
		p, ok := fc.scopeParents[node]
		if !ok || p == nil {
			return nil
		}
		if groupBranchHits(p, s) || !groupBranchPasses(p) {
			return p
		}
		node = p
	}
}

// groupBranchHits reports whether p is the statement the branch s
// names: the innermost loop for an unlabeled break or continue, the
// innermost switch or select for an unlabeled break, or the matching
// label.
func groupBranchHits(p syntax.Node, s *syntax.BranchStmt) bool {
	switch t := p.(type) {
	case *syntax.ForStmt:
		return s.Label == nil && (s.Tok == syntax.Break || s.Tok == syntax.Continue)
	case *syntax.SwitchStmt, *syntax.SelectStmt:
		return s.Label == nil && s.Tok == syntax.Break
	case *syntax.LabeledStmt:
		return s.Label != nil && t.Label.Value == s.Label.Value
	}
	return false
}

// groupBranchPasses reports whether the walk toward a branch target
// continues through p: statements and clauses nest, anything else ends
// the search.
func groupBranchPasses(p syntax.Node) bool {
	switch p.(type) {
	case *syntax.ForStmt, *syntax.SwitchStmt, *syntax.SelectStmt, *syntax.LabeledStmt,
		*syntax.BlockStmt, *syntax.IfStmt, *syntax.CaseClause, *syntax.CommClause:
		return true
	}
	return false
}

// groupBranchLeaves reports whether a break, continue, or goto leaves
// the group body under lowering: its target sits outside the body. Any
// other branch stays.
func (fc *funcLower) groupBranchLeaves(top groupFrame, s *syntax.BranchStmt) bool {
	switch s.Tok {
	case syntax.Break, syntax.Continue, syntax.Goto:
	default:
		return false
	}
	if s.Tok == syntax.Goto {
		if s.Label == nil {
			return true
		}
		target, ok := fc.scopeLabels[s.Label.Value]
		if !ok {
			return true
		}
		return !fc.scopeContains(top.body, target)
	}
	if fc.scopeParents == nil {
		fc.collectScopeInfo()
	}
	target := fc.groupBranchTarget(s)
	if target == nil {
		return true
	}
	return !fc.scopeContains(top.body, target)
}

// lowerGroupBranch prefixes a branch leaving a group body with the
// join-once block, so the jump only runs after the children join (and a
// group failure propagates instead of jumping).
func (fc *funcLower) lowerGroupBranch(s *syntax.BranchStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	prefix := fc.groupPrefixJoin(pos, top, fl.off(pos))
	return append(prefix, s)
}

// lowerGroupGo lowers a `go f(args)` child: the call keeps its source
// in place between a Go-closure prefix and an error-check suffix, while
// the native tree carries the same call node, so argument splices land
// in the uncovered span on both sides. The text wrapper renders with a
// reserved dummy name where the call sits and splits there; the dummy
// spelling cannot collide with user code, since `_logos_` names are
// rejected in .lgo, so the split lands exactly around the call span.
func (fc *funcLower) lowerGroupGo(s *syntax.CallStmt) []syntax.Stmt {
	top, ok := fc.groupOf()
	if !ok {
		return []syntax.Stmt{s}
	}
	fl := fc.fl
	pos := s.Pos()
	call, isCall := syntax.Unparen(s.Call).(*syntax.CallExpr)
	if !isCall {
		return []syntax.Stmt{s}
	}
	nvals, hasErr := types2.LogosErrorResults(fl.pkg.info.TypeOf(call))
	cr := fmt.Sprintf("_logos_cr%d", fc.groupSeq)
	fc.groupSeq++
	errSpelling := fc.nextErr()
	ft := groupChildFuncType(pos, top.gctx, cr)
	callStart := fl.off(leftmostOf(call))
	callEnd := groupExprEnd(fl.src, callStart)
	var checkN syntax.Stmt
	if hasErr {
		eName := nameOf(pos, errSpelling)
		checkN = errCheckInitOf(pos, blankNames(pos, nvals), call, eName,
			[]syntax.Stmt{returnOf(pos, []syntax.Expr{nameOf(pos, errSpelling)})})
	} else {
		checkN = exprStmtOf(pos, call)
	}
	bodyN := blockOf(pos, []syntax.Stmt{checkN, returnOf(pos, []syntax.Expr{nilOf(pos)})})
	cfc := fc.groupChildFunc(top, bodyN, ft)
	cfc.run()
	fc.groupTakeChild(cfc)
	lit := at(&syntax.FuncLit{Type: ft, Body: bodyN}, pos)
	fc.groupMarkDone(lit)
	callDummy := nameOf(pos, "_logos_CALL")
	var checkT syntax.Stmt
	if hasErr {
		checkT = errCheckInitOf(pos, blankNames(pos, nvals), callDummy, nameOf(pos, errSpelling),
			[]syntax.Stmt{returnOf(pos, []syntax.Expr{nameOf(pos, errSpelling)})})
	} else {
		checkT = exprStmtOf(pos, callDummy)
	}
	retT := returnOf(pos, []syntax.Expr{nilOf(pos)})
	text := renderText([]syntax.Stmt{checkT, retT}, nil)
	parts := strings.SplitN(text, "_logos_CALL", 2)
	if len(parts) != 2 {
		base.Fatalf("logos: group call text misses its call at %s", pos.String())
	}
	fl.add(fl.off(pos), callStart, top.g+".Go(func("+top.gctx+" "+rtAlias+".Context) ("+cr+" error) { "+parts[0])
	fl.add(callEnd, callEnd, parts[1]+"})")
	native := exprStmtOf(pos, callOf(pos, selOf(pos, nameOf(pos, top.g), "Go"), []syntax.Expr{lit}))
	return []syntax.Stmt{native}
}
