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

const groupOperandMark = "_logos_RET"

type groupFrame struct {
	g          string
	gctx       string
	j          string
	child      bool
	segment    bool
	segmentRes types2.Type
	closure    *groupClosure
	body       *syntax.BlockStmt
	planned    bool
}

func (top groupFrame) spawn() string {
	if top.child {
		return "GoNested"
	}
	return "Go"
}

func groupExprEnd(src []byte, start int) int {
	e := exprScan{end: start}
	scanSegment(src, start, len(src), e.step)
	return e.end
}

type exprScan struct {
	end     int
	depth   int
	opened  bool
	pending bool
}

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

func groupPostfix(k token.Token) bool {
	return k == token.LPAREN || k == token.LBRACK || k == token.PERIOD
}

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

func (fc *funcLower) groupTop() (groupFrame, bool) {
	if n := len(fc.groupStack); n > 0 {
		return fc.groupStack[n-1], true
	}
	return groupFrame{}, false
}

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

func (fc *funcLower) groupCallStmt(s *syntax.CallStmt) ([]syntax.Stmt, bool) {
	if s.Tok != syntax.Go {
		return fc.groupChanStmt(s)
	}
	if _, ok := fc.groupTop(); !ok {
		return nil, false
	}
	if out, handled := fc.groupChanStmt(s); handled {
		return out, true
	}
	return fc.lowerGroupGo(s), true
}

func (fc *funcLower) groupReturnStmt(s *syntax.ReturnStmt) ([]syntax.Stmt, bool) {
	top, ok := fc.groupTop()
	if !ok {
		return nil, false
	}
	if !top.child {
		return fc.lowerGroupReturn(s, top), true
	}
	return fc.groupChanStmt(s)
}

func (fc *funcLower) groupBranchStmt(s *syntax.BranchStmt) ([]syntax.Stmt, bool) {
	if top, ok := fc.groupTop(); ok && !top.child && fc.groupBranchLeaves(top, s) {
		return fc.lowerGroupBranch(s, top), true
	}
	return nil, false
}

func (fc *funcLower) scopeGroupStmt(s *syntax.GroupStmt, active []*scopeDeferCtx, stack []scopeFrame) []syntax.Stmt {
	if len(s.Body.List) == 0 {
		return []syntax.Stmt{s}
	}
	frame := fc.planGroup(s)
	ctx := &scopeDeferCtx{owner: s.Body, startOff: fc.fl.off(s.Body.Pos()), endOff: fc.fl.off(s.Body.Rbrace), depth: len(stack), join: &frame}
	fc.scopeCtxs = append(fc.scopeCtxs, ctx)
	if fc.groupCtxs == nil {
		fc.groupCtxs = make(map[syntax.Node]*scopeDeferCtx)
	}
	fc.groupCtxs[s.Body] = ctx
	s.Body.List = fc.scopeBlockList(s.Body.List, s.Body, fc.fl.off(s.Body.Rbrace), active, stack)
	return []syntax.Stmt{s}
}

func (fc *funcLower) yieldGroupStmt(s *syntax.GroupStmt) syntax.Stmt {
	s.Body.List = fc.rewriteYieldList(s.Body.List)
	return s
}

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

func (fc *funcLower) groupMarkDone(lit *syntax.FuncLit) {
	if fc.fl.groupDone == nil {
		fc.fl.groupDone = make(map[*syntax.FuncLit]bool)
	}
	fc.fl.groupDone[lit] = true
}

func (fc *funcLower) lowerGroupStmt(s *syntax.GroupStmt) []syntax.Stmt {
	if top, ok := fc.groupTop(); ok && !top.child {
		return fc.lowerGroupWrap(s, s.Pos(), top)
	}
	return fc.lowerGroupInline(s)
}

func (fc *funcLower) lowerGroupChild(s *syntax.GroupChild) []syntax.Stmt {
	top, ok := fc.groupTop()
	if !ok {
		return []syntax.Stmt{s}
	}
	return fc.lowerGroupWrap(s, s.Pos(), top)
}

func (fc *funcLower) lowerGroupWrap(s syntax.Stmt, pos syntax.Pos, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	cr := fmt.Sprintf("_logos_cr%d", fc.groupSeq)
	fc.groupSeq++
	ft := groupChildFuncType(pos, top.gctx, cr)
	prefix := top.g + "." + top.spawn() + "(func(" + top.gctx + " " + rtAlias + ".Context) (" + cr + " error) { "
	var body *syntax.BlockStmt
	var endText string
	var endStmts []syntax.Stmt
	if child, ok := s.(*syntax.GroupChild); ok {
		body = child.Body
		cfc := fc.groupChildFunc(top, body, ft)
		cfc.run()
		fc.groupTakeChild(cfc)
		endText, endStmts = cfc.groupChildEnd(pos)
		fl.add(fl.off(pos), fl.off(body.Pos())+1, prefix)
		fl.add(fl.off(body.Rbrace), fl.off(body.Rbrace)+1, lineBreakBefore(fl.src, fl.off(body.Pos())+1, fl.off(body.Rbrace))+endText)
	} else if nested, ok := s.(*syntax.GroupStmt); ok {
		body = blockOf(pos, nil)
		cfc := fc.groupChildFunc(top, body, ft)
		body.List = cfc.lowerGroupInline(nested)
		fc.groupTakeChild(cfc)
		endText, endStmts = cfc.groupChildEnd(pos)
		start := fl.off(syntax.StartPos(s))
		fl.add(start, start, prefix)
		endOff := stmtEnd(fl.src, start)
		fl.add(endOff, endOff, endText)
	} else {
		return []syntax.Stmt{s}
	}
	body.List = append(body.List, endStmts...)
	lit := at(&syntax.FuncLit{Type: ft, Body: body}, pos)
	fc.groupMarkDone(lit)
	native := exprStmtOf(pos, callOf(pos, selOf(pos, nameOf(pos, top.g), top.spawn()), []syntax.Expr{lit}))
	return []syntax.Stmt{native}
}

func (fc *funcLower) groupChildFunc(top groupFrame, body *syntax.BlockStmt, ft *syntax.FuncType) *funcLower {
	cfc := newFuncLower(fc.fl, ft, nil, body)
	cfc.ft, cfc.specs = ft, fc.fl.resultSpecs(ft, nil)
	cfc.errName = namedValue(cfc.specs[0].name)
	cfc.errSeq, cfc.funSeq, cfc.argSeq = fc.errSeq, fc.funSeq, fc.argSeq
	cfc.tmpSeq, cfc.scopeSeq, cfc.groupSeq = fc.tmpSeq, fc.scopeSeq, fc.groupSeq
	cfc.groupStack = []groupFrame{{g: top.g, gctx: top.gctx, child: true}}
	cfc.scopeParents = fc.scopeParents
	return cfc
}

func (fc *funcLower) groupChildEnd(pos syntax.Pos) (string, []syntax.Stmt) {
	ret := returnOf(pos, []syntax.Expr{nilOf(pos)})
	if !fc.hasErrdefer {
		return "return nil })", []syntax.Stmt{ret}
	}
	ok := assignOf(pos, []syntax.Expr{nameOf(pos, fc.okName)}, []syntax.Expr{nameOf(pos, "true")})
	return fc.okName + " = true; return nil })", []syntax.Stmt{ok, ret}
}

func (fc *funcLower) groupTakeChild(cfc *funcLower) {
	fc.errSeq, fc.funSeq, fc.argSeq = cfc.errSeq, cfc.funSeq, cfc.argSeq
	fc.tmpSeq, fc.scopeSeq, fc.groupSeq = cfc.tmpSeq, cfc.scopeSeq, cfc.groupSeq
}

func (fc *funcLower) groupJoinCheck(pos syntax.Pos, top groupFrame) (native, text []syntax.Stmt, subs []textSub) {
	errSpelling := fc.nextErr()
	siteSpelling := fc.fl.newSite(pos, pos.Line())
	exitNative, exitText, subs := fc.scopedExitPair(pos, errSpelling, siteSpelling)
	native, text = fc.groupJoinStmts(pos, top, errSpelling, exitNative, exitText)
	return native, text, subs
}

func (fc *funcLower) groupJoinWait(pos syntax.Pos, top groupFrame) (native, text []syntax.Stmt) {
	errSpelling := fc.nextErr()
	siteSpelling := fc.fl.newSite(pos, pos.Line())
	exitNative, exitText := fc.plainExitPair(pos, errSpelling, siteSpelling)
	return fc.groupJoinStmts(pos, top, errSpelling, exitNative, exitText)
}

func (fc *funcLower) groupJoinStmts(pos syntax.Pos, top groupFrame, errSpelling string, exitNative, exitText []syntax.Stmt) (native, text []syntax.Stmt) {
	mkJoin := func(exit []syntax.Stmt) []syntax.Stmt {
		set := assignOf(pos, []syntax.Expr{nameOf(pos, top.j)}, []syntax.Expr{nameOf(pos, "true")})
		wait := callOf(pos, selOf(pos, nameOf(pos, top.g), "Wait"), nil)
		check := errCheckInitOf(pos, nil, wait, nameOf(pos, errSpelling), exit)
		cond := at(&syntax.Operation{Op: syntax.Not, X: nameOf(pos, top.j)}, pos)
		join := ifOf(pos, nil, cond, []syntax.Stmt{set, check})
		fc.fl.markJoin(join)
		return []syntax.Stmt{join}
	}
	return mkJoin(exitNative), mkJoin(exitText)
}

func (fl *fileLower) markJoin(s syntax.Stmt) {
	if fl.groupJoins == nil {
		fl.groupJoins = make(map[syntax.Stmt]bool)
	}
	fl.groupJoins[s] = true
}

func groupWaitOnce(pos syntax.Pos, g, j string) *syntax.IfStmt {
	return ifOf(pos, nil,
		at(&syntax.Operation{Op: syntax.Not, X: nameOf(pos, j)}, pos),
		[]syntax.Stmt{
			assignOf(pos, []syntax.Expr{nameOf(pos, j)}, []syntax.Expr{nameOf(pos, "true")}),
			exprStmtOf(pos, callOf(pos, selOf(pos, nameOf(pos, g), "Wait"), nil)),
		})
}

func (fc *funcLower) newGroupFrame(s *syntax.GroupStmt) groupFrame {
	n := fc.groupSeq
	fc.groupSeq++
	return groupFrame{
		g:    fmt.Sprintf("_logos_g%d", n),
		gctx: fmt.Sprintf("_logos_gctx%d", n),
		j:    fmt.Sprintf("_logos_j%d", n),
		body: s.Body,
	}
}

func (fc *funcLower) lowerGroupInline(s *syntax.GroupStmt) []syntax.Stmt {
	fl := fc.fl
	frame, planned := fl.groupPlans[s.Body]
	if !planned {
		frame = fc.newGroupFrame(s)
	}
	g, gctx, j := frame.g, frame.gctx, frame.j
	fl.needRT = true
	pos := s.Pos()
	ctxSpan := span{start: fl.exprStart(s.Ctx), end: fl.exprEndOf(s.Ctx)}
	ctxText := fl.textOf(ctxSpan)
	pieces := []span{ctxSpan}
	limitText := gen("0")
	limitNode := syntax.Expr(litOf(pos, "0", syntax.IntLit))
	if s.Limit != nil {
		limitSpan := span{start: fl.exprStart(s.Limit), end: fl.exprEndOf(s.Limit)}
		limitText = fl.textOf(limitSpan)
		pieces = append(pieces, limitSpan)
		limitNode = s.Limit
	}
	headStart := fl.off(pos)
	braceOff := fl.off(s.Body.Pos())
	kept := leftoverComments(fl, span{start: headStart, end: braceOff + 1}, pieces)
	guardText := renderText([]syntax.Stmt{
		defineOf(pos, []syntax.Expr{nameOf(pos, j)}, []syntax.Expr{nameOf(pos, "false")}),
		deferOf(pos, callOf(pos, deferClosure(pos, []syntax.Stmt{groupWaitOnce(pos, g, j)}), nil)),
	}, nil)
	rebindText, rebindStmts := groupRebind(pos, s.Ctx, gctx)
	fl.addT(headStart, braceOff, cat(
		kept,
		gen("{\n"+g+", "+gctx+" := "+rtAlias+".NewGroup("),
		ctxText,
		gen(", "),
		limitText,
		gen(")\n_ = "+gctx+"\n"+rebindText),
		guardText,
	))
	header := defineOf(pos,
		[]syntax.Expr{nameOf(pos, g), nameOf(pos, gctx)},
		[]syntax.Expr{callOf(pos, rtSel(pos, "NewGroup"), []syntax.Expr{s.Ctx, limitNode})})
	blankUse := assignOf(pos, []syntax.Expr{nameOf(pos, "_")}, []syntax.Expr{nameOf(pos, gctx)})
	flagDef := defineOf(pos, []syntax.Expr{nameOf(pos, j)}, []syntax.Expr{nameOf(pos, "false")})
	guardDef := deferOf(pos, callOf(pos, deferClosure(pos, []syntax.Stmt{groupWaitOnce(pos, g, j)}), nil))
	fc.groupStack = append(fc.groupStack, frame)
	s.Body.List = fc.lowerList(s.Body.List)
	fc.groupStack = fc.groupStack[:len(fc.groupStack)-1]
	rbrace := s.Body.Rbrace
	joinNative, joinText, joinSubs := fc.groupEndJoin(rbrace, frame)
	rbraceOff := fl.off(rbrace)
	fl.addT(rbraceOff+1, rbraceOff+1, cat(gen("\n"), renderText(joinText, joinSubs), gen("}\n")))
	out := append([]syntax.Stmt{header, blankUse}, rebindStmts...)
	out = append(append(out, flagDef, guardDef, s.Body), joinNative...)
	block := blockOf(pos, out)
	block.Rbrace = rbrace
	return []syntax.Stmt{block}
}

func (fc *funcLower) groupEndJoin(pos syntax.Pos, top groupFrame) (native, text []syntax.Stmt, subs []textSub) {
	if top.planned {
		return nil, nil, nil
	}
	return fc.groupJoinCheck(pos, top)
}

func (fc *funcLower) planGroup(s *syntax.GroupStmt) groupFrame {
	frame := fc.newGroupFrame(s)
	frame.planned = true
	if fc.fl.groupPlans == nil {
		fc.fl.groupPlans = make(map[*syntax.BlockStmt]groupFrame)
	}
	fc.fl.groupPlans[s.Body] = frame
	return frame
}

func groupRebind(pos syntax.Pos, ctx syntax.Expr, gctx string) (string, []syntax.Stmt) {
	name, ok := plainContextName(ctx)
	if !ok {
		return "", nil
	}
	stmts := []syntax.Stmt{
		defineOf(pos, []syntax.Expr{nameOf(pos, name)}, []syntax.Expr{nameOf(pos, gctx)}),
		assignOf(pos, []syntax.Expr{nameOf(pos, "_")}, []syntax.Expr{nameOf(pos, name)}),
	}
	return name + " := " + gctx + "\n_ = " + name + "\n", stmts
}

func plainContextName(e syntax.Expr) (string, bool) {
	name, ok := syntax.Unparen(e).(*syntax.Name)
	if !ok || name.Value == "_" {
		return "", false
	}
	return name.Value, true
}

func (fc *funcLower) groupPrefixJoin(pos syntax.Pos, top groupFrame, off int) []syntax.Stmt {
	if top.planned {
		return nil
	}
	joinNative, joinText, joinSubs := fc.groupJoinCheck(pos, top)
	fc.fl.addT(off, off, renderText(joinText, joinSubs))
	return joinNative
}

func (fc *funcLower) lowerGroupReturn(s *syntax.ReturnStmt, top groupFrame) []syntax.Stmt {
	if s.Results == nil {
		return fc.lowerGroupBareReturn(s, top)
	}
	fl := fc.fl
	scanForLiterals(fl, s.Results)
	pos := s.Pos()
	retOff := fl.off(pos)
	operands := trimSpan(fl.src, span{start: retOff + len("return"), end: stmtEnd(fl.src, retOff)})
	temps := fc.nextTemps(len(fc.specs))
	joinNative, joinText, joinSubs := fc.groupJoinCheck(pos, top)
	native := fc.returnValues(pos, temps, slices.Clone(syntax.UnpackListExpr(s.Results)), joinNative)
	tw := &twin{}
	tw.absorb(joinSubs)
	text := renderText(fc.returnValues(pos, temps, []syntax.Expr{nameOf(pos, groupOperandMark)}, joinText), tw.subs)
	cut := strings.Index(text.s, groupOperandMark)
	head := cat(leftoverComments(fl, span{start: retOff, end: operands.start}, nil), text.slice(0, cut))
	fl.addT(retOff, operands.start, head)
	fl.addT(operands.end, operands.end, text.slice(cut+len(groupOperandMark), len(text.s)))
	return native
}

func (fc *funcLower) lowerGroupBareReturn(s *syntax.ReturnStmt, top groupFrame) []syntax.Stmt {
	prefix := fc.groupPrefixJoin(s.Pos(), top, fc.fl.off(s.Pos()))
	if fc.hasErrdefer {
		return append(prefix, fc.lowerReturn(s)...)
	}
	return append(prefix, s)
}

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

func groupBranchPasses(p syntax.Node) bool {
	switch p.(type) {
	case *syntax.ForStmt, *syntax.SwitchStmt, *syntax.SelectStmt, *syntax.LabeledStmt,
		*syntax.BlockStmt, *syntax.IfStmt, *syntax.CaseClause, *syntax.CommClause:
		return true
	}
	return false
}

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

func (fc *funcLower) lowerGroupBranch(s *syntax.BranchStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	prefix := fc.groupPrefixJoin(pos, top, fl.off(pos))
	return append(prefix, s)
}

func (fc *funcLower) lowerGroupGo(s *syntax.CallStmt) []syntax.Stmt {
	top, ok := fc.groupTop()
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
	callSpan := fl.callBodySpan(call)
	callStart, callEnd := callSpan.start, callSpan.end
	for _, raw := range call.ArgList {
		if constAsView(fl.pkg.info.TypeOf(raw)) != nil {
			fl.recordConstViewDecl(pos)
		}
	}
	capNative, capText := fc.groupCapture(pos, call)
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
	parts := strings.SplitN(text.s, "_logos_CALL", 2)
	if len(parts) != 2 {
		base.Fatalf("logos: group call text misses its call at %s", pos.String())
	}
	fl.add(fl.off(pos), callStart, "{ "+capText+top.g+"."+top.spawn()+"(func("+top.gctx+" "+rtAlias+".Context) ("+cr+" error) { "+parts[0])
	fl.add(callEnd, callEnd, parts[1]+"}) }")
	native := exprStmtOf(pos, callOf(pos, selOf(pos, nameOf(pos, top.g), top.spawn()), []syntax.Expr{lit}))
	return []syntax.Stmt{blockOf(pos, append(capNative, native))}
}
