// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const NextVersion = "logos-next-1"

type NextEdit struct {
	StartLine   int    `json:"start_line"`
	StartCol    int    `json:"start_col"`
	EndLine     int    `json:"end_line"`
	EndCol      int    `json:"end_col"`
	Replacement string `json:"replacement"`
}

type NextTask struct {
	Category    string `json:"category"`
	Rule        string `json:"rule"`
	StartLine   int    `json:"start_line"`
	EndLine     int    `json:"end_line"`
	FuncSig     string `json:"function_signature"`
	Instruction string `json:"ai_instruction"`
}

type NextFileReport struct {
	File  string     `json:"file"`
	Sum   string     `json:"sha256"`
	Edits []NextEdit `json:"edits"`
	Tasks []NextTask `json:"tasks"`
}

type NextPackageReport struct {
	Version string           `json:"version"`
	Package string           `json:"package"`
	Files   []NextFileReport `json:"files"`
}

const (
	nextRuleErrReturn = "err-return"
	nextRuleLoopChain = "loop-chain"
	nextRuleSprintf   = "sprintf-interp"
)

func ScanNext(info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte, names map[*syntax.File]string) []NextFileReport {
	errObj := types2.Universe.Lookup("error")
	if errObj == nil {
		return nil
	}
	s := &nextScan{info: info, errType: errObj.Type()}
	var out []NextFileReport
	for _, f := range files {
		data := src[f]
		if data == nil {
			continue
		}
		rep := s.scanFile(f, data, names[f])
		if len(rep.Edits) > 0 || len(rep.Tasks) > 0 {
			out = append(out, rep)
		}
	}
	return out
}

type nextScan struct {
	info           *types2.Info
	errType        types2.Type
	src            []byte
	starts         []int
	edits          []NextEdit
	tasks          []NextTask
	skip           [][2]int
	addrTaken      map[types2.Object]bool
	closureWritten map[types2.Object]bool
}

type nextCtx struct {
	fn      *syntax.FuncDecl
	lastErr bool
	safeErr bool
}

func (s *nextScan) scanFile(f *syntax.File, data []byte, name string) NextFileReport {
	sum := sha256.Sum256(data)
	if name == "" {
		name = f.Pos().Base().Filename()
	}
	rep := NextFileReport{File: name, Sum: hex.EncodeToString(sum[:])}
	s.src = data
	s.starts = lineStarts(data)
	s.edits = nil
	s.tasks = nil
	s.skip = nil
	s.initFileFacts(f)
	for _, decl := range f.DeclList {
		fn, ok := decl.(*syntax.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		ctx := nextCtx{fn: fn, lastErr: s.funcLastErr(fn), safeErr: s.safeResults(fn)}
		s.block(fn.Body.List, ctx)
		s.closures(fn, ctx)
	}
	s.sprintfFile(f)
	rep.Edits = s.edits
	rep.Tasks = s.tasks
	return rep
}

func (s *nextScan) closures(fn *syntax.FuncDecl, ctx nextCtx) {
	syntax.Inspect(fn.Body, func(n syntax.Node) bool {
		lit, ok := n.(*syntax.FuncLit)
		if !ok || lit.Body == nil {
			return true
		}
		if s.insideLines(int(lit.Pos().Line())) {
			return false
		}
		s.block(lit.Body.List, nextCtx{fn: fn, safeErr: ctx.safeErr})
		return true
	})
}

func (s *nextScan) safeResults(fn *syntax.FuncDecl) bool {
	named := false
	for _, field := range fn.Type.ResultList {
		if field.Name != nil {
			named = true
		}
	}
	if !named {
		return true
	}
	bare := false
	syntax.Inspect(fn.Body, func(n syntax.Node) bool {
		if _, ok := n.(*syntax.FuncLit); ok {
			return false
		}
		if ret, ok := n.(*syntax.ReturnStmt); ok && ret.Results == nil {
			bare = true
			return false
		}
		return true
	})
	return !bare
}

func (s *nextScan) insideLines(line int) bool {
	for _, span := range s.skip {
		if span[0] <= line && line <= span[1] {
			return true
		}
	}
	return false
}

func (s *nextScan) funcLastErr(fn *syntax.FuncDecl) bool {
	obj, ok := s.info.Defs[fn.Name].(*types2.Func)
	if !ok {
		return false
	}
	return s.sigLastErr(obj.Type())
}

func (s *nextScan) sigLastErr(t types2.Type) bool {
	sig, ok := t.(*types2.Signature)
	if !ok {
		return false
	}
	res := sig.Results()
	if res.Len() == 0 {
		return false
	}
	return types2.Identical(res.At(res.Len()-1).Type(), s.errType)
}

func (s *nextScan) block(list []syntax.Stmt, ctx nextCtx) {
	for i := 0; i < len(list); i++ {
		if consumed := s.errAssign(list, i, ctx); consumed > 0 {
			i += consumed - 1
			continue
		}
		if s.errInit(list[i], ctx) {
			continue
		}
		if s.loopAppend(list, i, ctx) {
			continue
		}
		s.children(list[i], ctx)
	}
}

func (s *nextScan) children(stmt syntax.Stmt, ctx nextCtx) {
	switch n := stmt.(type) {
	case *syntax.BlockStmt:
		s.block(n.List, ctx)
	case *syntax.IfStmt:
		if n.Then != nil {
			s.block(n.Then.List, ctx)
		}
		switch e := n.Else.(type) {
		case *syntax.BlockStmt:
			s.block(e.List, ctx)
		case *syntax.IfStmt:
			s.children(e, ctx)
		}
	case *syntax.ForStmt:
		if n.Body != nil {
			s.block(n.Body.List, ctx)
		}
	case *syntax.SwitchStmt:
		for _, c := range n.Body {
			s.block(c.Body, ctx)
		}
	case *syntax.SelectStmt:
		for _, c := range n.Body {
			s.block(c.Body, ctx)
		}
	case *syntax.LabeledStmt:
		s.children(n.Stmt, ctx)
	}
}

type nextErrMatch struct {
	lhs   []*syntax.Name
	call  *syntax.CallExpr
	op    syntax.Operator
	check *syntax.IfStmt
	last  *syntax.Name
	cond  *syntax.Name
}

type nextErrSpan struct {
	callSrc  string
	start    int
	ifEnd    int
	trailing string
}

func nextErrAssignShape(list []syntax.Stmt, i int) (nextErrMatch, bool) {
	if i+1 >= len(list) {
		return nextErrMatch{}, false
	}
	assign, ok := list[i].(*syntax.AssignStmt)
	if !ok {
		return nextErrMatch{}, false
	}
	check, ok := list[i+1].(*syntax.IfStmt)
	if !ok || check.Init != nil {
		return nextErrMatch{}, false
	}
	lhs, call, op, ok := assignCall(assign)
	if !ok {
		return nextErrMatch{}, false
	}
	last := lhs[len(lhs)-1]
	cond, ok := errCond(check, last.Value)
	if !ok {
		return nextErrMatch{}, false
	}
	return nextErrMatch{lhs: lhs, call: call, op: op, check: check, last: last, cond: cond}, true
}

func (s *nextScan) nextErrInitShape(stmt syntax.Stmt) (nextErrMatch, bool) {
	check, ok := stmt.(*syntax.IfStmt)
	if !ok || check.Init == nil {
		return nextErrMatch{}, false
	}
	assign, ok := check.Init.(*syntax.AssignStmt)
	if !ok {
		return nextErrMatch{}, false
	}
	lhs, call, op, ok := assignCall(assign)
	if !ok {
		return nextErrMatch{}, false
	}
	last, ok := initTarget(lhs)
	if !ok {
		return nextErrMatch{}, false
	}
	cond, ok := errCond(check, last.Value)
	if !ok {
		return nextErrMatch{}, false
	}
	if n, ok := s.callResults(call); !ok || n != len(lhs) {
		return nextErrMatch{}, false
	}
	return nextErrMatch{lhs: lhs, call: call, op: op, check: check, last: last, cond: cond}, true
}

func (s *nextScan) nextErrEligible(call *syntax.CallExpr, ctx nextCtx) bool {
	if !s.callLastErr(call) {
		return false
	}
	if !ctx.lastErr || !ctx.safeErr {
		return false
	}
	return true
}

func (s *nextScan) nextErrBodyRets(check *syntax.IfStmt, last *syntax.Name, ctx nextCtx) ([]syntax.Expr, bool) {
	rets, why, ok := checkBody(check)
	if !ok {
		s.children(check, ctx)
		s.task(check, ctx, nextRuleErrReturn, why)
		return nil, true
	}
	if !s.returnsFit(rets, last) {
		s.children(check, ctx)
		s.task(check, ctx, nextRuleErrReturn, "the check returns a different error or non-zero values, so ? cannot replace it")
		return nil, true
	}
	return rets, false
}

func (s *nextScan) nextErrLaterResolve(m nextErrMatch, rets []syntax.Expr, ctx nextCtx) (bool, bool) {
	obj := s.nameObj(m.last)
	if obj == nil {
		return true, false
	}
	allowed := map[*syntax.Name]bool{m.last: true, m.cond: true}
	for _, r := range rets {
		if n, ok := r.(*syntax.Name); ok {
			allowed[n] = true
		}
	}
	if s.usedElsewhere(ctx.fn, obj, allowed) {
		s.children(m.check, ctx)
		s.task(m.check, ctx, nextRuleErrReturn, "the error variable is read after the check, so ? cannot replace the assignment")
		return false, true
	}
	return false, false
}

func (s *nextScan) nextErrRewriteSpan(m nextErrMatch, startPos syntax.Pos, ctx nextCtx) (nextErrSpan, bool, bool) {
	callSrc, ok := s.callText(m.call)
	if !ok {
		return nextErrSpan{}, false, false
	}
	start, callEnd, ifEnd, ok := s.errSpan(startPos, m.call, m.check)
	if !ok {
		return nextErrSpan{}, false, false
	}
	trailing, blocked := scanComments(s.src, start, callEnd, ifEnd)
	if blocked {
		s.children(m.check, ctx)
		s.task(m.check, ctx, nextRuleErrReturn, "a comment sits inside the rewrite span, so the check needs a manual rewrite")
		return nextErrSpan{}, false, true
	}
	return nextErrSpan{callSrc: callSrc, start: start, ifEnd: ifEnd, trailing: trailing}, true, false
}

func (s *nextScan) nextErrAssignBuild(m nextErrMatch, callSrc string) string {
	if len(m.lhs) == 1 {
		return callSrc + "?"
	}
	names := make([]string, 0, len(m.lhs)-1)
	for _, n := range m.lhs[:len(m.lhs)-1] {
		names = append(names, n.Value)
	}
	joined := strings.Join(names, ", ")
	if m.op == syntax.Def && s.hasNew(m.lhs[:len(m.lhs)-1]) {
		return joined + " := " + callSrc + "?"
	}
	return joined + " = " + callSrc + "?"
}

func (s *nextScan) errAssign(list []syntax.Stmt, i int, ctx nextCtx) int {
	m, ok := nextErrAssignShape(list, i)
	if !ok {
		return 0
	}
	if !s.nextErrEligible(m.call, ctx) {
		return 0
	}
	rets, tasked := s.nextErrBodyRets(m.check, m.last, ctx)
	if tasked {
		return 2
	}
	noMatch, tasked := s.nextErrLaterResolve(m, rets, ctx)
	if noMatch {
		return 0
	}
	if tasked {
		return 2
	}
	span, ok, tasked := s.nextErrRewriteSpan(m, m.lhs[0].Pos(), ctx)
	if tasked {
		return 2
	}
	if !ok {
		return 0
	}
	s.emit(span.start, span.ifEnd, s.nextErrAssignBuild(m, span.callSrc)+span.trailing)
	return 2
}

func (s *nextScan) errInit(stmt syntax.Stmt, ctx nextCtx) bool {
	m, ok := s.nextErrInitShape(stmt)
	if !ok {
		return s.errInitValueTask(stmt, ctx)
	}
	if !s.nextErrEligible(m.call, ctx) {
		return false
	}
	rets, tasked := s.nextErrBodyRets(m.check, m.last, ctx)
	if tasked {
		return true
	}
	noMatch, tasked := s.nextErrLaterResolve(m, rets, ctx)
	if noMatch {
		return false
	}
	if tasked {
		return true
	}
	span, ok, tasked := s.nextErrRewriteSpan(m, m.check.Pos(), ctx)
	if tasked {
		return true
	}
	if !ok {
		return false
	}
	s.emit(span.start, span.ifEnd, span.callSrc+"?"+span.trailing)
	return true
}

func assignCall(a *syntax.AssignStmt) ([]*syntax.Name, *syntax.CallExpr, syntax.Operator, bool) {
	if a.Op != 0 && a.Op != syntax.Def {
		return nil, nil, 0, false
	}
	call, ok := unwrapParen(a.Rhs).(*syntax.CallExpr)
	if !ok {
		return nil, nil, 0, false
	}
	var lhs []*syntax.Name
	switch l := a.Lhs.(type) {
	case *syntax.Name:
		lhs = []*syntax.Name{l}
	case *syntax.ListExpr:
		for _, e := range l.ElemList {
			n, ok := e.(*syntax.Name)
			if !ok {
				return nil, nil, 0, false
			}
			lhs = append(lhs, n)
		}
	default:
		return nil, nil, 0, false
	}
	if len(lhs) == 0 {
		return nil, nil, 0, false
	}
	return lhs, call, a.Op, true
}

func nextErrCondSide(x syntax.Expr, y syntax.Expr, err string) (*syntax.Name, bool) {
	n, ok := x.(*syntax.Name)
	if !ok || !isNilName(y) || n.Value != err {
		return nil, false
	}
	return n, true
}

func errCond(t *syntax.IfStmt, err string) (*syntax.Name, bool) {
	cond, ok := unwrapParen(t.Cond).(*syntax.Operation)
	if !ok || cond.Op != syntax.Neq || cond.Y == nil {
		return nil, false
	}
	if n, ok := nextErrCondSide(cond.X, cond.Y, err); ok {
		return n, true
	}
	if n, ok := nextErrCondSide(cond.Y, cond.X, err); ok {
		return n, true
	}
	return nil, false
}

func checkBody(t *syntax.IfStmt) ([]syntax.Expr, string, bool) {
	if t.Else != nil {
		return nil, "the check carries an else branch, so ? cannot replace it", false
	}
	if t.Then == nil || len(t.Then.List) != 1 {
		return nil, "the check body does more than return, so ? cannot replace it", false
	}
	ret, ok := t.Then.List[0].(*syntax.ReturnStmt)
	if !ok {
		return nil, "the check body does more than return, so ? cannot replace it", false
	}
	if ret.Results == nil {
		return nil, "the check returns no values, so ? cannot replace it", false
	}
	var rets []syntax.Expr
	if l, ok := ret.Results.(*syntax.ListExpr); ok {
		rets = l.ElemList
	} else {
		rets = []syntax.Expr{ret.Results}
	}
	if len(rets) == 0 {
		return nil, "the check returns no values, so ? cannot replace it", false
	}
	return rets, "", true
}

func initTarget(lhs []*syntax.Name) (*syntax.Name, bool) {
	if len(lhs) == 1 {
		return lhs[0], true
	}
	if len(lhs) == 2 && lhs[0].Value == "_" {
		return lhs[1], true
	}
	return nil, false
}
func (s *nextScan) errInitValueTask(stmt syntax.Stmt, ctx nextCtx) bool {
	check, ok := stmt.(*syntax.IfStmt)
	if !ok || check.Init == nil {
		return false
	}
	assign, ok := check.Init.(*syntax.AssignStmt)
	if !ok {
		return false
	}
	lhs, call, _, ok := assignCall(assign)
	if !ok || len(lhs) != 2 {
		return false
	}
	if lhs[0].Value == "_" || lhs[1].Value == "_" {
		return false
	}
	if lhs[0].Value == lhs[1].Value {
		return false
	}
	if _, ok := errCond(check, lhs[1].Value); !ok {
		return false
	}
	if !s.nextErrEligible(call, ctx) {
		return false
	}
	if n, ok := s.callResults(call); !ok || n != len(lhs) {
		return false
	}
	s.children(check, ctx)
	s.task(check, ctx, nextRuleErrReturn, "the if-init binds a value, so the check needs a manual rewrite")
	return true
}

func (s *nextScan) returnsFit(rets []syntax.Expr, last *syntax.Name) bool {
	if len(rets) == 0 {
		return false
	}
	for _, r := range rets[:len(rets)-1] {
		if !s.isZeroExpr(r) {
			return false
		}
	}
	return s.sameErr(rets[len(rets)-1], last)
}

func (s *nextScan) isZeroExpr(e syntax.Expr) bool {
	switch n := unwrapParen(e).(type) {
	case *syntax.BasicLit:
		switch n.Kind {
		case syntax.IntLit:
			return n.Value == "0"
		case syntax.FloatLit:
			return n.Value == "0.0" || n.Value == "0."
		case syntax.StringLit:
			return n.Value == `""`
		}
	case *syntax.Name:
		switch n.Value {
		case "false":
			return s.nameObj(n) == types2.Universe.Lookup("false")
		case "nil":
			return s.nameObj(n) == types2.Universe.Lookup("nil")
		}
	}
	return false
}

func (s *nextScan) hasNew(lhs []*syntax.Name) bool {
	for _, n := range lhs {
		if n.Value == "_" {
			continue
		}
		if _, ok := s.info.Defs[n]; ok {
			return true
		}
	}
	return false
}

func (s *nextScan) errSpan(startPos syntax.Pos, call *syntax.CallExpr, check *syntax.IfStmt) (int, int, int, bool) {
	start, err := offsetOf(s.starts, len(s.src), startPos)
	if err != nil {
		return 0, 0, 0, false
	}
	callEnd, ok := s.endOf(call)
	if !ok {
		return 0, 0, 0, false
	}
	if check.Then == nil {
		return 0, 0, 0, false
	}
	rbrace, err := offsetOf(s.starts, len(s.src), check.Then.Rbrace)
	if err != nil {
		return 0, 0, 0, false
	}
	return start, callEnd, rbrace + 1, true
}

func (s *nextScan) task(n syntax.Node, ctx nextCtx, rule, instruction string) {
	s.tasks = append(s.tasks, NextTask{
		Category:    "ai-patch",
		Rule:        rule,
		StartLine:   int(n.Pos().Line()),
		EndLine:     int(syntax.EndPos(n).Line()),
		FuncSig:     s.funcSig(ctx.fn),
		Instruction: instruction,
	})
}

func (s *nextScan) funcSig(fn *syntax.FuncDecl) string {
	if fn == nil || fn.Body == nil {
		return ""
	}
	start, ok := s.funcStart(fn)
	if !ok {
		return ""
	}
	end, err := offsetOf(s.starts, len(s.src), fn.Body.Pos())
	if err != nil || end <= start {
		return ""
	}
	sig := string(s.src[start:end])
	if len(sig) > 0 && (sig[len(sig)-1] == '{' || sig[len(sig)-1] == ' ' || sig[len(sig)-1] == '\n' || sig[len(sig)-1] == '\t') {
		sig = sig[:len(sig)-1]
	}
	return strings.Trim(sig, " \t\n\r")
}

func (s *nextScan) funcStart(fn *syntax.FuncDecl) (int, bool) {
	name, err := offsetOf(s.starts, len(s.src), fn.Name.Pos())
	if err != nil {
		return 0, false
	}
	line := int(fn.Name.Pos().Line())
	head := s.starts[line-1]
	if idx := lastFunc(head, name, s.src); idx >= 0 {
		return head + idx, true
	}
	if line > 1 {
		prev := s.starts[line-2]
		if idx := lastFunc(prev, head, s.src); idx >= 0 {
			return prev + idx, true
		}
	}
	return name, true
}

func nextFuncBefore(i int, src []byte) bool {
	return i == 0 || src[i-1] == ' ' || src[i-1] == '\t' || src[i-1] == '\n' || src[i-1] == ';'
}

func nextFuncAfter(i int, src []byte) bool {
	return i+4 >= len(src) || src[i+4] == ' ' || src[i+4] == '\t' || src[i+4] == '\n' || src[i+4] == '('
}

func nextFuncWord(src []byte, i int) bool {
	if string(src[i:i+4]) != "func" {
		return false
	}
	return nextFuncBefore(i, src) && nextFuncAfter(i, src)
}

func lastFunc(lo, hi int, src []byte) int {
	found := -1
	for i := lo; i+4 <= hi; i++ {
		if nextFuncWord(src, i) {
			found = i - lo
		}
	}
	return found
}

func (s *nextScan) emit(start, end int, replacement string) {
	sl, sc := s.lineCol(start)
	el, ec := s.lineCol(end)
	s.edits = append(s.edits, NextEdit{
		StartLine:   sl,
		StartCol:    sc,
		EndLine:     el,
		EndCol:      ec,
		Replacement: replacement,
	})
	s.skip = append(s.skip, [2]int{sl, el})
}

func (s *nextScan) lineCol(off int) (int, int) {
	line := 0
	for i, st := range s.starts {
		if st > off {
			break
		}
		line = i
	}
	return line + 1, off - s.starts[line] + 1
}

func (s *nextScan) nameObj(n *syntax.Name) types2.Object {
	if obj, ok := s.info.Defs[n]; ok {
		return obj
	}
	if obj, ok := s.info.Uses[n]; ok {
		return obj
	}
	return nil
}

func (s *nextScan) sameErr(ret syntax.Expr, last *syntax.Name) bool {
	n, ok := ret.(*syntax.Name)
	if !ok || n.Value != last.Value {
		return false
	}
	return s.nameObj(n) == s.nameObj(last)
}

func (s *nextScan) usedElsewhere(fn *syntax.FuncDecl, obj types2.Object, allowed map[*syntax.Name]bool) bool {
	used := false
	syntax.Inspect(fn, func(n syntax.Node) bool {
		if used {
			return false
		}
		name, ok := n.(*syntax.Name)
		if !ok || allowed[name] {
			return true
		}
		if _, ok := s.info.Defs[name]; ok {
			return true
		}
		if s.info.Uses[name] == obj {
			used = true
			return false
		}
		return true
	})
	return used
}

func (s *nextScan) callLastErr(call *syntax.CallExpr) bool {
	obj := s.callee(call)
	if obj == nil {
		return false
	}
	return s.sigLastErr(obj.Type())
}

func (s *nextScan) callResults(call *syntax.CallExpr) (int, bool) {
	obj := s.callee(call)
	if obj == nil {
		return 0, false
	}
	sig, ok := obj.Type().(*types2.Signature)
	if !ok {
		return 0, false
	}
	return sig.Results().Len(), true
}

func (s *nextScan) callee(call *syntax.CallExpr) types2.Object {
	switch fun := unwrapParen(call.Fun).(type) {
	case *syntax.Name:
		return s.nameObj(fun)
	case *syntax.SelectorExpr:
		if obj, ok := s.info.Uses[fun.Sel]; ok {
			return obj
		}
		if sel, ok := s.info.Selections[fun]; ok {
			return sel.Obj()
		}
	}
	return nil
}

func unwrapParen(e syntax.Expr) syntax.Expr {
	for {
		p, ok := e.(*syntax.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}
