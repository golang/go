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

// This file lowers M8-S3 channel operations in group children (SPEC Sec
// 2.6, T-206): a plain send, receive, or range directly in a child
// selects on the group context, so a failing sibling wakes blocked
// peers, and a user `select` without `default` gains the same arm. A
// user `select` with `default` stays unchanged.
//
// Scope: statement-level operations only (`v := <-ch`, `var v = <-ch`,
// `ch <- v`, `for v := range ch`, bare `<-ch`, user selects). A call or
// receive inside an assignment target evaluates after the right-hand
// side here, where Go evaluates it before: keep side effects out of
// targets of a receiving assignment. A receive in a grouped
// `var ( ... )` declaration, or nested in
// a larger expression keeps blocking; bind it first. Statement spans
// are covered like every call template (with a literal scan first, so
// function literals inside lower before the cover drains their
// splices); tuples, lambdas, and interpolations inside a rewritten
// statement would overlap that cover, so keep them in child bodies.

// groupChanTop reports the child frame when lowering inside a child
// body: only there do channel operations select on cancellation.
func (fc *funcLower) groupChanTop() (groupFrame, bool) {
	if top, ok := fc.groupTop(); ok && top.child {
		return top, true
	}
	return groupFrame{}, false
}

// groupDoneText spells the cancel arm shared by rewritten
// communication: `case <-gctx.Done(): return gctx.Err()`.
func groupDoneText(gctx string) string {
	return "case <-" + gctx + ".Done(): return " + gctx + ".Err()"
}

// groupDoneClause builds the same arm as a syntax node.
func groupDoneClause(pos syntax.Pos, gctx string) *syntax.CommClause {
	done := callOf(pos, selOf(pos, nameOf(pos, gctx), "Done"), nil)
	recv := at(&syntax.Operation{Op: syntax.Recv, X: done}, pos)
	ret := returnOf(pos, []syntax.Expr{
		callOf(pos, selOf(pos, nameOf(pos, gctx), "Err"), nil),
	})
	return at(&syntax.CommClause{Comm: exprStmtOf(pos, recv), Body: []syntax.Stmt{ret}}, pos)
}

// groupRecvOp matches a bare channel receive serving as a whole
// right-hand element: `<-ch`, without the comma-ok form (which the
// assignment shape supplies).
func groupRecvOp(e syntax.Expr) (*syntax.Operation, bool) {
	op, ok := syntax.Unparen(e).(*syntax.Operation)
	if !ok || op.Op != syntax.Recv || op.Y != nil {
		return nil, false
	}
	return op, true
}

// groupChanElem returns the element type of the channel type t. A
// comma-ok receive records a tuple on the operation, so the element type
// always comes from the channel operand.
func groupChanElem(t types2.Type) types2.Type {
	ch, ok := t.Underlying().(*types2.Chan)
	if !ok {
		base.Fatalf("logos: receive from non-channel type %s", t)
	}
	return ch.Elem()
}

// groupHasRecv reports whether an assignment holds a direct receive on
// its right-hand side, the shape the M8-S3 rewrite covers.
func groupHasRecv(s *syntax.AssignStmt) bool {
	if s.Rhs == nil {
		return false
	}
	for _, r := range syntax.UnpackListExpr(s.Rhs) {
		if _, ok := groupRecvOp(r); ok {
			return true
		}
	}
	return false
}

// groupCallValues counts the values of a right-hand element: a tuple
// supplies its length, anything else one.
func groupCallValues(t types2.Type) int {
	if tup, ok := t.(*types2.Tuple); ok {
		return tup.Len()
	}
	return 1
}

// groupSpellType respells the runtime type t for eject text, qualifying
// outside packages the way the file denotes them.
func (fc *funcLower) groupSpellType(t types2.Type, pos syntax.Pos) syntax.Expr {
	text := types2.TypeString(t, fc.yieldQualifier)
	texpr, ok := reparseType(text)
	if !ok {
		base.Fatalf("logos: cannot respell channel element type %q", text)
	}
	syntax.Inspect(texpr, func(n syntax.Node) bool {
		if n != nil {
			n.SetPos(pos)
		}
		return true
	})
	return texpr
}

// groupRecvElem is one right-hand element of a receiving statement:
// recv is set for a direct receive, vals counts the values it supplies,
// and tmp and otmp name the value and ok temporaries of a receive.
type groupRecvElem struct {
	recv *syntax.Operation
	vals int
	tmp  string
	otmp string
}

// groupRecvLayout classifies the right-hand elements of a statement with
// lhsN targets. It reports false when no element receives or the
// elements do not cover the targets one for one, so the statement keeps
// its original form.
func (fc *funcLower) groupRecvLayout(lhsN int, rhs []syntax.Expr) ([]groupRecvElem, bool) {
	elems := make([]groupRecvElem, len(rhs))
	total, recvCount := 0, 0
	for i, r := range rhs {
		if op, ok := groupRecvOp(r); ok {
			elems[i].recv = op
			elems[i].vals = 1
			if len(rhs) == 1 && lhsN == 2 {
				elems[i].vals = 2
			}
			recvCount++
		} else {
			elems[i].vals = groupCallValues(fc.fl.pkg.info.TypeOf(r))
		}
		total += elems[i].vals
	}
	return elems, recvCount > 0 && total == lhsN
}

// groupTempName spells the next temporary with the given prefix.
func (fc *funcLower) groupTempName(prefix string) string {
	nm := fmt.Sprintf("_logos_%s%d", prefix, fc.tmpSeq)
	fc.tmpSeq++
	return nm
}

// groupNames builds name expressions for the given spellings.
func groupNames(pos syntax.Pos, spellings []string) []syntax.Expr {
	out := make([]syntax.Expr, len(spellings))
	for i, sp := range spellings {
		out[i] = nameOf(pos, sp)
	}
	return out
}

// groupPlainTemps binds a non-receiving right-hand element to vals
// temporaries, keeping its source text, so evaluation order is kept.
func (fc *funcLower) groupPlainTemps(pos syntax.Pos, r syntax.Expr, vals int) (syntax.Stmt, string, []string) {
	fl := fc.fl
	start := fl.off(leftmostOf(r))
	elemText := fl.textOf(span{start: start, end: groupExprEnd(fl.src, start)})
	names := make([]string, vals)
	for i := range names {
		names[i] = fc.groupTempName("t")
	}
	return defineOf(pos, groupNames(pos, names), []syntax.Expr{r}), strings.Join(names, ", ") + " := " + elemText, names
}

// groupRecvSelect declares the temporaries of a receiving element and
// receives into them through a select with the cancel arm. It returns
// the native statements, the text lines, and the temporary spellings.
func (fc *funcLower) groupRecvSelect(pos syntax.Pos, top groupFrame, e *groupRecvElem) ([]syntax.Stmt, []string, []string) {
	fl := fc.fl
	ch := e.recv.X
	chStart := fl.off(leftmostOf(ch))
	chText := fl.textOf(span{start: chStart, end: groupExprEnd(fl.src, chStart)})
	elemT := groupChanElem(fl.pkg.info.TypeOf(ch))
	e.tmp = fc.groupTempName("t")
	names := []string{e.tmp}
	native := []syntax.Stmt{typedVarDecl(pos, nameOf(pos, e.tmp), fc.groupSpellType(elemT, pos))}
	text := []string{"var " + e.tmp + " " + types2.TypeString(elemT, fc.yieldQualifier)}
	caseLHS := e.tmp
	if e.vals == 2 {
		e.otmp = fc.groupTempName("o")
		names = append(names, e.otmp)
		native = append(native, typedVarDecl(pos, nameOf(pos, e.otmp), nameOf(pos, "bool")))
		text = append(text, "var "+e.otmp+" bool")
		caseLHS += ", " + e.otmp
	}
	recvN := assignOf(pos, groupNames(pos, names), []syntax.Expr{at(&syntax.Operation{Op: syntax.Recv, X: ch}, pos)})
	native = append(native, at(&syntax.SelectStmt{
		Body: []*syntax.CommClause{at(&syntax.CommClause{Comm: recvN}, pos), groupDoneClause(pos, top.gctx)},
	}, pos))
	text = append(text, "select { case "+caseLHS+" = <-"+chText+": "+groupDoneText(top.gctx)+" }")
	return native, text, names
}

// groupRecvTemps evaluates every right-hand element into temporaries
// in source order: receives go through a cancel select, everything else
// binds plainly. It returns the native statements, the text lines, and
// the temporary spellings in target order.
func (fc *funcLower) groupRecvTemps(pos syntax.Pos, top groupFrame, rhs []syntax.Expr, elems []groupRecvElem) ([]syntax.Stmt, []string, []string) {
	var native []syntax.Stmt
	var text, temps []string
	for i := range elems {
		if elems[i].recv == nil {
			n, t, names := fc.groupPlainTemps(pos, rhs[i], elems[i].vals)
			native, text, temps = append(native, n), append(text, t), append(temps, names...)
			continue
		}
		n, t, names := fc.groupRecvSelect(pos, top, &elems[i])
		native, text, temps = append(native, n...), append(text, t...), append(temps, names...)
	}
	return native, text, temps
}

// groupAssignFinal builds the statement that moves the temporaries into
// the original targets, with the operator text it spells.
func groupAssignFinal(s *syntax.AssignStmt, lhs, temps []syntax.Expr) (syntax.Stmt, string) {
	pos := s.Pos()
	switch s.Op {
	case syntax.Def:
		return defineOf(pos, lhs, temps), " := "
	case 0:
		return assignOf(pos, lhs, temps), " = "
	}
	return at(&syntax.AssignStmt{Op: s.Op, Lhs: listOf(pos, lhs), Rhs: listOf(pos, temps)}, pos), " " + s.Op.String() + " "
}

// lowerGroupRecv rewrites an assignment that receives directly: every
// right-hand element evaluates into a temporary (receives through a
// cancel select), then the original assignment moves the temporaries.
func (fc *funcLower) lowerGroupRecv(s *syntax.AssignStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	scanForLiterals(fl, s)
	lhs := syntax.UnpackListExpr(s.Lhs)
	rhs := syntax.UnpackListExpr(s.Rhs)
	elems, ok := fc.groupRecvLayout(len(lhs), rhs)
	if !ok {
		return []syntax.Stmt{s}
	}
	pos := s.Pos()
	native, text, temps := fc.groupRecvTemps(pos, top, rhs, elems)
	finalN, sep := groupAssignFinal(s, lhs, groupNames(pos, temps))
	start := fl.off(leftmostOf(lhs[0]))
	text = append(text, fl.textOf(span{start: start, end: fl.off(pos)})+sep+strings.Join(temps, ", "))
	fl.add(start, stmtEnd(fl.src, start), strings.Join(text, "\n"))
	return append(native, finalN)
}

// groupRecvDecl matches `var v = <-ch` and its typed and comma-ok
// forms: a single ungrouped variable declaration with values.
func groupRecvDecl(s *syntax.DeclStmt) (*syntax.VarDecl, bool) {
	if len(s.DeclList) != 1 {
		return nil, false
	}
	d, ok := s.DeclList[0].(*syntax.VarDecl)
	if !ok || d.Group != nil || d.Values == nil {
		return nil, false
	}
	for _, v := range syntax.UnpackListExpr(d.Values) {
		if _, isRecv := groupRecvOp(v); isRecv {
			return d, true
		}
	}
	return nil, false
}

// groupAssignOffset returns the offset of the first `=` token at or
// after start, which separates the names and type of a declaration from
// its values.
func groupAssignOffset(src []byte, start int) int {
	off := start
	scanSegment(src, start, len(src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		if t.kind == token.ASSIGN {
			off = t.start
			return false
		}
		return true
	})
	return off
}

// groupVarDeclOf rebuilds a variable declaration over new values.
func groupVarDeclOf(pos syntax.Pos, d *syntax.VarDecl, values []syntax.Expr) syntax.Stmt {
	nd := at(&syntax.VarDecl{NameList: d.NameList, Type: d.Type, Values: listOf(pos, values)}, pos)
	return at(&syntax.DeclStmt{DeclList: []syntax.Decl{nd}}, pos)
}

// lowerGroupVarRecv rewrites a declaration that receives directly the
// way lowerGroupRecv rewrites an assignment: temporaries first, then the
// original declaration over the temporaries.
func (fc *funcLower) lowerGroupVarRecv(s *syntax.DeclStmt, d *syntax.VarDecl, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	scanForLiterals(fl, s)
	rhs := syntax.UnpackListExpr(d.Values)
	elems, ok := fc.groupRecvLayout(len(d.NameList), rhs)
	if !ok {
		return []syntax.Stmt{s}
	}
	pos := s.Pos()
	native, text, temps := fc.groupRecvTemps(pos, top, rhs, elems)
	start := fl.off(pos)
	head := strings.TrimRight(fl.textOf(span{start: start, end: groupAssignOffset(fl.src, start)}), " \t\n")
	text = append(text, head+" = "+strings.Join(temps, ", "))
	fl.add(start, stmtEnd(fl.src, start), strings.Join(text, "\n"))
	return append(native, groupVarDeclOf(pos, d, groupNames(pos, temps)))
}

// groupChanStmt rewrites the statements that communicate directly in a
// child body: receiving assignments and declarations, bare receives,
// sends, and ranges over channels. It reports false for every other
// statement, and outside child bodies.
func (fc *funcLower) groupChanStmt(s syntax.Stmt) ([]syntax.Stmt, bool) {
	top, ok := fc.groupChanTop()
	if !ok {
		return nil, false
	}
	switch s := s.(type) {
	case *syntax.AssignStmt:
		return fc.groupChanAssign(s, top)
	case *syntax.DeclStmt:
		return fc.groupChanDecl(s, top)
	case *syntax.ExprStmt:
		return fc.groupChanExpr(s, top)
	case *syntax.SendStmt:
		return fc.lowerGroupSend(s, top), true
	case *syntax.ForStmt:
		return fc.groupChanFor(s, top)
	}
	return nil, false
}

// groupChanAssign rewrites an assignment with a direct receive.
func (fc *funcLower) groupChanAssign(s *syntax.AssignStmt, top groupFrame) ([]syntax.Stmt, bool) {
	if !groupHasRecv(s) {
		return nil, false
	}
	return fc.lowerGroupRecv(s, top), true
}

// groupChanDecl rewrites a variable declaration with a direct receive.
func (fc *funcLower) groupChanDecl(s *syntax.DeclStmt, top groupFrame) ([]syntax.Stmt, bool) {
	d, ok := groupRecvDecl(s)
	if !ok {
		return nil, false
	}
	return fc.lowerGroupVarRecv(s, d, top), true
}

// groupChanExpr rewrites a bare receive statement.
func (fc *funcLower) groupChanExpr(s *syntax.ExprStmt, top groupFrame) ([]syntax.Stmt, bool) {
	op, ok := groupRecvOp(s.X)
	if !ok {
		return nil, false
	}
	return fc.lowerGroupRecvExpr(s, op, top), true
}

// groupChanFor rewrites a range over a channel.
func (fc *funcLower) groupChanFor(s *syntax.ForStmt, top groupFrame) ([]syntax.Stmt, bool) {
	rc, ok := s.Init.(*syntax.RangeClause)
	if !ok || !fc.groupRangeOverChan(rc) {
		return nil, false
	}
	return fc.lowerGroupRange(s, rc, top), true
}

// lowerGroupRecvExpr rewrites a bare `<-ch` in a child: a select that
// either receives or returns the context error.
func (fc *funcLower) lowerGroupRecvExpr(s *syntax.ExprStmt, op *syntax.Operation, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	scanForLiterals(fl, s)
	recvText := fl.textOf(span{start: fl.off(leftmostOf(op)), end: groupExprEnd(fl.src, fl.off(leftmostOf(op)))})
	native := at(&syntax.SelectStmt{
		Body: []*syntax.CommClause{
			at(&syntax.CommClause{Comm: exprStmtOf(pos, op)}, pos),
			groupDoneClause(pos, top.gctx),
		},
	}, pos)
	start := fl.off(leftmostOf(op))
	end := stmtEnd(fl.src, start)
	fl.add(start, end, "select { case "+recvText+": "+groupDoneText(top.gctx)+" }")
	return []syntax.Stmt{native}
}

// lowerGroupSend rewrites `ch <- v` in a child: a select that either
// sends or returns the context error.
func (fc *funcLower) lowerGroupSend(s *syntax.SendStmt, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	scanForLiterals(fl, s)
	chStart := fl.off(leftmostOf(s.Chan))
	chText := fl.textOf(span{start: chStart, end: fl.off(s.Pos())})
	vStart := fl.off(leftmostOf(s.Value))
	vText := fl.textOf(span{start: vStart, end: groupExprEnd(fl.src, vStart)})
	sendN := at(&syntax.SendStmt{Chan: s.Chan, Value: s.Value}, pos)
	native := at(&syntax.SelectStmt{
		Body: []*syntax.CommClause{
			at(&syntax.CommClause{Comm: sendN}, pos),
			groupDoneClause(pos, top.gctx),
		},
	}, pos)
	start := fl.off(leftmostOf(s.Chan))
	end := groupExprEnd(fl.src, vStart)
	fl.add(start, end, "select { case "+chText+" <- "+vText+": "+groupDoneText(top.gctx)+" }")
	return []syntax.Stmt{native}
}

// groupRangeOverChan reports whether a range clause ranges over a
// channel: only those iterate through the cancel adapter.
func (fc *funcLower) groupRangeOverChan(rc *syntax.RangeClause) bool {
	t := fc.fl.pkg.info.TypeOf(syntax.Unparen(rc.X))
	if t == nil {
		return false
	}
	_, ok := t.Underlying().(*types2.Chan)
	return ok
}

// lowerGroupRange rewrites `for v := range ch` in a child through the
// channel adapter, with a post-loop check that reports cancellation:
// a closed channel ends the range silently, a cancelled context ends
// it with the context error.
func (fc *funcLower) lowerGroupRange(s *syntax.ForStmt, rc *syntax.RangeClause, top groupFrame) []syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	scanForLiterals(fl, rc.X)
	chStart := fl.off(leftmostOf(rc.X))
	chText := strings.TrimRight(string(fl.src[chStart:fl.off(s.Body.Pos())]), " \t\n")
	fl.needRT = true
	adapter := callOf(pos, selOf(pos, nameOf(pos, rtAlias), "Chan"), []syntax.Expr{
		nameOf(pos, top.gctx), rc.X,
	})
	rc.X = adapter
	scanForLiterals(fl, s.Cond)
	scanForLiterals(fl, s.Post)
	s.Body.List = fc.lowerList(s.Body.List)
	lhsPart := ""
	if rc.Lhs != nil {
		lhsStart := fl.off(leftmostOf(rc.Lhs))
		lhsPart = strings.TrimRight(string(fl.src[lhsStart:fl.off(rc.Pos())]), " \t\n:=")
		if rc.Def {
			lhsPart += " := "
		} else {
			lhsPart += " = "
		}
	}
	headStart := fl.off(pos)
	braceOff := fl.off(s.Body.Pos())
	fl.add(headStart, braceOff, "for "+lhsPart+"range "+rtAlias+".Chan("+top.gctx+", "+chText+") ")
	checkN := ifOf(pos, nil,
		at(&syntax.Operation{
			Op: syntax.Neq,
			X:  callOf(pos, selOf(pos, nameOf(pos, top.gctx), "Err"), nil),
			Y:  nilOf(pos),
		}, pos),
		[]syntax.Stmt{
			returnOf(pos, []syntax.Expr{
				callOf(pos, selOf(pos, nameOf(pos, top.gctx), "Err"), nil),
			}),
		})
	end := fl.off(s.Body.Rbrace) + 1
	fl.add(end, end, "\n"+renderText([]syntax.Stmt{checkN}, nil))
	return []syntax.Stmt{s, checkN}
}

// groupSelectHasDefault reports whether a select already defaults.
func groupSelectHasDefault(s *syntax.SelectStmt) bool {
	for _, c := range s.Body {
		if c.Comm == nil {
			return true
		}
	}
	return false
}

// groupAugmentSelect adds the cancel arm to a user select without
// `default` in a child body: it fires only when no other case is ready
// and the group is done. A select with `default` never blocks, so it
// stays exactly as written, and a select outside a child is untouched.
func (fc *funcLower) groupAugmentSelect(s *syntax.SelectStmt) {
	top, ok := fc.groupChanTop()
	if !ok || groupSelectHasDefault(s) {
		return
	}
	pos := s.Pos()
	s.Body = append(s.Body, groupDoneClause(pos, top.gctx))
	rbraceOff := fc.fl.off(s.Rbrace)
	fc.fl.add(rbraceOff, rbraceOff, "\n"+groupDoneText(top.gctx)+"\n")
}
