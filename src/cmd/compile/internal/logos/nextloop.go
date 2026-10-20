// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strconv"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// loopMatch is one index-range append loop whose element passed the
// purity walk.
type loopMatch struct {
	loop  *syntax.ForStmt
	iv    string
	xs    string
	ys    string
	elem  syntax.Expr
	spans [][2]int
}

// loopAppend matches an index-range loop appending one pure element per
// iteration to a slice declared right before it in the same block. It
// reports whether the loop was rewritten or listed as a task.
func (s *nextScan) loopAppend(list []syntax.Stmt, i int, ctx nextCtx) bool {
	loop, ok := list[i].(*syntax.ForStmt)
	if !ok || loop.Body == nil {
		return false
	}
	iv, xs, ok := s.loopHeader(loop)
	if !ok {
		return false
	}
	ys, call, ok := s.loopBody(loop.Body)
	if !ok {
		if !s.appends(loop.Body) {
			return false
		}
		return s.taskLoop(loop, ctx, "the loop body does more than append one element, so no single chain expresses it")
	}
	m := loopMatch{loop: loop, iv: iv, xs: xs, ys: ys, elem: call.ArgList[1]}
	m.spans, ok = s.elemSpans(m.elem, xs, iv, ys)
	if !ok {
		return s.taskLoop(loop, ctx, "the appended element escapes the chain form, so the loop needs a manual rewrite")
	}
	if len(m.spans) == 0 {
		return false
	}
	return s.loopPair(list, i, m, ctx)
}

// appends reports whether a loop body calls the append builtin.
func (s *nextScan) appends(body *syntax.BlockStmt) bool {
	found := false
	syntax.Inspect(body, func(n syntax.Node) bool {
		call, ok := n.(*syntax.CallExpr)
		if ok && s.isBuiltin(call, "append") {
			found = true
		}
		return !found
	})
	return found
}

// loopPair pairs the loop at list[i] with the declaration of its target.
// The rewrite replaces every byte from the declaration to the end of the
// loop, so it applies only when the declaration is the statement right
// before the loop. A distant declaration whose target is named in
// between lists the loop as a task; otherwise the loop stays as written.
func (s *nextScan) loopPair(list []syntax.Stmt, i int, m loopMatch, ctx nextCtx) bool {
	start, declIdx, declEnd := s.findDecl(list, i, m.ys)
	if start < 0 {
		return false
	}
	if declIdx != i-1 {
		if !touchedBetween(list, declIdx, i, m.ys) {
			return false
		}
		return s.taskLoop(m.loop, ctx, "the target accumulates before the loop, so no single chain expresses it")
	}
	return s.spliceLoop(m, start, declEnd, ctx)
}

// spliceLoop emits the map chain replacing the declaration starting at
// start and the loop, or lists the loop when a comment sits inside.
func (s *nextScan) spliceLoop(m loopMatch, start, declEnd int, ctx nextCtx) bool {
	end, err := offsetOf(s.starts, len(s.src), m.loop.Body.Rbrace)
	if err != nil {
		return false
	}
	trailing, blocked := scanComments(s.src, start, declEnd, end+1)
	if blocked {
		return s.taskLoop(m.loop, ctx, "a comment sits inside the rewrite span, so the loop needs a manual rewrite")
	}
	param := chainParam(m.elem, m.xs, m.iv)
	built, ok := s.substitute(m.elem, m.spans, param)
	if !ok {
		return false
	}
	s.emit(start, end+1, m.ys+" := "+m.xs+".map("+param+" => "+built+")"+trailing)
	return true
}

// taskLoop lists one loop for AI patching and keeps scanning inside it.
func (s *nextScan) taskLoop(loop syntax.Stmt, ctx nextCtx, instruction string) bool {
	s.children(loop, ctx)
	s.task(loop, ctx, nextRuleLoopChain, instruction)
	return true
}

// findDecl finds the nearest earlier declaration of the loop target,
// returning its start and end offsets and its index. Declaration
// positions name keywords or operators, so bounds come from the
// declared name and type instead. Missing declarations report -1.
func (s *nextScan) findDecl(list []syntax.Stmt, i int, ys string) (int, int, int) {
	for j := i - 1; j >= 0; j-- {
		start, declEnd, ok := s.declBounds(list[j], ys)
		if !ok {
			continue
		}
		return start, j, declEnd
	}
	return -1, 0, 0
}

// touchedBetween reports whether the target is named between its
// declaration and the loop.
func touchedBetween(list []syntax.Stmt, decl, i int, ys string) bool {
	for _, between := range list[decl+1 : i] {
		if mentions(between, ys) {
			return true
		}
	}
	return false
}

// mentions reports whether a statement names a variable.
func mentions(stmt syntax.Stmt, name string) bool {
	found := false
	syntax.Inspect(stmt, func(n syntax.Node) bool {
		if found {
			return false
		}
		if id, ok := n.(*syntax.Name); ok && id.Value == name {
			found = true
			return false
		}
		return true
	})
	return found
}

// declBounds matches one empty slice declaration of the loop target,
// returning its start and end offsets for splicing and comments.
func (s *nextScan) declBounds(prev syntax.Stmt, ys string) (int, int, bool) {
	switch n := prev.(type) {
	case *syntax.DeclStmt:
		return s.declBoundsVarDecl(n, ys)
	case *syntax.AssignStmt:
		return s.declBoundsAssign(n, ys)
	}
	return 0, 0, false
}

// declBoundsVarDecl matches a var declaration of the loop target,
// returning its start and end offsets for splicing and comments.
func (s *nextScan) declBoundsVarDecl(n *syntax.DeclStmt, ys string) (int, int, bool) {
	if len(n.DeclList) != 1 {
		return 0, 0, false
	}
	decl, ok := n.DeclList[0].(*syntax.VarDecl)
	if !ok {
		return 0, 0, false
	}
	if !declBoundsVarForm(decl, ys) {
		return 0, 0, false
	}
	start, err := offsetOf(s.starts, len(s.src), n.Pos())
	if err != nil {
		return 0, 0, false
	}
	end, ok := s.endOf(decl.Type)
	if !ok {
		return 0, 0, false
	}
	return start, end, true
}

// declBoundsVarForm reports whether a variable declaration introduces
// the loop target as a valueless slice.
func declBoundsVarForm(decl *syntax.VarDecl, ys string) bool {
	if len(decl.NameList) != 1 {
		return false
	}
	if decl.Values != nil {
		return false
	}
	if _, ok := decl.Type.(*syntax.SliceType); !ok {
		return false
	}
	return decl.NameList[0].Value == ys
}

// declBoundsAssign matches a short declaration of the loop target,
// returning its start and end offsets for splicing and comments.
func (s *nextScan) declBoundsAssign(n *syntax.AssignStmt, ys string) (int, int, bool) {
	if !declBoundsEmptySlice(n, ys) {
		return 0, 0, false
	}
	start, ok := s.startOf(n.Lhs)
	if !ok {
		return 0, 0, false
	}
	end, ok := s.endOf(n.Rhs)
	if !ok {
		return 0, 0, false
	}
	return start, end, true
}

// declBoundsEmptySlice reports whether an assignment declares the loop
// target as an empty slice literal.
func declBoundsEmptySlice(n *syntax.AssignStmt, ys string) bool {
	if n.Op != syntax.Def {
		return false
	}
	name, ok := n.Lhs.(*syntax.Name)
	if !ok || name.Value != ys {
		return false
	}
	lit, ok := n.Rhs.(*syntax.CompositeLit)
	if !ok || len(lit.ElemList) != 0 {
		return false
	}
	_, ok = lit.Type.(*syntax.SliceType)
	return ok
}

// loopHeader verifies the classic index loop over len(xs), returning
// the index and receiver names.
func (s *nextScan) loopHeader(loop *syntax.ForStmt) (string, string, bool) {
	iv, ok := loopHeaderIndex(loop)
	if !ok {
		return "", "", false
	}
	xs, ok := s.loopHeaderBound(loop, iv)
	if !ok {
		return "", "", false
	}
	if !loopHeaderPost(loop, iv) {
		return "", "", false
	}
	return iv, xs, true
}

// loopHeaderIndex verifies the zero-based index initialization,
// returning the index name.
func loopHeaderIndex(loop *syntax.ForStmt) (string, bool) {
	init, ok := loop.Init.(*syntax.AssignStmt)
	if !ok || init.Op != syntax.Def {
		return "", false
	}
	iv, ok := init.Lhs.(*syntax.Name)
	if !ok {
		return "", false
	}
	zero, ok := init.Rhs.(*syntax.BasicLit)
	if !ok || zero.Kind != syntax.IntLit || zero.Value != "0" {
		return "", false
	}
	return iv.Value, true
}

// loopHeaderBound verifies the loop condition against the receiver
// length, returning the receiver name.
func (s *nextScan) loopHeaderBound(loop *syntax.ForStmt, iv string) (string, bool) {
	cond, ok := unwrapParen(loop.Cond).(*syntax.Operation)
	if !ok || cond.Op != syntax.Lss || cond.Y == nil {
		return "", false
	}
	ix, ok := cond.X.(*syntax.Name)
	if !ok || ix.Value != iv {
		return "", false
	}
	bound, ok := cond.Y.(*syntax.CallExpr)
	if !ok {
		return "", false
	}
	return s.loopHeaderLen(bound, iv)
}

// loopHeaderLen verifies one len call over the receiver, returning the
// receiver name.
func (s *nextScan) loopHeaderLen(bound *syntax.CallExpr, iv string) (string, bool) {
	if len(bound.ArgList) != 1 || bound.HasDots {
		return "", false
	}
	if !s.isBuiltin(bound, "len") {
		return "", false
	}
	xs, ok := bound.ArgList[0].(*syntax.Name)
	if !ok || xs.Value == iv {
		return "", false
	}
	return xs.Value, true
}

// loopHeaderPost verifies the index increment of the loop post
// statement.
func loopHeaderPost(loop *syntax.ForStmt, iv string) bool {
	post, ok := loop.Post.(*syntax.AssignStmt)
	if !ok || post.Op != syntax.Add || post.Rhs != nil {
		return false
	}
	pi, ok := post.Lhs.(*syntax.Name)
	if !ok || pi.Value != iv {
		return false
	}
	return true
}

// loopBody verifies a body holding one append of one element, returning
// the target and call.
func (s *nextScan) loopBody(body *syntax.BlockStmt) (string, *syntax.CallExpr, bool) {
	assign, ok := loopBodySingle(body)
	if !ok {
		return "", nil, false
	}
	target, ok := assign.Lhs.(*syntax.Name)
	if !ok {
		return "", nil, false
	}
	call, ok := s.loopBodyAppend(assign, target)
	if !ok {
		return "", nil, false
	}
	return target.Value, call, true
}

// loopBodySingle extracts the single plain assignment from a loop body.
func loopBodySingle(body *syntax.BlockStmt) (*syntax.AssignStmt, bool) {
	if len(body.List) != 1 {
		return nil, false
	}
	assign, ok := body.List[0].(*syntax.AssignStmt)
	if !ok || assign.Op != 0 {
		return nil, false
	}
	return assign, true
}

// loopBodyAppend verifies the append call assigning to the target,
// returning the call.
func (s *nextScan) loopBodyAppend(assign *syntax.AssignStmt, target *syntax.Name) (*syntax.CallExpr, bool) {
	call, ok := unwrapParen(assign.Rhs).(*syntax.CallExpr)
	if !ok {
		return nil, false
	}
	if !s.isBuiltin(call, "append") {
		return nil, false
	}
	if len(call.ArgList) != 2 || call.HasDots {
		return nil, false
	}
	dst, ok := call.ArgList[0].(*syntax.Name)
	if !ok || dst.Value != target.Value {
		return nil, false
	}
	return call, true
}

// isBuiltin reports whether a call names one language builtin.
func (s *nextScan) isBuiltin(call *syntax.CallExpr, name string) bool {
	fun, ok := call.Fun.(*syntax.Name)
	if !ok || fun.Value != name {
		return false
	}
	obj, ok := s.info.Uses[fun].(*types2.Builtin)
	return ok && obj.Name() == name
}

// elemSpans checks the appended element for chain purity and records
// the spans of xs[i] for substitution. A pure element references the
// index only through xs[i], never takes addresses, and calls only
// plain or package-qualified functions.
func (s *nextScan) elemSpans(elem syntax.Expr, xs, iv, ys string) ([][2]int, bool) {
	c := &elemCheck{scan: s, xs: xs, iv: iv, ys: ys}
	if !c.expr(elem) {
		return nil, false
	}
	return c.spans, true
}

// elemCheck carries one purity walk.
type elemCheck struct {
	scan  *nextScan
	xs    string
	iv    string
	ys    string
	spans [][2]int
}

// expr verifies one element expression. The loop index may appear only
// as the subscript of the receiver.
func (c *elemCheck) expr(e syntax.Expr) bool {
	switch n := unwrapParen(e).(type) {
	case *syntax.Name:
		return exprName(n, c.ys, c.iv)
	case *syntax.BasicLit:
		return true
	case *syntax.SelectorExpr:
		return c.expr(n.X)
	case *syntax.IndexExpr:
		return c.index(n)
	case *syntax.CallExpr:
		return c.call(n)
	case *syntax.Operation:
		return c.exprOp(n)
	case *syntax.ParenExpr:
		return c.expr(n.X)
	}
	return false
}

// exprName reports whether a name avoids the loop target and index.
func exprName(n *syntax.Name, ys, iv string) bool {
	return n.Value != ys && n.Value != iv
}

// exprOp verifies one operator expression, rejecting address and
// receive operators.
func (c *elemCheck) exprOp(n *syntax.Operation) bool {
	if n.Op == syntax.And || n.Op == syntax.Recv {
		return false
	}
	if n.Y == nil {
		return c.expr(n.X)
	}
	return c.expr(n.X) && c.expr(n.Y)
}

// index verifies one index expression and records xs[i].
func (c *elemCheck) index(n *syntax.IndexExpr) bool {
	x, ok := n.X.(*syntax.Name)
	if !ok || x.Value != c.xs {
		return c.expr(n.X) && c.expr(n.Index)
	}
	i, ok := unwrapParen(n.Index).(*syntax.Name)
	if !ok || i.Value != c.iv {
		return false
	}
	start, err := offsetOf(c.scan.starts, len(c.scan.src), x.Pos())
	if err != nil {
		return false
	}
	end, ok := c.scan.endOf(n)
	if !ok {
		return false
	}
	c.spans = append(c.spans, [2]int{start, end})
	return true
}

// call verifies one element call.
func (c *elemCheck) call(n *syntax.CallExpr) bool {
	if n.HasDots {
		return false
	}
	if !c.callFun(n.Fun) {
		return false
	}
	return c.callArgs(n)
}

// callFun verifies the function of one element call.
func (c *elemCheck) callFun(fun syntax.Expr) bool {
	switch f := fun.(type) {
	case *syntax.Name:
		return c.callName(f)
	case *syntax.SelectorExpr:
		return c.callSelector(f)
	}
	return false
}

// callName verifies a plain function name of one element call.
func (c *elemCheck) callName(fun *syntax.Name) bool {
	if fun.Value == "append" {
		return false
	}
	if fun.Value == "len" || fun.Value == "cap" {
		return true
	}
	return callObjFunc(c.scan.nameObj(fun))
}

// callObjFunc reports whether a resolved object is a function or a
// builtin callable from an element.
func callObjFunc(obj types2.Object) bool {
	if _, ok := obj.(*types2.Func); ok {
		return true
	}
	_, ok := obj.(*types2.Builtin)
	return ok
}

// callSelector verifies a package-qualified function of one element
// call.
func (c *elemCheck) callSelector(fun *syntax.SelectorExpr) bool {
	base, ok := fun.X.(*syntax.Name)
	if !ok {
		return false
	}
	_, ok = c.scan.info.Uses[base].(*types2.PkgName)
	return ok
}

// callArgs verifies every argument of one element call.
func (c *elemCheck) callArgs(n *syntax.CallExpr) bool {
	for _, arg := range n.ArgList {
		if !c.expr(arg) {
			return false
		}
	}
	return true
}

// chainParam picks the lambda parameter name.
func chainParam(elem syntax.Expr, xs, iv string) string {
	used := map[string]bool{xs: true, iv: true}
	syntax.Inspect(elem, func(n syntax.Node) bool {
		if name, ok := n.(*syntax.Name); ok {
			used[name.Value] = true
		}
		return true
	})
	for _, cand := range []string{"x", "v", "e", "elem"} {
		if !used[cand] {
			return cand
		}
	}
	for i := 1; ; i++ {
		cand := "elem" + strconv.Itoa(i)
		if !used[cand] {
			return cand
		}
	}
}

// substitute splices the element source with the parameter.
func (s *nextScan) substitute(elem syntax.Expr, spans [][2]int, param string) (string, bool) {
	start, ok := s.startOf(elem)
	if !ok {
		return "", false
	}
	end, ok := s.endOf(elem)
	if !ok {
		return "", false
	}
	var out strings.Builder
	pos := start
	for _, span := range spans {
		if span[0] < pos || span[1] > end {
			return "", false
		}
		out.Write(s.src[pos:span[0]])
		out.WriteString(param)
		pos = span[1]
	}
	out.Write(s.src[pos:end])
	return out.String(), true
}
