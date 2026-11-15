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

type loopMatch struct {
	loop  *syntax.ForStmt
	iv    string
	xs    string
	ys    string
	elem  syntax.Expr
	spans [][2]int
}

func (s *nextScan) loopAppend(list []syntax.Stmt, i int, ctx nextCtx) bool {
	loop, ok := list[i].(*syntax.ForStmt)
	if !ok || loop.Body == nil {
		return false
	}
	if s.rangeTwoVar(loop) {
		return s.taskLoop(loop, ctx, "the range binds both index and value, so the loop needs a manual rewrite")
	}
	if s.isRangeFunc(loop) {
		return s.taskLoop(loop, ctx, "the ranged value is a function, so the loop needs a manual rewrite")
	}
	iv, xs, val, ok := s.loopHeader(loop)
	if !ok {
		if _, ok := loop.Init.(*syntax.RangeClause); ok {
			return s.taskLoop(loop, ctx, "the range shape needs a manual rewrite")
		}
		return false
	}
	if !s.loopSliceOk(loop, xs) {
		return s.taskLoop(loop, ctx, "the ranged value is not a slice, so the loop needs a manual rewrite")
	}
	ys, call, ok := s.loopBody(loop.Body)
	if !ok {
		if !s.appends(loop.Body) {
			return false
		}
		return s.taskLoop(loop, ctx, "the loop body does more than append one element, so no single chain expresses it")
	}
	m := loopMatch{loop: loop, iv: iv, xs: xs, ys: ys, elem: call.ArgList[1]}
	c := &elemCheck{scan: s, xs: xs, iv: iv, ys: ys, val: val, unsafeCalls: s.receiverUnsafe(loop, xs, ctx)}
	if !c.expr(m.elem) {
		return s.taskLoop(loop, ctx, "the appended element escapes the chain form, so the loop needs a manual rewrite")
	}
	m.spans = c.spans
	if len(m.spans) == 0 {
		return false
	}
	return s.loopPair(list, i, m, ctx)
}

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

func (s *nextScan) loopPair(list []syntax.Stmt, i int, m loopMatch, ctx nextCtx) bool {
	start, declIdx, declEnd := s.findDecl(list, i, m.ys)
	if start < 0 {
		return false
	}
	if declIdx != i-1 {
		if !touchedBetween(list, declIdx, i, m.ys) {
			return s.taskLoop(m.loop, ctx, "an unrelated statement sits between the declaration and the loop, so the loop needs a manual rewrite")
		}
		return s.taskLoop(m.loop, ctx, "the target accumulates before the loop, so no single chain expresses it")
	}
	if !declNilSlice(list[declIdx], m.ys) {
		return false
	}
	return s.spliceLoop(m, start, declEnd, ctx)
}

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

func (s *nextScan) taskLoop(loop syntax.Stmt, ctx nextCtx, instruction string) bool {
	s.children(loop, ctx)
	s.task(loop, ctx, nextRuleLoopChain, instruction)
	return true
}
func (s *nextScan) rangeTwoVar(loop *syntax.ForStmt) bool {
	rc, ok := loop.Init.(*syntax.RangeClause)
	if !ok {
		return false
	}
	l, ok := rc.Lhs.(*syntax.ListExpr)
	if !ok || len(l.ElemList) != 2 {
		return false
	}
	idx, ok := l.ElemList[0].(*syntax.Name)
	if !ok {
		return false
	}
	return idx.Value != "_"
}
func (s *nextScan) loopSliceOk(loop *syntax.ForStmt, xs string) bool {
	obj := s.loopReceiverObj(loop, xs)
	if obj == nil {
		return false
	}
	t := obj.Type()
	if t == nil {
		return false
	}
	if _, ok := t.(*types2.TypeParam); ok {
		return false
	}
	_, ok := t.(*types2.Slice)
	return ok
}
func (s *nextScan) isRangeFunc(loop *syntax.ForStmt) bool {
	rc, ok := loop.Init.(*syntax.RangeClause)
	if !ok || s.info == nil {
		return false
	}
	if t := s.info.TypeOf(rc.X); t != nil {
		_, ok = t.Underlying().(*types2.Signature)
		return ok
	}
	name, ok := rc.X.(*syntax.Name)
	if !ok {
		return false
	}
	obj := s.loopReceiverObj(loop, name.Value)
	if obj == nil || obj.Type() == nil {
		return true
	}
	_, ok = obj.Type().Underlying().(*types2.Signature)
	return ok
}

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

func touchedBetween(list []syntax.Stmt, decl, i int, ys string) bool {
	for _, between := range list[decl+1 : i] {
		if mentions(between, ys) {
			return true
		}
	}
	return false
}

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

func (s *nextScan) declBounds(prev syntax.Stmt, ys string) (int, int, bool) {
	switch n := prev.(type) {
	case *syntax.DeclStmt:
		return s.declBoundsVarDecl(n, ys)
	case *syntax.AssignStmt:
		return s.declBoundsAssign(n, ys)
	}
	return 0, 0, false
}

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

func declNilSlice(stmt syntax.Stmt, ys string) bool {
	n, ok := stmt.(*syntax.DeclStmt)
	if !ok || len(n.DeclList) != 1 {
		return false
	}
	decl, ok := n.DeclList[0].(*syntax.VarDecl)
	return ok && declBoundsVarForm(decl, ys)
}

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

func (s *nextScan) loopHeader(loop *syntax.ForStmt) (string, string, string, bool) {
	if rc, ok := loop.Init.(*syntax.RangeClause); ok {
		return s.loopHeaderRange(loop, rc)
	}
	iv, ok := loopHeaderIndex(loop)
	if !ok {
		return "", "", "", false
	}
	xs, ok := s.loopHeaderBound(loop, iv)
	if !ok {
		return "", "", "", false
	}
	if !loopHeaderPost(loop, iv) {
		return "", "", "", false
	}
	return iv, xs, "", true
}

func (s *nextScan) loopHeaderRange(loop *syntax.ForStmt, rc *syntax.RangeClause) (string, string, string, bool) {
	if !rc.Def || loop.Cond != nil || loop.Post != nil {
		return "", "", "", false
	}
	xsName, ok := rc.X.(*syntax.Name)
	if !ok || xsName.Value == "_" {
		return "", "", "", false
	}
	var iv, val string
	switch lhs := rc.Lhs.(type) {
	case *syntax.Name:
		iv = lhs.Value
	case *syntax.ListExpr:
		if len(lhs.ElemList) != 2 {
			return "", "", "", false
		}
		idx, ok := lhs.ElemList[0].(*syntax.Name)
		if !ok || idx.Value != "_" {
			return "", "", "", false
		}
		v, ok := lhs.ElemList[1].(*syntax.Name)
		if !ok {
			return "", "", "", false
		}
		iv, val = "_", v.Value
	default:
		return "", "", "", false
	}
	if iv == "_" && val == "" || iv == xsName.Value || val == "_" || val == xsName.Value {
		return "", "", "", false
	}
	return iv, xsName.Value, val, true
}

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

func (s *nextScan) isBuiltin(call *syntax.CallExpr, name string) bool {
	fun, ok := call.Fun.(*syntax.Name)
	if !ok || fun.Value != name {
		return false
	}
	obj, ok := s.info.Uses[fun].(*types2.Builtin)
	return ok && obj.Name() == name
}

func (s *nextScan) initFileFacts(f *syntax.File) {
	s.addrTaken = make(map[types2.Object]bool)
	s.closureWritten = make(map[types2.Object]bool)
	if s.info == nil {
		return
	}
	for _, decl := range f.DeclList {
		fn, ok := decl.(*syntax.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		s.scanFuncFacts(fn)
	}
}

func (s *nextScan) scanFuncFacts(fn *syntax.FuncDecl) {
	syntax.Walk(fn.Body, factVisitor{scan: s})
}

type factVisitor struct {
	scan         *nextScan
	inClosure    bool
	lit          *syntax.FuncLit
	closureScope *types2.Scope
}

func (v factVisitor) Visit(n syntax.Node) syntax.Visitor {
	if n == nil {
		return nil
	}
	switch n := n.(type) {
	case *syntax.FuncLit:
		v.inClosure = true
		v.lit = n
		if v.scan.info != nil && v.scan.info.Scopes != nil {
			v.closureScope = v.scan.info.Scopes[n.Type]
		}
		return v
	case *syntax.Operation:
		if n.Op == syntax.And && n.Y == nil {
			if name, ok := unwrapParen(n.X).(*syntax.Name); ok {
				if obj := v.scan.nameObj(name); obj != nil {
					v.scan.addrTaken[obj] = true
				}
			}
		}
	case *syntax.AssignStmt:
		if v.inClosure {
			v.checkAssign(n)
		}
	case *syntax.ForStmt:
		if v.inClosure {
			v.checkFor(n)
		}
	}
	return v
}

func (v factVisitor) checkAssign(as *syntax.AssignStmt) {
	for _, name := range assignedNames(as.Lhs) {
		if as.Op == syntax.Def && v.scan.info != nil && v.scan.info.Defs[name] != nil {
			continue
		}
		obj := v.scan.nameObj(name)
		if obj == nil {
			continue
		}
		if v.isOuterVar(obj) {
			v.scan.closureWritten[obj] = true
		}
	}
}

func (v factVisitor) checkFor(loop *syntax.ForStmt) {
	rc, ok := loop.Init.(*syntax.RangeClause)
	if !ok || rc.Def {
		return
	}
	for _, name := range assignedNames(rc.Lhs) {
		obj := v.scan.nameObj(name)
		if obj == nil {
			continue
		}
		if v.isOuterVar(obj) {
			v.scan.closureWritten[obj] = true
		}
	}
}

func (v factVisitor) isOuterVar(obj types2.Object) bool {
	if v.closureScope != nil {
		return !scopeEncloses(v.closureScope, obj.Parent())
	}
	if v.lit != nil && obj.Pos().IsKnown() && v.lit.Pos().IsKnown() {
		return obj.Pos().Cmp(v.lit.Pos()) < 0
	}
	return true
}

func scopeEncloses(ancestor, child *types2.Scope) bool {
	for cur := child; cur != nil; cur = cur.Parent() {
		if cur == ancestor {
			return true
		}
	}
	return false
}

func assignedNames(lhs syntax.Expr) []*syntax.Name {
	switch n := lhs.(type) {
	case *syntax.Name:
		return []*syntax.Name{n}
	case *syntax.ListExpr:
		var names []*syntax.Name
		for _, elem := range n.ElemList {
			names = append(names, assignedNames(elem)...)
		}
		return names
	case *syntax.ParenExpr:
		return assignedNames(n.X)
	}
	return nil
}

type elemCheck struct {
	scan        *nextScan
	xs          string
	iv          string
	ys          string
	val         string
	unsafeCalls bool
	spans       [][2]int
}

func (s *nextScan) receiverUnsafe(loop *syntax.ForStmt, xs string, ctx nextCtx) bool {
	obj := s.loopReceiverObj(loop, xs)
	if obj == nil {
		return true
	}
	if isPackageVar(obj) {
		return true
	}
	return s.addrTaken[obj] || s.closureWritten[obj]
}

func (s *nextScan) loopReceiverObj(loop *syntax.ForStmt, xs string) types2.Object {
	var found types2.Object
	inspect := func(n syntax.Node) {
		if n == nil || found != nil {
			return
		}
		syntax.Inspect(n, func(n syntax.Node) bool {
			if name, ok := n.(*syntax.Name); ok && name.Value == xs && found == nil {
				found = s.nameObj(name)
			}
			return found == nil
		})
	}
	inspect(loop.Cond)
	inspect(loop.Init)
	return found
}

func isPackageVar(obj types2.Object) bool {
	v, ok := obj.(*types2.Var)
	if !ok || v.Name() == "" {
		return false
	}
	pkg := v.Pkg()
	if pkg == nil {
		return false
	}
	return pkg.Scope().Lookup(v.Name()) == obj
}
func (c *elemCheck) expr(e syntax.Expr) bool {
	switch n := unwrapParen(e).(type) {
	case *syntax.Name:
		if c.val != "" && n.Value == c.val {
			return c.elemRef(n)
		}
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
func (c *elemCheck) elemRef(n *syntax.Name) bool {
	start, err := offsetOf(c.scan.starts, len(c.scan.src), n.Pos())
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

func exprName(n *syntax.Name, ys, iv string) bool {
	return n.Value != ys && n.Value != iv
}

func (c *elemCheck) exprOp(n *syntax.Operation) bool {
	if n.Op == syntax.And || n.Op == syntax.Recv {
		return false
	}
	if n.Y == nil {
		return c.expr(n.X)
	}
	return c.expr(n.X) && c.expr(n.Y)
}

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

func (c *elemCheck) call(n *syntax.CallExpr) bool {
	if n.HasDots {
		return false
	}
	if !c.callFun(n.Fun) {
		return false
	}
	return c.callArgs(n)
}

func (c *elemCheck) callFun(fun syntax.Expr) bool {
	switch f := fun.(type) {
	case *syntax.Name:
		return c.callName(f)
	case *syntax.SelectorExpr:
		return c.callSelector(f)
	}
	return false
}

func (c *elemCheck) callName(fun *syntax.Name) bool {
	if fun.Value == "append" {
		return false
	}
	if fun.Value == "len" || fun.Value == "cap" {
		return true
	}
	obj := c.scan.nameObj(fun)
	if _, ok := obj.(*types2.Func); ok {
		if c.unsafeCalls {
			return false
		}
		return true
	}
	return callObjFunc(obj)
}

func callObjFunc(obj types2.Object) bool {
	if _, ok := obj.(*types2.Func); ok {
		return true
	}
	_, ok := obj.(*types2.Builtin)
	return ok
}

func (c *elemCheck) callSelector(fun *syntax.SelectorExpr) bool {
	if c.unsafeCalls {
		return false
	}
	base, ok := fun.X.(*syntax.Name)
	if !ok {
		return false
	}
	_, ok = c.scan.info.Uses[base].(*types2.PkgName)
	return ok
}

func (c *elemCheck) callArgs(n *syntax.CallExpr) bool {
	for _, arg := range n.ArgList {
		if !c.expr(arg) {
			return false
		}
	}
	return true
}

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
