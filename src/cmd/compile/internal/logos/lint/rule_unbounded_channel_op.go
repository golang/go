// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"go/constant"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// unboundedChannelOp is correctness::unbounded_channel_op (SPEC Sec 1.2,
// #163): a channel send or receive that has no cancellation path. An
// operation is bounded in two cases only:
//
//   - it is a case of a select statement that has a <-x.Done() case, where
//     x is any expression and the call returns a channel; a receive of
//     x.Done() on its own is the wait for cancellation itself;
//   - its channel is a local variable made once, in the function, by
//     make(chan T, k) with a constant capacity k, and the function sends on
//     the channel at most k times in total, never inside a loop.
//
// For the second case the function is the declaration or top-level literal
// together with every literal nested in it. A literal inside a loop counts
// as inside it, and a send in a select case counts. The two conditions are
// properties of the channel: when the sends exceed k, or any send sits in a
// loop, every operation on that channel is reported, including the
// receives. A receive is bounded by the same conditions as the sends. The
// statement bodies of a done select are ordinary code. A range over a
// channel, close, and len are not operations of this rule. A select with a
// default case but no done case is not exempt. In loGo/s, a plain channel
// operation directly inside a go group child block is bounded too, because
// lowering turns it into a select on the group context's Done channel; a
// function literal inside the block is not covered.
var unboundedChannelOp = Rule{
	ID:      "correctness::unbounded_channel_op",
	Group:   Correctness,
	Default: Deny,
	Go:      true,
	Check:   checkUnboundedChannelOp,
}

// checkUnboundedChannelOp is the Check of unboundedChannelOp. Every function
// declaration, and every function literal outside one, is analyzed with the
// literals nested in it.
func checkUnboundedChannelOp(pass *Pass) {
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.FuncDecl:
				if n.Body != nil {
					newChanUnit(pass, f, n).run()
				}
				return false
			case *syntax.FuncLit:
				newChanUnit(pass, f, n).run()
				return false
			}
			return true
		})
	}
}

// chanOp is one operation not covered by a done select: its node, the
// position just after its last token, the local channel variable it names or
// nil, and its description.
type chanOp struct {
	node syntax.Node
	end  syntax.Pos
	ch   *types2.Var
	what string
}

// chanUnit is the analysis of one function and the literals inside it.
type chanUnit struct {
	pass *Pass
	file *syntax.File
	root syntax.Node
	// free is set while walking code whose channel operations are covered:
	// the cases of a done select and a go group child block.
	free  bool
	loops int
	// defs counts the assignments to each local variable; caps holds the
	// capacity each one was made with, -1 when it is not constant.
	defs   map[*types2.Var]int
	caps   map[*types2.Var]int64
	sent   map[*types2.Var]int
	looped map[*types2.Var]bool
	ops    []chanOp
}

// newChanUnit prepares the analysis of the function rooted at root, a node of
// file f.
func newChanUnit(pass *Pass, f *syntax.File, root syntax.Node) *chanUnit {
	return &chanUnit{pass: pass, file: f, root: root}
}

// run collects the made channels, walks the operations, and reports those on
// a channel that is not bounded.
func (u *chanUnit) run() {
	syntax.Inspect(u.root, u.visit)
	if len(u.ops) == 0 {
		return
	}
	u.defs = make(map[*types2.Var]int)
	u.caps = make(map[*types2.Var]int64)
	syntax.Inspect(u.root, u.collect)
	for _, op := range u.ops {
		if op.ch != nil && u.bounded(op.ch) {
			continue
		}
		u.pass.Report(Diagnostic{
			Pos:     syntax.StartPos(op.node),
			End:     op.end,
			Message: fmt.Sprintf("%s may block forever: put it in a select with a <-ctx.Done() case, or make the channel locally with a constant capacity that covers every send", op.what),
		})
	}
}

// bounded reports whether ch is a local channel made once with a constant
// capacity that covers every send of the function.
func (u *chanUnit) bounded(ch *types2.Var) bool {
	k, ok := u.caps[ch]
	return ok && u.defs[ch] == 1 && !u.looped[ch] && int64(u.sent[ch]) <= k
}

// collect records the assignments to local variables.
func (u *chanUnit) collect(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.AssignStmt:
		if lhs, rhs, ok := assignPairs(n); ok {
			u.note(lhs, rhs)
		}
	case *syntax.VarDecl:
		if lhs, rhs, ok := declPairs(n); ok {
			u.note(lhs, rhs)
		}
	}
	return true
}

// note records each variable of lhs as assigned, with the capacity of the
// channel the matching value makes.
func (u *chanUnit) note(lhs, rhs []syntax.Expr) {
	for i, l := range lhs {
		id, ok := bareExpr(l).(*syntax.Name)
		if !ok {
			continue
		}
		v := localVar(u.pass, id)
		if v == nil {
			continue
		}
		u.defs[v]++
		if len(rhs) != len(lhs) {
			continue
		}
		if k, ok := u.capacity(rhs[i]); ok {
			u.caps[v] = k
		}
	}
}

// capacity returns the capacity of the channel x makes, when x is a call of
// make for a channel; -1 stands for a capacity that is not constant.
func (u *chanUnit) capacity(x syntax.Expr) (int64, bool) {
	call := callOf(x)
	if call == nil || !callsBuiltin(u.pass, call, "make") {
		return 0, false
	}
	if _, ok := underOrNil(TypeOf(call)).(*types2.Chan); !ok {
		return 0, false
	}
	if len(call.ArgList) < 2 {
		return 0, true
	}
	val := call.ArgList[1].GetTypeInfo().Value
	if val == nil {
		return -1, true
	}
	k, exact := constant.Int64Val(constant.ToInt(val))
	if !exact {
		return -1, true
	}
	return k, true
}

// visit walks the operations of the function.
func (u *chanUnit) visit(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.ForStmt:
		u.loop(n)
		return false
	case *syntax.SelectStmt:
		u.selectStmt(n)
		return false
	case *syntax.GroupChild:
		u.child(n)
		return false
	case *syntax.FuncLit:
		u.nested(n.Body)
		return false
	case *syntax.LambdaExpr:
		u.nested(n.Body)
		return false
	case *syntax.SendStmt:
		u.send(n)
	case *syntax.Operation:
		u.operation(n)
	}
	return true
}

// operation records n when it is a receive.
func (u *chanUnit) operation(n *syntax.Operation) {
	if op := recvOf(n); op != nil {
		u.recv(op)
	}
}

// walk visits the operations under n, when there is a node.
func (u *chanUnit) walk(n syntax.Node) {
	if n != nil {
		syntax.Inspect(n, u.visit)
	}
}

// loop walks a for statement: the initialization once, the rest in a loop.
func (u *chanUnit) loop(f *syntax.ForStmt) {
	u.walk(f.Init)
	u.loops++
	u.walk(f.Cond)
	u.walk(f.Post)
	u.walk(f.Body)
	u.loops--
}

// nested walks a function literal or lambda body, whose operations are not
// covered by the block around it.
func (u *chanUnit) nested(body syntax.Node) {
	outer := u.free
	u.free = false
	u.walk(body)
	u.free = outer
}

// child walks a go group child block, whose operations are covered.
func (u *chanUnit) child(g *syntax.GroupChild) {
	outer := u.free
	u.free = true
	u.walk(g.Body)
	u.free = outer
}

// selectStmt walks a select statement: the operations of its cases are
// covered when the select has a done case, and the bodies are ordinary code.
func (u *chanUnit) selectStmt(s *syntax.SelectStmt) {
	outer := u.free
	covered := outer || selectHasDone(s)
	for _, cc := range s.Body {
		u.free = covered
		operands := u.comm(cc.Comm)
		u.free = outer
		for _, x := range operands {
			u.walk(x)
		}
		for _, st := range cc.Body {
			u.walk(st)
		}
	}
}

// comm records the operation of a select case and returns the operands left
// to walk.
func (u *chanUnit) comm(st syntax.SimpleStmt) []syntax.Node {
	switch st := st.(type) {
	case *syntax.SendStmt:
		u.send(st)
		return []syntax.Node{st.Chan, st.Value}
	case *syntax.ExprStmt:
		if op := recvOf(st.X); op != nil {
			u.recv(op)
			return []syntax.Node{op.X}
		}
		return []syntax.Node{st.X}
	case *syntax.AssignStmt:
		if op := commRecv(st); op != nil {
			u.recv(op)
			return []syntax.Node{st.Lhs, op.X}
		}
		return []syntax.Node{st.Lhs, st.Rhs}
	}
	return nil
}

// send counts a send on a local channel and records it.
func (u *chanUnit) send(s *syntax.SendStmt) {
	ch := u.chanVar(s.Chan)
	if ch != nil {
		u.count(ch)
	}
	u.record(s, u.exprEnd(s.Value), ch, "send on "+syntax.String(s.Chan))
}

// count adds a send to the channel's total, or marks it as sent in a loop.
func (u *chanUnit) count(ch *types2.Var) {
	if u.sent == nil {
		u.sent = make(map[*types2.Var]int)
		u.looped = make(map[*types2.Var]bool)
	}
	if u.loops > 0 {
		u.looped[ch] = true
		return
	}
	u.sent[ch]++
}

// recv records a receive, except the wait on x.Done().
func (u *chanUnit) recv(op *syntax.Operation) {
	if isDoneCall(op.X) {
		return
	}
	u.record(op, u.exprEnd(op.X), u.chanVar(op.X), "receive from "+syntax.String(op.X))
}

// record keeps an operation that is not covered; end is the position just
// after its last token.
func (u *chanUnit) record(n syntax.Node, end syntax.Pos, ch *types2.Var, what string) {
	if !u.free {
		u.ops = append(u.ops, chanOp{node: n, end: end, ch: ch, what: what})
	}
}

// exprEnd returns the position just after the last token of x, an operand of
// a channel operation in the file of u. syntax.EndPos stops before the closing
// bracket or parenthesis of a call, index, slice, parenthesized expression, or
// type assertion, so each of those ends at the matching token in the source. A
// unary or binary operation ends where its last operand ends. Without the
// source of the file it falls back to syntax.EndPos.
func (u *chanUnit) exprEnd(x syntax.Expr) syntax.Pos {
	mode := syntax.Mode(0)
	if u.pass.Kind == KindLgo {
		mode = syntax.LogosMode
	}
	switch x := x.(type) {
	case *syntax.CallExpr:
		return u.pass.CallEnd(u.file, x)
	case *syntax.IndexExpr, *syntax.SliceExpr:
		if end, ok := syntax.BracketEnd(u.pass.Source(u.file), x.Pos(), mode); ok {
			return end
		}
	case *syntax.ParenExpr:
		if end, ok := syntax.ParenEnd(u.pass.Source(u.file), x.Pos(), mode); ok {
			return end
		}
	case *syntax.AssertExpr:
		// The "(" of x.(T) follows the "." at the node's position.
		open := syntax.MakePos(x.Pos().Base(), x.Pos().Line(), x.Pos().Col()+1)
		if end, ok := syntax.ParenEnd(u.pass.Source(u.file), open, mode); ok {
			return end
		}
	case *syntax.Operation:
		if x.Y != nil {
			return u.exprEnd(x.Y)
		}
		return u.exprEnd(x.X)
	}
	return syntax.EndPos(x)
}

// chanVar returns the local variable x names, or nil.
func (u *chanUnit) chanVar(x syntax.Expr) *types2.Var {
	id, ok := bareExpr(x).(*syntax.Name)
	if !ok {
		return nil
	}
	return localVar(u.pass, id)
}

// recvOf returns x as a receive operation, or nil.
func recvOf(x syntax.Expr) *syntax.Operation {
	op, ok := bareExpr(x).(*syntax.Operation)
	if !ok || op.Op != syntax.Recv || op.Y != nil {
		return nil
	}
	return op
}

// commRecv returns the receive of a select case that assigns its value.
func commRecv(a *syntax.AssignStmt) *syntax.Operation {
	if a.Rhs == nil {
		return nil
	}
	return recvOf(a.Rhs)
}

// selectHasDone reports whether a case of s receives from x.Done().
func selectHasDone(s *syntax.SelectStmt) bool {
	for _, cc := range s.Body {
		if op := commOperation(cc.Comm); op != nil && isDoneCall(op.X) {
			return true
		}
	}
	return false
}

// commOperation returns the receive that a select case performs, or nil.
func commOperation(st syntax.SimpleStmt) *syntax.Operation {
	switch st := st.(type) {
	case *syntax.ExprStmt:
		return recvOf(st.X)
	case *syntax.AssignStmt:
		return commRecv(st)
	}
	return nil
}

// isDoneCall reports whether x is a call x.Done() that returns a channel.
func isDoneCall(x syntax.Expr) bool {
	call, ok := bareExpr(x).(*syntax.CallExpr)
	if !ok || len(call.ArgList) != 0 {
		return false
	}
	sel, ok := bareExpr(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Done" {
		return false
	}
	_, isChan := underOrNil(TypeOf(call)).(*types2.Chan)
	return isChan
}
