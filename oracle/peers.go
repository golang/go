// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/oracle/serial"
)

// peers enumerates, for a given channel send (or receive) operation,
// the set of possible receives (or sends) that correspond to it.
//
// TODO(adonovan): support reflect.{Select,Recv,Send,Close}.
// TODO(adonovan): permit the user to query based on a MakeChan (not send/recv),
// or the implicit receive in "for v := range ch".
func peers(q *Query) error {
	lconf := loader.Config{Build: q.Build}

	if err := setPTAScope(&lconf, q.Scope); err != nil {
		return err
	}

	// Load/parse/type-check the program.
	lprog, err := lconf.Load()
	if err != nil {
		return err
	}
	q.Fset = lprog.Fset

	qpos, err := parseQueryPos(lprog, q.Pos, false)
	if err != nil {
		return err
	}

	prog := ssautil.CreateProgram(lprog, ssa.GlobalDebug)

	ptaConfig, err := setupPTA(prog, lprog, q.PTALog, q.Reflection)
	if err != nil {
		return err
	}

	opPos := findOp(qpos)
	if opPos == token.NoPos {
		return fmt.Errorf("there is no channel operation here")
	}

	// Defer SSA construction till after errors are reported.
	prog.Build()

	var queryOp chanOp // the originating send or receive operation
	var ops []chanOp   // all sends/receives of opposite direction

	// Look at all channel operations in the whole ssa.Program.
	// Build a list of those of same type as the query.
	allFuncs := ssautil.AllFunctions(prog)
	for fn := range allFuncs {
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				for _, op := range chanOps(instr) {
					ops = append(ops, op)
					if op.pos == opPos {
						queryOp = op // we found the query op
					}
				}
			}
		}
	}
	if queryOp.ch == nil {
		return fmt.Errorf("ssa.Instruction for send/receive not found")
	}

	// Discard operations of wrong channel element type.
	// Build set of channel ssa.Values as query to pointer analysis.
	// We compare channels by element types, not channel types, to
	// ignore both directionality and type names.
	queryType := queryOp.ch.Type()
	queryElemType := queryType.Underlying().(*types.Chan).Elem()
	ptaConfig.AddQuery(queryOp.ch)
	i := 0
	for _, op := range ops {
		if types.Identical(op.ch.Type().Underlying().(*types.Chan).Elem(), queryElemType) {
			ptaConfig.AddQuery(op.ch)
			ops[i] = op
			i++
		}
	}
	ops = ops[:i]

	// Run the pointer analysis.
	ptares := ptrAnalysis(ptaConfig)

	// Find the points-to set.
	queryChanPtr := ptares.Queries[queryOp.ch]

	// Ascertain which make(chan) labels the query's channel can alias.
	var makes []token.Pos
	for _, label := range queryChanPtr.PointsTo().Labels() {
		makes = append(makes, label.Pos())
	}
	sort.Sort(byPos(makes))

	// Ascertain which channel operations can alias the same make(chan) labels.
	var sends, receives, closes []token.Pos
	for _, op := range ops {
		if ptr, ok := ptares.Queries[op.ch]; ok && ptr.MayAlias(queryChanPtr) {
			switch op.dir {
			case types.SendOnly:
				sends = append(sends, op.pos)
			case types.RecvOnly:
				receives = append(receives, op.pos)
			case types.SendRecv:
				closes = append(closes, op.pos)
			}
		}
	}
	sort.Sort(byPos(sends))
	sort.Sort(byPos(receives))
	sort.Sort(byPos(closes))

	q.result = &peersResult{
		queryPos:  opPos,
		queryType: queryType,
		makes:     makes,
		sends:     sends,
		receives:  receives,
		closes:    closes,
	}
	return nil
}

// findOp returns the position of the enclosing send/receive/close op.
// For send and receive operations, this is the position of the <- token;
// for close operations, it's the Lparen of the function call.
//
// TODO(adonovan): handle implicit receive operations from 'for...range chan' statements.
func findOp(qpos *queryPos) token.Pos {
	for _, n := range qpos.path {
		switch n := n.(type) {
		case *ast.UnaryExpr:
			if n.Op == token.ARROW {
				return n.OpPos
			}
		case *ast.SendStmt:
			return n.Arrow
		case *ast.CallExpr:
			// close function call can only exist as a direct identifier
			if close, ok := unparen(n.Fun).(*ast.Ident); ok {
				if b, ok := qpos.info.Info.Uses[close].(*types.Builtin); ok && b.Name() == "close" {
					return n.Lparen
				}
			}
		}
	}
	return token.NoPos
}

// chanOp abstracts an ssa.Send, ssa.Unop(ARROW), or a SelectState.
type chanOp struct {
	ch  ssa.Value
	dir types.ChanDir // SendOnly=send, RecvOnly=recv, SendRecv=close
	pos token.Pos
}

// chanOps returns a slice of all the channel operations in the instruction.
func chanOps(instr ssa.Instruction) []chanOp {
	// TODO(adonovan): handle calls to reflect.{Select,Recv,Send,Close} too.
	var ops []chanOp
	switch instr := instr.(type) {
	case *ssa.UnOp:
		if instr.Op == token.ARROW {
			ops = append(ops, chanOp{instr.X, types.RecvOnly, instr.Pos()})
		}
	case *ssa.Send:
		ops = append(ops, chanOp{instr.Chan, types.SendOnly, instr.Pos()})
	case *ssa.Select:
		for _, st := range instr.States {
			ops = append(ops, chanOp{st.Chan, st.Dir, st.Pos})
		}
	case ssa.CallInstruction:
		cc := instr.Common()
		if b, ok := cc.Value.(*ssa.Builtin); ok && b.Name() == "close" {
			ops = append(ops, chanOp{cc.Args[0], types.SendRecv, cc.Pos()})
		}
	}
	return ops
}

type peersResult struct {
	queryPos                       token.Pos   // of queried channel op
	queryType                      types.Type  // type of queried channel
	makes, sends, receives, closes []token.Pos // positions of aliased makechan/send/receive/close instrs
}

func (r *peersResult) display(printf printfFunc) {
	if len(r.makes) == 0 {
		printf(r.queryPos, "This channel can't point to anything.")
		return
	}
	printf(r.queryPos, "This channel of type %s may be:", r.queryType)
	for _, alloc := range r.makes {
		printf(alloc, "\tallocated here")
	}
	for _, send := range r.sends {
		printf(send, "\tsent to, here")
	}
	for _, receive := range r.receives {
		printf(receive, "\treceived from, here")
	}
	for _, clos := range r.closes {
		printf(clos, "\tclosed, here")
	}
}

func (r *peersResult) toSerial(res *serial.Result, fset *token.FileSet) {
	peers := &serial.Peers{
		Pos:  fset.Position(r.queryPos).String(),
		Type: r.queryType.String(),
	}
	for _, alloc := range r.makes {
		peers.Allocs = append(peers.Allocs, fset.Position(alloc).String())
	}
	for _, send := range r.sends {
		peers.Sends = append(peers.Sends, fset.Position(send).String())
	}
	for _, receive := range r.receives {
		peers.Receives = append(peers.Receives, fset.Position(receive).String())
	}
	for _, clos := range r.closes {
		peers.Closes = append(peers.Closes, fset.Position(clos).String())
	}
	res.Peers = peers
}

// -------- utils --------

// NB: byPos is not deterministic across packages since it depends on load order.
// Use lessPos if the tests need it.
type byPos []token.Pos

func (p byPos) Len() int           { return len(p) }
func (p byPos) Less(i, j int) bool { return p[i] < p[j] }
func (p byPos) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
