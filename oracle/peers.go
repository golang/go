// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/oracle/serial"
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
	"code.google.com/p/go.tools/ssa/ssautil"
)

// peers enumerates, for a given channel send (or receive) operation,
// the set of possible receives (or sends) that correspond to it.
//
// TODO(adonovan): support reflect.{Select,Recv,Send}.
// TODO(adonovan): permit the user to query based on a MakeChan (not send/recv),
// or the implicit receive in "for v := range ch".
//
func peers(o *Oracle, qpos *QueryPos) (queryResult, error) {
	arrowPos := findArrow(qpos)
	if arrowPos == token.NoPos {
		return nil, fmt.Errorf("there is no send/receive here")
	}

	buildSSA(o)

	var queryOp chanOp // the originating send or receive operation
	var ops []chanOp   // all sends/receives of opposite direction

	// Look at all send/receive instructions in the whole ssa.Program.
	// Build a list of those of same type to query.
	allFuncs := ssautil.AllFunctions(o.prog)
	for fn := range allFuncs {
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				for _, op := range chanOps(instr) {
					ops = append(ops, op)
					if op.pos == arrowPos {
						queryOp = op // we found the query op
					}
				}
			}
		}
	}
	if queryOp.ch == nil {
		return nil, fmt.Errorf("ssa.Instruction for send/receive not found")
	}

	// Discard operations of wrong channel element type.
	// Build set of channel ssa.Values as query to pointer analysis.
	// We compare channels by element types, not channel types, to
	// ignore both directionality and type names.
	queryType := queryOp.ch.Type()
	queryElemType := queryType.Underlying().(*types.Chan).Elem()
	o.ptaConfig.AddQuery(queryOp.ch)
	i := 0
	for _, op := range ops {
		if types.IsIdentical(op.ch.Type().Underlying().(*types.Chan).Elem(), queryElemType) {
			o.ptaConfig.AddQuery(op.ch)
			ops[i] = op
			i++
		}
	}
	ops = ops[:i]

	// Run the pointer analysis.
	ptares := ptrAnalysis(o)

	// Combine the PT sets from all contexts.
	queryChanPts := pointer.PointsToCombined(ptares.Queries[queryOp.ch])

	// Ascertain which make(chan) labels the query's channel can alias.
	var makes []token.Pos
	for _, label := range queryChanPts.Labels() {
		makes = append(makes, label.Pos())
	}
	sort.Sort(byPos(makes))

	// Ascertain which send/receive operations can alias the same make(chan) labels.
	var sends, receives []token.Pos
	for _, op := range ops {
		for _, ptr := range ptares.Queries[op.ch] {
			if ptr.PointsTo().Intersects(queryChanPts) {
				if op.dir == ast.SEND {
					sends = append(sends, op.pos)
				} else {
					receives = append(receives, op.pos)
				}
			}
		}
	}
	sort.Sort(byPos(sends))
	sort.Sort(byPos(receives))

	return &peersResult{
		queryPos:  arrowPos,
		queryType: queryType,
		makes:     makes,
		sends:     sends,
		receives:  receives,
	}, nil
}

// findArrow returns the position of the enclosing send/receive op
// (<-) for the query position, or token.NoPos if not found.
//
func findArrow(qpos *QueryPos) token.Pos {
	for _, n := range qpos.path {
		switch n := n.(type) {
		case *ast.UnaryExpr:
			if n.Op == token.ARROW {
				return n.OpPos
			}
		case *ast.SendStmt:
			return n.Arrow
		}
	}
	return token.NoPos
}

// chanOp abstracts an ssa.Send, ssa.Unop(ARROW), or a SelectState.
type chanOp struct {
	ch  ssa.Value
	dir ast.ChanDir
	pos token.Pos
}

// chanOps returns a slice of all the channel operations in the instruction.
func chanOps(instr ssa.Instruction) []chanOp {
	// TODO(adonovan): handle calls to reflect.{Select,Recv,Send} too.
	var ops []chanOp
	switch instr := instr.(type) {
	case *ssa.UnOp:
		if instr.Op == token.ARROW {
			ops = append(ops, chanOp{instr.X, ast.RECV, instr.Pos()})
		}
	case *ssa.Send:
		ops = append(ops, chanOp{instr.Chan, ast.SEND, instr.Pos()})
	case *ssa.Select:
		for _, st := range instr.States {
			ops = append(ops, chanOp{st.Chan, st.Dir, st.Pos})
		}
	}
	return ops
}

type peersResult struct {
	queryPos               token.Pos   // of queried '<-' token
	queryType              types.Type  // type of queried channel
	makes, sends, receives []token.Pos // positions of alisaed makechan/send/receive instrs
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
	res.Peers = peers
}

// -------- utils --------

type byPos []token.Pos

func (p byPos) Len() int           { return len(p) }
func (p byPos) Less(i, j int) bool { return p[i] < p[j] }
func (p byPos) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
