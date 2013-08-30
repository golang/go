// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
)

// peers enumerates, for a given channel send (or receive) operation,
// the set of possible receives (or sends) that correspond to it.
//
// TODO(adonovan): support reflect.{Select,Recv,Send}.
// TODO(adonovan): permit the user to query based on a MakeChan (not send/recv),
// or the implicit receive in "for v := range ch".
//
func peers(o *oracle) (queryResult, error) {
	arrowPos := findArrow(o)
	if arrowPos == token.NoPos {
		return nil, o.errorf(o.queryPath[0], "there is no send/receive here")
	}

	buildSSA(o)

	var queryOp chanOp // the originating send or receive operation
	var ops []chanOp   // all sends/receives of opposite direction

	// Look at all send/receive instructions in the whole ssa.Program.
	// Build a list of those of same type to query.
	allFuncs := ssa.AllFunctions(o.prog)
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
		return nil, o.errorf(arrowPos, "ssa.Instruction for send/receive not found")
	}

	// Discard operations of wrong channel element type.
	// Build set of channel ssa.Values as query to pointer analysis.
	queryElemType := queryOp.ch.Type().Underlying().(*types.Chan).Elem()
	channels := map[ssa.Value][]pointer.Pointer{queryOp.ch: nil}
	i := 0
	for _, op := range ops {
		if types.IsIdentical(op.ch.Type().Underlying().(*types.Chan).Elem(), queryElemType) {
			channels[op.ch] = nil
			ops[i] = op
			i++
		}
	}
	ops = ops[:i]

	// Run the pointer analysis.
	o.config.QueryValues = channels
	ptrAnalysis(o)

	// Combine the PT sets from all contexts.
	queryChanPts := pointer.PointsToCombined(channels[queryOp.ch])

	return &peersResult{
		queryOp:      queryOp,
		ops:          ops,
		queryChanPts: queryChanPts,
	}, nil
}

// findArrow returns the position of the enclosing send/receive op
// (<-) for the query position, or token.NoPos if not found.
//
func findArrow(o *oracle) token.Pos {
	for _, n := range o.queryPath {
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
	queryOp      chanOp
	ops          []chanOp
	queryChanPts pointer.PointsToSet
}

func (r *peersResult) display(o *oracle) {
	// Report which make(chan) labels the query's channel can alias.
	labels := r.queryChanPts.Labels()
	if len(labels) == 0 {
		o.printf(r.queryOp.pos, "This channel can't point to anything.")
		return
	}
	o.printf(r.queryOp.pos, "This channel of type %s may be:", r.queryOp.ch.Type())
	// TODO(adonovan): sort, to ensure test determinism.
	for _, label := range labels {
		o.printf(label, "\tallocated here")
	}

	// Report which send/receive operations can alias the same make(chan) labels.
	for _, op := range r.ops {
		// TODO(adonovan): sort, to ensure test determinism.
		for _, ptr := range o.config.QueryValues[op.ch] {
			if ptr != nil && ptr.PointsTo().Intersects(r.queryChanPts) {
				verb := "received from"
				if op.dir == ast.SEND {
					verb = "sent to"
				}
				o.printf(op.pos, "\t%s, here", verb)
			}
		}
	}
}
