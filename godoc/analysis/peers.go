// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package analysis

// This file computes the channel "peers" relation over all pairs of
// channel operations in the program.  The peers are displayed in the
// lower pane when a channel operation (make, <-, close) is clicked.

// TODO(adonovan): handle calls to reflect.{Select,Recv,Send,Close} too,
// then enable reflection in PTA.

import (
	"fmt"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/pointer"
	"golang.org/x/tools/go/ssa"
)

func (a *analysis) doChannelPeers(ptsets map[ssa.Value]pointer.Pointer) {
	addSendRecv := func(j *commJSON, op chanOp) {
		j.Ops = append(j.Ops, commOpJSON{
			Op: anchorJSON{
				Text: op.mode,
				Href: a.posURL(op.pos, op.len),
			},
			Fn: prettyFunc(nil, op.fn),
		})
	}

	// Build an undirected bipartite multigraph (binary relation)
	// of MakeChan ops and send/recv/close ops.
	//
	// TODO(adonovan): opt: use channel element types to partition
	// the O(n^2) problem into subproblems.
	aliasedOps := make(map[*ssa.MakeChan][]chanOp)
	opToMakes := make(map[chanOp][]*ssa.MakeChan)
	for _, op := range a.ops {
		// Combine the PT sets from all contexts.
		var makes []*ssa.MakeChan // aliased ops
		ptr, ok := ptsets[op.ch]
		if !ok {
			continue // e.g. channel op in dead code
		}
		for _, label := range ptr.PointsTo().Labels() {
			makechan, ok := label.Value().(*ssa.MakeChan)
			if !ok {
				continue // skip intrinsically-created channels for now
			}
			if makechan.Pos() == token.NoPos {
				continue // not possible?
			}
			makes = append(makes, makechan)
			aliasedOps[makechan] = append(aliasedOps[makechan], op)
		}
		opToMakes[op] = makes
	}

	// Now that complete relation is built, build links for ops.
	for _, op := range a.ops {
		v := commJSON{
			Ops: []commOpJSON{}, // (JS wants non-nil)
		}
		ops := make(map[chanOp]bool)
		for _, makechan := range opToMakes[op] {
			v.Ops = append(v.Ops, commOpJSON{
				Op: anchorJSON{
					Text: "made",
					Href: a.posURL(makechan.Pos()-token.Pos(len("make")),
						len("make")),
				},
				Fn: makechan.Parent().RelString(op.fn.Package().Pkg),
			})
			for _, op := range aliasedOps[makechan] {
				ops[op] = true
			}
		}
		for op := range ops {
			addSendRecv(&v, op)
		}

		// Add links for each aliased op.
		fi, offset := a.fileAndOffset(op.pos)
		fi.addLink(aLink{
			start:   offset,
			end:     offset + op.len,
			title:   "show channel ops",
			onclick: fmt.Sprintf("onClickComm(%d)", fi.addData(v)),
		})
	}
	// Add links for makechan ops themselves.
	for makechan, ops := range aliasedOps {
		v := commJSON{
			Ops: []commOpJSON{}, // (JS wants non-nil)
		}
		for _, op := range ops {
			addSendRecv(&v, op)
		}

		fi, offset := a.fileAndOffset(makechan.Pos())
		fi.addLink(aLink{
			start:   offset - len("make"),
			end:     offset,
			title:   "show channel ops",
			onclick: fmt.Sprintf("onClickComm(%d)", fi.addData(v)),
		})
	}
}

// -- utilities --------------------------------------------------------

// chanOp abstracts an ssa.Send, ssa.Unop(ARROW), close(), or a SelectState.
// Derived from oracle/peers.go.
type chanOp struct {
	ch   ssa.Value
	mode string // sent|received|closed
	pos  token.Pos
	len  int
	fn   *ssa.Function
}

// chanOps returns a slice of all the channel operations in the instruction.
// Derived from oracle/peers.go.
func chanOps(instr ssa.Instruction) []chanOp {
	fn := instr.Parent()
	var ops []chanOp
	switch instr := instr.(type) {
	case *ssa.UnOp:
		if instr.Op == token.ARROW {
			// TODO(adonovan): don't assume <-ch; could be 'range ch'.
			ops = append(ops, chanOp{instr.X, "received", instr.Pos(), len("<-"), fn})
		}
	case *ssa.Send:
		ops = append(ops, chanOp{instr.Chan, "sent", instr.Pos(), len("<-"), fn})
	case *ssa.Select:
		for _, st := range instr.States {
			mode := "received"
			if st.Dir == types.SendOnly {
				mode = "sent"
			}
			ops = append(ops, chanOp{st.Chan, mode, st.Pos, len("<-"), fn})
		}
	case ssa.CallInstruction:
		call := instr.Common()
		if blt, ok := call.Value.(*ssa.Builtin); ok && blt.Name() == "close" {
			pos := instr.Common().Pos()
			ops = append(ops, chanOp{call.Args[0], "closed", pos - token.Pos(len("close")), len("close("), fn})
		}
	}
	return ops
}
