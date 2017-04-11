// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// checkcontrolflow checks fn's control flow structures for correctness.
// It catches:
//   * misplaced breaks and continues
//   * bad labeled break and continues
//   * invalid, unused, duplicate, and missing labels
//   * gotos jumping over declarations and into blocks
func checkcontrolflow(fn *Node) {
	c := controlflow{
		labels:       make(map[string]*cfLabel),
		labeledNodes: make(map[*Node]*cfLabel),
	}
	c.pushPos(fn.Pos)
	c.stmtList(fn.Nbody)

	// Check that we used all labels.
	for name, lab := range c.labels {
		if !lab.used() && !lab.reported && !lab.defNode.Used() {
			yyerrorl(lab.defNode.Pos, "label %v defined and not used", name)
			lab.reported = true
		}
		if lab.used() && !lab.defined() && !lab.reported {
			yyerrorl(lab.useNode.Pos, "label %v not defined", name)
			lab.reported = true
		}
	}

	// Check any forward gotos. Non-forward gotos have already been checked.
	for _, n := range c.fwdGotos {
		lab := c.labels[n.Left.Sym.Name]
		// If the label is undefined, we have already have printed an error.
		if lab.defined() {
			c.checkgoto(n, lab.defNode)
		}
	}
}

type controlflow struct {
	// Labels and labeled control flow nodes (OFOR, OFORUNTIL, OSWITCH, OSELECT) in f.
	labels       map[string]*cfLabel
	labeledNodes map[*Node]*cfLabel

	// Gotos that jump forward; required for deferred checkgoto calls.
	fwdGotos []*Node

	// Breaks are allowed in loops, switches, and selects.
	allowBreak bool
	// Continues are allowed only in loops.
	allowContinue bool

	// Position stack. The current position is top of stack.
	pos []src.XPos
}

// cfLabel is a label tracked by a controlflow.
type cfLabel struct {
	ctlNode *Node // associated labeled control flow node
	defNode *Node // label definition Node (OLABEL)
	// Label use Node (OGOTO, OBREAK, OCONTINUE).
	// There might be multiple uses, but we only need to track one.
	useNode  *Node
	reported bool // reported indicates whether an error has already been reported for this label
}

// defined reports whether the label has a definition (OLABEL node).
func (l *cfLabel) defined() bool { return l.defNode != nil }

// used reports whether the label has a use (OGOTO, OBREAK, or OCONTINUE node).
func (l *cfLabel) used() bool { return l.useNode != nil }

// label returns the label associated with sym, creating it if necessary.
func (c *controlflow) label(sym *types.Sym) *cfLabel {
	lab := c.labels[sym.Name]
	if lab == nil {
		lab = new(cfLabel)
		c.labels[sym.Name] = lab
	}
	return lab
}

// stmtList checks l.
func (c *controlflow) stmtList(l Nodes) {
	for _, n := range l.Slice() {
		c.stmt(n)
	}
}

// stmt checks n.
func (c *controlflow) stmt(n *Node) {
	c.pushPos(n.Pos)
	defer c.popPos()
	c.stmtList(n.Ninit)

	checkedNbody := false

	switch n.Op {
	case OLABEL:
		sym := n.Left.Sym
		lab := c.label(sym)
		// Associate label with its control flow node, if any
		if ctl := n.labeledControl(); ctl != nil {
			c.labeledNodes[ctl] = lab
		}

		if !lab.defined() {
			lab.defNode = n
		} else {
			c.err("label %v already defined at %v", sym, linestr(lab.defNode.Pos))
			lab.reported = true
		}

	case OGOTO:
		lab := c.label(n.Left.Sym)
		if !lab.used() {
			lab.useNode = n
		}
		if lab.defined() {
			c.checkgoto(n, lab.defNode)
		} else {
			c.fwdGotos = append(c.fwdGotos, n)
		}

	case OCONTINUE, OBREAK:
		if n.Left == nil {
			// plain break/continue
			if n.Op == OCONTINUE && !c.allowContinue {
				c.err("%v is not in a loop", n.Op)
			} else if !c.allowBreak {
				c.err("%v is not in a loop, switch, or select", n.Op)
			}
			break
		}

		// labeled break/continue; look up the target
		sym := n.Left.Sym
		lab := c.label(sym)
		if !lab.used() {
			lab.useNode = n.Left
		}
		if !lab.defined() {
			c.err("%v label not defined: %v", n.Op, sym)
			lab.reported = true
			break
		}
		ctl := lab.ctlNode
		if n.Op == OCONTINUE && ctl != nil && (ctl.Op == OSWITCH || ctl.Op == OSELECT) {
			// Cannot continue in a switch or select.
			ctl = nil
		}
		if ctl == nil {
			// Valid label but not usable with a break/continue here, e.g.:
			// for {
			// 	continue abc
			// }
			// abc:
			// for {}
			c.err("invalid %v label %v", n.Op, sym)
			lab.reported = true
		}

	case OFOR, OFORUNTIL, OSWITCH, OSELECT:
		// set up for continue/break in body
		allowBreak := c.allowBreak
		allowContinue := c.allowContinue
		c.allowBreak = true
		switch n.Op {
		case OFOR, OFORUNTIL:
			c.allowContinue = true
		}
		lab := c.labeledNodes[n]
		if lab != nil {
			// labeled for loop
			lab.ctlNode = n
		}

		// check body
		c.stmtList(n.Nbody)
		checkedNbody = true

		// tear down continue/break
		c.allowBreak = allowBreak
		c.allowContinue = allowContinue
		if lab != nil {
			lab.ctlNode = nil
		}
	}

	if !checkedNbody {
		c.stmtList(n.Nbody)
	}
	c.stmtList(n.List)
	c.stmtList(n.Rlist)
}

// pushPos pushes a position onto the position stack.
func (c *controlflow) pushPos(pos src.XPos) {
	if !pos.IsKnown() {
		pos = c.peekPos()
		if Debug['K'] != 0 {
			Warn("controlflow: unknown position")
		}
	}
	c.pos = append(c.pos, pos)
}

// popLine pops the top of the position stack.
func (c *controlflow) popPos() { c.pos = c.pos[:len(c.pos)-1] }

// peekPos peeks at the top of the position stack.
func (c *controlflow) peekPos() src.XPos { return c.pos[len(c.pos)-1] }

// err reports a control flow error at the current position.
func (c *controlflow) err(msg string, args ...interface{}) {
	yyerrorl(c.peekPos(), msg, args...)
}

// checkgoto checks that a goto from from to to does not
// jump into a block or jump over variable declarations.
func (c *controlflow) checkgoto(from *Node, to *Node) {
	if from.Op != OGOTO || to.Op != OLABEL {
		Fatalf("bad from/to in checkgoto: %v -> %v", from, to)
	}

	// from and to's Sym fields record dclstack's value at their
	// position, which implicitly encodes their block nesting
	// level and variable declaration position within that block.
	//
	// For valid gotos, to.Sym will be a tail of from.Sym.
	// Otherwise, any link in to.Sym not also in from.Sym
	// indicates a block/declaration being jumped into/over.
	//
	// TODO(mdempsky): We should only complain about jumping over
	// variable declarations, but currently we reject type and
	// constant declarations too (#8042).

	if from.Sym == to.Sym {
		return
	}

	nf := dcldepth(from.Sym)
	nt := dcldepth(to.Sym)

	// Unwind from.Sym so it's no longer than to.Sym. It's okay to
	// jump out of blocks or backwards past variable declarations.
	fs := from.Sym
	for ; nf > nt; nf-- {
		fs = fs.Link
	}

	if fs == to.Sym {
		return
	}

	// Decide what to complain about. Unwind to.Sym until where it
	// forked from from.Sym, and keep track of the innermost block
	// and declaration we jumped into/over.
	var block *types.Sym
	var dcl *types.Sym

	// If to.Sym is longer, unwind until it's the same length.
	ts := to.Sym
	for ; nt > nf; nt-- {
		if ts.Pkg == nil {
			block = ts
		} else {
			dcl = ts
		}
		ts = ts.Link
	}

	// Same length; unwind until we find their common ancestor.
	for ts != fs {
		if ts.Pkg == nil {
			block = ts
		} else {
			dcl = ts
		}
		ts = ts.Link
		fs = fs.Link
	}

	// Prefer to complain about 'into block' over declarations.
	pos := from.Left.Pos
	if block != nil {
		yyerrorl(pos, "goto %v jumps into block starting at %v", from.Left.Sym, linestr(block.Lastlineno))
	} else {
		yyerrorl(pos, "goto %v jumps over declaration of %v at %v", from.Left.Sym, dcl, linestr(dcl.Lastlineno))
	}
}

// dcldepth returns the declaration depth for a dclstack Sym; that is,
// the sum of the block nesting level and the number of declarations
// in scope.
func dcldepth(s *types.Sym) int {
	n := 0
	for ; s != nil; s = s.Link {
		n++
	}
	return n
}
