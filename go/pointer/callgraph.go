// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file defines the internal (context-sensitive) call graph.

import (
	"fmt"
	"go/token"

	"golang.org/x/tools/go/ssa"
)

type cgnode struct {
	fn         *ssa.Function
	obj        nodeid      // start of this contour's object block
	sites      []*callsite // ordered list of callsites within this function
	callersite *callsite   // where called from, if known; nil for shared contours
}

// contour returns a description of this node's contour.
func (n *cgnode) contour() string {
	if n.callersite == nil {
		return "shared contour"
	}
	if n.callersite.instr != nil {
		return fmt.Sprintf("as called from %s", n.callersite.instr.Parent())
	}
	return fmt.Sprintf("as called from intrinsic (targets=n%d)", n.callersite.targets)
}

func (n *cgnode) String() string {
	return fmt.Sprintf("cg%d:%s", n.obj, n.fn)
}

// A callsite represents a single call site within a cgnode;
// it is implicitly context-sensitive.
// callsites never represent calls to built-ins;
// they are handled as intrinsics.
//
type callsite struct {
	targets nodeid              // pts(·) contains objects for dynamically called functions
	instr   ssa.CallInstruction // the call instruction; nil for synthetic/intrinsic
}

func (c *callsite) String() string {
	if c.instr != nil {
		return c.instr.Common().Description()
	}
	return "synthetic function call"
}

// pos returns the source position of this callsite, or token.NoPos if implicit.
func (c *callsite) pos() token.Pos {
	if c.instr != nil {
		return c.instr.Pos()
	}
	return token.NoPos
}
