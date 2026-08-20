// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"fmt"
	"slices"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// This file contains the algorithm to place phi nodes in a function.
// For small functions, we use Braun, Buchwald, Hack, Leißa, Mallon, and Zwinkau.
// https://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
// For large functions, we use Sreedhar & Gao: A Linear Time Algorithm for Placing Φ-Nodes.
// http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.1979&rep=rep1&type=pdf

const smallBlocks = 500

const debugPhi = false

// fwdRefAux wraps an arbitrary ir.Node as an ssa.Aux for use with OpFwdref.
type fwdRefAux struct {
	_ [0]func() // ensure ir.Node isn't compared for equality
	N ir.Node
}

func (fwdRefAux) CanBeAnSSAAux() {}

// insertPhis finds all the places in the function where a phi is
// necessary and inserts them.
// Uses FwdRef ops to find all uses of variables, and s.defvars to find
// all definitions.
// Phi values are inserted, and all FwdRefs are changed to a Copy
// of the appropriate phi or definition.
// TODO: make this part of cmd/compile/internal/ssa somehow?
func (s *state) insertPhis() {
	if len(s.f.Blocks) <= smallBlocks {
		sps := simplePhiState{s: s, f: s.f, defvars: s.defvars}
		sps.insertPhis()
		return
	}
	ps := phiState{s: s, f: s.f, defvars: s.defvars}
	ps.insertPhis()
}

type phiState struct {
	s       *state                   // SSA state
	f       *ssa.Func                // function to work on
	defvars []map[ir.Node]*ssa.Value // defined variables at end of each block

	varnum map[ir.Node]int32 // variable numbering

	// miscellaneous
	placeholder *ssa.Value // value to use as a "not set yet" placeholder.
}

func (s *phiState) insertPhis() {
	if debugPhi {
		fmt.Println(s.f.String())
	}

	// Find all the variables for which we need to match up reads & writes.
	// This step prunes any basic-block-only variables from consideration.
	// Generate a numbering for these variables.
	s.varnum = map[ir.Node]int32{}
	var vars []ir.Node
	var vartypes []*types.Type
	for _, b := range s.f.Blocks {
		for _, v := range b.Values {
			if v.Op != ssaop.OpFwdRef {
				continue
			}
			var_ := v.Aux.(fwdRefAux).N

			// Optimization: look back 1 block for the definition.
			if len(b.Preds) == 1 {
				c := b.Preds[0].Block()
				if w := s.defvars[c.ID][var_]; w != nil {
					v.Op = ssaop.OpCopy
					v.Aux = nil
					v.AddArg(w)
					continue
				}
			}

			if _, ok := s.varnum[var_]; ok {
				continue
			}
			s.varnum[var_] = int32(len(vartypes))
			if debugPhi {
				fmt.Printf("var%d = %v\n", len(vartypes), var_)
			}
			vars = append(vars, var_)
			vartypes = append(vartypes, v.Type)
		}
	}

	if len(vartypes) == 0 {
		return
	}

	// Find all definitions of the variables we need to process.
	// defs[n] contains all the blocks in which variable number n is assigned.
	defs := make([][]*ssa.Block, len(vartypes))
	for _, b := range s.f.Blocks {
		for var_ := range s.defvars[b.ID] { // TODO: encode defvars some other way (explicit ops)? make defvars[n] a slice instead of a map.
			if n, ok := s.varnum[var_]; ok {
				defs[n] = append(defs[n], b)
			}
		}
	}

	// Allocate scratch locations.
	s.placeholder = s.s.entryNewValue0(ssaop.OpUnknown, types.TypeInvalid)

	// Generate phi ops for each variable.
	for n := range vartypes {
		s.insertVarPhis(n, vars[n], defs[n], vartypes[n])
	}

	// Resolve FwdRefs to the correct write or phi.
	s.resolveFwdRefs()

	// Erase variable numbers stored in AuxInt fields of phi ops. They are no longer needed.
	for _, b := range s.f.Blocks {
		for _, v := range b.Values {
			if v.Op == ssaop.OpPhi {
				v.AuxInt = 0
			}
			// Any remaining FwdRefs are dead code.
			if v.Op == ssaop.OpFwdRef {
				v.Op = ssaop.OpUnknown
				v.Aux = nil
			}
		}
	}
}

func (s *phiState) insertVarPhis(n int, var_ ir.Node, defs []*ssa.Block, typ *types.Type) {
	// Iterate DF+ of the defining blocks.
	for c := range s.f.IterDomFrontierPlus(slices.Values(defs)) {
		// Add a phi to block c for variable n.
		v := c.NewValue0I(s.s.blockStarts[c.ID], ssaop.OpPhi, typ, int64(n))
		// Note: we store the variable number in the phi's AuxInt field. Used temporarily by phi building.
		if var_.Op() == ir.ONAME {
			s.s.addNamedValue(var_.(*ir.Name), v)
		}
		for range c.Preds {
			v.AddArg(s.placeholder) // Actual args will be filled in by resolveFwdRefs.
		}
		if debugPhi {
			fmt.Printf("new phi for var%d in %s: %s\n", n, c, v)
		}
	}
}

// resolveFwdRefs links all FwdRef uses up to their nearest dominating definition.
func (s *phiState) resolveFwdRefs() {
	// Do a depth-first walk of the dominator tree, keeping track
	// of the most-recently-seen value for each variable.
	sdom := s.f.Sdom()

	// Map from variable ID to SSA value at the current point of the walk.
	values := make([]*ssa.Value, len(s.varnum))
	for i := range values {
		values[i] = s.placeholder
	}

	// Stack of work to do.
	type stackEntry struct {
		b *ssa.Block // block to explore

		// variable/value pair to reinstate on exit
		n int32 // variable ID
		v *ssa.Value

		// Note: only one of b or n,v will be set.
	}
	var stk []stackEntry

	stk = append(stk, stackEntry{b: s.f.Entry})
	for len(stk) > 0 {
		work := stk[len(stk)-1]
		stk = stk[:len(stk)-1]

		b := work.b
		if b == nil {
			// On exit from a block, this case will undo any assignments done below.
			values[work.n] = work.v
			continue
		}

		// Process phis as new defs. They come before FwdRefs in this block.
		for _, v := range b.Values {
			if v.Op != ssaop.OpPhi {
				continue
			}
			n := int32(v.AuxInt)
			// Remember the old assignment so we can undo it when we exit b.
			stk = append(stk, stackEntry{n: n, v: values[n]})
			// Record the new assignment.
			values[n] = v
		}

		// Replace a FwdRef op with the current incoming value for its variable.
		for _, v := range b.Values {
			if v.Op != ssaop.OpFwdRef {
				continue
			}
			n := s.varnum[v.Aux.(fwdRefAux).N]
			v.Op = ssaop.OpCopy
			v.Aux = nil
			v.AddArg(values[n])
		}

		// Establish values for variables defined in b.
		for var_, v := range s.defvars[b.ID] {
			n, ok := s.varnum[var_]
			if !ok {
				// some variable not live across a basic block boundary.
				continue
			}
			// Remember the old assignment so we can undo it when we exit b.
			stk = append(stk, stackEntry{n: n, v: values[n]})
			// Record the new assignment.
			values[n] = v
		}

		// Replace phi args in successors with the current incoming value.
		for _, e := range b.Succs {
			c, i := e.Block(), e.Index()
			for j := len(c.Values) - 1; j >= 0; j-- {
				v := c.Values[j]
				if v.Op != ssaop.OpPhi {
					break // All phis will be at the end of the block during phi building.
				}
				// Only set arguments that have been resolved.
				// For very wide CFGs, this significantly speeds up phi resolution.
				// See golang.org/issue/8225.
				if w := values[v.AuxInt]; w.Op != ssaop.OpUnknown {
					v.SetArg(i, w)
				}
			}
		}

		// Walk children in dominator tree.
		for c := sdom.Child(b); c != nil; c = sdom.Sibling(c) {
			stk = append(stk, stackEntry{b: c})
		}
	}
}

// TODO: stop walking the iterated domininance frontier when
// the variable is dead. Maybe detect that by checking if the
// node we're on is reverse dominated by all the reads?
// Reverse dominated by the highest common successor of all the reads?

// Variant to use for small functions.
type simplePhiState struct {
	s         *state                   // SSA state
	f         *ssa.Func                // function to work on
	fwdrefs   []*ssa.Value             // list of FwdRefs to be processed
	defvars   []map[ir.Node]*ssa.Value // defined variables at end of each block
	reachable []bool                   // which blocks are reachable
}

func (s *simplePhiState) insertPhis() {
	s.reachable = ssa.ReachableBlocks(s.f)

	// Find FwdRef ops.
	for _, b := range s.f.Blocks {
		for _, v := range b.Values {
			if v.Op != ssaop.OpFwdRef {
				continue
			}
			s.fwdrefs = append(s.fwdrefs, v)
			var_ := v.Aux.(fwdRefAux).N
			if _, ok := s.defvars[b.ID][var_]; !ok {
				s.defvars[b.ID][var_] = v // treat FwdDefs as definitions.
			}
		}
	}

	var args []*ssa.Value

loop:
	for len(s.fwdrefs) > 0 {
		v := s.fwdrefs[len(s.fwdrefs)-1]
		s.fwdrefs = s.fwdrefs[:len(s.fwdrefs)-1]
		b := v.Block
		var_ := v.Aux.(fwdRefAux).N
		if b == s.f.Entry {
			// No variable should be live at entry.
			s.s.Fatalf("value %v (%v) incorrectly live at entry", var_, v)
		}
		if !s.reachable[b.ID] {
			// This block is dead.
			// It doesn't matter what we use here as long as it is well-formed.
			v.Op = ssaop.OpUnknown
			v.Aux = nil
			continue
		}
		// Find variable value on each predecessor.
		args = args[:0]
		for _, e := range b.Preds {
			args = append(args, s.lookupVarOutgoing(e.Block(), v.Type, var_, v.Pos))
		}

		// Decide if we need a phi or not. We need a phi if there
		// are two different args (which are both not v).
		var w *ssa.Value
		for _, a := range args {
			if a == v {
				continue // self-reference
			}
			if a == w {
				continue // already have this witness
			}
			if w != nil {
				// two witnesses, need a phi value
				v.Op = ssaop.OpPhi
				v.AddArgs(args...)
				v.Aux = nil
				v.Pos = s.s.blockStarts[b.ID]
				continue loop
			}
			w = a // save witness
		}
		if w == nil {
			s.s.Fatalf("no witness for reachable phi %s", v)
		}
		// One witness. Make v a copy of w.
		v.Op = ssaop.OpCopy
		v.Aux = nil
		v.AddArg(w)
	}
}

// lookupVarOutgoing finds the variable's value at the end of block b.
func (s *simplePhiState) lookupVarOutgoing(b *ssa.Block, t *types.Type, var_ ir.Node, line src.XPos) *ssa.Value {
	for {
		if v := s.defvars[b.ID][var_]; v != nil {
			return v
		}
		// The variable is not defined by b and we haven't looked it up yet.
		// If b has exactly one predecessor, loop to look it up there.
		// Otherwise, give up and insert a new FwdRef and resolve it later.
		if len(b.Preds) != 1 {
			break
		}
		b = b.Preds[0].Block()
		if !s.reachable[b.ID] {
			// This is rare; it happens with oddly interleaved infinite loops in dead code.
			// See issue 19783.
			break
		}
	}
	// Generate a FwdRef for the variable and return that.
	v := b.NewValue0A(line, ssaop.OpFwdRef, t, fwdRefAux{N: var_})
	s.defvars[b.ID][var_] = v
	if var_.Op() == ir.ONAME {
		s.s.addNamedValue(var_.(*ir.Name), v)
	}
	s.fwdrefs = append(s.fwdrefs, v)
	return v
}
