// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"log"

	"cmd/internal/ssa"
)

func buildssa(fn *Node) {
	dumplist("buildssa", Curfn.Nbody)

	var s ssaState

	// TODO(khr): build config just once at the start of the compiler binary
	s.config = ssa.NewConfig(Thearch.Thestring)
	s.f = s.config.NewFunc()
	s.f.Name = fn.Nname.Sym.Name

	// We construct SSA using an algorithm similar to
	// Brau, Buchwald, Hack, Leißa, Mallon, and Zwinkau
	// http://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
	// TODO: check this comment

	// Allocate starting block
	s.f.Entry = s.f.NewBlock(ssa.BlockPlain)

	// Allocate exit block
	s.exit = s.f.NewBlock(ssa.BlockExit)

	// TODO(khr): all args.  Make a struct containing args/returnvals, declare
	// an FP which contains a pointer to that struct.

	s.vars = map[string]*ssa.Value{}
	s.labels = map[string]*ssa.Block{}
	s.argOffsets = map[string]int64{}

	// Convert the AST-based IR to the SSA-based IR
	s.startBlock(s.f.Entry)
	s.stmtList(fn.Nbody)

	// Finish up exit block
	s.startBlock(s.exit)
	s.exit.Control = s.mem()
	s.endBlock()

	// Link up variable uses to variable definitions
	s.linkForwardReferences()

	ssa.Compile(s.f)

	// TODO(khr): Use the resulting s.f to generate code
}

type ssaState struct {
	// configuration (arch) information
	config *ssa.Config

	// function we're building
	f *ssa.Func

	// exit block that "return" jumps to (and panics jump to)
	exit *ssa.Block

	// the target block for each label in f
	labels map[string]*ssa.Block

	// current location where we're interpreting the AST
	curBlock *ssa.Block

	// variable assignments in the current block (map from variable name to ssa value)
	vars map[string]*ssa.Value

	// all defined variables at the end of each block.  Indexed by block ID.
	defvars []map[string]*ssa.Value

	// offsets of argument slots
	// unnamed and unused args are not listed.
	argOffsets map[string]int64
}

// startBlock sets the current block we're generating code in to b.
func (s *ssaState) startBlock(b *ssa.Block) {
	s.curBlock = b
	s.vars = map[string]*ssa.Value{}
}

// endBlock marks the end of generating code for the current block.
// Returns the (former) current block.  Returns nil if there is no current
// block, i.e. if no code flows to the current execution point.
func (s *ssaState) endBlock() *ssa.Block {
	b := s.curBlock
	if b == nil {
		return nil
	}
	for len(s.defvars) <= int(b.ID) {
		s.defvars = append(s.defvars, nil)
	}
	s.defvars[b.ID] = s.vars
	s.curBlock = nil
	s.vars = nil
	return b
}

// ssaStmtList converts the statement n to SSA and adds it to s.
func (s *ssaState) stmtList(l *NodeList) {
	for ; l != nil; l = l.Next {
		s.stmt(l.N)
	}
}

// ssaStmt converts the statement n to SSA and adds it to s.
func (s *ssaState) stmt(n *Node) {
	s.stmtList(n.Ninit)
	switch n.Op {

	case OBLOCK:
		s.stmtList(n.List)

	case ODCL:
		// TODO: ???  Assign 0?

	case OLABEL, OGOTO:
		// get block at label, or make one
		t := s.labels[n.Left.Sym.Name]
		if t == nil {
			t = s.f.NewBlock(ssa.BlockPlain)
			s.labels[n.Left.Sym.Name] = t
		}
		// go to that label (we pretend "label:" is preceded by "goto label")
		b := s.endBlock()
		addEdge(b, t)

		if n.Op == OLABEL {
			// next we work on the label's target block
			s.startBlock(t)
		}

	case OAS:
		// TODO(khr): colas?
		val := s.expr(n.Right)
		if n.Left.Op == OINDREG {
			// indirect off a register (TODO: always SP?)
			// used for storing arguments to callees
			addr := s.f.Entry.NewValue(ssa.OpSPAddr, Ptrto(n.Right.Type), n.Left.Xoffset)
			s.vars[".mem"] = s.curBlock.NewValue3(ssa.OpStore, ssa.TypeMem, nil, addr, val, s.mem())
		} else if n.Left.Op != ONAME {
			// some more complicated expression.  Rewrite to a store.  TODO
			addr := s.expr(n.Left) // TODO: wrap in &

			// TODO(khr): nil check
			s.vars[".mem"] = s.curBlock.NewValue3(ssa.OpStore, n.Right.Type, nil, addr, val, s.mem())
		} else if !n.Left.Addable {
			// TODO
			log.Fatalf("assignment to non-addable value")
		} else if n.Left.Class&PHEAP != 0 {
			// TODO
			log.Fatalf("assignment to heap value")
		} else if n.Left.Class == PEXTERN {
			// assign to global variable
			addr := s.f.Entry.NewValue(ssa.OpGlobal, Ptrto(n.Left.Type), n.Left.Sym)
			s.vars[".mem"] = s.curBlock.NewValue3(ssa.OpStore, ssa.TypeMem, nil, addr, val, s.mem())
		} else if n.Left.Class == PPARAMOUT {
			// store to parameter slot
			addr := s.f.Entry.NewValue(ssa.OpFPAddr, Ptrto(n.Right.Type), n.Left.Xoffset)
			s.vars[".mem"] = s.curBlock.NewValue3(ssa.OpStore, ssa.TypeMem, nil, addr, val, s.mem())
		} else {
			// normal variable
			s.vars[n.Left.Sym.Name] = val
		}
	case OIF:
		cond := s.expr(n.Ntest)
		b := s.endBlock()
		b.Kind = ssa.BlockIf
		b.Control = cond
		// TODO(khr): likely direction

		bThen := s.f.NewBlock(ssa.BlockPlain)
		bEnd := s.f.NewBlock(ssa.BlockPlain)
		var bElse *ssa.Block

		if n.Nelse == nil {
			addEdge(b, bThen)
			addEdge(b, bEnd)
		} else {
			bElse = s.f.NewBlock(ssa.BlockPlain)
			addEdge(b, bThen)
			addEdge(b, bElse)
		}

		s.startBlock(bThen)
		s.stmtList(n.Nbody)
		b = s.endBlock()
		if b != nil {
			addEdge(b, bEnd)
		}

		if n.Nelse != nil {
			s.startBlock(bElse)
			s.stmtList(n.Nelse)
			b = s.endBlock()
			if b != nil {
				addEdge(b, bEnd)
			}
		}
		s.startBlock(bEnd)

	case ORETURN:
		s.stmtList(n.List)
		b := s.endBlock()
		addEdge(b, s.exit)

	case OFOR:
		bCond := s.f.NewBlock(ssa.BlockPlain)
		bBody := s.f.NewBlock(ssa.BlockPlain)
		bEnd := s.f.NewBlock(ssa.BlockPlain)

		// first, jump to condition test
		b := s.endBlock()
		addEdge(b, bCond)

		// generate code to test condition
		// TODO(khr): Ntest == nil exception
		s.startBlock(bCond)
		cond := s.expr(n.Ntest)
		b = s.endBlock()
		b.Kind = ssa.BlockIf
		b.Control = cond
		// TODO(khr): likely direction
		addEdge(b, bBody)
		addEdge(b, bEnd)

		// generate body
		s.startBlock(bBody)
		s.stmtList(n.Nbody)
		s.stmt(n.Nincr)
		b = s.endBlock()
		addEdge(b, bCond)

		s.startBlock(bEnd)

	case OVARKILL:
		// TODO(khr): ??? anything to do here?  Only for addrtaken variables?
		// Maybe just link it in the store chain?
	default:
		log.Fatalf("unhandled stmt %s", opnames[n.Op])
	}
}

// expr converts the expression n to ssa, adds it to s and returns the ssa result.
func (s *ssaState) expr(n *Node) *ssa.Value {
	if n == nil {
		// TODO(khr): is this nil???
		return s.f.Entry.NewValue(ssa.OpConst, n.Type, nil)
	}
	switch n.Op {
	case ONAME:
		// TODO: remember offsets for PPARAM names
		if n.Class == PEXTERN {
			// global variable
			addr := s.f.Entry.NewValue(ssa.OpGlobal, Ptrto(n.Type), n.Sym)
			return s.curBlock.NewValue2(ssa.OpLoad, n.Type, nil, addr, s.mem())
		}
		s.argOffsets[n.Sym.Name] = n.Xoffset
		return s.variable(n.Sym.Name, n.Type)
		// binary ops
	case OLITERAL:
		switch n.Val.Ctype {
		case CTINT:
			return s.f.ConstInt(n.Type, Mpgetfix(n.Val.U.Xval))
		default:
			log.Fatalf("unhandled OLITERAL %v", n.Val.Ctype)
			return nil
		}
	case OLT:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.curBlock.NewValue2(ssa.OpLess, ssa.TypeBool, nil, a, b)
	case OADD:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.curBlock.NewValue2(ssa.OpAdd, a.Type, nil, a, b)

	case OSUB:
		// TODO:(khr) fold code for all binary ops together somehow
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.curBlock.NewValue2(ssa.OpSub, a.Type, nil, a, b)

	case OIND:
		p := s.expr(n.Left)
		c := s.curBlock.NewValue1(ssa.OpIsNonNil, ssa.TypeBool, nil, p)
		b := s.endBlock()
		b.Kind = ssa.BlockIf
		b.Control = c
		bNext := s.f.NewBlock(ssa.BlockPlain)
		addEdge(b, bNext)
		addEdge(b, s.exit)
		s.startBlock(bNext)
		// TODO(khr): if ptr check fails, don't go directly to exit.
		// Instead, go to a call to panicnil or something.
		// TODO: implicit nil checks somehow?

		return s.curBlock.NewValue2(ssa.OpLoad, n.Type, nil, p, s.mem())
	case ODOTPTR:
		p := s.expr(n.Left)
		// TODO: nilcheck
		p = s.curBlock.NewValue2(ssa.OpAdd, p.Type, nil, p, s.f.ConstInt(s.config.UIntPtr, n.Xoffset))
		return s.curBlock.NewValue2(ssa.OpLoad, n.Type, nil, p, s.mem())

	case OINDEX:
		// TODO: slice vs array?  Map index is already reduced to a function call
		a := s.expr(n.Left)
		i := s.expr(n.Right)
		// convert index to full width
		// TODO: if index is 64-bit and we're compiling to 32-bit, check that high
		// 32 bits are zero (and use a low32 op instead of convnop here).
		i = s.curBlock.NewValue1(ssa.OpConvNop, s.config.UIntPtr, nil, i)

		// bounds check
		len := s.curBlock.NewValue1(ssa.OpSliceLen, s.config.UIntPtr, nil, a)
		cmp := s.curBlock.NewValue2(ssa.OpIsInBounds, ssa.TypeBool, nil, i, len)
		b := s.endBlock()
		b.Kind = ssa.BlockIf
		b.Control = cmp
		bNext := s.f.NewBlock(ssa.BlockPlain)
		addEdge(b, bNext)
		addEdge(b, s.exit)
		s.startBlock(bNext)
		// TODO: don't go directly to s.exit.  Go to a stub that calls panicindex first.

		return s.curBlock.NewValue3(ssa.OpSliceIndex, n.Left.Type.Type, nil, a, i, s.mem())

	case OCALLFUNC:
		// run all argument assignments
		// TODO(khr): do we need to evaluate function first?
		// Or is it already side-effect-free and does not require a call?
		s.stmtList(n.List)

		if n.Left.Op != ONAME {
			// TODO(khr): closure calls?
			log.Fatalf("can't handle CALLFUNC with non-ONAME fn %s", opnames[n.Left.Op])
		}
		bNext := s.f.NewBlock(ssa.BlockPlain)
		call := s.curBlock.NewValue1(ssa.OpStaticCall, ssa.TypeMem, n.Left.Sym, s.mem())
		b := s.endBlock()
		b.Kind = ssa.BlockCall
		b.Control = call
		addEdge(b, bNext)
		addEdge(b, s.exit)

		// read result from stack at the start of the fallthrough block
		s.startBlock(bNext)
		var titer Iter
		fp := Structfirst(&titer, Getoutarg(n.Left.Type))
		a := s.f.Entry.NewValue(ssa.OpSPAddr, Ptrto(fp.Type), fp.Width)
		return s.curBlock.NewValue2(ssa.OpLoad, fp.Type, nil, a, call)
	default:
		log.Fatalf("unhandled expr %s", opnames[n.Op])
		return nil
	}
}

// variable returns the value of a variable at the current location.
func (s *ssaState) variable(name string, t ssa.Type) *ssa.Value {
	if s.curBlock == nil {
		log.Fatalf("nil curblock!")
	}
	v := s.vars[name]
	if v == nil {
		// TODO: get type?  Take Sym as arg?
		v = s.curBlock.NewValue(ssa.OpFwdRef, t, name)
		s.vars[name] = v
	}
	return v
}

func (s *ssaState) mem() *ssa.Value {
	return s.variable(".mem", ssa.TypeMem)
}

func (s *ssaState) linkForwardReferences() {
	// Build ssa graph.  Each variable on its first use in a basic block
	// leaves a FwdRef in that block representing the incoming value
	// of that variable.  This function links that ref up with possible definitions,
	// inserting Phi values as needed.  This is essentially the algorithm
	// described by Brau, Buchwald, Hack, Leißa, Mallon, and Zwinkau:
	// http://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
	for _, b := range s.f.Blocks {
		for _, v := range b.Values {
			if v.Op != ssa.OpFwdRef {
				continue
			}
			name := v.Aux.(string)
			v.Op = ssa.OpCopy
			v.Aux = nil
			v.SetArgs1(s.lookupVarIncoming(b, v.Type, name))
		}
	}
}

// lookupVarIncoming finds the variable's value at the start of block b.
func (s *ssaState) lookupVarIncoming(b *ssa.Block, t ssa.Type, name string) *ssa.Value {
	// TODO(khr): have lookupVarIncoming overwrite the fwdRef or copy it
	// will be used in, instead of having the result used in a copy value.
	if b == s.f.Entry {
		if name == ".mem" {
			return b.NewValue(ssa.OpArg, t, name)
		}
		// variable is live at the entry block.  Load it.
		a := s.f.Entry.NewValue(ssa.OpFPAddr, Ptrto(t.(*Type)), s.argOffsets[name])
		m := b.NewValue(ssa.OpArg, ssa.TypeMem, ".mem") // TODO: reuse mem starting value
		return b.NewValue2(ssa.OpLoad, t, nil, a, m)
	}
	var vals []*ssa.Value
	for _, p := range b.Preds {
		vals = append(vals, s.lookupVarOutgoing(p, t, name))
	}
	v0 := vals[0]
	for i := 1; i < len(vals); i++ {
		if vals[i] != v0 {
			// need a phi value
			v := b.NewValue(ssa.OpPhi, t, nil)
			v.AddArgs(vals...)
			return v
		}
	}
	return v0
}

// lookupVarOutgoing finds the variable's value at the end of block b.
func (s *ssaState) lookupVarOutgoing(b *ssa.Block, t ssa.Type, name string) *ssa.Value {
	m := s.defvars[b.ID]
	if v, ok := m[name]; ok {
		return v
	}
	// The variable is not defined by b and we haven't
	// looked it up yet.  Generate v, a copy value which
	// will be the outgoing value of the variable.  Then
	// look up w, the incoming value of the variable.
	// Make v = copy(w).  We need the extra copy to
	// prevent infinite recursion when looking up the
	// incoming value of the variable.
	v := b.NewValue(ssa.OpCopy, t, nil)
	m[name] = v
	v.AddArg(s.lookupVarIncoming(b, t, name))
	return v
}

// TODO: the above mutually recursive functions can lead to very deep stacks.  Fix that.

// addEdge adds an edge from b to c.
func addEdge(b, c *ssa.Block) {
	b.Succs = append(b.Succs, c)
	c.Preds = append(c.Preds, b)
}
