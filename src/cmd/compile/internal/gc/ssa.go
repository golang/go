// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"

	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

// buildssa builds an SSA function
// and reports whether it should be used.
// Once the SSA implementation is complete,
// it will never return nil, and the bool can be removed.
func buildssa(fn *Node) (ssafn *ssa.Func, usessa bool) {
	name := fn.Func.Nname.Sym.Name
	usessa = len(name) > 4 && name[len(name)-4:] == "_ssa"

	if usessa {
		dumplist("buildssa-enter", fn.Func.Enter)
		dumplist("buildssa-body", fn.Nbody)
	}

	var s state
	s.pushLine(fn.Lineno)
	defer s.popLine()

	// TODO(khr): build config just once at the start of the compiler binary

	var e ssaExport
	e.log = usessa
	s.config = ssa.NewConfig(Thearch.Thestring, &e)
	s.f = s.config.NewFunc()
	s.f.Name = name

	// If SSA support for the function is incomplete,
	// assume that any panics are due to violated
	// invariants. Swallow them silently.
	defer func() {
		if err := recover(); err != nil {
			if !e.unimplemented {
				panic(err)
			}
		}
	}()

	// We construct SSA using an algorithm similar to
	// Brau, Buchwald, Hack, Leißa, Mallon, and Zwinkau
	// http://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
	// TODO: check this comment

	// Allocate starting block
	s.f.Entry = s.f.NewBlock(ssa.BlockPlain)

	// Allocate exit block
	s.exit = s.f.NewBlock(ssa.BlockExit)

	// Allocate starting values
	s.vars = map[*Node]*ssa.Value{}
	s.labels = map[string]*ssa.Block{}
	s.startmem = s.entryNewValue0(ssa.OpArg, ssa.TypeMem)
	s.sp = s.entryNewValue0(ssa.OpSP, s.config.Uintptr) // TODO: use generic pointer type (unsafe.Pointer?) instead
	s.sb = s.entryNewValue0(ssa.OpSB, s.config.Uintptr)

	// Generate addresses of local declarations
	s.decladdrs = map[*Node]*ssa.Value{}
	for d := fn.Func.Dcl; d != nil; d = d.Next {
		n := d.N
		switch n.Class {
		case PPARAM, PPARAMOUT:
			aux := &ssa.ArgSymbol{Typ: n.Type, Offset: n.Xoffset, Sym: n.Sym}
			s.decladdrs[n] = s.entryNewValue1A(ssa.OpAddr, Ptrto(n.Type), aux, s.sp)
		case PAUTO:
			aux := &ssa.AutoSymbol{Typ: n.Type, Offset: -1, Sym: n.Sym} // offset TBD by SSA pass
			s.decladdrs[n] = s.entryNewValue1A(ssa.OpAddr, Ptrto(n.Type), aux, s.sp)
		}
	}
	// nodfp is a special argument which is the function's FP.
	aux := &ssa.ArgSymbol{Typ: s.config.Uintptr, Offset: 0, Sym: nodfp.Sym}
	s.decladdrs[nodfp] = s.entryNewValue1A(ssa.OpAddr, s.config.Uintptr, aux, s.sp)

	// Convert the AST-based IR to the SSA-based IR
	s.startBlock(s.f.Entry)
	s.stmtList(fn.Func.Enter)
	s.stmtList(fn.Nbody)

	// fallthrough to exit
	if b := s.endBlock(); b != nil {
		addEdge(b, s.exit)
	}

	// Finish up exit block
	s.startBlock(s.exit)
	s.exit.Control = s.mem()
	s.endBlock()

	// Link up variable uses to variable definitions
	s.linkForwardReferences()

	// Main call to ssa package to compile function
	ssa.Compile(s.f)

	// Calculate stats about what percentage of functions SSA handles.
	if false {
		fmt.Printf("SSA implemented: %t\n", !e.unimplemented)
	}

	if e.unimplemented {
		return nil, false
	}
	return s.f, usessa // TODO: return s.f, true once runtime support is in (gc maps, write barriers, etc.)
}

type state struct {
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

	// variable assignments in the current block (map from variable symbol to ssa value)
	// *Node is the unique identifier (an ONAME Node) for the variable.
	vars map[*Node]*ssa.Value

	// all defined variables at the end of each block.  Indexed by block ID.
	defvars []map[*Node]*ssa.Value

	// addresses of PPARAM, PPARAMOUT, and PAUTO variables.
	decladdrs map[*Node]*ssa.Value

	// starting values.  Memory, frame pointer, and stack pointer
	startmem *ssa.Value
	sp       *ssa.Value
	sb       *ssa.Value

	// line number stack.  The current line number is top of stack
	line []int32
}

func (s *state) Fatalf(msg string, args ...interface{})         { s.config.Fatalf(msg, args...) }
func (s *state) Unimplementedf(msg string, args ...interface{}) { s.config.Unimplementedf(msg, args...) }

// dummy node for the memory variable
var memvar = Node{Op: ONAME, Sym: &Sym{Name: "mem"}}

// startBlock sets the current block we're generating code in to b.
func (s *state) startBlock(b *ssa.Block) {
	if s.curBlock != nil {
		s.Fatalf("starting block %v when block %v has not ended", b, s.curBlock)
	}
	s.curBlock = b
	s.vars = map[*Node]*ssa.Value{}
}

// endBlock marks the end of generating code for the current block.
// Returns the (former) current block.  Returns nil if there is no current
// block, i.e. if no code flows to the current execution point.
func (s *state) endBlock() *ssa.Block {
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
	b.Line = s.peekLine()
	return b
}

// pushLine pushes a line number on the line number stack.
func (s *state) pushLine(line int32) {
	s.line = append(s.line, line)
}

// popLine pops the top of the line number stack.
func (s *state) popLine() {
	s.line = s.line[:len(s.line)-1]
}

// peekLine peek the top of the line number stack.
func (s *state) peekLine() int32 {
	return s.line[len(s.line)-1]
}

// newValue0 adds a new value with no arguments to the current block.
func (s *state) newValue0(op ssa.Op, t ssa.Type) *ssa.Value {
	return s.curBlock.NewValue0(s.peekLine(), op, t)
}

// newValue0A adds a new value with no arguments and an aux value to the current block.
func (s *state) newValue0A(op ssa.Op, t ssa.Type, aux interface{}) *ssa.Value {
	return s.curBlock.NewValue0A(s.peekLine(), op, t, aux)
}

// newValue1 adds a new value with one argument to the current block.
func (s *state) newValue1(op ssa.Op, t ssa.Type, arg *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue1(s.peekLine(), op, t, arg)
}

// newValue1A adds a new value with one argument and an aux value to the current block.
func (s *state) newValue1A(op ssa.Op, t ssa.Type, aux interface{}, arg *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue1A(s.peekLine(), op, t, aux, arg)
}

// newValue2 adds a new value with two arguments to the current block.
func (s *state) newValue2(op ssa.Op, t ssa.Type, arg0, arg1 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue2(s.peekLine(), op, t, arg0, arg1)
}

// newValue2I adds a new value with two arguments and an auxint value to the current block.
func (s *state) newValue2I(op ssa.Op, t ssa.Type, aux int64, arg0, arg1 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue2I(s.peekLine(), op, t, aux, arg0, arg1)
}

// newValue3 adds a new value with three arguments to the current block.
func (s *state) newValue3(op ssa.Op, t ssa.Type, arg0, arg1, arg2 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue3(s.peekLine(), op, t, arg0, arg1, arg2)
}

// entryNewValue adds a new value with no arguments to the entry block.
func (s *state) entryNewValue0(op ssa.Op, t ssa.Type) *ssa.Value {
	return s.f.Entry.NewValue0(s.peekLine(), op, t)
}

// entryNewValue adds a new value with no arguments and an aux value to the entry block.
func (s *state) entryNewValue0A(op ssa.Op, t ssa.Type, aux interface{}) *ssa.Value {
	return s.f.Entry.NewValue0A(s.peekLine(), op, t, aux)
}

// entryNewValue1 adds a new value with one argument to the entry block.
func (s *state) entryNewValue1(op ssa.Op, t ssa.Type, arg *ssa.Value) *ssa.Value {
	return s.f.Entry.NewValue1(s.peekLine(), op, t, arg)
}

// entryNewValue1 adds a new value with one argument and an auxint value to the entry block.
func (s *state) entryNewValue1I(op ssa.Op, t ssa.Type, auxint int64, arg *ssa.Value) *ssa.Value {
	return s.f.Entry.NewValue1I(s.peekLine(), op, t, auxint, arg)
}

// entryNewValue1A adds a new value with one argument and an aux value to the entry block.
func (s *state) entryNewValue1A(op ssa.Op, t ssa.Type, aux interface{}, arg *ssa.Value) *ssa.Value {
	return s.f.Entry.NewValue1A(s.peekLine(), op, t, aux, arg)
}

// entryNewValue2 adds a new value with two arguments to the entry block.
func (s *state) entryNewValue2(op ssa.Op, t ssa.Type, arg0, arg1 *ssa.Value) *ssa.Value {
	return s.f.Entry.NewValue2(s.peekLine(), op, t, arg0, arg1)
}

// constInt adds a new const int value to the entry block.
func (s *state) constInt(t ssa.Type, c int64) *ssa.Value {
	return s.f.ConstInt(s.peekLine(), t, c)
}

// ssaStmtList converts the statement n to SSA and adds it to s.
func (s *state) stmtList(l *NodeList) {
	for ; l != nil; l = l.Next {
		s.stmt(l.N)
	}
}

// ssaStmt converts the statement n to SSA and adds it to s.
func (s *state) stmt(n *Node) {
	s.pushLine(n.Lineno)
	defer s.popLine()

	s.stmtList(n.Ninit)
	switch n.Op {

	case OBLOCK:
		s.stmtList(n.List)

	case ODCL:
		if n.Left.Class&PHEAP == 0 {
			return
		}
		if compiling_runtime != 0 {
			Fatal("%v escapes to heap, not allowed in runtime.", n)
		}

		// TODO: the old pass hides the details of PHEAP
		// variables behind ONAME nodes. Figure out if it's better
		// to rewrite the tree and make the heapaddr construct explicit
		// or to keep this detail hidden behind the scenes.
		palloc := prealloc[n.Left]
		if palloc == nil {
			palloc = callnew(n.Left.Type)
			prealloc[n.Left] = palloc
		}
		s.assign(OAS, n.Left.Name.Heapaddr, palloc)

	case OLABEL, OGOTO:
		// get block at label, or make one
		t := s.labels[n.Left.Sym.Name]
		if t == nil {
			t = s.f.NewBlock(ssa.BlockPlain)
			s.labels[n.Left.Sym.Name] = t
		}
		// go to that label (we pretend "label:" is preceded by "goto label")
		if b := s.endBlock(); b != nil {
			addEdge(b, t)
		}

		if n.Op == OLABEL {
			// next we work on the label's target block
			s.startBlock(t)
		}
		if n.Op == OGOTO && s.curBlock == nil {
			s.Unimplementedf("goto at start of function; see test/goto.go")
		}

	case OAS, OASWB:
		s.assign(n.Op, n.Left, n.Right)

	case OIF:
		cond := s.expr(n.Left)
		b := s.endBlock()
		b.Kind = ssa.BlockIf
		b.Control = cond
		// TODO(khr): likely direction

		bThen := s.f.NewBlock(ssa.BlockPlain)
		bEnd := s.f.NewBlock(ssa.BlockPlain)
		var bElse *ssa.Block

		if n.Rlist == nil {
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

		if n.Rlist != nil {
			s.startBlock(bElse)
			s.stmtList(n.Rlist)
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
		// TODO(khr): Left == nil exception
		if n.Left == nil {
			s.Unimplementedf("cond n.Left == nil: %v", n)
		}
		s.startBlock(bCond)
		s.stmtList(n.Left.Ninit)
		cond := s.expr(n.Left)
		b = s.endBlock()
		b.Kind = ssa.BlockIf
		b.Control = cond
		// TODO(khr): likely direction
		addEdge(b, bBody)
		addEdge(b, bEnd)

		// generate body
		s.startBlock(bBody)
		s.stmtList(n.Nbody)
		if n.Right != nil {
			s.stmt(n.Right)
		}
		b = s.endBlock()
		addEdge(b, bCond)

		s.startBlock(bEnd)

	case OCALLFUNC:
		s.expr(n)

	case OVARKILL:
		// TODO(khr): ??? anything to do here?  Only for addrtaken variables?
		// Maybe just link it in the store chain?
	default:
		s.Unimplementedf("unhandled stmt %s", opnames[n.Op])
	}
}

var binOpToSSA = [...]ssa.Op{
	// Comparisons
	OEQ: ssa.OpEq,
	ONE: ssa.OpNeq,
	OLT: ssa.OpLess,
	OLE: ssa.OpLeq,
	OGT: ssa.OpGreater,
	OGE: ssa.OpGeq,
	// Arithmetic
	OADD: ssa.OpAdd,
	OSUB: ssa.OpSub,
	OLSH: ssa.OpLsh,
	ORSH: ssa.OpRsh,
}

// expr converts the expression n to ssa, adds it to s and returns the ssa result.
func (s *state) expr(n *Node) *ssa.Value {
	s.pushLine(n.Lineno)
	defer s.popLine()

	switch n.Op {
	case ONAME:
		if n.Class == PFUNC {
			// "value" of a function is the address of the function's closure
			sym := funcsym(n.Sym)
			aux := &ssa.ExternSymbol{n.Type, sym}
			return s.entryNewValue1A(ssa.OpAddr, Ptrto(n.Type), aux, s.sb)
		}
		if canSSA(n) {
			return s.variable(n, n.Type)
		}
		addr := s.addr(n)
		return s.newValue2(ssa.OpLoad, n.Type, addr, s.mem())
	case OLITERAL:
		switch n.Val().Ctype() {
		case CTINT:
			return s.constInt(n.Type, Mpgetfix(n.Val().U.(*Mpint)))
		case CTSTR:
			return s.entryNewValue0A(ssa.OpConst, n.Type, n.Val().U)
		default:
			s.Unimplementedf("unhandled OLITERAL %v", n.Val().Ctype())
			return nil
		}
	case OCONVNOP:
		x := s.expr(n.Left)
		return s.newValue1(ssa.OpConvNop, n.Type, x)
	case OCONV:
		x := s.expr(n.Left)
		return s.newValue1(ssa.OpConvert, n.Type, x)

	// binary ops
	case OLT, OEQ, ONE, OLE, OGE, OGT:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.newValue2(binOpToSSA[n.Op], ssa.TypeBool, a, b)
	case OADD, OSUB, OLSH, ORSH:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.newValue2(binOpToSSA[n.Op], a.Type, a, b)

	case OADDR:
		return s.addr(n.Left)

	case OIND:
		p := s.expr(n.Left)
		s.nilCheck(p)
		return s.newValue2(ssa.OpLoad, n.Type, p, s.mem())

	case ODOTPTR:
		p := s.expr(n.Left)
		s.nilCheck(p)
		p = s.newValue2(ssa.OpAdd, p.Type, p, s.constInt(s.config.Uintptr, n.Xoffset))
		return s.newValue2(ssa.OpLoad, n.Type, p, s.mem())

	case OINDEX:
		if n.Left.Type.Bound >= 0 { // array or string
			a := s.expr(n.Left)
			i := s.expr(n.Right)
			var elemtype *Type
			var len *ssa.Value
			if n.Left.Type.IsString() {
				len = s.newValue1(ssa.OpStringLen, s.config.Uintptr, a)
				elemtype = Types[TUINT8]
			} else {
				len = s.constInt(s.config.Uintptr, n.Left.Type.Bound)
				elemtype = n.Left.Type.Type
			}
			s.boundsCheck(i, len)
			return s.newValue2(ssa.OpArrayIndex, elemtype, a, i)
		} else { // slice
			p := s.addr(n)
			return s.newValue2(ssa.OpLoad, n.Left.Type.Type, p, s.mem())
		}

	case OCALLFUNC:
		static := n.Left.Op == ONAME && n.Left.Class == PFUNC

		// evaluate closure
		var closure *ssa.Value
		if !static {
			closure = s.expr(n.Left)
		}

		// run all argument assignments
		s.stmtList(n.List)

		bNext := s.f.NewBlock(ssa.BlockPlain)
		var call *ssa.Value
		if static {
			call = s.newValue1A(ssa.OpStaticCall, ssa.TypeMem, n.Left.Sym, s.mem())
		} else {
			entry := s.newValue2(ssa.OpLoad, s.config.Uintptr, closure, s.mem())
			call = s.newValue3(ssa.OpClosureCall, ssa.TypeMem, entry, closure, s.mem())
		}
		b := s.endBlock()
		b.Kind = ssa.BlockCall
		b.Control = call
		addEdge(b, bNext)
		addEdge(b, s.exit)

		// read result from stack at the start of the fallthrough block
		s.startBlock(bNext)
		var titer Iter
		fp := Structfirst(&titer, Getoutarg(n.Left.Type))
		if fp == nil {
			// CALLFUNC has no return value. Continue with the next statement.
			return nil
		}
		a := s.entryNewValue1I(ssa.OpOffPtr, Ptrto(fp.Type), fp.Width, s.sp)
		return s.newValue2(ssa.OpLoad, fp.Type, a, call)
	default:
		s.Unimplementedf("unhandled expr %s", opnames[n.Op])
		return nil
	}
}

func (s *state) assign(op uint8, left *Node, right *Node) {
	// TODO: do write barrier
	// if op == OASWB
	var val *ssa.Value
	if right == nil {
		// right == nil means use the zero value of the assigned type.
		t := left.Type
		if !canSSA(left) {
			// if we can't ssa this memory, treat it as just zeroing out the backing memory
			addr := s.addr(left)
			s.vars[&memvar] = s.newValue2I(ssa.OpZero, ssa.TypeMem, t.Size(), addr, s.mem())
			return
		}
		switch {
		case t.IsString():
			val = s.entryNewValue0A(ssa.OpConst, left.Type, "")
		case t.IsInteger():
			val = s.entryNewValue0(ssa.OpConst, left.Type)
		case t.IsBoolean():
			val = s.entryNewValue0A(ssa.OpConst, left.Type, false) // TODO: store bools as 0/1 in AuxInt?
		default:
			s.Unimplementedf("zero for type %v not implemented", t)
		}
	} else {
		val = s.expr(right)
	}
	if left.Op == ONAME && canSSA(left) {
		// Update variable assignment.
		s.vars[left] = val
		return
	}
	// not ssa-able.  Treat as a store.
	addr := s.addr(left)
	s.vars[&memvar] = s.newValue3(ssa.OpStore, ssa.TypeMem, addr, val, s.mem())
}

// addr converts the address of the expression n to SSA, adds it to s and returns the SSA result.
func (s *state) addr(n *Node) *ssa.Value {
	switch n.Op {
	case ONAME:
		switch n.Class {
		case PEXTERN:
			// global variable
			aux := &ssa.ExternSymbol{n.Type, n.Sym}
			return s.entryNewValue1A(ssa.OpAddr, Ptrto(n.Type), aux, s.sb)
		case PPARAM, PPARAMOUT, PAUTO:
			// parameter/result slot or local variable
			v := s.decladdrs[n]
			if v == nil {
				s.Fatalf("addr of undeclared ONAME %v. declared: %v", n, s.decladdrs)
			}
			return v
		case PAUTO | PHEAP:
			return s.expr(n.Name.Heapaddr)
		default:
			s.Unimplementedf("variable address of %v not implemented", n)
			return nil
		}
	case OINDREG:
		// indirect off a register (TODO: always SP?)
		// used for storing/loading arguments/returns to/from callees
		return s.entryNewValue1I(ssa.OpOffPtr, Ptrto(n.Type), n.Xoffset, s.sp)
	case OINDEX:
		if n.Left.Type.Bound >= 0 { // array
			a := s.addr(n.Left)
			i := s.expr(n.Right)
			len := s.constInt(s.config.Uintptr, n.Left.Type.Bound)
			s.boundsCheck(i, len)
			return s.newValue2(ssa.OpPtrIndex, Ptrto(n.Left.Type.Type), a, i)
		} else { // slice
			a := s.expr(n.Left)
			i := s.expr(n.Right)
			len := s.newValue1(ssa.OpSliceLen, s.config.Uintptr, a)
			s.boundsCheck(i, len)
			p := s.newValue1(ssa.OpSlicePtr, Ptrto(n.Left.Type.Type), a)
			return s.newValue2(ssa.OpPtrIndex, Ptrto(n.Left.Type.Type), p, i)
		}
	default:
		s.Unimplementedf("addr: bad op %v", Oconv(int(n.Op), 0))
		return nil
	}
}

// canSSA reports whether n is SSA-able.
// n must be an ONAME.
func canSSA(n *Node) bool {
	if n.Op != ONAME {
		return false
	}
	if n.Addrtaken {
		return false
	}
	if n.Class&PHEAP != 0 {
		return false
	}
	if n.Class == PEXTERN {
		return false
	}
	if n.Class == PPARAMOUT {
		return false
	}
	if Isfat(n.Type) {
		return false
	}
	return true
	// TODO: try to make more variables SSAable.
}

// nilCheck generates nil pointer checking code.
// Starts a new block on return.
func (s *state) nilCheck(ptr *ssa.Value) {
	c := s.newValue1(ssa.OpIsNonNil, ssa.TypeBool, ptr)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = c
	bNext := s.f.NewBlock(ssa.BlockPlain)
	addEdge(b, bNext)
	addEdge(b, s.exit)
	s.startBlock(bNext)
	// TODO(khr): Don't go directly to exit.  Go to a stub that calls panicmem first.
	// TODO: implicit nil checks somehow?
}

// boundsCheck generates bounds checking code.  Checks if 0 <= idx < len, branches to exit if not.
// Starts a new block on return.
func (s *state) boundsCheck(idx, len *ssa.Value) {
	// TODO: convert index to full width?
	// TODO: if index is 64-bit and we're compiling to 32-bit, check that high 32 bits are zero.

	// bounds check
	cmp := s.newValue2(ssa.OpIsInBounds, ssa.TypeBool, idx, len)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = cmp
	bNext := s.f.NewBlock(ssa.BlockPlain)
	addEdge(b, bNext)
	addEdge(b, s.exit)
	// TODO: don't go directly to s.exit.  Go to a stub that calls panicindex first.
	s.startBlock(bNext)
}

// variable returns the value of a variable at the current location.
func (s *state) variable(name *Node, t ssa.Type) *ssa.Value {
	if s.curBlock == nil {
		// Unimplemented instead of Fatal because fixedbugs/bug303.go
		// demonstrates a case in which this appears to happen legitimately.
		// TODO: decide on the correct behavior here.
		s.Unimplementedf("nil curblock adding variable %v (%v)", name, t)
	}
	v := s.vars[name]
	if v == nil {
		// TODO: get type?  Take Sym as arg?
		v = s.newValue0A(ssa.OpFwdRef, t, name)
		s.vars[name] = v
	}
	return v
}

func (s *state) mem() *ssa.Value {
	return s.variable(&memvar, ssa.TypeMem)
}

func (s *state) linkForwardReferences() {
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
			name := v.Aux.(*Node)
			v.Op = ssa.OpCopy
			v.Aux = nil
			v.SetArgs1(s.lookupVarIncoming(b, v.Type, name))
		}
	}
}

// lookupVarIncoming finds the variable's value at the start of block b.
func (s *state) lookupVarIncoming(b *ssa.Block, t ssa.Type, name *Node) *ssa.Value {
	// TODO(khr): have lookupVarIncoming overwrite the fwdRef or copy it
	// will be used in, instead of having the result used in a copy value.
	if b == s.f.Entry {
		if name == &memvar {
			return s.startmem
		}
		// variable is live at the entry block.  Load it.
		addr := s.decladdrs[name]
		if addr == nil {
			// TODO: closure args reach here.
			s.Unimplementedf("variable %s not found", name)
		}
		if _, ok := addr.Aux.(*ssa.ArgSymbol); !ok {
			s.Fatalf("variable live at start of function %s is not an argument %s", b.Func.Name, name)
		}
		return s.entryNewValue2(ssa.OpLoad, t, addr, s.startmem)
	}
	var vals []*ssa.Value
	for _, p := range b.Preds {
		vals = append(vals, s.lookupVarOutgoing(p, t, name))
	}
	if len(vals) == 0 {
		s.Unimplementedf("TODO: Handle fixedbugs/bug076.go")
		return nil
	}
	v0 := vals[0]
	for i := 1; i < len(vals); i++ {
		if vals[i] != v0 {
			// need a phi value
			v := b.NewValue0(s.peekLine(), ssa.OpPhi, t)
			v.AddArgs(vals...)
			return v
		}
	}
	return v0
}

// lookupVarOutgoing finds the variable's value at the end of block b.
func (s *state) lookupVarOutgoing(b *ssa.Block, t ssa.Type, name *Node) *ssa.Value {
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
	v := b.NewValue0(s.peekLine(), ssa.OpCopy, t)
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

// an unresolved branch
type branch struct {
	p *obj.Prog  // branch instruction
	b *ssa.Block // target
}

// genssa appends entries to ptxt for each instruction in f.
// gcargs and gclocals are filled in with pointer maps for the frame.
func genssa(f *ssa.Func, ptxt *obj.Prog, gcargs, gclocals *Sym) {
	// TODO: line numbers

	if f.FrameSize > 1<<31 {
		Yyerror("stack frame too large (>2GB)")
		return
	}

	ptxt.To.Type = obj.TYPE_TEXTSIZE
	ptxt.To.Val = int32(Rnd(Curfn.Type.Argwid, int64(Widthptr))) // arg size
	ptxt.To.Offset = f.FrameSize - 8                             // TODO: arch-dependent

	// Remember where each block starts.
	bstart := make([]*obj.Prog, f.NumBlocks())

	// Remember all the branch instructions we've seen
	// and where they would like to go
	var branches []branch

	// Emit basic blocks
	for i, b := range f.Blocks {
		bstart[b.ID] = Pc
		// Emit values in block
		for _, v := range b.Values {
			genValue(v)
		}
		// Emit control flow instructions for block
		var next *ssa.Block
		if i < len(f.Blocks)-1 {
			next = f.Blocks[i+1]
		}
		branches = genBlock(b, next, branches)
	}

	// Resolve branches
	for _, br := range branches {
		br.p.To.Val = bstart[br.b.ID]
	}

	Pc.As = obj.ARET // overwrite AEND

	// TODO: liveness
	// TODO: gcargs
	// TODO: gclocals

	// TODO: dump frame if -f

	// Emit garbage collection symbols.  TODO: put something in them
	//liveness(Curfn, ptxt, gcargs, gclocals)
	duint32(gcargs, 0, 0)
	ggloblsym(gcargs, 4, obj.RODATA|obj.DUPOK)
	duint32(gclocals, 0, 0)
	ggloblsym(gclocals, 4, obj.RODATA|obj.DUPOK)
}

func genValue(v *ssa.Value) {
	lineno = v.Line
	switch v.Op {
	case ssa.OpAMD64ADDQ:
		// TODO: use addq instead of leaq if target is in the right register.
		p := Prog(x86.ALEAQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		p.From.Scale = 1
		p.From.Index = regnum(v.Args[1])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64ADDL:
		p := Prog(x86.ALEAL)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		p.From.Scale = 1
		p.From.Index = regnum(v.Args[1])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64ADDW:
		p := Prog(x86.ALEAW)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		p.From.Scale = 1
		p.From.Index = regnum(v.Args[1])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64ADDB, ssa.OpAMD64ANDQ:
		r := regnum(v)
		x := regnum(v.Args[0])
		y := regnum(v.Args[1])
		if x != r && y != r {
			p := Prog(x86.AMOVQ)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			x = r
		}
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		if x == r {
			p.From.Reg = y
		} else {
			p.From.Reg = x
		}
	case ssa.OpAMD64ADDQconst:
		// TODO: use addq instead of leaq if target is in the right register.
		p := Prog(x86.ALEAQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64MULQconst:
		v.Unimplementedf("IMULQ doasm")
		return
		// TODO: this isn't right.  doasm fails on it.  I don't think obj
		// has ever been taught to compile imul $c, r1, r2.
		p := Prog(x86.AIMULQ)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.From3 = new(obj.Addr)
		p.From3.Type = obj.TYPE_REG
		p.From3.Reg = regnum(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64SUBQconst:
		// This code compensates for the fact that the register allocator
		// doesn't understand 2-address instructions yet.  TODO: fix that.
		x := regnum(v.Args[0])
		r := regnum(v)
		if x != r {
			p := Prog(x86.AMOVQ)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			x = r
		}
		p := Prog(x86.ASUBQ)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpAMD64SHLQ, ssa.OpAMD64SHRQ, ssa.OpAMD64SARQ:
		x := regnum(v.Args[0])
		r := regnum(v)
		if x != r {
			if r == x86.REG_CX {
				v.Fatalf("can't implement %s, target and shift both in CX", v.LongString())
			}
			p := Prog(x86.AMOVQ)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			x = r
		}
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[1]) // should be CX
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpAMD64SHLQconst, ssa.OpAMD64SHRQconst, ssa.OpAMD64SARQconst:
		x := regnum(v.Args[0])
		r := regnum(v)
		if x != r {
			p := Prog(x86.AMOVQ)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			x = r
		}
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpAMD64SBBQcarrymask:
		r := regnum(v)
		p := Prog(x86.ASBBQ)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpAMD64CMOVQCC:
		r := regnum(v)
		x := regnum(v.Args[1])
		y := regnum(v.Args[2])
		if x != r && y != r {
			p := Prog(x86.AMOVQ)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			x = r
		}
		var p *obj.Prog
		if x == r {
			p = Prog(x86.ACMOVQCS)
			p.From.Reg = y
		} else {
			p = Prog(x86.ACMOVQCC)
			p.From.Reg = x
		}
		p.From.Type = obj.TYPE_REG
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpAMD64LEAQ1:
		p := Prog(x86.ALEAQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		p.From.Scale = 1
		p.From.Index = regnum(v.Args[1])
		addAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64LEAQ:
		p := Prog(x86.ALEAQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		addAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64CMPQ, ssa.OpAMD64TESTB, ssa.OpAMD64TESTQ:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v.Args[1])
	case ssa.OpAMD64CMPQconst:
		p := Prog(x86.ACMPQ)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[0])
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssa.OpAMD64MOVQconst:
		x := regnum(v)
		p := Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssa.OpAMD64MOVQload, ssa.OpAMD64MOVLload, ssa.OpAMD64MOVWload, ssa.OpAMD64MOVBload:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		addAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64MOVQloadidx8:
		p := Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		addAux(&p.From, v)
		p.From.Scale = 8
		p.From.Index = regnum(v.Args[1])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64MOVQstore, ssa.OpAMD64MOVLstore, ssa.OpAMD64MOVWstore, ssa.OpAMD64MOVBstore:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[1])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = regnum(v.Args[0])
		addAux(&p.To, v)
	case ssa.OpAMD64MOVLQSX, ssa.OpAMD64MOVWQSX, ssa.OpAMD64MOVBQSX:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64MOVXzero:
		nb := v.AuxInt
		offset := int64(0)
		reg := regnum(v.Args[0])
		for nb >= 8 {
			nb, offset = movZero(x86.AMOVQ, 8, nb, offset, reg)
		}
		for nb >= 4 {
			nb, offset = movZero(x86.AMOVL, 4, nb, offset, reg)
		}
		for nb >= 2 {
			nb, offset = movZero(x86.AMOVW, 2, nb, offset, reg)
		}
		for nb >= 1 {
			nb, offset = movZero(x86.AMOVB, 1, nb, offset, reg)
		}
	case ssa.OpCopy: // TODO: lower to MOVQ earlier?
		if v.Type.IsMemory() {
			return
		}
		x := regnum(v.Args[0])
		y := regnum(v)
		if x != y {
			p := Prog(x86.AMOVQ)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = y
		}
	case ssa.OpLoadReg8:
		p := Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = x86.REG_SP
		p.From.Offset = localOffset(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpStoreReg8:
		p := Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[0])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = x86.REG_SP
		p.To.Offset = localOffset(v)
	case ssa.OpPhi:
		// just check to make sure regalloc did it right
		f := v.Block.Func
		loc := f.RegAlloc[v.ID]
		for _, a := range v.Args {
			if f.RegAlloc[a.ID] != loc { // TODO: .Equal() instead?
				v.Fatalf("phi arg at different location than phi %v %v %v %v", v, loc, a, f.RegAlloc[a.ID])
			}
		}
	case ssa.OpConst:
		if v.Block.Func.RegAlloc[v.ID] != nil {
			v.Fatalf("const value %v shouldn't have a location", v)
		}
	case ssa.OpArg:
		// memory arg needs no code
		// TODO: check that only mem arg goes here.
	case ssa.OpAMD64CALLstatic:
		p := Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = Linksym(v.Aux.(*Sym))
	case ssa.OpAMD64CALLclosure:
		p := Prog(obj.ACALL)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v.Args[0])
	case ssa.OpSP, ssa.OpSB:
		// nothing to do
	default:
		v.Unimplementedf("value %s not implemented", v.LongString())
	}
}

// movZero generates a register indirect move with a 0 immediate and keeps track of bytes left and next offset
func movZero(as int, width int64, nbytes int64, offset int64, regnum int16) (nleft int64, noff int64) {
	p := Prog(as)
	// TODO: use zero register on archs that support it.
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = 0
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = regnum
	p.To.Offset = offset
	offset += width
	nleft = nbytes - width
	return nleft, offset
}

func genBlock(b, next *ssa.Block, branches []branch) []branch {
	lineno = b.Line
	switch b.Kind {
	case ssa.BlockPlain:
		if b.Succs[0] != next {
			p := Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
		}
	case ssa.BlockExit:
		Prog(obj.ARET)
	case ssa.BlockCall:
		if b.Succs[0] != next {
			p := Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
		}
	case ssa.BlockAMD64EQ:
		if b.Succs[0] == next {
			p := Prog(x86.AJNE)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[1]})
		} else if b.Succs[1] == next {
			p := Prog(x86.AJEQ)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
		} else {
			p := Prog(x86.AJEQ)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
			q := Prog(obj.AJMP)
			q.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{q, b.Succs[1]})
		}
	case ssa.BlockAMD64NE:
		if b.Succs[0] == next {
			p := Prog(x86.AJEQ)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[1]})
		} else if b.Succs[1] == next {
			p := Prog(x86.AJNE)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
		} else {
			p := Prog(x86.AJNE)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
			q := Prog(obj.AJMP)
			q.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{q, b.Succs[1]})
		}
	case ssa.BlockAMD64LT:
		if b.Succs[0] == next {
			p := Prog(x86.AJGE)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[1]})
		} else if b.Succs[1] == next {
			p := Prog(x86.AJLT)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
		} else {
			p := Prog(x86.AJLT)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
			q := Prog(obj.AJMP)
			q.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{q, b.Succs[1]})
		}
	case ssa.BlockAMD64ULT:
		if b.Succs[0] == next {
			p := Prog(x86.AJCC)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[1]})
		} else if b.Succs[1] == next {
			p := Prog(x86.AJCS)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
		} else {
			p := Prog(x86.AJCS)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
			q := Prog(obj.AJMP)
			q.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{q, b.Succs[1]})
		}
	case ssa.BlockAMD64UGT:
		if b.Succs[0] == next {
			p := Prog(x86.AJLS)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[1]})
		} else if b.Succs[1] == next {
			p := Prog(x86.AJHI)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
		} else {
			p := Prog(x86.AJHI)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
			q := Prog(obj.AJMP)
			q.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{q, b.Succs[1]})
		}

	default:
		b.Unimplementedf("branch %s not implemented", b.LongString())
	}
	return branches
}

// addAux adds the offset in the aux fields (AuxInt and Aux) of v to a.
func addAux(a *obj.Addr, v *ssa.Value) {
	if a.Type != obj.TYPE_MEM {
		v.Fatalf("bad addAux addr %s", a)
	}
	// add integer offset
	a.Offset += v.AuxInt

	// If no additional symbol offset, we're done.
	if v.Aux == nil {
		return
	}
	// Add symbol's offset from its base register.
	switch sym := v.Aux.(type) {
	case *ssa.ExternSymbol:
		a.Name = obj.NAME_EXTERN
		a.Sym = Linksym(sym.Sym.(*Sym))
	case *ssa.ArgSymbol:
		a.Offset += v.Block.Func.FrameSize + sym.Offset
	case *ssa.AutoSymbol:
		if sym.Offset == -1 {
			v.Fatalf("auto symbol %s offset not calculated", sym.Sym)
		}
		a.Offset += sym.Offset
	default:
		v.Fatalf("aux in %s not implemented %#v", v, v.Aux)
	}
}

// ssaRegToReg maps ssa register numbers to obj register numbers.
var ssaRegToReg = [...]int16{
	x86.REG_AX,
	x86.REG_CX,
	x86.REG_DX,
	x86.REG_BX,
	x86.REG_SP,
	x86.REG_BP,
	x86.REG_SI,
	x86.REG_DI,
	x86.REG_R8,
	x86.REG_R9,
	x86.REG_R10,
	x86.REG_R11,
	x86.REG_R12,
	x86.REG_R13,
	x86.REG_R14,
	x86.REG_R15,
	x86.REG_X0,
	x86.REG_X1,
	x86.REG_X2,
	x86.REG_X3,
	x86.REG_X4,
	x86.REG_X5,
	x86.REG_X6,
	x86.REG_X7,
	x86.REG_X8,
	x86.REG_X9,
	x86.REG_X10,
	x86.REG_X11,
	x86.REG_X12,
	x86.REG_X13,
	x86.REG_X14,
	x86.REG_X15,
	0, // SB isn't a real register.  We fill an Addr.Reg field with 0 in this case.
	// TODO: arch-dependent
}

// regnum returns the register (in cmd/internal/obj numbering) to
// which v has been allocated.  Panics if v is not assigned to a
// register.
func regnum(v *ssa.Value) int16 {
	return ssaRegToReg[v.Block.Func.RegAlloc[v.ID].(*ssa.Register).Num]
}

// localOffset returns the offset below the frame pointer where
// a stack-allocated local has been allocated.  Panics if v
// is not assigned to a local slot.
func localOffset(v *ssa.Value) int64 {
	return v.Block.Func.RegAlloc[v.ID].(*ssa.LocalSlot).Idx
}

// ssaExport exports a bunch of compiler services for the ssa backend.
type ssaExport struct {
	log           bool
	unimplemented bool
}

// StringSym returns a symbol (a *Sym wrapped in an interface) which
// is a global string constant containing s.
func (*ssaExport) StringSym(s string) interface{} {
	// TODO: is idealstring correct?  It might not matter...
	hdr, _ := stringsym(s)
	return &ssa.ExternSymbol{Typ: idealstring, Sym: hdr}
}

// Log logs a message from the compiler.
func (e *ssaExport) Logf(msg string, args ...interface{}) {
	// If e was marked as unimplemented, anything could happen. Ignore.
	if e.log && !e.unimplemented {
		fmt.Printf(msg, args...)
	}
}

// Fatal reports a compiler error and exits.
func (e *ssaExport) Fatalf(msg string, args ...interface{}) {
	// If e was marked as unimplemented, anything could happen. Ignore.
	if !e.unimplemented {
		Fatal(msg, args...)
	}
}

// Unimplemented reports that the function cannot be compiled.
// It will be removed once SSA work is complete.
func (e *ssaExport) Unimplementedf(msg string, args ...interface{}) {
	const alwaysLog = false // enable to calculate top unimplemented features
	if !e.unimplemented && (e.log || alwaysLog) {
		// first implementation failure, print explanation
		fmt.Printf("SSA unimplemented: "+msg+"\n", args...)
	}
	e.unimplemented = true
}
