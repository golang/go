// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"os"
	"strings"

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
	usessa = strings.HasSuffix(name, "_ssa")

	if usessa {
		fmt.Println("generating SSA for", name)
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
	s.labels = map[string]*ssaLabel{}
	s.labeledNodes = map[*Node]*ssaLabel{}
	s.startmem = s.entryNewValue0(ssa.OpArg, ssa.TypeMem)
	s.sp = s.entryNewValue0(ssa.OpSP, Types[TUINTPTR]) // TODO: use generic pointer type (unsafe.Pointer?) instead
	s.sb = s.entryNewValue0(ssa.OpSB, Types[TUINTPTR])

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
		default:
			str := ""
			if n.Class&PHEAP != 0 {
				str = ",heap"
			}
			s.Unimplementedf("local variable %v with class %s%s unimplemented", n, classnames[n.Class&^PHEAP], str)
		}
	}
	// nodfp is a special argument which is the function's FP.
	aux := &ssa.ArgSymbol{Typ: Types[TUINTPTR], Offset: 0, Sym: nodfp.Sym}
	s.decladdrs[nodfp] = s.entryNewValue1A(ssa.OpAddr, Types[TUINTPTR], aux, s.sp)

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

	// Check that we used all labels
	for name, lab := range s.labels {
		if !lab.used() && !lab.reported {
			yyerrorl(int(lab.defNode.Lineno), "label %v defined and not used", name)
			lab.reported = true
		}
		if lab.used() && !lab.defined() && !lab.reported {
			yyerrorl(int(lab.useNode.Lineno), "label %v not defined", name)
			lab.reported = true
		}
	}

	// Check any forward gotos. Non-forward gotos have already been checked.
	for _, n := range s.fwdGotos {
		lab := s.labels[n.Left.Sym.Name]
		// If the label is undefined, we have already have printed an error.
		if lab.defined() {
			s.checkgoto(n, lab.defNode)
		}
	}

	if nerrors > 0 {
		return nil, false
	}

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

	// TODO: enable codegen more broadly once the codegen stabilizes
	// and runtime support is in (gc maps, write barriers, etc.)
	return s.f, usessa || name == os.Getenv("GOSSAFUNC") || localpkg.Name == os.Getenv("GOSSAPKG")
}

type state struct {
	// configuration (arch) information
	config *ssa.Config

	// function we're building
	f *ssa.Func

	// exit block that "return" jumps to (and panics jump to)
	exit *ssa.Block

	// labels and labeled control flow nodes (OFOR, OSWITCH, OSELECT) in f
	labels       map[string]*ssaLabel
	labeledNodes map[*Node]*ssaLabel

	// gotos that jump forward; required for deferred checkgoto calls
	fwdGotos []*Node

	// unlabeled break and continue statement tracking
	breakTo    *ssa.Block // current target for plain break statement
	continueTo *ssa.Block // current target for plain continue statement

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

type ssaLabel struct {
	target         *ssa.Block // block identified by this label
	breakTarget    *ssa.Block // block to break to in control flow node identified by this label
	continueTarget *ssa.Block // block to continue to in control flow node identified by this label
	defNode        *Node      // label definition Node (OLABEL)
	// Label use Node (OGOTO, OBREAK, OCONTINUE).
	// Used only for error detection and reporting.
	// There might be multiple uses, but we only need to track one.
	useNode  *Node
	reported bool // reported indicates whether an error has already been reported for this label
}

// defined reports whether the label has a definition (OLABEL node).
func (l *ssaLabel) defined() bool { return l.defNode != nil }

// used reports whether the label has a use (OGOTO, OBREAK, or OCONTINUE node).
func (l *ssaLabel) used() bool { return l.useNode != nil }

// label returns the label associated with sym, creating it if necessary.
func (s *state) label(sym *Sym) *ssaLabel {
	lab := s.labels[sym.Name]
	if lab == nil {
		lab = new(ssaLabel)
		s.labels[sym.Name] = lab
	}
	return lab
}

func (s *state) Logf(msg string, args ...interface{})           { s.config.Logf(msg, args...) }
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

func (s *state) Error(msg string, args ...interface{}) {
	yyerrorl(int(s.peekLine()), msg, args...)
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

// newValue1I adds a new value with one argument and an auxint value to the current block.
func (s *state) newValue1I(op ssa.Op, t ssa.Type, aux int64, arg *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue1I(s.peekLine(), op, t, aux, arg)
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

// constInt* routines add a new const int value to the entry block.
func (s *state) constInt8(t ssa.Type, c int8) *ssa.Value {
	return s.f.ConstInt8(s.peekLine(), t, c)
}
func (s *state) constInt16(t ssa.Type, c int16) *ssa.Value {
	return s.f.ConstInt16(s.peekLine(), t, c)
}
func (s *state) constInt32(t ssa.Type, c int32) *ssa.Value {
	return s.f.ConstInt32(s.peekLine(), t, c)
}
func (s *state) constInt64(t ssa.Type, c int64) *ssa.Value {
	return s.f.ConstInt64(s.peekLine(), t, c)
}
func (s *state) constIntPtr(t ssa.Type, c int64) *ssa.Value {
	if s.config.PtrSize == 4 && int64(int32(c)) != c {
		s.Fatalf("pointer constant too big %d", c)
	}
	return s.f.ConstIntPtr(s.peekLine(), t, c)
}
func (s *state) constInt(t ssa.Type, c int64) *ssa.Value {
	if s.config.IntSize == 8 {
		return s.constInt64(t, c)
	}
	if int64(int32(c)) != c {
		s.Fatalf("integer constant too big %d", c)
	}
	return s.constInt32(t, int32(c))
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

	// If s.curBlock is nil, then we're about to generate dead code.
	// We can't just short-circuit here, though,
	// because we check labels and gotos as part of SSA generation.
	// Provide a block for the dead code so that we don't have
	// to add special cases everywhere else.
	if s.curBlock == nil {
		dead := s.f.NewBlock(ssa.BlockPlain)
		s.startBlock(dead)
	}

	s.stmtList(n.Ninit)
	switch n.Op {

	case OBLOCK:
		s.stmtList(n.List)

	// No-ops
	case OEMPTY, ODCLCONST, ODCLTYPE:

	// Expression statements
	case OCALLFUNC, OCALLMETH, OCALLINTER:
		s.expr(n)

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

	case OLABEL:
		sym := n.Left.Sym

		if isblanksym(sym) {
			// Empty identifier is valid but useless.
			// See issues 11589, 11593.
			return
		}

		lab := s.label(sym)

		// Associate label with its control flow node, if any
		if ctl := n.Name.Defn; ctl != nil {
			switch ctl.Op {
			case OFOR, OSWITCH, OSELECT:
				s.labeledNodes[ctl] = lab
			}
		}

		if !lab.defined() {
			lab.defNode = n
		} else {
			s.Error("label %v already defined at %v", sym, Ctxt.Line(int(lab.defNode.Lineno)))
			lab.reported = true
		}
		// The label might already have a target block via a goto.
		if lab.target == nil {
			lab.target = s.f.NewBlock(ssa.BlockPlain)
		}

		// go to that label (we pretend "label:" is preceded by "goto label")
		b := s.endBlock()
		addEdge(b, lab.target)
		s.startBlock(lab.target)

	case OGOTO:
		sym := n.Left.Sym

		lab := s.label(sym)
		if lab.target == nil {
			lab.target = s.f.NewBlock(ssa.BlockPlain)
		}
		if !lab.used() {
			lab.useNode = n
		}

		if lab.defined() {
			s.checkgoto(n, lab.defNode)
		} else {
			s.fwdGotos = append(s.fwdGotos, n)
		}

		b := s.endBlock()
		addEdge(b, lab.target)

	case OAS, OASWB:
		// Check whether we can generate static data rather than code.
		// If so, ignore n and defer data generation until codegen.
		// Failure to do this causes writes to readonly symbols.
		if gen_as_init(n, true) {
			var data []*Node
			if s.f.StaticData != nil {
				data = s.f.StaticData.([]*Node)
			}
			s.f.StaticData = append(data, n)
			return
		}
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
		if b := s.endBlock(); b != nil {
			addEdge(b, bEnd)
		}

		if n.Rlist != nil {
			s.startBlock(bElse)
			s.stmtList(n.Rlist)
			if b := s.endBlock(); b != nil {
				addEdge(b, bEnd)
			}
		}
		s.startBlock(bEnd)

	case ORETURN:
		s.stmtList(n.List)
		b := s.endBlock()
		addEdge(b, s.exit)

	case OCONTINUE, OBREAK:
		var op string
		var to *ssa.Block
		switch n.Op {
		case OCONTINUE:
			op = "continue"
			to = s.continueTo
		case OBREAK:
			op = "break"
			to = s.breakTo
		}
		if n.Left == nil {
			// plain break/continue
			if to == nil {
				s.Error("%s is not in a loop", op)
				return
			}
			// nothing to do; "to" is already the correct target
		} else {
			// labeled break/continue; look up the target
			sym := n.Left.Sym
			lab := s.label(sym)
			if !lab.used() {
				lab.useNode = n.Left
			}
			if !lab.defined() {
				s.Error("%s label not defined: %v", op, sym)
				lab.reported = true
				return
			}
			switch n.Op {
			case OCONTINUE:
				to = lab.continueTarget
			case OBREAK:
				to = lab.breakTarget
			}
			if to == nil {
				// Valid label but not usable with a break/continue here, e.g.:
				// for {
				// 	continue abc
				// }
				// abc:
				// for {}
				s.Error("invalid %s label %v", op, sym)
				lab.reported = true
				return
			}
		}

		b := s.endBlock()
		addEdge(b, to)

	case OFOR:
		// OFOR: for Ninit; Left; Right { Nbody }
		bCond := s.f.NewBlock(ssa.BlockPlain)
		bBody := s.f.NewBlock(ssa.BlockPlain)
		bIncr := s.f.NewBlock(ssa.BlockPlain)
		bEnd := s.f.NewBlock(ssa.BlockPlain)

		// first, jump to condition test
		b := s.endBlock()
		addEdge(b, bCond)

		// generate code to test condition
		s.startBlock(bCond)
		var cond *ssa.Value
		if n.Left != nil {
			cond = s.expr(n.Left)
		} else {
			cond = s.entryNewValue0A(ssa.OpConstBool, Types[TBOOL], true)
		}
		b = s.endBlock()
		b.Kind = ssa.BlockIf
		b.Control = cond
		// TODO(khr): likely direction
		addEdge(b, bBody)
		addEdge(b, bEnd)

		// set up for continue/break in body
		prevContinue := s.continueTo
		prevBreak := s.breakTo
		s.continueTo = bIncr
		s.breakTo = bEnd
		lab := s.labeledNodes[n]
		if lab != nil {
			// labeled for loop
			lab.continueTarget = bIncr
			lab.breakTarget = bEnd
		}

		// generate body
		s.startBlock(bBody)
		s.stmtList(n.Nbody)

		// tear down continue/break
		s.continueTo = prevContinue
		s.breakTo = prevBreak
		if lab != nil {
			lab.continueTarget = nil
			lab.breakTarget = nil
		}

		// done with body, goto incr
		if b := s.endBlock(); b != nil {
			addEdge(b, bIncr)
		}

		// generate incr
		s.startBlock(bIncr)
		if n.Right != nil {
			s.stmt(n.Right)
		}
		if b := s.endBlock(); b != nil {
			addEdge(b, bCond)
		}
		s.startBlock(bEnd)

	case OSWITCH, OSELECT:
		// These have been mostly rewritten by the front end into their Nbody fields.
		// Our main task is to correctly hook up any break statements.
		bEnd := s.f.NewBlock(ssa.BlockPlain)

		prevBreak := s.breakTo
		s.breakTo = bEnd
		lab := s.labeledNodes[n]
		if lab != nil {
			// labeled
			lab.breakTarget = bEnd
		}

		// generate body code
		s.stmtList(n.Nbody)

		s.breakTo = prevBreak
		if lab != nil {
			lab.breakTarget = nil
		}

		if b := s.endBlock(); b != nil {
			addEdge(b, bEnd)
		}
		s.startBlock(bEnd)

	case OVARKILL:
		// TODO(khr): ??? anything to do here?  Only for addrtaken variables?
		// Maybe just link it in the store chain?
	default:
		s.Unimplementedf("unhandled stmt %s", opnames[n.Op])
	}
}

type opAndType struct {
	op    uint8
	etype uint8
}

var opToSSA = map[opAndType]ssa.Op{
	opAndType{OADD, TINT8}:   ssa.OpAdd8,
	opAndType{OADD, TUINT8}:  ssa.OpAdd8,
	opAndType{OADD, TINT16}:  ssa.OpAdd16,
	opAndType{OADD, TUINT16}: ssa.OpAdd16,
	opAndType{OADD, TINT32}:  ssa.OpAdd32,
	opAndType{OADD, TUINT32}: ssa.OpAdd32,
	opAndType{OADD, TPTR32}:  ssa.OpAdd32,
	opAndType{OADD, TINT64}:  ssa.OpAdd64,
	opAndType{OADD, TUINT64}: ssa.OpAdd64,
	opAndType{OADD, TPTR64}:  ssa.OpAdd64,

	opAndType{OSUB, TINT8}:   ssa.OpSub8,
	opAndType{OSUB, TUINT8}:  ssa.OpSub8,
	opAndType{OSUB, TINT16}:  ssa.OpSub16,
	opAndType{OSUB, TUINT16}: ssa.OpSub16,
	opAndType{OSUB, TINT32}:  ssa.OpSub32,
	opAndType{OSUB, TUINT32}: ssa.OpSub32,
	opAndType{OSUB, TINT64}:  ssa.OpSub64,
	opAndType{OSUB, TUINT64}: ssa.OpSub64,

	opAndType{ONOT, TBOOL}: ssa.OpNot,

	opAndType{OMINUS, TINT8}:   ssa.OpNeg8,
	opAndType{OMINUS, TUINT8}:  ssa.OpNeg8,
	opAndType{OMINUS, TINT16}:  ssa.OpNeg16,
	opAndType{OMINUS, TUINT16}: ssa.OpNeg16,
	opAndType{OMINUS, TINT32}:  ssa.OpNeg32,
	opAndType{OMINUS, TUINT32}: ssa.OpNeg32,
	opAndType{OMINUS, TINT64}:  ssa.OpNeg64,
	opAndType{OMINUS, TUINT64}: ssa.OpNeg64,

	opAndType{OCOM, TINT8}:   ssa.OpCom8,
	opAndType{OCOM, TUINT8}:  ssa.OpCom8,
	opAndType{OCOM, TINT16}:  ssa.OpCom16,
	opAndType{OCOM, TUINT16}: ssa.OpCom16,
	opAndType{OCOM, TINT32}:  ssa.OpCom32,
	opAndType{OCOM, TUINT32}: ssa.OpCom32,
	opAndType{OCOM, TINT64}:  ssa.OpCom64,
	opAndType{OCOM, TUINT64}: ssa.OpCom64,

	opAndType{OMUL, TINT8}:   ssa.OpMul8,
	opAndType{OMUL, TUINT8}:  ssa.OpMul8,
	opAndType{OMUL, TINT16}:  ssa.OpMul16,
	opAndType{OMUL, TUINT16}: ssa.OpMul16,
	opAndType{OMUL, TINT32}:  ssa.OpMul32,
	opAndType{OMUL, TUINT32}: ssa.OpMul32,
	opAndType{OMUL, TINT64}:  ssa.OpMul64,
	opAndType{OMUL, TUINT64}: ssa.OpMul64,

	opAndType{OAND, TINT8}:   ssa.OpAnd8,
	opAndType{OAND, TUINT8}:  ssa.OpAnd8,
	opAndType{OAND, TINT16}:  ssa.OpAnd16,
	opAndType{OAND, TUINT16}: ssa.OpAnd16,
	opAndType{OAND, TINT32}:  ssa.OpAnd32,
	opAndType{OAND, TUINT32}: ssa.OpAnd32,
	opAndType{OAND, TINT64}:  ssa.OpAnd64,
	opAndType{OAND, TUINT64}: ssa.OpAnd64,

	opAndType{OOR, TINT8}:   ssa.OpOr8,
	opAndType{OOR, TUINT8}:  ssa.OpOr8,
	opAndType{OOR, TINT16}:  ssa.OpOr16,
	opAndType{OOR, TUINT16}: ssa.OpOr16,
	opAndType{OOR, TINT32}:  ssa.OpOr32,
	opAndType{OOR, TUINT32}: ssa.OpOr32,
	opAndType{OOR, TINT64}:  ssa.OpOr64,
	opAndType{OOR, TUINT64}: ssa.OpOr64,

	opAndType{OXOR, TINT8}:   ssa.OpXor8,
	opAndType{OXOR, TUINT8}:  ssa.OpXor8,
	opAndType{OXOR, TINT16}:  ssa.OpXor16,
	opAndType{OXOR, TUINT16}: ssa.OpXor16,
	opAndType{OXOR, TINT32}:  ssa.OpXor32,
	opAndType{OXOR, TUINT32}: ssa.OpXor32,
	opAndType{OXOR, TINT64}:  ssa.OpXor64,
	opAndType{OXOR, TUINT64}: ssa.OpXor64,

	opAndType{OEQ, TBOOL}:      ssa.OpEq8,
	opAndType{OEQ, TINT8}:      ssa.OpEq8,
	opAndType{OEQ, TUINT8}:     ssa.OpEq8,
	opAndType{OEQ, TINT16}:     ssa.OpEq16,
	opAndType{OEQ, TUINT16}:    ssa.OpEq16,
	opAndType{OEQ, TINT32}:     ssa.OpEq32,
	opAndType{OEQ, TUINT32}:    ssa.OpEq32,
	opAndType{OEQ, TINT64}:     ssa.OpEq64,
	opAndType{OEQ, TUINT64}:    ssa.OpEq64,
	opAndType{OEQ, TPTR64}:     ssa.OpEq64,
	opAndType{OEQ, TINTER}:     ssa.OpEqFat, // e == nil only
	opAndType{OEQ, TARRAY}:     ssa.OpEqFat, // slice only; a == nil only
	opAndType{OEQ, TFUNC}:      ssa.OpEqPtr,
	opAndType{OEQ, TMAP}:       ssa.OpEqPtr,
	opAndType{OEQ, TCHAN}:      ssa.OpEqPtr,
	opAndType{OEQ, TUINTPTR}:   ssa.OpEqPtr,
	opAndType{OEQ, TUNSAFEPTR}: ssa.OpEqPtr,

	opAndType{ONE, TBOOL}:      ssa.OpNeq8,
	opAndType{ONE, TINT8}:      ssa.OpNeq8,
	opAndType{ONE, TUINT8}:     ssa.OpNeq8,
	opAndType{ONE, TINT16}:     ssa.OpNeq16,
	opAndType{ONE, TUINT16}:    ssa.OpNeq16,
	opAndType{ONE, TINT32}:     ssa.OpNeq32,
	opAndType{ONE, TUINT32}:    ssa.OpNeq32,
	opAndType{ONE, TINT64}:     ssa.OpNeq64,
	opAndType{ONE, TUINT64}:    ssa.OpNeq64,
	opAndType{ONE, TPTR64}:     ssa.OpNeq64,
	opAndType{ONE, TINTER}:     ssa.OpNeqFat, // e != nil only
	opAndType{ONE, TARRAY}:     ssa.OpNeqFat, // slice only; a != nil only
	opAndType{ONE, TFUNC}:      ssa.OpNeqPtr,
	opAndType{ONE, TMAP}:       ssa.OpNeqPtr,
	opAndType{ONE, TCHAN}:      ssa.OpNeqPtr,
	opAndType{ONE, TUINTPTR}:   ssa.OpNeqPtr,
	opAndType{ONE, TUNSAFEPTR}: ssa.OpNeqPtr,

	opAndType{OLT, TINT8}:   ssa.OpLess8,
	opAndType{OLT, TUINT8}:  ssa.OpLess8U,
	opAndType{OLT, TINT16}:  ssa.OpLess16,
	opAndType{OLT, TUINT16}: ssa.OpLess16U,
	opAndType{OLT, TINT32}:  ssa.OpLess32,
	opAndType{OLT, TUINT32}: ssa.OpLess32U,
	opAndType{OLT, TINT64}:  ssa.OpLess64,
	opAndType{OLT, TUINT64}: ssa.OpLess64U,

	opAndType{OGT, TINT8}:   ssa.OpGreater8,
	opAndType{OGT, TUINT8}:  ssa.OpGreater8U,
	opAndType{OGT, TINT16}:  ssa.OpGreater16,
	opAndType{OGT, TUINT16}: ssa.OpGreater16U,
	opAndType{OGT, TINT32}:  ssa.OpGreater32,
	opAndType{OGT, TUINT32}: ssa.OpGreater32U,
	opAndType{OGT, TINT64}:  ssa.OpGreater64,
	opAndType{OGT, TUINT64}: ssa.OpGreater64U,

	opAndType{OLE, TINT8}:   ssa.OpLeq8,
	opAndType{OLE, TUINT8}:  ssa.OpLeq8U,
	opAndType{OLE, TINT16}:  ssa.OpLeq16,
	opAndType{OLE, TUINT16}: ssa.OpLeq16U,
	opAndType{OLE, TINT32}:  ssa.OpLeq32,
	opAndType{OLE, TUINT32}: ssa.OpLeq32U,
	opAndType{OLE, TINT64}:  ssa.OpLeq64,
	opAndType{OLE, TUINT64}: ssa.OpLeq64U,

	opAndType{OGE, TINT8}:   ssa.OpGeq8,
	opAndType{OGE, TUINT8}:  ssa.OpGeq8U,
	opAndType{OGE, TINT16}:  ssa.OpGeq16,
	opAndType{OGE, TUINT16}: ssa.OpGeq16U,
	opAndType{OGE, TINT32}:  ssa.OpGeq32,
	opAndType{OGE, TUINT32}: ssa.OpGeq32U,
	opAndType{OGE, TINT64}:  ssa.OpGeq64,
	opAndType{OGE, TUINT64}: ssa.OpGeq64U,
}

func (s *state) concreteEtype(t *Type) uint8 {
	e := t.Etype
	switch e {
	default:
		return e
	case TINT:
		if s.config.IntSize == 8 {
			return TINT64
		}
		return TINT32
	case TUINT:
		if s.config.IntSize == 8 {
			return TUINT64
		}
		return TUINT32
	case TUINTPTR:
		if s.config.PtrSize == 8 {
			return TUINT64
		}
		return TUINT32
	}
}

func (s *state) ssaOp(op uint8, t *Type) ssa.Op {
	etype := s.concreteEtype(t)
	x, ok := opToSSA[opAndType{op, etype}]
	if !ok {
		s.Unimplementedf("unhandled binary op %s etype=%s", opnames[op], Econv(int(etype), 0))
	}
	return x
}

type opAndTwoTypes struct {
	op     uint8
	etype1 uint8
	etype2 uint8
}

var shiftOpToSSA = map[opAndTwoTypes]ssa.Op{
	opAndTwoTypes{OLSH, TINT8, TUINT8}:   ssa.OpLsh8x8,
	opAndTwoTypes{OLSH, TUINT8, TUINT8}:  ssa.OpLsh8x8,
	opAndTwoTypes{OLSH, TINT8, TUINT16}:  ssa.OpLsh8x16,
	opAndTwoTypes{OLSH, TUINT8, TUINT16}: ssa.OpLsh8x16,
	opAndTwoTypes{OLSH, TINT8, TUINT32}:  ssa.OpLsh8x32,
	opAndTwoTypes{OLSH, TUINT8, TUINT32}: ssa.OpLsh8x32,
	opAndTwoTypes{OLSH, TINT8, TUINT64}:  ssa.OpLsh8x64,
	opAndTwoTypes{OLSH, TUINT8, TUINT64}: ssa.OpLsh8x64,

	opAndTwoTypes{OLSH, TINT16, TUINT8}:   ssa.OpLsh16x8,
	opAndTwoTypes{OLSH, TUINT16, TUINT8}:  ssa.OpLsh16x8,
	opAndTwoTypes{OLSH, TINT16, TUINT16}:  ssa.OpLsh16x16,
	opAndTwoTypes{OLSH, TUINT16, TUINT16}: ssa.OpLsh16x16,
	opAndTwoTypes{OLSH, TINT16, TUINT32}:  ssa.OpLsh16x32,
	opAndTwoTypes{OLSH, TUINT16, TUINT32}: ssa.OpLsh16x32,
	opAndTwoTypes{OLSH, TINT16, TUINT64}:  ssa.OpLsh16x64,
	opAndTwoTypes{OLSH, TUINT16, TUINT64}: ssa.OpLsh16x64,

	opAndTwoTypes{OLSH, TINT32, TUINT8}:   ssa.OpLsh32x8,
	opAndTwoTypes{OLSH, TUINT32, TUINT8}:  ssa.OpLsh32x8,
	opAndTwoTypes{OLSH, TINT32, TUINT16}:  ssa.OpLsh32x16,
	opAndTwoTypes{OLSH, TUINT32, TUINT16}: ssa.OpLsh32x16,
	opAndTwoTypes{OLSH, TINT32, TUINT32}:  ssa.OpLsh32x32,
	opAndTwoTypes{OLSH, TUINT32, TUINT32}: ssa.OpLsh32x32,
	opAndTwoTypes{OLSH, TINT32, TUINT64}:  ssa.OpLsh32x64,
	opAndTwoTypes{OLSH, TUINT32, TUINT64}: ssa.OpLsh32x64,

	opAndTwoTypes{OLSH, TINT64, TUINT8}:   ssa.OpLsh64x8,
	opAndTwoTypes{OLSH, TUINT64, TUINT8}:  ssa.OpLsh64x8,
	opAndTwoTypes{OLSH, TINT64, TUINT16}:  ssa.OpLsh64x16,
	opAndTwoTypes{OLSH, TUINT64, TUINT16}: ssa.OpLsh64x16,
	opAndTwoTypes{OLSH, TINT64, TUINT32}:  ssa.OpLsh64x32,
	opAndTwoTypes{OLSH, TUINT64, TUINT32}: ssa.OpLsh64x32,
	opAndTwoTypes{OLSH, TINT64, TUINT64}:  ssa.OpLsh64x64,
	opAndTwoTypes{OLSH, TUINT64, TUINT64}: ssa.OpLsh64x64,

	opAndTwoTypes{ORSH, TINT8, TUINT8}:   ssa.OpRsh8x8,
	opAndTwoTypes{ORSH, TUINT8, TUINT8}:  ssa.OpRsh8Ux8,
	opAndTwoTypes{ORSH, TINT8, TUINT16}:  ssa.OpRsh8x16,
	opAndTwoTypes{ORSH, TUINT8, TUINT16}: ssa.OpRsh8Ux16,
	opAndTwoTypes{ORSH, TINT8, TUINT32}:  ssa.OpRsh8x32,
	opAndTwoTypes{ORSH, TUINT8, TUINT32}: ssa.OpRsh8Ux32,
	opAndTwoTypes{ORSH, TINT8, TUINT64}:  ssa.OpRsh8x64,
	opAndTwoTypes{ORSH, TUINT8, TUINT64}: ssa.OpRsh8Ux64,

	opAndTwoTypes{ORSH, TINT16, TUINT8}:   ssa.OpRsh16x8,
	opAndTwoTypes{ORSH, TUINT16, TUINT8}:  ssa.OpRsh16Ux8,
	opAndTwoTypes{ORSH, TINT16, TUINT16}:  ssa.OpRsh16x16,
	opAndTwoTypes{ORSH, TUINT16, TUINT16}: ssa.OpRsh16Ux16,
	opAndTwoTypes{ORSH, TINT16, TUINT32}:  ssa.OpRsh16x32,
	opAndTwoTypes{ORSH, TUINT16, TUINT32}: ssa.OpRsh16Ux32,
	opAndTwoTypes{ORSH, TINT16, TUINT64}:  ssa.OpRsh16x64,
	opAndTwoTypes{ORSH, TUINT16, TUINT64}: ssa.OpRsh16Ux64,

	opAndTwoTypes{ORSH, TINT32, TUINT8}:   ssa.OpRsh32x8,
	opAndTwoTypes{ORSH, TUINT32, TUINT8}:  ssa.OpRsh32Ux8,
	opAndTwoTypes{ORSH, TINT32, TUINT16}:  ssa.OpRsh32x16,
	opAndTwoTypes{ORSH, TUINT32, TUINT16}: ssa.OpRsh32Ux16,
	opAndTwoTypes{ORSH, TINT32, TUINT32}:  ssa.OpRsh32x32,
	opAndTwoTypes{ORSH, TUINT32, TUINT32}: ssa.OpRsh32Ux32,
	opAndTwoTypes{ORSH, TINT32, TUINT64}:  ssa.OpRsh32x64,
	opAndTwoTypes{ORSH, TUINT32, TUINT64}: ssa.OpRsh32Ux64,

	opAndTwoTypes{ORSH, TINT64, TUINT8}:   ssa.OpRsh64x8,
	opAndTwoTypes{ORSH, TUINT64, TUINT8}:  ssa.OpRsh64Ux8,
	opAndTwoTypes{ORSH, TINT64, TUINT16}:  ssa.OpRsh64x16,
	opAndTwoTypes{ORSH, TUINT64, TUINT16}: ssa.OpRsh64Ux16,
	opAndTwoTypes{ORSH, TINT64, TUINT32}:  ssa.OpRsh64x32,
	opAndTwoTypes{ORSH, TUINT64, TUINT32}: ssa.OpRsh64Ux32,
	opAndTwoTypes{ORSH, TINT64, TUINT64}:  ssa.OpRsh64x64,
	opAndTwoTypes{ORSH, TUINT64, TUINT64}: ssa.OpRsh64Ux64,
}

func (s *state) ssaShiftOp(op uint8, t *Type, u *Type) ssa.Op {
	etype1 := s.concreteEtype(t)
	etype2 := s.concreteEtype(u)
	x, ok := shiftOpToSSA[opAndTwoTypes{op, etype1, etype2}]
	if !ok {
		s.Unimplementedf("unhandled shift op %s etype=%s/%s", opnames[op], Econv(int(etype1), 0), Econv(int(etype2), 0))
	}
	return x
}

// expr converts the expression n to ssa, adds it to s and returns the ssa result.
func (s *state) expr(n *Node) *ssa.Value {
	s.pushLine(n.Lineno)
	defer s.popLine()

	s.stmtList(n.Ninit)
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
			i := Mpgetfix(n.Val().U.(*Mpint))
			switch n.Type.Size() {
			case 1:
				return s.constInt8(n.Type, int8(i))
			case 2:
				return s.constInt16(n.Type, int16(i))
			case 4:
				return s.constInt32(n.Type, int32(i))
			case 8:
				return s.constInt64(n.Type, i)
			default:
				s.Fatalf("bad integer size %d", n.Type.Size())
				return nil
			}
		case CTSTR:
			return s.entryNewValue0A(ssa.OpConstString, n.Type, n.Val().U)
		case CTBOOL:
			return s.entryNewValue0A(ssa.OpConstBool, n.Type, n.Val().U)
		case CTNIL:
			return s.entryNewValue0(ssa.OpConstNil, n.Type)
		default:
			s.Unimplementedf("unhandled OLITERAL %v", n.Val().Ctype())
			return nil
		}
	case OCONVNOP:
		to := n.Type
		from := n.Left.Type
		if to.Etype == TFUNC {
			s.Unimplementedf("CONVNOP closure %v -> %v", n.Type, n.Left.Type)
			return nil
		}

		// Assume everything will work out, so set up our return value.
		// Anything interesting that happens from here is a fatal.
		x := s.expr(n.Left)
		v := s.newValue1(ssa.OpCopy, to, x) // ensure that v has the right type

		// named <--> unnamed type or typed <--> untyped const
		if from.Etype == to.Etype {
			return v
		}
		// unsafe.Pointer <--> *T
		if to.Etype == TUNSAFEPTR && from.IsPtr() || from.Etype == TUNSAFEPTR && to.IsPtr() {
			return v
		}

		dowidth(from)
		dowidth(to)
		if from.Width != to.Width {
			s.Fatalf("CONVNOP width mismatch %v (%d) -> %v (%d)\n", from, from.Width, to, to.Width)
			return nil
		}
		if etypesign(from.Etype) != etypesign(to.Etype) {
			s.Fatalf("CONVNOP sign mismatch %v (%s) -> %v (%s)\n", from, Econv(int(from.Etype), 0), to, Econv(int(to.Etype), 0))
			return nil
		}

		if flag_race != 0 {
			s.Unimplementedf("questionable CONVNOP from race detector %v -> %v\n", from, to)
			return nil
		}

		if etypesign(from.Etype) == 0 {
			s.Fatalf("CONVNOP unrecognized non-integer %v -> %v\n", from, to)
			return nil
		}

		// integer, same width, same sign
		return v

	case OCONV:
		x := s.expr(n.Left)
		ft := n.Left.Type // from type
		tt := n.Type      // to type
		if ft.IsInteger() && tt.IsInteger() {
			var op ssa.Op
			if tt.Size() == ft.Size() {
				op = ssa.OpCopy
			} else if tt.Size() < ft.Size() {
				// truncation
				switch 10*ft.Size() + tt.Size() {
				case 21:
					op = ssa.OpTrunc16to8
				case 41:
					op = ssa.OpTrunc32to8
				case 42:
					op = ssa.OpTrunc32to16
				case 81:
					op = ssa.OpTrunc64to8
				case 82:
					op = ssa.OpTrunc64to16
				case 84:
					op = ssa.OpTrunc64to32
				default:
					s.Fatalf("weird integer truncation %s -> %s", ft, tt)
				}
			} else if ft.IsSigned() {
				// sign extension
				switch 10*ft.Size() + tt.Size() {
				case 12:
					op = ssa.OpSignExt8to16
				case 14:
					op = ssa.OpSignExt8to32
				case 18:
					op = ssa.OpSignExt8to64
				case 24:
					op = ssa.OpSignExt16to32
				case 28:
					op = ssa.OpSignExt16to64
				case 48:
					op = ssa.OpSignExt32to64
				default:
					s.Fatalf("bad integer sign extension %s -> %s", ft, tt)
				}
			} else {
				// zero extension
				switch 10*ft.Size() + tt.Size() {
				case 12:
					op = ssa.OpZeroExt8to16
				case 14:
					op = ssa.OpZeroExt8to32
				case 18:
					op = ssa.OpZeroExt8to64
				case 24:
					op = ssa.OpZeroExt16to32
				case 28:
					op = ssa.OpZeroExt16to64
				case 48:
					op = ssa.OpZeroExt32to64
				default:
					s.Fatalf("weird integer sign extension %s -> %s", ft, tt)
				}
			}
			return s.newValue1(op, n.Type, x)
		}
		s.Unimplementedf("unhandled OCONV %s -> %s", n.Left.Type, n.Type)
		return nil

	// binary ops
	case OLT, OEQ, ONE, OLE, OGE, OGT:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.newValue2(s.ssaOp(n.Op, n.Left.Type), Types[TBOOL], a, b)
	case OADD, OAND, OMUL, OOR, OSUB, OXOR:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.newValue2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
	case OLSH, ORSH:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.newValue2(s.ssaShiftOp(n.Op, n.Type, n.Right.Type), a.Type, a, b)
	case OANDAND, OOROR:
		// To implement OANDAND (and OOROR), we introduce a
		// new temporary variable to hold the result. The
		// variable is associated with the OANDAND node in the
		// s.vars table (normally variables are only
		// associated with ONAME nodes). We convert
		//     A && B
		// to
		//     var = A
		//     if var {
		//         var = B
		//     }
		// Using var in the subsequent block introduces the
		// necessary phi variable.
		el := s.expr(n.Left)
		s.vars[n] = el

		b := s.endBlock()
		b.Kind = ssa.BlockIf
		b.Control = el

		bRight := s.f.NewBlock(ssa.BlockPlain)
		bResult := s.f.NewBlock(ssa.BlockPlain)
		if n.Op == OANDAND {
			addEdge(b, bRight)
			addEdge(b, bResult)
		} else if n.Op == OOROR {
			addEdge(b, bResult)
			addEdge(b, bRight)
		}

		s.startBlock(bRight)
		er := s.expr(n.Right)
		s.vars[n] = er

		b = s.endBlock()
		addEdge(b, bResult)

		s.startBlock(bResult)
		return s.variable(n, n.Type)

	// unary ops
	case ONOT, OMINUS, OCOM:
		a := s.expr(n.Left)
		return s.newValue1(s.ssaOp(n.Op, n.Type), a.Type, a)

	case OADDR:
		return s.addr(n.Left)

	case OINDREG:
		if int(n.Reg) != Thearch.REGSP {
			s.Unimplementedf("OINDREG of non-SP register %s in expr: %v", obj.Rconv(int(n.Reg)), n)
			return nil
		}
		addr := s.entryNewValue1I(ssa.OpOffPtr, Ptrto(n.Type), n.Xoffset, s.sp)
		return s.newValue2(ssa.OpLoad, n.Type, addr, s.mem())

	case OIND:
		p := s.expr(n.Left)
		s.nilCheck(p)
		return s.newValue2(ssa.OpLoad, n.Type, p, s.mem())

	case ODOT:
		v := s.expr(n.Left)
		return s.newValue1I(ssa.OpStructSelect, n.Type, n.Xoffset, v)

	case ODOTPTR:
		p := s.expr(n.Left)
		s.nilCheck(p)
		p = s.newValue2(ssa.OpAddPtr, p.Type, p, s.constIntPtr(Types[TUINTPTR], n.Xoffset))
		return s.newValue2(ssa.OpLoad, n.Type, p, s.mem())

	case OINDEX:
		if n.Left.Type.Bound >= 0 { // array or string
			a := s.expr(n.Left)
			i := s.expr(n.Right)
			i = s.extendIndex(i)
			var elemtype *Type
			var len *ssa.Value
			if n.Left.Type.IsString() {
				len = s.newValue1(ssa.OpStringLen, Types[TINT], a)
				elemtype = Types[TUINT8]
			} else {
				len = s.constInt(Types[TINT], n.Left.Type.Bound)
				elemtype = n.Left.Type.Type
			}
			s.boundsCheck(i, len)
			return s.newValue2(ssa.OpArrayIndex, elemtype, a, i)
		} else { // slice
			p := s.addr(n)
			return s.newValue2(ssa.OpLoad, n.Left.Type.Type, p, s.mem())
		}

	case OLEN, OCAP:
		switch {
		case n.Left.Type.IsSlice():
			op := ssa.OpSliceLen
			if n.Op == OCAP {
				op = ssa.OpSliceCap
			}
			return s.newValue1(op, Types[TINT], s.expr(n.Left))
		case n.Left.Type.IsString(): // string; not reachable for OCAP
			return s.newValue1(ssa.OpStringLen, Types[TINT], s.expr(n.Left))
		default: // array
			return s.constInt(Types[TINT], n.Left.Type.Bound)
		}

	case OCALLFUNC, OCALLMETH:
		left := n.Left
		static := left.Op == ONAME && left.Class == PFUNC

		if n.Op == OCALLMETH {
			// Rewrite to an OCALLFUNC: (p.f)(...) becomes (f)(p, ...)
			// Take care not to modify the original AST.
			if left.Op != ODOTMETH {
				Fatal("OCALLMETH: n.Left not an ODOTMETH: %v", left)
			}

			newLeft := *left.Right
			newLeft.Type = left.Type
			if newLeft.Op == ONAME {
				newLeft.Class = PFUNC
			}
			left = &newLeft
			static = true
		}

		// evaluate closure
		var closure *ssa.Value
		if !static {
			closure = s.expr(left)
		}

		// run all argument assignments
		s.stmtList(n.List)

		bNext := s.f.NewBlock(ssa.BlockPlain)
		var call *ssa.Value
		if static {
			call = s.newValue1A(ssa.OpStaticCall, ssa.TypeMem, left.Sym, s.mem())
		} else {
			entry := s.newValue2(ssa.OpLoad, Types[TUINTPTR], closure, s.mem())
			call = s.newValue3(ssa.OpClosureCall, ssa.TypeMem, entry, closure, s.mem())
		}
		dowidth(left.Type)
		call.AuxInt = left.Type.Argwid // call operations carry the argsize of the callee along with them
		b := s.endBlock()
		b.Kind = ssa.BlockCall
		b.Control = call
		addEdge(b, bNext)
		addEdge(b, s.exit)

		// read result from stack at the start of the fallthrough block
		s.startBlock(bNext)
		var titer Iter
		fp := Structfirst(&titer, Getoutarg(left.Type))
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
		val = s.zeroVal(t)
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

// zeroVal returns the zero value for type t.
func (s *state) zeroVal(t *Type) *ssa.Value {
	switch {
	case t.IsInteger():
		switch t.Size() {
		case 1:
			return s.constInt8(t, 0)
		case 2:
			return s.constInt16(t, 0)
		case 4:
			return s.constInt32(t, 0)
		case 8:
			return s.constInt64(t, 0)
		default:
			s.Fatalf("bad sized integer type %s", t)
		}
	case t.IsString():
		return s.entryNewValue0A(ssa.OpConstString, t, "")
	case t.IsPtr():
		return s.entryNewValue0(ssa.OpConstNil, t)
	case t.IsBoolean():
		return s.entryNewValue0A(ssa.OpConstBool, t, false) // TODO: store bools as 0/1 in AuxInt?
	}
	s.Unimplementedf("zero for type %v not implemented", t)
	return nil
}

// etypesign returns the signed-ness of e, for integer/pointer etypes.
// -1 means signed, +1 means unsigned, 0 means non-integer/non-pointer.
func etypesign(e uint8) int8 {
	switch e {
	case TINT8, TINT16, TINT32, TINT64, TINT:
		return -1
	case TUINT8, TUINT16, TUINT32, TUINT64, TUINT, TUINTPTR, TUNSAFEPTR:
		return +1
	}
	return 0
}

// addr converts the address of the expression n to SSA, adds it to s and returns the SSA result.
// The value that the returned Value represents is guaranteed to be non-nil.
func (s *state) addr(n *Node) *ssa.Value {
	switch n.Op {
	case ONAME:
		switch n.Class {
		case PEXTERN:
			// global variable
			aux := &ssa.ExternSymbol{n.Type, n.Sym}
			v := s.entryNewValue1A(ssa.OpAddr, Ptrto(n.Type), aux, s.sb)
			// TODO: Make OpAddr use AuxInt as well as Aux.
			if n.Xoffset != 0 {
				v = s.entryNewValue1I(ssa.OpOffPtr, v.Type, n.Xoffset, v)
			}
			return v
		case PPARAM, PPARAMOUT, PAUTO:
			// parameter/result slot or local variable
			v := s.decladdrs[n]
			if v == nil {
				if flag_race != 0 && n.String() == ".fp" {
					s.Unimplementedf("race detector mishandles nodfp")
				}
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
		// indirect off a register
		// used for storing/loading arguments/returns to/from callees
		if int(n.Reg) != Thearch.REGSP {
			s.Unimplementedf("OINDREG of non-SP register %s in addr: %v", obj.Rconv(int(n.Reg)), n)
			return nil
		}
		return s.entryNewValue1I(ssa.OpOffPtr, Ptrto(n.Type), n.Xoffset, s.sp)
	case OINDEX:
		if n.Left.Type.IsSlice() {
			a := s.expr(n.Left)
			i := s.expr(n.Right)
			i = s.extendIndex(i)
			len := s.newValue1(ssa.OpSliceLen, Types[TUINTPTR], a)
			s.boundsCheck(i, len)
			p := s.newValue1(ssa.OpSlicePtr, Ptrto(n.Left.Type.Type), a)
			return s.newValue2(ssa.OpPtrIndex, Ptrto(n.Left.Type.Type), p, i)
		} else { // array
			a := s.addr(n.Left)
			i := s.expr(n.Right)
			i = s.extendIndex(i)
			len := s.constInt(Types[TINT], n.Left.Type.Bound)
			s.boundsCheck(i, len)
			return s.newValue2(ssa.OpPtrIndex, Ptrto(n.Left.Type.Type), a, i)
		}
	case OIND:
		p := s.expr(n.Left)
		s.nilCheck(p)
		return p
	case ODOT:
		p := s.addr(n.Left)
		return s.newValue2(ssa.OpAddPtr, p.Type, p, s.constIntPtr(Types[TUINTPTR], n.Xoffset))
	case ODOTPTR:
		p := s.expr(n.Left)
		s.nilCheck(p)
		return s.newValue2(ssa.OpAddPtr, p.Type, p, s.constIntPtr(Types[TUINTPTR], n.Xoffset))
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
// Used only for automatically inserted nil checks,
// not for user code like 'x != nil'.
func (s *state) nilCheck(ptr *ssa.Value) {
	c := s.newValue1(ssa.OpIsNonNil, Types[TBOOL], ptr)
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
	cmp := s.newValue2(ssa.OpIsInBounds, Types[TBOOL], idx, len)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = cmp
	bNext := s.f.NewBlock(ssa.BlockPlain)
	addEdge(b, bNext)
	addEdge(b, s.exit)
	// TODO: don't go directly to s.exit.  Go to a stub that calls panicindex first.
	s.startBlock(bNext)
}

// checkgoto checks that a goto from from to to does not
// jump into a block or jump over variable declarations.
// It is a copy of checkgoto in the pre-SSA backend,
// modified only for line number handling.
// TODO: document how this works and why it is designed the way it is.
func (s *state) checkgoto(from *Node, to *Node) {
	if from.Sym == to.Sym {
		return
	}

	nf := 0
	for fs := from.Sym; fs != nil; fs = fs.Link {
		nf++
	}
	nt := 0
	for fs := to.Sym; fs != nil; fs = fs.Link {
		nt++
	}
	fs := from.Sym
	for ; nf > nt; nf-- {
		fs = fs.Link
	}
	if fs != to.Sym {
		// decide what to complain about.
		// prefer to complain about 'into block' over declarations,
		// so scan backward to find most recent block or else dcl.
		var block *Sym

		var dcl *Sym
		ts := to.Sym
		for ; nt > nf; nt-- {
			if ts.Pkg == nil {
				block = ts
			} else {
				dcl = ts
			}
			ts = ts.Link
		}

		for ts != fs {
			if ts.Pkg == nil {
				block = ts
			} else {
				dcl = ts
			}
			ts = ts.Link
			fs = fs.Link
		}

		lno := int(from.Left.Lineno)
		if block != nil {
			yyerrorl(lno, "goto %v jumps into block starting at %v", from.Left.Sym, Ctxt.Line(int(block.Lastlineno)))
		} else {
			yyerrorl(lno, "goto %v jumps over declaration of %v at %v", from.Left.Sym, dcl, Ctxt.Line(int(dcl.Lastlineno)))
		}
	}
}

// variable returns the value of a variable at the current location.
func (s *state) variable(name *Node, t ssa.Type) *ssa.Value {
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
		// This block is dead; we have no predecessors and we're not the entry block.
		// It doesn't matter what we use here as long as it is well-formed,
		// so use the default/zero value.
		if name == &memvar {
			return s.startmem
		}
		return s.zeroVal(name.Type)
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

	e := f.Config.Frontend().(*ssaExport)
	// We're about to emit a bunch of Progs.
	// Since the only way to get here is to explicitly request it,
	// just fail on unimplemented instead of trying to unwind our mess.
	e.mustImplement = true

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

	// Emit static data
	if f.StaticData != nil {
		for _, n := range f.StaticData.([]*Node) {
			if !gen_as_init(n, false) {
				Fatal("non-static data marked as static: %v\n\n", n, f)
			}
		}
	}

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
	// 2-address opcode arithmetic, symmetric
	case ssa.OpAMD64ADDB,
		ssa.OpAMD64ANDQ, ssa.OpAMD64ANDL, ssa.OpAMD64ANDW, ssa.OpAMD64ANDB,
		ssa.OpAMD64ORQ, ssa.OpAMD64ORL, ssa.OpAMD64ORW, ssa.OpAMD64ORB,
		ssa.OpAMD64XORQ, ssa.OpAMD64XORL, ssa.OpAMD64XORW, ssa.OpAMD64XORB,
		ssa.OpAMD64MULQ, ssa.OpAMD64MULL, ssa.OpAMD64MULW:
		r := regnum(v)
		x := regnum(v.Args[0])
		y := regnum(v.Args[1])
		if x != r && y != r {
			p := Prog(regMoveAMD64(v.Type.Size()))
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
	// 2-address opcode arithmetic, not symmetric
	case ssa.OpAMD64SUBQ, ssa.OpAMD64SUBL, ssa.OpAMD64SUBW, ssa.OpAMD64SUBB:
		r := regnum(v)
		x := regnum(v.Args[0])
		y := regnum(v.Args[1])
		var neg bool
		if y == r {
			// compute -(y-x) instead
			x, y = y, x
			neg = true
		}
		if x != r {
			p := Prog(regMoveAMD64(v.Type.Size()))
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}

		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p.From.Reg = y
		if neg {
			p := Prog(x86.ANEGQ) // TODO: use correct size?  This is mostly a hack until regalloc does 2-address correctly
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
	case ssa.OpAMD64SHLQ, ssa.OpAMD64SHLL, ssa.OpAMD64SHLW, ssa.OpAMD64SHLB,
		ssa.OpAMD64SHRQ, ssa.OpAMD64SHRL, ssa.OpAMD64SHRW, ssa.OpAMD64SHRB,
		ssa.OpAMD64SARQ, ssa.OpAMD64SARL, ssa.OpAMD64SARW, ssa.OpAMD64SARB:
		x := regnum(v.Args[0])
		r := regnum(v)
		if x != r {
			if r == x86.REG_CX {
				v.Fatalf("can't implement %s, target and shift both in CX", v.LongString())
			}
			p := Prog(regMoveAMD64(v.Type.Size()))
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[1]) // should be CX
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpAMD64ADDQconst, ssa.OpAMD64ADDLconst, ssa.OpAMD64ADDWconst:
		// TODO: use addq instead of leaq if target is in the right register.
		var asm int
		switch v.Op {
		case ssa.OpAMD64ADDQconst:
			asm = x86.ALEAQ
		case ssa.OpAMD64ADDLconst:
			asm = x86.ALEAL
		case ssa.OpAMD64ADDWconst:
			asm = x86.ALEAW
		}
		p := Prog(asm)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64MULQconst, ssa.OpAMD64MULLconst, ssa.OpAMD64MULWconst:
		r := regnum(v)
		x := regnum(v.Args[0])
		if r != x {
			p := Prog(regMoveAMD64(v.Type.Size()))
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		// TODO: Teach doasm to compile the three-address multiply imul $c, r1, r2
		// instead of using the MOVQ above.
		//p.From3 = new(obj.Addr)
		//p.From3.Type = obj.TYPE_REG
		//p.From3.Reg = regnum(v.Args[0])
	case ssa.OpAMD64ADDBconst,
		ssa.OpAMD64ANDQconst, ssa.OpAMD64ANDLconst, ssa.OpAMD64ANDWconst, ssa.OpAMD64ANDBconst,
		ssa.OpAMD64ORQconst, ssa.OpAMD64ORLconst, ssa.OpAMD64ORWconst, ssa.OpAMD64ORBconst,
		ssa.OpAMD64XORQconst, ssa.OpAMD64XORLconst, ssa.OpAMD64XORWconst, ssa.OpAMD64XORBconst,
		ssa.OpAMD64SUBQconst, ssa.OpAMD64SUBLconst, ssa.OpAMD64SUBWconst, ssa.OpAMD64SUBBconst,
		ssa.OpAMD64SHLQconst, ssa.OpAMD64SHLLconst, ssa.OpAMD64SHLWconst, ssa.OpAMD64SHLBconst,
		ssa.OpAMD64SHRQconst, ssa.OpAMD64SHRLconst, ssa.OpAMD64SHRWconst, ssa.OpAMD64SHRBconst,
		ssa.OpAMD64SARQconst, ssa.OpAMD64SARLconst, ssa.OpAMD64SARWconst, ssa.OpAMD64SARBconst:
		// This code compensates for the fact that the register allocator
		// doesn't understand 2-address instructions yet.  TODO: fix that.
		x := regnum(v.Args[0])
		r := regnum(v)
		if x != r {
			p := Prog(regMoveAMD64(v.Type.Size()))
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpAMD64SBBQcarrymask, ssa.OpAMD64SBBLcarrymask:
		r := regnum(v)
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpAMD64LEAQ1, ssa.OpAMD64LEAQ2, ssa.OpAMD64LEAQ4, ssa.OpAMD64LEAQ8:
		p := Prog(x86.ALEAQ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		switch v.Op {
		case ssa.OpAMD64LEAQ1:
			p.From.Scale = 1
		case ssa.OpAMD64LEAQ2:
			p.From.Scale = 2
		case ssa.OpAMD64LEAQ4:
			p.From.Scale = 4
		case ssa.OpAMD64LEAQ8:
			p.From.Scale = 8
		}
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
	case ssa.OpAMD64CMPQ, ssa.OpAMD64CMPL, ssa.OpAMD64CMPW, ssa.OpAMD64CMPB,
		ssa.OpAMD64TESTQ, ssa.OpAMD64TESTL, ssa.OpAMD64TESTW, ssa.OpAMD64TESTB:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v.Args[1])
	case ssa.OpAMD64CMPQconst, ssa.OpAMD64CMPLconst, ssa.OpAMD64CMPWconst, ssa.OpAMD64CMPBconst,
		ssa.OpAMD64TESTQconst, ssa.OpAMD64TESTLconst, ssa.OpAMD64TESTWconst, ssa.OpAMD64TESTBconst:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[0])
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssa.OpAMD64MOVBconst, ssa.OpAMD64MOVWconst, ssa.OpAMD64MOVLconst, ssa.OpAMD64MOVQconst:
		x := regnum(v)
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		var i int64
		switch v.Op {
		case ssa.OpAMD64MOVBconst:
			i = int64(int8(v.AuxInt))
		case ssa.OpAMD64MOVWconst:
			i = int64(int16(v.AuxInt))
		case ssa.OpAMD64MOVLconst:
			i = int64(int32(v.AuxInt))
		case ssa.OpAMD64MOVQconst:
			i = v.AuxInt
		}
		p.From.Offset = i
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssa.OpAMD64MOVQload, ssa.OpAMD64MOVLload, ssa.OpAMD64MOVWload, ssa.OpAMD64MOVBload, ssa.OpAMD64MOVBQSXload, ssa.OpAMD64MOVBQZXload:
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
	case ssa.OpAMD64MOVQstoreidx8:
		p := Prog(x86.AMOVQ)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[2])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = regnum(v.Args[0])
		p.To.Scale = 8
		p.To.Index = regnum(v.Args[1])
		addAux(&p.To, v)
	case ssa.OpAMD64MOVLQSX, ssa.OpAMD64MOVWQSX, ssa.OpAMD64MOVBQSX, ssa.OpAMD64MOVLQZX, ssa.OpAMD64MOVWQZX, ssa.OpAMD64MOVBQZX:
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
	case ssa.OpLoadReg:
		if v.Type.IsFlags() {
			v.Unimplementedf("load flags not implemented: %v", v.LongString())
			return
		}
		p := Prog(movSize(v.Type.Size()))
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = x86.REG_SP
		p.From.Offset = localOffset(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Unimplementedf("store flags not implemented: %v", v.LongString())
			return
		}
		p := Prog(movSize(v.Type.Size()))
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
	case ssa.OpConst8, ssa.OpConst16, ssa.OpConst32, ssa.OpConst64, ssa.OpConstString, ssa.OpConstNil, ssa.OpConstBool:
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
	case ssa.OpAMD64NEGQ, ssa.OpAMD64NEGL, ssa.OpAMD64NEGW, ssa.OpAMD64NEGB,
		ssa.OpAMD64NOTQ, ssa.OpAMD64NOTL, ssa.OpAMD64NOTW, ssa.OpAMD64NOTB:
		x := regnum(v.Args[0])
		r := regnum(v)
		if x != r {
			p := Prog(regMoveAMD64(v.Type.Size()))
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
		p := Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpSP, ssa.OpSB:
		// nothing to do
	case ssa.OpAMD64SETEQ, ssa.OpAMD64SETNE,
		ssa.OpAMD64SETL, ssa.OpAMD64SETLE,
		ssa.OpAMD64SETG, ssa.OpAMD64SETGE,
		ssa.OpAMD64SETB, ssa.OpAMD64SETBE,
		ssa.OpAMD64SETA, ssa.OpAMD64SETAE:
		p := Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v)
	case ssa.OpAMD64REPSTOSQ:
		Prog(x86.AREP)
		Prog(x86.ASTOSQ)
		v.Unimplementedf("REPSTOSQ clobbers not implemented: %s", v.LongString())
	case ssa.OpAMD64REPMOVSB:
		Prog(x86.AREP)
		Prog(x86.AMOVSB)
		v.Unimplementedf("REPMOVSB clobbers not implemented: %s", v.LongString())
	default:
		v.Unimplementedf("genValue not implemented: %s", v.LongString())
	}
}

// movSize returns the MOV instruction of the given width.
func movSize(width int64) (asm int) {
	switch width {
	case 1:
		asm = x86.AMOVB
	case 2:
		asm = x86.AMOVW
	case 4:
		asm = x86.AMOVL
	case 8:
		asm = x86.AMOVQ
	default:
		panic(fmt.Errorf("bad movSize %d", width))
	}
	return asm
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

var blockJump = [...]struct{ asm, invasm int }{
	ssa.BlockAMD64EQ:  {x86.AJEQ, x86.AJNE},
	ssa.BlockAMD64NE:  {x86.AJNE, x86.AJEQ},
	ssa.BlockAMD64LT:  {x86.AJLT, x86.AJGE},
	ssa.BlockAMD64GE:  {x86.AJGE, x86.AJLT},
	ssa.BlockAMD64LE:  {x86.AJLE, x86.AJGT},
	ssa.BlockAMD64GT:  {x86.AJGT, x86.AJLE},
	ssa.BlockAMD64ULT: {x86.AJCS, x86.AJCC},
	ssa.BlockAMD64UGE: {x86.AJCC, x86.AJCS},
	ssa.BlockAMD64UGT: {x86.AJHI, x86.AJLS},
	ssa.BlockAMD64ULE: {x86.AJLS, x86.AJHI},
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
	case ssa.BlockAMD64EQ, ssa.BlockAMD64NE,
		ssa.BlockAMD64LT, ssa.BlockAMD64GE,
		ssa.BlockAMD64LE, ssa.BlockAMD64GT,
		ssa.BlockAMD64ULT, ssa.BlockAMD64UGT,
		ssa.BlockAMD64ULE, ssa.BlockAMD64UGE:

		jmp := blockJump[b.Kind]
		switch next {
		case b.Succs[0]:
			p := Prog(jmp.invasm)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[1]})
		case b.Succs[1]:
			p := Prog(jmp.asm)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
		default:
			p := Prog(jmp.asm)
			p.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{p, b.Succs[0]})
			q := Prog(obj.AJMP)
			q.To.Type = obj.TYPE_BRANCH
			branches = append(branches, branch{q, b.Succs[1]})
		}

	default:
		b.Unimplementedf("branch not implemented: %s. Control: %s", b.LongString(), b.Control.LongString())
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

// extendIndex extends v to a full pointer width.
func (s *state) extendIndex(v *ssa.Value) *ssa.Value {
	size := v.Type.Size()
	if size == s.config.PtrSize {
		return v
	}
	if size > s.config.PtrSize {
		// TODO: truncate 64-bit indexes on 32-bit pointer archs.  We'd need to test
		// the high word and branch to out-of-bounds failure if it is not 0.
		s.Unimplementedf("64->32 index truncation not implemented")
		return v
	}

	// Extend value to the required size
	var op ssa.Op
	if v.Type.IsSigned() {
		switch 10*size + s.config.PtrSize {
		case 14:
			op = ssa.OpSignExt8to32
		case 18:
			op = ssa.OpSignExt8to64
		case 24:
			op = ssa.OpSignExt16to32
		case 28:
			op = ssa.OpSignExt16to64
		case 48:
			op = ssa.OpSignExt32to64
		default:
			s.Fatalf("bad signed index extension %s", v.Type)
		}
	} else {
		switch 10*size + s.config.PtrSize {
		case 14:
			op = ssa.OpZeroExt8to32
		case 18:
			op = ssa.OpZeroExt8to64
		case 24:
			op = ssa.OpZeroExt16to32
		case 28:
			op = ssa.OpZeroExt16to64
		case 48:
			op = ssa.OpZeroExt32to64
		default:
			s.Fatalf("bad unsigned index extension %s", v.Type)
		}
	}
	return s.newValue1(op, Types[TUINTPTR], v)
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

// regMoveAMD64 returns the register->register move opcode for the given width.
// TODO: generalize for all architectures?
func regMoveAMD64(width int64) int {
	switch width {
	case 1:
		return x86.AMOVB
	case 2:
		return x86.AMOVW
	case 4:
		return x86.AMOVL
	case 8:
		return x86.AMOVQ
	default:
		panic("bad register width")
	}
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
// TODO: Make this panic again once it stops happening routinely.
func localOffset(v *ssa.Value) int64 {
	reg := v.Block.Func.RegAlloc[v.ID]
	slot, ok := reg.(*ssa.LocalSlot)
	if !ok {
		v.Unimplementedf("localOffset of non-LocalSlot value: %s", v.LongString())
		return 0
	}
	return slot.Idx
}

// ssaExport exports a bunch of compiler services for the ssa backend.
type ssaExport struct {
	log           bool
	unimplemented bool
	mustImplement bool
}

func (s *ssaExport) TypeBool() ssa.Type    { return Types[TBOOL] }
func (s *ssaExport) TypeInt8() ssa.Type    { return Types[TINT8] }
func (s *ssaExport) TypeInt16() ssa.Type   { return Types[TINT16] }
func (s *ssaExport) TypeInt32() ssa.Type   { return Types[TINT32] }
func (s *ssaExport) TypeInt64() ssa.Type   { return Types[TINT64] }
func (s *ssaExport) TypeUInt8() ssa.Type   { return Types[TUINT8] }
func (s *ssaExport) TypeUInt16() ssa.Type  { return Types[TUINT16] }
func (s *ssaExport) TypeUInt32() ssa.Type  { return Types[TUINT32] }
func (s *ssaExport) TypeUInt64() ssa.Type  { return Types[TUINT64] }
func (s *ssaExport) TypeInt() ssa.Type     { return Types[TINT] }
func (s *ssaExport) TypeUintptr() ssa.Type { return Types[TUINTPTR] }
func (s *ssaExport) TypeString() ssa.Type  { return Types[TSTRING] }
func (s *ssaExport) TypeBytePtr() ssa.Type { return Ptrto(Types[TUINT8]) }

// StringData returns a symbol (a *Sym wrapped in an interface) which
// is the data component of a global string constant containing s.
func (*ssaExport) StringData(s string) interface{} {
	// TODO: is idealstring correct?  It might not matter...
	_, data := stringsym(s)
	return &ssa.ExternSymbol{Typ: idealstring, Sym: data}
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
	if e.mustImplement {
		Fatal(msg, args...)
	}
	const alwaysLog = false // enable to calculate top unimplemented features
	if !e.unimplemented && (e.log || alwaysLog) {
		// first implementation failure, print explanation
		fmt.Printf("SSA unimplemented: "+msg+"\n", args...)
	}
	e.unimplemented = true
}
