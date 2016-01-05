// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"crypto/sha1"
	"fmt"
	"html"
	"math"
	"os"
	"strings"

	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

// Smallest possible faulting page at address zero.
const minZeroPage = 4096

func shouldssa(fn *Node) bool {
	if Thearch.Thestring != "amd64" {
		return false
	}

	// Environment variable control of SSA CG
	// 1. IF GOSSAFUNC == current function name THEN
	//       compile this function with SSA and log output to ssa.html

	// 2. IF GOSSAHASH == "" THEN
	//       compile this function (and everything else) with SSA

	// 3. IF GOSSAHASH == "n" or "N"
	//       IF GOSSAPKG == current package name THEN
	//          compile this function (and everything in this package) with SSA
	//       ELSE
	//          use the old back end for this function.
	//       This is for compatibility with existing test harness and should go away.

	// 4. IF GOSSAHASH is a suffix of the binary-rendered SHA1 hash of the function name THEN
	//          compile this function with SSA
	//       ELSE
	//          compile this function with the old back end.

	// Plan is for 3 to be removed when the tests are revised.
	// SSA is now default, and is disabled by setting
	// GOSSAHASH to n or N, or selectively with strings of
	// 0 and 1.

	name := fn.Func.Nname.Sym.Name

	funcname := os.Getenv("GOSSAFUNC")
	if funcname != "" {
		// If GOSSAFUNC is set, compile only that function.
		return name == funcname
	}

	pkg := os.Getenv("GOSSAPKG")
	if pkg != "" {
		// If GOSSAPKG is set, compile only that package.
		return localpkg.Name == pkg
	}

	gossahash := os.Getenv("GOSSAHASH")
	if gossahash == "" || gossahash == "y" || gossahash == "Y" {
		return true
	}
	if gossahash == "n" || gossahash == "N" {
		return false
	}

	// Check the hash of the name against a partial input hash.
	// We use this feature to do a binary search within a package to
	// find a function that is incorrectly compiled.
	hstr := ""
	for _, b := range sha1.Sum([]byte(name)) {
		hstr += fmt.Sprintf("%08b", b)
	}

	if strings.HasSuffix(hstr, gossahash) {
		fmt.Printf("GOSSAHASH triggered %s\n", name)
		return true
	}

	// Iteratively try additional hashes to allow tests for multi-point
	// failure.
	for i := 0; true; i++ {
		ev := fmt.Sprintf("GOSSAHASH%d", i)
		evv := os.Getenv(ev)
		if evv == "" {
			break
		}
		if strings.HasSuffix(hstr, evv) {
			fmt.Printf("%s triggered %s\n", ev, name)
			return true
		}
	}

	return false
}

// buildssa builds an SSA function.
func buildssa(fn *Node) *ssa.Func {
	name := fn.Func.Nname.Sym.Name
	printssa := strings.HasSuffix(name, "_ssa") || strings.Contains(name, "_ssa.") || name == os.Getenv("GOSSAFUNC")
	if printssa {
		fmt.Println("generating SSA for", name)
		dumplist("buildssa-enter", fn.Func.Enter)
		dumplist("buildssa-body", fn.Nbody)
		dumplist("buildssa-exit", fn.Func.Exit)
	}

	var s state
	s.pushLine(fn.Lineno)
	defer s.popLine()

	// TODO(khr): build config just once at the start of the compiler binary

	var e ssaExport
	e.log = printssa
	s.config = ssa.NewConfig(Thearch.Thestring, &e, Ctxt)
	s.f = s.config.NewFunc()
	s.f.Name = name
	s.exitCode = fn.Func.Exit
	s.panics = map[funcLine]*ssa.Block{}

	if name == os.Getenv("GOSSAFUNC") {
		// TODO: tempfile? it is handy to have the location
		// of this file be stable, so you can just reload in the browser.
		s.config.HTML = ssa.NewHTMLWriter("ssa.html", &s, name)
		// TODO: generate and print a mapping from nodes to values and blocks
	}
	defer func() {
		if !printssa {
			s.config.HTML.Close()
		}
	}()

	// We construct SSA using an algorithm similar to
	// Brau, Buchwald, Hack, Leißa, Mallon, and Zwinkau
	// http://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
	// TODO: check this comment

	// Allocate starting block
	s.f.Entry = s.f.NewBlock(ssa.BlockPlain)

	// Allocate starting values
	s.labels = map[string]*ssaLabel{}
	s.labeledNodes = map[*Node]*ssaLabel{}
	s.startmem = s.entryNewValue0(ssa.OpInitMem, ssa.TypeMem)
	s.sp = s.entryNewValue0(ssa.OpSP, Types[TUINTPTR]) // TODO: use generic pointer type (unsafe.Pointer?) instead
	s.sb = s.entryNewValue0(ssa.OpSB, Types[TUINTPTR])

	s.startBlock(s.f.Entry)
	s.vars[&memVar] = s.startmem

	s.varsyms = map[*Node]interface{}{}

	// Generate addresses of local declarations
	s.decladdrs = map[*Node]*ssa.Value{}
	for d := fn.Func.Dcl; d != nil; d = d.Next {
		n := d.N
		switch n.Class {
		case PPARAM:
			aux := s.lookupSymbol(n, &ssa.ArgSymbol{Typ: n.Type, Node: n})
			s.decladdrs[n] = s.entryNewValue1A(ssa.OpAddr, Ptrto(n.Type), aux, s.sp)
		case PAUTO | PHEAP:
			// TODO this looks wrong for PAUTO|PHEAP, no vardef, but also no definition
			aux := s.lookupSymbol(n, &ssa.AutoSymbol{Typ: n.Type, Node: n})
			s.decladdrs[n] = s.entryNewValue1A(ssa.OpAddr, Ptrto(n.Type), aux, s.sp)
		case PPARAM | PHEAP, PPARAMOUT | PHEAP:
		// This ends up wrong, have to do it at the PARAM node instead.
		case PAUTO, PPARAMOUT:
			// processed at each use, to prevent Addr coming
			// before the decl.
		case PFUNC:
			// local function - already handled by frontend
		default:
			str := ""
			if n.Class&PHEAP != 0 {
				str = ",heap"
			}
			s.Unimplementedf("local variable with class %s%s unimplemented", classnames[n.Class&^PHEAP], str)
		}
	}

	// Convert the AST-based IR to the SSA-based IR
	s.stmtList(fn.Func.Enter)
	s.stmtList(fn.Nbody)

	// fallthrough to exit
	if s.curBlock != nil {
		s.stmtList(s.exitCode)
		m := s.mem()
		b := s.endBlock()
		b.Kind = ssa.BlockRet
		b.Control = m
	}

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
		return nil
	}

	// Link up variable uses to variable definitions
	s.linkForwardReferences()

	// Don't carry reference this around longer than necessary
	s.exitCode = nil

	// Main call to ssa package to compile function
	ssa.Compile(s.f)

	return s.f
}

type state struct {
	// configuration (arch) information
	config *ssa.Config

	// function we're building
	f *ssa.Func

	// labels and labeled control flow nodes (OFOR, OSWITCH, OSELECT) in f
	labels       map[string]*ssaLabel
	labeledNodes map[*Node]*ssaLabel

	// gotos that jump forward; required for deferred checkgoto calls
	fwdGotos []*Node
	// Code that must precede any return
	// (e.g., copying value of heap-escaped paramout back to true paramout)
	exitCode *NodeList

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

	// addresses of PPARAM and PPARAMOUT variables.
	decladdrs map[*Node]*ssa.Value

	// symbols for PEXTERN, PAUTO and PPARAMOUT variables so they can be reused.
	varsyms map[*Node]interface{}

	// starting values.  Memory, stack pointer, and globals pointer
	startmem *ssa.Value
	sp       *ssa.Value
	sb       *ssa.Value

	// line number stack.  The current line number is top of stack
	line []int32

	// list of panic calls by function name and line number.
	// Used to deduplicate panic calls.
	panics map[funcLine]*ssa.Block
}

type funcLine struct {
	f    *Node
	line int32
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

func (s *state) Logf(msg string, args ...interface{})            { s.config.Logf(msg, args...) }
func (s *state) Fatalf(msg string, args ...interface{})          { s.config.Fatalf(msg, args...) }
func (s *state) Unimplementedf(msg string, args ...interface{})  { s.config.Unimplementedf(msg, args...) }
func (s *state) Warnl(line int, msg string, args ...interface{}) { s.config.Warnl(line, msg, args...) }
func (s *state) Debug_checknil() bool                            { return s.config.Debug_checknil() }

var (
	// dummy node for the memory variable
	memVar = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "mem"}}

	// dummy nodes for temporary variables
	ptrVar   = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "ptr"}}
	capVar   = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "cap"}}
	typVar   = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "typ"}}
	idataVar = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "idata"}}
	okVar    = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "ok"}}
)

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

// newValue0I adds a new value with no arguments and an auxint value to the current block.
func (s *state) newValue0I(op ssa.Op, t ssa.Type, auxint int64) *ssa.Value {
	return s.curBlock.NewValue0I(s.peekLine(), op, t, auxint)
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

// newValue3I adds a new value with three arguments and an auxint value to the current block.
func (s *state) newValue3I(op ssa.Op, t ssa.Type, aux int64, arg0, arg1, arg2 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue3I(s.peekLine(), op, t, aux, arg0, arg1, arg2)
}

// entryNewValue0 adds a new value with no arguments to the entry block.
func (s *state) entryNewValue0(op ssa.Op, t ssa.Type) *ssa.Value {
	return s.f.Entry.NewValue0(s.peekLine(), op, t)
}

// entryNewValue0A adds a new value with no arguments and an aux value to the entry block.
func (s *state) entryNewValue0A(op ssa.Op, t ssa.Type, aux interface{}) *ssa.Value {
	return s.f.Entry.NewValue0A(s.peekLine(), op, t, aux)
}

// entryNewValue0I adds a new value with no arguments and an auxint value to the entry block.
func (s *state) entryNewValue0I(op ssa.Op, t ssa.Type, auxint int64) *ssa.Value {
	return s.f.Entry.NewValue0I(s.peekLine(), op, t, auxint)
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

// const* routines add a new const value to the entry block.
func (s *state) constBool(c bool) *ssa.Value {
	return s.f.ConstBool(s.peekLine(), Types[TBOOL], c)
}
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
func (s *state) constFloat32(t ssa.Type, c float64) *ssa.Value {
	return s.f.ConstFloat32(s.peekLine(), t, c)
}
func (s *state) constFloat64(t ssa.Type, c float64) *ssa.Value {
	return s.f.ConstFloat64(s.peekLine(), t, c)
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
	case OEMPTY, ODCLCONST, ODCLTYPE, OFALL:

	// Expression statements
	case OCALLFUNC, OCALLMETH, OCALLINTER:
		s.call(n, callNormal)
	case ODEFER:
		s.call(n.Left, callDefer)
	case OPROC:
		s.call(n.Left, callGo)

	case OAS2DOTTYPE:
		res, resok := s.dottype(n.Rlist.N, true)
		s.assign(n.List.N, res, false, n.Lineno)
		s.assign(n.List.Next.N, resok, false, n.Lineno)
		return

	case ODCL:
		if n.Left.Class&PHEAP == 0 {
			return
		}
		if compiling_runtime != 0 {
			Fatalf("%v escapes to heap, not allowed in runtime.", n)
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
		r := s.expr(palloc)
		s.assign(n.Left.Name.Heapaddr, r, false, n.Lineno)

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
		b.AddEdgeTo(lab.target)
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
		b.AddEdgeTo(lab.target)

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
		var r *ssa.Value
		if n.Right != nil {
			if n.Right.Op == OSTRUCTLIT || n.Right.Op == OARRAYLIT {
				// All literals with nonzero fields have already been
				// rewritten during walk.  Any that remain are just T{}
				// or equivalents.  Leave r = nil to get zeroing behavior.
				if !iszero(n.Right) {
					Fatalf("literal with nonzero value in SSA: %v", n.Right)
				}
			} else {
				r = s.expr(n.Right)
			}
		}
		if n.Right != nil && n.Right.Op == OAPPEND {
			// Yuck!  The frontend gets rid of the write barrier, but we need it!
			// At least, we need it in the case where growslice is called.
			// TODO: Do the write barrier on just the growslice branch.
			// TODO: just add a ptr graying to the end of growslice?
			// TODO: check whether we need to do this for ODOTTYPE and ORECV also.
			// They get similar wb-removal treatment in walk.go:OAS.
			s.assign(n.Left, r, true, n.Lineno)
			return
		}
		s.assign(n.Left, r, n.Op == OASWB, n.Lineno)

	case OIF:
		bThen := s.f.NewBlock(ssa.BlockPlain)
		bEnd := s.f.NewBlock(ssa.BlockPlain)
		var bElse *ssa.Block
		if n.Rlist != nil {
			bElse = s.f.NewBlock(ssa.BlockPlain)
			s.condBranch(n.Left, bThen, bElse, n.Likely)
		} else {
			s.condBranch(n.Left, bThen, bEnd, n.Likely)
		}

		s.startBlock(bThen)
		s.stmtList(n.Nbody)
		if b := s.endBlock(); b != nil {
			b.AddEdgeTo(bEnd)
		}

		if n.Rlist != nil {
			s.startBlock(bElse)
			s.stmtList(n.Rlist)
			if b := s.endBlock(); b != nil {
				b.AddEdgeTo(bEnd)
			}
		}
		s.startBlock(bEnd)

	case ORETURN:
		s.stmtList(n.List)
		s.stmtList(s.exitCode)
		m := s.mem()
		b := s.endBlock()
		b.Kind = ssa.BlockRet
		b.Control = m
	case ORETJMP:
		s.stmtList(n.List)
		s.stmtList(s.exitCode)
		m := s.mem()
		b := s.endBlock()
		b.Kind = ssa.BlockRetJmp
		b.Aux = n.Left.Sym
		b.Control = m

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
		b.AddEdgeTo(to)

	case OFOR:
		// OFOR: for Ninit; Left; Right { Nbody }
		bCond := s.f.NewBlock(ssa.BlockPlain)
		bBody := s.f.NewBlock(ssa.BlockPlain)
		bIncr := s.f.NewBlock(ssa.BlockPlain)
		bEnd := s.f.NewBlock(ssa.BlockPlain)

		// first, jump to condition test
		b := s.endBlock()
		b.AddEdgeTo(bCond)

		// generate code to test condition
		s.startBlock(bCond)
		if n.Left != nil {
			s.condBranch(n.Left, bBody, bEnd, 1)
		} else {
			b := s.endBlock()
			b.Kind = ssa.BlockPlain
			b.AddEdgeTo(bBody)
		}

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
			b.AddEdgeTo(bIncr)
		}

		// generate incr
		s.startBlock(bIncr)
		if n.Right != nil {
			s.stmt(n.Right)
		}
		if b := s.endBlock(); b != nil {
			b.AddEdgeTo(bCond)
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
			b.AddEdgeTo(bEnd)
		}
		s.startBlock(bEnd)

	case OVARKILL:
		// Insert a varkill op to record that a variable is no longer live.
		// We only care about liveness info at call sites, so putting the
		// varkill in the store chain is enough to keep it correctly ordered
		// with respect to call ops.
		if !canSSA(n.Left) {
			s.vars[&memVar] = s.newValue1A(ssa.OpVarKill, ssa.TypeMem, n.Left, s.mem())
		}

	case OCHECKNIL:
		p := s.expr(n.Left)
		s.nilCheck(p)

	default:
		s.Unimplementedf("unhandled stmt %s", opnames[n.Op])
	}
}

type opAndType struct {
	op    Op
	etype EType
}

var opToSSA = map[opAndType]ssa.Op{
	opAndType{OADD, TINT8}:    ssa.OpAdd8,
	opAndType{OADD, TUINT8}:   ssa.OpAdd8,
	opAndType{OADD, TINT16}:   ssa.OpAdd16,
	opAndType{OADD, TUINT16}:  ssa.OpAdd16,
	opAndType{OADD, TINT32}:   ssa.OpAdd32,
	opAndType{OADD, TUINT32}:  ssa.OpAdd32,
	opAndType{OADD, TPTR32}:   ssa.OpAdd32,
	opAndType{OADD, TINT64}:   ssa.OpAdd64,
	opAndType{OADD, TUINT64}:  ssa.OpAdd64,
	opAndType{OADD, TPTR64}:   ssa.OpAdd64,
	opAndType{OADD, TFLOAT32}: ssa.OpAdd32F,
	opAndType{OADD, TFLOAT64}: ssa.OpAdd64F,

	opAndType{OSUB, TINT8}:    ssa.OpSub8,
	opAndType{OSUB, TUINT8}:   ssa.OpSub8,
	opAndType{OSUB, TINT16}:   ssa.OpSub16,
	opAndType{OSUB, TUINT16}:  ssa.OpSub16,
	opAndType{OSUB, TINT32}:   ssa.OpSub32,
	opAndType{OSUB, TUINT32}:  ssa.OpSub32,
	opAndType{OSUB, TINT64}:   ssa.OpSub64,
	opAndType{OSUB, TUINT64}:  ssa.OpSub64,
	opAndType{OSUB, TFLOAT32}: ssa.OpSub32F,
	opAndType{OSUB, TFLOAT64}: ssa.OpSub64F,

	opAndType{ONOT, TBOOL}: ssa.OpNot,

	opAndType{OMINUS, TINT8}:    ssa.OpNeg8,
	opAndType{OMINUS, TUINT8}:   ssa.OpNeg8,
	opAndType{OMINUS, TINT16}:   ssa.OpNeg16,
	opAndType{OMINUS, TUINT16}:  ssa.OpNeg16,
	opAndType{OMINUS, TINT32}:   ssa.OpNeg32,
	opAndType{OMINUS, TUINT32}:  ssa.OpNeg32,
	opAndType{OMINUS, TINT64}:   ssa.OpNeg64,
	opAndType{OMINUS, TUINT64}:  ssa.OpNeg64,
	opAndType{OMINUS, TFLOAT32}: ssa.OpNeg32F,
	opAndType{OMINUS, TFLOAT64}: ssa.OpNeg64F,

	opAndType{OCOM, TINT8}:   ssa.OpCom8,
	opAndType{OCOM, TUINT8}:  ssa.OpCom8,
	opAndType{OCOM, TINT16}:  ssa.OpCom16,
	opAndType{OCOM, TUINT16}: ssa.OpCom16,
	opAndType{OCOM, TINT32}:  ssa.OpCom32,
	opAndType{OCOM, TUINT32}: ssa.OpCom32,
	opAndType{OCOM, TINT64}:  ssa.OpCom64,
	opAndType{OCOM, TUINT64}: ssa.OpCom64,

	opAndType{OIMAG, TCOMPLEX64}:  ssa.OpComplexImag,
	opAndType{OIMAG, TCOMPLEX128}: ssa.OpComplexImag,
	opAndType{OREAL, TCOMPLEX64}:  ssa.OpComplexReal,
	opAndType{OREAL, TCOMPLEX128}: ssa.OpComplexReal,

	opAndType{OMUL, TINT8}:    ssa.OpMul8,
	opAndType{OMUL, TUINT8}:   ssa.OpMul8,
	opAndType{OMUL, TINT16}:   ssa.OpMul16,
	opAndType{OMUL, TUINT16}:  ssa.OpMul16,
	opAndType{OMUL, TINT32}:   ssa.OpMul32,
	opAndType{OMUL, TUINT32}:  ssa.OpMul32,
	opAndType{OMUL, TINT64}:   ssa.OpMul64,
	opAndType{OMUL, TUINT64}:  ssa.OpMul64,
	opAndType{OMUL, TFLOAT32}: ssa.OpMul32F,
	opAndType{OMUL, TFLOAT64}: ssa.OpMul64F,

	opAndType{ODIV, TFLOAT32}: ssa.OpDiv32F,
	opAndType{ODIV, TFLOAT64}: ssa.OpDiv64F,

	opAndType{OHMUL, TINT8}:   ssa.OpHmul8,
	opAndType{OHMUL, TUINT8}:  ssa.OpHmul8u,
	opAndType{OHMUL, TINT16}:  ssa.OpHmul16,
	opAndType{OHMUL, TUINT16}: ssa.OpHmul16u,
	opAndType{OHMUL, TINT32}:  ssa.OpHmul32,
	opAndType{OHMUL, TUINT32}: ssa.OpHmul32u,

	opAndType{ODIV, TINT8}:   ssa.OpDiv8,
	opAndType{ODIV, TUINT8}:  ssa.OpDiv8u,
	opAndType{ODIV, TINT16}:  ssa.OpDiv16,
	opAndType{ODIV, TUINT16}: ssa.OpDiv16u,
	opAndType{ODIV, TINT32}:  ssa.OpDiv32,
	opAndType{ODIV, TUINT32}: ssa.OpDiv32u,
	opAndType{ODIV, TINT64}:  ssa.OpDiv64,
	opAndType{ODIV, TUINT64}: ssa.OpDiv64u,

	opAndType{OMOD, TINT8}:   ssa.OpMod8,
	opAndType{OMOD, TUINT8}:  ssa.OpMod8u,
	opAndType{OMOD, TINT16}:  ssa.OpMod16,
	opAndType{OMOD, TUINT16}: ssa.OpMod16u,
	opAndType{OMOD, TINT32}:  ssa.OpMod32,
	opAndType{OMOD, TUINT32}: ssa.OpMod32u,
	opAndType{OMOD, TINT64}:  ssa.OpMod64,
	opAndType{OMOD, TUINT64}: ssa.OpMod64u,

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
	opAndType{OEQ, TINTER}:     ssa.OpEqInter,
	opAndType{OEQ, TARRAY}:     ssa.OpEqSlice,
	opAndType{OEQ, TFUNC}:      ssa.OpEqPtr,
	opAndType{OEQ, TMAP}:       ssa.OpEqPtr,
	opAndType{OEQ, TCHAN}:      ssa.OpEqPtr,
	opAndType{OEQ, TPTR64}:     ssa.OpEqPtr,
	opAndType{OEQ, TUINTPTR}:   ssa.OpEqPtr,
	opAndType{OEQ, TUNSAFEPTR}: ssa.OpEqPtr,
	opAndType{OEQ, TFLOAT64}:   ssa.OpEq64F,
	opAndType{OEQ, TFLOAT32}:   ssa.OpEq32F,

	opAndType{ONE, TBOOL}:      ssa.OpNeq8,
	opAndType{ONE, TINT8}:      ssa.OpNeq8,
	opAndType{ONE, TUINT8}:     ssa.OpNeq8,
	opAndType{ONE, TINT16}:     ssa.OpNeq16,
	opAndType{ONE, TUINT16}:    ssa.OpNeq16,
	opAndType{ONE, TINT32}:     ssa.OpNeq32,
	opAndType{ONE, TUINT32}:    ssa.OpNeq32,
	opAndType{ONE, TINT64}:     ssa.OpNeq64,
	opAndType{ONE, TUINT64}:    ssa.OpNeq64,
	opAndType{ONE, TINTER}:     ssa.OpNeqInter,
	opAndType{ONE, TARRAY}:     ssa.OpNeqSlice,
	opAndType{ONE, TFUNC}:      ssa.OpNeqPtr,
	opAndType{ONE, TMAP}:       ssa.OpNeqPtr,
	opAndType{ONE, TCHAN}:      ssa.OpNeqPtr,
	opAndType{ONE, TPTR64}:     ssa.OpNeqPtr,
	opAndType{ONE, TUINTPTR}:   ssa.OpNeqPtr,
	opAndType{ONE, TUNSAFEPTR}: ssa.OpNeqPtr,
	opAndType{ONE, TFLOAT64}:   ssa.OpNeq64F,
	opAndType{ONE, TFLOAT32}:   ssa.OpNeq32F,

	opAndType{OLT, TINT8}:    ssa.OpLess8,
	opAndType{OLT, TUINT8}:   ssa.OpLess8U,
	opAndType{OLT, TINT16}:   ssa.OpLess16,
	opAndType{OLT, TUINT16}:  ssa.OpLess16U,
	opAndType{OLT, TINT32}:   ssa.OpLess32,
	opAndType{OLT, TUINT32}:  ssa.OpLess32U,
	opAndType{OLT, TINT64}:   ssa.OpLess64,
	opAndType{OLT, TUINT64}:  ssa.OpLess64U,
	opAndType{OLT, TFLOAT64}: ssa.OpLess64F,
	opAndType{OLT, TFLOAT32}: ssa.OpLess32F,

	opAndType{OGT, TINT8}:    ssa.OpGreater8,
	opAndType{OGT, TUINT8}:   ssa.OpGreater8U,
	opAndType{OGT, TINT16}:   ssa.OpGreater16,
	opAndType{OGT, TUINT16}:  ssa.OpGreater16U,
	opAndType{OGT, TINT32}:   ssa.OpGreater32,
	opAndType{OGT, TUINT32}:  ssa.OpGreater32U,
	opAndType{OGT, TINT64}:   ssa.OpGreater64,
	opAndType{OGT, TUINT64}:  ssa.OpGreater64U,
	opAndType{OGT, TFLOAT64}: ssa.OpGreater64F,
	opAndType{OGT, TFLOAT32}: ssa.OpGreater32F,

	opAndType{OLE, TINT8}:    ssa.OpLeq8,
	opAndType{OLE, TUINT8}:   ssa.OpLeq8U,
	opAndType{OLE, TINT16}:   ssa.OpLeq16,
	opAndType{OLE, TUINT16}:  ssa.OpLeq16U,
	opAndType{OLE, TINT32}:   ssa.OpLeq32,
	opAndType{OLE, TUINT32}:  ssa.OpLeq32U,
	opAndType{OLE, TINT64}:   ssa.OpLeq64,
	opAndType{OLE, TUINT64}:  ssa.OpLeq64U,
	opAndType{OLE, TFLOAT64}: ssa.OpLeq64F,
	opAndType{OLE, TFLOAT32}: ssa.OpLeq32F,

	opAndType{OGE, TINT8}:    ssa.OpGeq8,
	opAndType{OGE, TUINT8}:   ssa.OpGeq8U,
	opAndType{OGE, TINT16}:   ssa.OpGeq16,
	opAndType{OGE, TUINT16}:  ssa.OpGeq16U,
	opAndType{OGE, TINT32}:   ssa.OpGeq32,
	opAndType{OGE, TUINT32}:  ssa.OpGeq32U,
	opAndType{OGE, TINT64}:   ssa.OpGeq64,
	opAndType{OGE, TUINT64}:  ssa.OpGeq64U,
	opAndType{OGE, TFLOAT64}: ssa.OpGeq64F,
	opAndType{OGE, TFLOAT32}: ssa.OpGeq32F,

	opAndType{OLROT, TUINT8}:  ssa.OpLrot8,
	opAndType{OLROT, TUINT16}: ssa.OpLrot16,
	opAndType{OLROT, TUINT32}: ssa.OpLrot32,
	opAndType{OLROT, TUINT64}: ssa.OpLrot64,

	opAndType{OSQRT, TFLOAT64}: ssa.OpSqrt,
}

func (s *state) concreteEtype(t *Type) EType {
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

func (s *state) ssaOp(op Op, t *Type) ssa.Op {
	etype := s.concreteEtype(t)
	x, ok := opToSSA[opAndType{op, etype}]
	if !ok {
		s.Unimplementedf("unhandled binary op %s %s", opnames[op], Econv(etype))
	}
	return x
}

func floatForComplex(t *Type) *Type {
	if t.Size() == 8 {
		return Types[TFLOAT32]
	} else {
		return Types[TFLOAT64]
	}
}

type opAndTwoTypes struct {
	op     Op
	etype1 EType
	etype2 EType
}

type twoTypes struct {
	etype1 EType
	etype2 EType
}

type twoOpsAndType struct {
	op1              ssa.Op
	op2              ssa.Op
	intermediateType EType
}

var fpConvOpToSSA = map[twoTypes]twoOpsAndType{

	twoTypes{TINT8, TFLOAT32}:  twoOpsAndType{ssa.OpSignExt8to32, ssa.OpCvt32to32F, TINT32},
	twoTypes{TINT16, TFLOAT32}: twoOpsAndType{ssa.OpSignExt16to32, ssa.OpCvt32to32F, TINT32},
	twoTypes{TINT32, TFLOAT32}: twoOpsAndType{ssa.OpCopy, ssa.OpCvt32to32F, TINT32},
	twoTypes{TINT64, TFLOAT32}: twoOpsAndType{ssa.OpCopy, ssa.OpCvt64to32F, TINT64},

	twoTypes{TINT8, TFLOAT64}:  twoOpsAndType{ssa.OpSignExt8to32, ssa.OpCvt32to64F, TINT32},
	twoTypes{TINT16, TFLOAT64}: twoOpsAndType{ssa.OpSignExt16to32, ssa.OpCvt32to64F, TINT32},
	twoTypes{TINT32, TFLOAT64}: twoOpsAndType{ssa.OpCopy, ssa.OpCvt32to64F, TINT32},
	twoTypes{TINT64, TFLOAT64}: twoOpsAndType{ssa.OpCopy, ssa.OpCvt64to64F, TINT64},

	twoTypes{TFLOAT32, TINT8}:  twoOpsAndType{ssa.OpCvt32Fto32, ssa.OpTrunc32to8, TINT32},
	twoTypes{TFLOAT32, TINT16}: twoOpsAndType{ssa.OpCvt32Fto32, ssa.OpTrunc32to16, TINT32},
	twoTypes{TFLOAT32, TINT32}: twoOpsAndType{ssa.OpCvt32Fto32, ssa.OpCopy, TINT32},
	twoTypes{TFLOAT32, TINT64}: twoOpsAndType{ssa.OpCvt32Fto64, ssa.OpCopy, TINT64},

	twoTypes{TFLOAT64, TINT8}:  twoOpsAndType{ssa.OpCvt64Fto32, ssa.OpTrunc32to8, TINT32},
	twoTypes{TFLOAT64, TINT16}: twoOpsAndType{ssa.OpCvt64Fto32, ssa.OpTrunc32to16, TINT32},
	twoTypes{TFLOAT64, TINT32}: twoOpsAndType{ssa.OpCvt64Fto32, ssa.OpCopy, TINT32},
	twoTypes{TFLOAT64, TINT64}: twoOpsAndType{ssa.OpCvt64Fto64, ssa.OpCopy, TINT64},
	// unsigned
	twoTypes{TUINT8, TFLOAT32}:  twoOpsAndType{ssa.OpZeroExt8to32, ssa.OpCvt32to32F, TINT32},
	twoTypes{TUINT16, TFLOAT32}: twoOpsAndType{ssa.OpZeroExt16to32, ssa.OpCvt32to32F, TINT32},
	twoTypes{TUINT32, TFLOAT32}: twoOpsAndType{ssa.OpZeroExt32to64, ssa.OpCvt64to32F, TINT64}, // go wide to dodge unsigned
	twoTypes{TUINT64, TFLOAT32}: twoOpsAndType{ssa.OpCopy, ssa.OpInvalid, TUINT64},            // Cvt64Uto32F, branchy code expansion instead

	twoTypes{TUINT8, TFLOAT64}:  twoOpsAndType{ssa.OpZeroExt8to32, ssa.OpCvt32to64F, TINT32},
	twoTypes{TUINT16, TFLOAT64}: twoOpsAndType{ssa.OpZeroExt16to32, ssa.OpCvt32to64F, TINT32},
	twoTypes{TUINT32, TFLOAT64}: twoOpsAndType{ssa.OpZeroExt32to64, ssa.OpCvt64to64F, TINT64}, // go wide to dodge unsigned
	twoTypes{TUINT64, TFLOAT64}: twoOpsAndType{ssa.OpCopy, ssa.OpInvalid, TUINT64},            // Cvt64Uto64F, branchy code expansion instead

	twoTypes{TFLOAT32, TUINT8}:  twoOpsAndType{ssa.OpCvt32Fto32, ssa.OpTrunc32to8, TINT32},
	twoTypes{TFLOAT32, TUINT16}: twoOpsAndType{ssa.OpCvt32Fto32, ssa.OpTrunc32to16, TINT32},
	twoTypes{TFLOAT32, TUINT32}: twoOpsAndType{ssa.OpCvt32Fto64, ssa.OpTrunc64to32, TINT64}, // go wide to dodge unsigned
	twoTypes{TFLOAT32, TUINT64}: twoOpsAndType{ssa.OpInvalid, ssa.OpCopy, TUINT64},          // Cvt32Fto64U, branchy code expansion instead

	twoTypes{TFLOAT64, TUINT8}:  twoOpsAndType{ssa.OpCvt64Fto32, ssa.OpTrunc32to8, TINT32},
	twoTypes{TFLOAT64, TUINT16}: twoOpsAndType{ssa.OpCvt64Fto32, ssa.OpTrunc32to16, TINT32},
	twoTypes{TFLOAT64, TUINT32}: twoOpsAndType{ssa.OpCvt64Fto64, ssa.OpTrunc64to32, TINT64}, // go wide to dodge unsigned
	twoTypes{TFLOAT64, TUINT64}: twoOpsAndType{ssa.OpInvalid, ssa.OpCopy, TUINT64},          // Cvt64Fto64U, branchy code expansion instead

	// float
	twoTypes{TFLOAT64, TFLOAT32}: twoOpsAndType{ssa.OpCvt64Fto32F, ssa.OpCopy, TFLOAT32},
	twoTypes{TFLOAT64, TFLOAT64}: twoOpsAndType{ssa.OpCopy, ssa.OpCopy, TFLOAT64},
	twoTypes{TFLOAT32, TFLOAT32}: twoOpsAndType{ssa.OpCopy, ssa.OpCopy, TFLOAT32},
	twoTypes{TFLOAT32, TFLOAT64}: twoOpsAndType{ssa.OpCvt32Fto64F, ssa.OpCopy, TFLOAT64},
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

func (s *state) ssaShiftOp(op Op, t *Type, u *Type) ssa.Op {
	etype1 := s.concreteEtype(t)
	etype2 := s.concreteEtype(u)
	x, ok := shiftOpToSSA[opAndTwoTypes{op, etype1, etype2}]
	if !ok {
		s.Unimplementedf("unhandled shift op %s etype=%s/%s", opnames[op], Econv(etype1), Econv(etype2))
	}
	return x
}

func (s *state) ssaRotateOp(op Op, t *Type) ssa.Op {
	etype1 := s.concreteEtype(t)
	x, ok := opToSSA[opAndType{op, etype1}]
	if !ok {
		s.Unimplementedf("unhandled rotate op %s etype=%s", opnames[op], Econv(etype1))
	}
	return x
}

// expr converts the expression n to ssa, adds it to s and returns the ssa result.
func (s *state) expr(n *Node) *ssa.Value {
	s.pushLine(n.Lineno)
	defer s.popLine()

	s.stmtList(n.Ninit)
	switch n.Op {
	case OCFUNC:
		aux := s.lookupSymbol(n, &ssa.ExternSymbol{n.Type, n.Left.Sym})
		return s.entryNewValue1A(ssa.OpAddr, n.Type, aux, s.sb)
	case OPARAM:
		addr := s.addr(n, false)
		return s.newValue2(ssa.OpLoad, n.Left.Type, addr, s.mem())
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
		addr := s.addr(n, false)
		return s.newValue2(ssa.OpLoad, n.Type, addr, s.mem())
	case OCLOSUREVAR:
		addr := s.addr(n, false)
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
			return s.constBool(n.Val().U.(bool))
		case CTNIL:
			t := n.Type
			switch {
			case t.IsSlice():
				return s.entryNewValue0(ssa.OpConstSlice, t)
			case t.IsInterface():
				return s.entryNewValue0(ssa.OpConstInterface, t)
			default:
				return s.entryNewValue0(ssa.OpConstNil, t)
			}
		case CTFLT:
			f := n.Val().U.(*Mpflt)
			switch n.Type.Size() {
			case 4:
				// -0.0 literals need to be treated as if they were 0.0, adding 0.0 here
				// accomplishes this while not affecting other values.
				return s.constFloat32(n.Type, mpgetflt32(f)+0.0)
			case 8:
				return s.constFloat64(n.Type, mpgetflt(f)+0.0)
			default:
				s.Fatalf("bad float size %d", n.Type.Size())
				return nil
			}
		case CTCPLX:
			c := n.Val().U.(*Mpcplx)
			r := &c.Real
			i := &c.Imag
			switch n.Type.Size() {
			case 8:
				{
					pt := Types[TFLOAT32]
					// -0.0 literals need to be treated as if they were 0.0, adding 0.0 here
					// accomplishes this while not affecting other values.
					return s.newValue2(ssa.OpComplexMake, n.Type,
						s.constFloat32(pt, mpgetflt32(r)+0.0),
						s.constFloat32(pt, mpgetflt32(i)+0.0))
				}
			case 16:
				{
					pt := Types[TFLOAT64]
					return s.newValue2(ssa.OpComplexMake, n.Type,
						s.constFloat64(pt, mpgetflt(r)+0.0),
						s.constFloat64(pt, mpgetflt(i)+0.0))
				}
			default:
				s.Fatalf("bad float size %d", n.Type.Size())
				return nil
			}

		default:
			s.Unimplementedf("unhandled OLITERAL %v", n.Val().Ctype())
			return nil
		}
	case OCONVNOP:
		to := n.Type
		from := n.Left.Type

		// Assume everything will work out, so set up our return value.
		// Anything interesting that happens from here is a fatal.
		x := s.expr(n.Left)

		// Special case for not confusing GC and liveness.
		// We don't want pointers accidentally classified
		// as not-pointers or vice-versa because of copy
		// elision.
		if to.IsPtr() != from.IsPtr() {
			return s.newValue2(ssa.OpConvert, to, x, s.mem())
		}

		v := s.newValue1(ssa.OpCopy, to, x) // ensure that v has the right type

		// CONVNOP closure
		if to.Etype == TFUNC && from.IsPtr() {
			return v
		}

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
			s.Fatalf("CONVNOP sign mismatch %v (%s) -> %v (%s)\n", from, Econv(from.Etype), to, Econv(to.Etype))
			return nil
		}

		if flag_race != 0 {
			// These appear to be fine, but they fail the
			// integer constraint below, so okay them here.
			// Sample non-integer conversion: map[string]string -> *uint8
			return v
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

		if ft.IsFloat() || tt.IsFloat() {
			conv, ok := fpConvOpToSSA[twoTypes{s.concreteEtype(ft), s.concreteEtype(tt)}]
			if !ok {
				s.Fatalf("weird float conversion %s -> %s", ft, tt)
			}
			op1, op2, it := conv.op1, conv.op2, conv.intermediateType

			if op1 != ssa.OpInvalid && op2 != ssa.OpInvalid {
				// normal case, not tripping over unsigned 64
				if op1 == ssa.OpCopy {
					if op2 == ssa.OpCopy {
						return x
					}
					return s.newValue1(op2, n.Type, x)
				}
				if op2 == ssa.OpCopy {
					return s.newValue1(op1, n.Type, x)
				}
				return s.newValue1(op2, n.Type, s.newValue1(op1, Types[it], x))
			}
			// Tricky 64-bit unsigned cases.
			if ft.IsInteger() {
				// therefore tt is float32 or float64, and ft is also unsigned
				if tt.Size() == 4 {
					return s.uint64Tofloat32(n, x, ft, tt)
				}
				if tt.Size() == 8 {
					return s.uint64Tofloat64(n, x, ft, tt)
				}
				s.Fatalf("weird unsigned integer to float conversion %s -> %s", ft, tt)
			}
			// therefore ft is float32 or float64, and tt is unsigned integer
			if ft.Size() == 4 {
				return s.float32ToUint64(n, x, ft, tt)
			}
			if ft.Size() == 8 {
				return s.float64ToUint64(n, x, ft, tt)
			}
			s.Fatalf("weird float to unsigned integer conversion %s -> %s", ft, tt)
			return nil
		}

		if ft.IsComplex() && tt.IsComplex() {
			var op ssa.Op
			if ft.Size() == tt.Size() {
				op = ssa.OpCopy
			} else if ft.Size() == 8 && tt.Size() == 16 {
				op = ssa.OpCvt32Fto64F
			} else if ft.Size() == 16 && tt.Size() == 8 {
				op = ssa.OpCvt64Fto32F
			} else {
				s.Fatalf("weird complex conversion %s -> %s", ft, tt)
			}
			ftp := floatForComplex(ft)
			ttp := floatForComplex(tt)
			return s.newValue2(ssa.OpComplexMake, tt,
				s.newValue1(op, ttp, s.newValue1(ssa.OpComplexReal, ftp, x)),
				s.newValue1(op, ttp, s.newValue1(ssa.OpComplexImag, ftp, x)))
		}

		s.Unimplementedf("unhandled OCONV %s -> %s", Econv(n.Left.Type.Etype), Econv(n.Type.Etype))
		return nil

	case ODOTTYPE:
		res, _ := s.dottype(n, false)
		return res

	// binary ops
	case OLT, OEQ, ONE, OLE, OGE, OGT:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		if n.Left.Type.IsComplex() {
			pt := floatForComplex(n.Left.Type)
			op := s.ssaOp(OEQ, pt)
			r := s.newValue2(op, Types[TBOOL], s.newValue1(ssa.OpComplexReal, pt, a), s.newValue1(ssa.OpComplexReal, pt, b))
			i := s.newValue2(op, Types[TBOOL], s.newValue1(ssa.OpComplexImag, pt, a), s.newValue1(ssa.OpComplexImag, pt, b))
			c := s.newValue2(ssa.OpAnd8, Types[TBOOL], r, i)
			switch n.Op {
			case OEQ:
				return c
			case ONE:
				return s.newValue1(ssa.OpNot, Types[TBOOL], c)
			default:
				s.Fatalf("ordered complex compare %s", opnames[n.Op])
			}
		}
		return s.newValue2(s.ssaOp(n.Op, n.Left.Type), Types[TBOOL], a, b)
	case OMUL:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		if n.Type.IsComplex() {
			mulop := ssa.OpMul64F
			addop := ssa.OpAdd64F
			subop := ssa.OpSub64F
			pt := floatForComplex(n.Type) // Could be Float32 or Float64
			wt := Types[TFLOAT64]         // Compute in Float64 to minimize cancellation error

			areal := s.newValue1(ssa.OpComplexReal, pt, a)
			breal := s.newValue1(ssa.OpComplexReal, pt, b)
			aimag := s.newValue1(ssa.OpComplexImag, pt, a)
			bimag := s.newValue1(ssa.OpComplexImag, pt, b)

			if pt != wt { // Widen for calculation
				areal = s.newValue1(ssa.OpCvt32Fto64F, wt, areal)
				breal = s.newValue1(ssa.OpCvt32Fto64F, wt, breal)
				aimag = s.newValue1(ssa.OpCvt32Fto64F, wt, aimag)
				bimag = s.newValue1(ssa.OpCvt32Fto64F, wt, bimag)
			}

			xreal := s.newValue2(subop, wt, s.newValue2(mulop, wt, areal, breal), s.newValue2(mulop, wt, aimag, bimag))
			ximag := s.newValue2(addop, wt, s.newValue2(mulop, wt, areal, bimag), s.newValue2(mulop, wt, aimag, breal))

			if pt != wt { // Narrow to store back
				xreal = s.newValue1(ssa.OpCvt64Fto32F, pt, xreal)
				ximag = s.newValue1(ssa.OpCvt64Fto32F, pt, ximag)
			}

			return s.newValue2(ssa.OpComplexMake, n.Type, xreal, ximag)
		}
		return s.newValue2(s.ssaOp(n.Op, n.Type), a.Type, a, b)

	case ODIV:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		if n.Type.IsComplex() {
			// TODO this is not executed because the front-end substitutes a runtime call.
			// That probably ought to change; with modest optimization the widen/narrow
			// conversions could all be elided in larger expression trees.
			mulop := ssa.OpMul64F
			addop := ssa.OpAdd64F
			subop := ssa.OpSub64F
			divop := ssa.OpDiv64F
			pt := floatForComplex(n.Type) // Could be Float32 or Float64
			wt := Types[TFLOAT64]         // Compute in Float64 to minimize cancellation error

			areal := s.newValue1(ssa.OpComplexReal, pt, a)
			breal := s.newValue1(ssa.OpComplexReal, pt, b)
			aimag := s.newValue1(ssa.OpComplexImag, pt, a)
			bimag := s.newValue1(ssa.OpComplexImag, pt, b)

			if pt != wt { // Widen for calculation
				areal = s.newValue1(ssa.OpCvt32Fto64F, wt, areal)
				breal = s.newValue1(ssa.OpCvt32Fto64F, wt, breal)
				aimag = s.newValue1(ssa.OpCvt32Fto64F, wt, aimag)
				bimag = s.newValue1(ssa.OpCvt32Fto64F, wt, bimag)
			}

			denom := s.newValue2(addop, wt, s.newValue2(mulop, wt, breal, breal), s.newValue2(mulop, wt, bimag, bimag))
			xreal := s.newValue2(addop, wt, s.newValue2(mulop, wt, areal, breal), s.newValue2(mulop, wt, aimag, bimag))
			ximag := s.newValue2(subop, wt, s.newValue2(mulop, wt, aimag, breal), s.newValue2(mulop, wt, areal, bimag))

			// TODO not sure if this is best done in wide precision or narrow
			// Double-rounding might be an issue.
			// Note that the pre-SSA implementation does the entire calculation
			// in wide format, so wide is compatible.
			xreal = s.newValue2(divop, wt, xreal, denom)
			ximag = s.newValue2(divop, wt, ximag, denom)

			if pt != wt { // Narrow to store back
				xreal = s.newValue1(ssa.OpCvt64Fto32F, pt, xreal)
				ximag = s.newValue1(ssa.OpCvt64Fto32F, pt, ximag)
			}
			return s.newValue2(ssa.OpComplexMake, n.Type, xreal, ximag)
		}
		if n.Type.IsFloat() {
			return s.newValue2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
		} else {
			// do a size-appropriate check for zero
			cmp := s.newValue2(s.ssaOp(ONE, n.Type), Types[TBOOL], b, s.zeroVal(n.Type))
			s.check(cmp, panicdivide)
			return s.newValue2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
		}
	case OMOD:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		// do a size-appropriate check for zero
		cmp := s.newValue2(s.ssaOp(ONE, n.Type), Types[TBOOL], b, s.zeroVal(n.Type))
		s.check(cmp, panicdivide)
		return s.newValue2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
	case OADD, OSUB:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		if n.Type.IsComplex() {
			pt := floatForComplex(n.Type)
			op := s.ssaOp(n.Op, pt)
			return s.newValue2(ssa.OpComplexMake, n.Type,
				s.newValue2(op, pt, s.newValue1(ssa.OpComplexReal, pt, a), s.newValue1(ssa.OpComplexReal, pt, b)),
				s.newValue2(op, pt, s.newValue1(ssa.OpComplexImag, pt, a), s.newValue1(ssa.OpComplexImag, pt, b)))
		}
		return s.newValue2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
	case OAND, OOR, OHMUL, OXOR:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.newValue2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
	case OLSH, ORSH:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.newValue2(s.ssaShiftOp(n.Op, n.Type, n.Right.Type), a.Type, a, b)
	case OLROT:
		a := s.expr(n.Left)
		i := n.Right.Int()
		if i <= 0 || i >= n.Type.Size()*8 {
			s.Fatalf("Wrong rotate distance for LROT, expected 1 through %d, saw %d", n.Type.Size()*8-1, i)
		}
		return s.newValue1I(s.ssaRotateOp(n.Op, n.Type), a.Type, i, a)
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
		// In theory, we should set b.Likely here based on context.
		// However, gc only gives us likeliness hints
		// in a single place, for plain OIF statements,
		// and passing around context is finnicky, so don't bother for now.

		bRight := s.f.NewBlock(ssa.BlockPlain)
		bResult := s.f.NewBlock(ssa.BlockPlain)
		if n.Op == OANDAND {
			b.AddEdgeTo(bRight)
			b.AddEdgeTo(bResult)
		} else if n.Op == OOROR {
			b.AddEdgeTo(bResult)
			b.AddEdgeTo(bRight)
		}

		s.startBlock(bRight)
		er := s.expr(n.Right)
		s.vars[n] = er

		b = s.endBlock()
		b.AddEdgeTo(bResult)

		s.startBlock(bResult)
		return s.variable(n, Types[TBOOL])
	case OCOMPLEX:
		r := s.expr(n.Left)
		i := s.expr(n.Right)
		return s.newValue2(ssa.OpComplexMake, n.Type, r, i)

	// unary ops
	case OMINUS:
		a := s.expr(n.Left)
		if n.Type.IsComplex() {
			tp := floatForComplex(n.Type)
			negop := s.ssaOp(n.Op, tp)
			return s.newValue2(ssa.OpComplexMake, n.Type,
				s.newValue1(negop, tp, s.newValue1(ssa.OpComplexReal, tp, a)),
				s.newValue1(negop, tp, s.newValue1(ssa.OpComplexImag, tp, a)))
		}
		return s.newValue1(s.ssaOp(n.Op, n.Type), a.Type, a)
	case ONOT, OCOM, OSQRT:
		a := s.expr(n.Left)
		return s.newValue1(s.ssaOp(n.Op, n.Type), a.Type, a)
	case OIMAG, OREAL:
		a := s.expr(n.Left)
		return s.newValue1(s.ssaOp(n.Op, n.Left.Type), n.Type, a)
	case OPLUS:
		return s.expr(n.Left)

	case OADDR:
		return s.addr(n.Left, n.Bounded)

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
		// TODO: fix when we can SSA struct types.
		p := s.addr(n, false)
		return s.newValue2(ssa.OpLoad, n.Type, p, s.mem())

	case ODOTPTR:
		p := s.expr(n.Left)
		s.nilCheck(p)
		p = s.newValue2(ssa.OpAddPtr, p.Type, p, s.constInt(Types[TINT], n.Xoffset))
		return s.newValue2(ssa.OpLoad, n.Type, p, s.mem())

	case OINDEX:
		switch {
		case n.Left.Type.IsString():
			a := s.expr(n.Left)
			i := s.expr(n.Right)
			i = s.extendIndex(i)
			if !n.Bounded {
				len := s.newValue1(ssa.OpStringLen, Types[TINT], a)
				s.boundsCheck(i, len)
			}
			ptrtyp := Ptrto(Types[TUINT8])
			ptr := s.newValue1(ssa.OpStringPtr, ptrtyp, a)
			ptr = s.newValue2(ssa.OpAddPtr, ptrtyp, ptr, i)
			return s.newValue2(ssa.OpLoad, Types[TUINT8], ptr, s.mem())
		case n.Left.Type.IsSlice():
			p := s.addr(n, false)
			return s.newValue2(ssa.OpLoad, n.Left.Type.Type, p, s.mem())
		case n.Left.Type.IsArray():
			// TODO: fix when we can SSA arrays of length 1.
			p := s.addr(n, false)
			return s.newValue2(ssa.OpLoad, n.Left.Type.Type, p, s.mem())
		default:
			s.Fatalf("bad type for index %v", n.Left.Type)
			return nil
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
		case n.Left.Type.IsMap(), n.Left.Type.IsChan():
			return s.referenceTypeBuiltin(n, s.expr(n.Left))
		default: // array
			return s.constInt(Types[TINT], n.Left.Type.Bound)
		}

	case OSPTR:
		a := s.expr(n.Left)
		if n.Left.Type.IsSlice() {
			return s.newValue1(ssa.OpSlicePtr, n.Type, a)
		} else {
			return s.newValue1(ssa.OpStringPtr, n.Type, a)
		}

	case OITAB:
		a := s.expr(n.Left)
		return s.newValue1(ssa.OpITab, n.Type, a)

	case OEFACE:
		tab := s.expr(n.Left)
		data := s.expr(n.Right)
		// The frontend allows putting things like struct{*byte} in
		// the data portion of an eface.  But we don't want struct{*byte}
		// as a register type because (among other reasons) the liveness
		// analysis is confused by the "fat" variables that result from
		// such types being spilled.
		// So here we ensure that we are selecting the underlying pointer
		// when we build an eface.
		for !data.Type.IsPtr() {
			switch {
			case data.Type.IsArray():
				data = s.newValue2(ssa.OpArrayIndex, data.Type.Elem(), data, s.constInt(Types[TINT], 0))
			case data.Type.IsStruct():
				for i := data.Type.NumFields() - 1; i >= 0; i-- {
					f := data.Type.FieldType(i)
					if f.Size() == 0 {
						// eface type could also be struct{p *byte; q [0]int}
						continue
					}
					data = s.newValue1I(ssa.OpStructSelect, f, data.Type.FieldOff(i), data)
					break
				}
			default:
				s.Fatalf("type being put into an eface isn't a pointer")
			}
		}
		return s.newValue2(ssa.OpIMake, n.Type, tab, data)

	case OSLICE, OSLICEARR:
		v := s.expr(n.Left)
		var i, j *ssa.Value
		if n.Right.Left != nil {
			i = s.extendIndex(s.expr(n.Right.Left))
		}
		if n.Right.Right != nil {
			j = s.extendIndex(s.expr(n.Right.Right))
		}
		p, l, c := s.slice(n.Left.Type, v, i, j, nil)
		return s.newValue3(ssa.OpSliceMake, n.Type, p, l, c)
	case OSLICESTR:
		v := s.expr(n.Left)
		var i, j *ssa.Value
		if n.Right.Left != nil {
			i = s.extendIndex(s.expr(n.Right.Left))
		}
		if n.Right.Right != nil {
			j = s.extendIndex(s.expr(n.Right.Right))
		}
		p, l, _ := s.slice(n.Left.Type, v, i, j, nil)
		return s.newValue2(ssa.OpStringMake, n.Type, p, l)
	case OSLICE3, OSLICE3ARR:
		v := s.expr(n.Left)
		var i *ssa.Value
		if n.Right.Left != nil {
			i = s.extendIndex(s.expr(n.Right.Left))
		}
		j := s.extendIndex(s.expr(n.Right.Right.Left))
		k := s.extendIndex(s.expr(n.Right.Right.Right))
		p, l, c := s.slice(n.Left.Type, v, i, j, k)
		return s.newValue3(ssa.OpSliceMake, n.Type, p, l, c)

	case OCALLFUNC, OCALLINTER, OCALLMETH:
		return s.call(n, callNormal)

	case OGETG:
		return s.newValue1(ssa.OpGetG, n.Type, s.mem())

	case OAPPEND:
		// append(s, e1, e2, e3).  Compile like:
		// ptr,len,cap := s
		// newlen := len + 3
		// if newlen > s.cap {
		//     ptr,_,cap = growslice(s, newlen)
		// }
		// *(ptr+len) = e1
		// *(ptr+len+1) = e2
		// *(ptr+len+2) = e3
		// makeslice(ptr,newlen,cap)

		et := n.Type.Type
		pt := Ptrto(et)

		// Evaluate slice
		slice := s.expr(n.List.N)

		// Allocate new blocks
		grow := s.f.NewBlock(ssa.BlockPlain)
		assign := s.f.NewBlock(ssa.BlockPlain)

		// Decide if we need to grow
		nargs := int64(count(n.List) - 1)
		p := s.newValue1(ssa.OpSlicePtr, pt, slice)
		l := s.newValue1(ssa.OpSliceLen, Types[TINT], slice)
		c := s.newValue1(ssa.OpSliceCap, Types[TINT], slice)
		nl := s.newValue2(s.ssaOp(OADD, Types[TINT]), Types[TINT], l, s.constInt(Types[TINT], nargs))
		cmp := s.newValue2(s.ssaOp(OGT, Types[TINT]), Types[TBOOL], nl, c)
		s.vars[&ptrVar] = p
		s.vars[&capVar] = c
		b := s.endBlock()
		b.Kind = ssa.BlockIf
		b.Likely = ssa.BranchUnlikely
		b.Control = cmp
		b.AddEdgeTo(grow)
		b.AddEdgeTo(assign)

		// Call growslice
		s.startBlock(grow)
		taddr := s.newValue1A(ssa.OpAddr, Types[TUINTPTR], &ssa.ExternSymbol{Types[TUINTPTR], typenamesym(n.Type)}, s.sb)

		r := s.rtcall(growslice, true, []*Type{pt, Types[TINT], Types[TINT]}, taddr, p, l, c, nl)

		s.vars[&ptrVar] = r[0]
		// Note: we don't need to read r[1], the result's length.  It will be nl.
		// (or maybe we should, we just have to spill/restore nl otherwise?)
		s.vars[&capVar] = r[2]
		b = s.endBlock()
		b.AddEdgeTo(assign)

		// assign new elements to slots
		s.startBlock(assign)

		// Evaluate args
		args := make([]*ssa.Value, 0, nargs)
		store := make([]bool, 0, nargs)
		for l := n.List.Next; l != nil; l = l.Next {
			if canSSAType(l.N.Type) {
				args = append(args, s.expr(l.N))
				store = append(store, true)
			} else {
				args = append(args, s.addr(l.N, false))
				store = append(store, false)
			}
		}

		p = s.variable(&ptrVar, pt)          // generates phi for ptr
		c = s.variable(&capVar, Types[TINT]) // generates phi for cap
		p2 := s.newValue2(ssa.OpPtrIndex, pt, p, l)
		for i, arg := range args {
			addr := s.newValue2(ssa.OpPtrIndex, pt, p2, s.constInt(Types[TINT], int64(i)))
			if store[i] {
				s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, et.Size(), addr, arg, s.mem())
			} else {
				s.vars[&memVar] = s.newValue3I(ssa.OpMove, ssa.TypeMem, et.Size(), addr, arg, s.mem())
			}
			if haspointers(et) {
				// TODO: just one write barrier call for all of these writes?
				// TODO: maybe just one writeBarrier.enabled check?
				s.insertWB(et, addr, n.Lineno)
			}
		}

		// make result
		delete(s.vars, &ptrVar)
		delete(s.vars, &capVar)
		return s.newValue3(ssa.OpSliceMake, n.Type, p, nl, c)

	default:
		s.Unimplementedf("unhandled expr %s", opnames[n.Op])
		return nil
	}
}

// condBranch evaluates the boolean expression cond and branches to yes
// if cond is true and no if cond is false.
// This function is intended to handle && and || better than just calling
// s.expr(cond) and branching on the result.
func (s *state) condBranch(cond *Node, yes, no *ssa.Block, likely int8) {
	if cond.Op == OANDAND {
		mid := s.f.NewBlock(ssa.BlockPlain)
		s.stmtList(cond.Ninit)
		s.condBranch(cond.Left, mid, no, max8(likely, 0))
		s.startBlock(mid)
		s.condBranch(cond.Right, yes, no, likely)
		return
		// Note: if likely==1, then both recursive calls pass 1.
		// If likely==-1, then we don't have enough information to decide
		// whether the first branch is likely or not.  So we pass 0 for
		// the likeliness of the first branch.
		// TODO: have the frontend give us branch prediction hints for
		// OANDAND and OOROR nodes (if it ever has such info).
	}
	if cond.Op == OOROR {
		mid := s.f.NewBlock(ssa.BlockPlain)
		s.stmtList(cond.Ninit)
		s.condBranch(cond.Left, yes, mid, min8(likely, 0))
		s.startBlock(mid)
		s.condBranch(cond.Right, yes, no, likely)
		return
		// Note: if likely==-1, then both recursive calls pass -1.
		// If likely==1, then we don't have enough info to decide
		// the likelihood of the first branch.
	}
	if cond.Op == ONOT {
		s.stmtList(cond.Ninit)
		s.condBranch(cond.Left, no, yes, -likely)
		return
	}
	c := s.expr(cond)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = c
	b.Likely = ssa.BranchPrediction(likely) // gc and ssa both use -1/0/+1 for likeliness
	b.AddEdgeTo(yes)
	b.AddEdgeTo(no)
}

func (s *state) assign(left *Node, right *ssa.Value, wb bool, line int32) {
	if left.Op == ONAME && isblank(left) {
		return
	}
	t := left.Type
	dowidth(t)
	if right == nil {
		// right == nil means use the zero value of the assigned type.
		if !canSSA(left) {
			// if we can't ssa this memory, treat it as just zeroing out the backing memory
			addr := s.addr(left, false)
			if left.Op == ONAME {
				s.vars[&memVar] = s.newValue1A(ssa.OpVarDef, ssa.TypeMem, left, s.mem())
			}
			s.vars[&memVar] = s.newValue2I(ssa.OpZero, ssa.TypeMem, t.Size(), addr, s.mem())
			return
		}
		right = s.zeroVal(t)
	}
	if left.Op == ONAME && canSSA(left) {
		// Update variable assignment.
		s.vars[left] = right
		s.addNamedValue(left, right)
		return
	}
	// not ssa-able.  Treat as a store.
	addr := s.addr(left, false)
	if left.Op == ONAME {
		s.vars[&memVar] = s.newValue1A(ssa.OpVarDef, ssa.TypeMem, left, s.mem())
	}
	s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, t.Size(), addr, right, s.mem())
	if wb {
		s.insertWB(left.Type, addr, line)
	}
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
	case t.IsFloat():
		switch t.Size() {
		case 4:
			return s.constFloat32(t, 0)
		case 8:
			return s.constFloat64(t, 0)
		default:
			s.Fatalf("bad sized float type %s", t)
		}
	case t.IsComplex():
		switch t.Size() {
		case 8:
			z := s.constFloat32(Types[TFLOAT32], 0)
			return s.entryNewValue2(ssa.OpComplexMake, t, z, z)
		case 16:
			z := s.constFloat64(Types[TFLOAT64], 0)
			return s.entryNewValue2(ssa.OpComplexMake, t, z, z)
		default:
			s.Fatalf("bad sized complex type %s", t)
		}

	case t.IsString():
		return s.entryNewValue0A(ssa.OpConstString, t, "")
	case t.IsPtr():
		return s.entryNewValue0(ssa.OpConstNil, t)
	case t.IsBoolean():
		return s.constBool(false)
	case t.IsInterface():
		return s.entryNewValue0(ssa.OpConstInterface, t)
	case t.IsSlice():
		return s.entryNewValue0(ssa.OpConstSlice, t)
	}
	s.Unimplementedf("zero for type %v not implemented", t)
	return nil
}

type callKind int8

const (
	callNormal callKind = iota
	callDefer
	callGo
)

func (s *state) call(n *Node, k callKind) *ssa.Value {
	var sym *Sym           // target symbol (if static)
	var closure *ssa.Value // ptr to closure to run (if dynamic)
	var codeptr *ssa.Value // ptr to target code (if dynamic)
	var rcvr *ssa.Value    // receiver to set
	fn := n.Left
	switch n.Op {
	case OCALLFUNC:
		if k == callNormal && fn.Op == ONAME && fn.Class == PFUNC {
			sym = fn.Sym
			break
		}
		closure = s.expr(fn)
		if closure == nil {
			return nil // TODO: remove when expr always returns non-nil
		}
	case OCALLMETH:
		if fn.Op != ODOTMETH {
			Fatalf("OCALLMETH: n.Left not an ODOTMETH: %v", fn)
		}
		if fn.Right.Op != ONAME {
			Fatalf("OCALLMETH: n.Left.Right not a ONAME: %v", fn.Right)
		}
		if k == callNormal {
			sym = fn.Right.Sym
			break
		}
		n2 := *fn.Right
		n2.Class = PFUNC
		closure = s.expr(&n2)
		// Note: receiver is already assigned in n.List, so we don't
		// want to set it here.
	case OCALLINTER:
		if fn.Op != ODOTINTER {
			Fatalf("OCALLINTER: n.Left not an ODOTINTER: %v", Oconv(int(fn.Op), 0))
		}
		i := s.expr(fn.Left)
		itab := s.newValue1(ssa.OpITab, Types[TUINTPTR], i)
		itabidx := fn.Xoffset + 3*int64(Widthptr) + 8 // offset of fun field in runtime.itab
		itab = s.newValue1I(ssa.OpOffPtr, Types[TUINTPTR], itabidx, itab)
		if k == callNormal {
			codeptr = s.newValue2(ssa.OpLoad, Types[TUINTPTR], itab, s.mem())
		} else {
			closure = itab
		}
		rcvr = s.newValue1(ssa.OpIData, Types[TUINTPTR], i)
	}
	dowidth(fn.Type)
	stksize := fn.Type.Argwid // includes receiver

	// Run all argument assignments.  The arg slots have already
	// been offset by the appropriate amount (+2*widthptr for go/defer,
	// +widthptr for interface calls).
	// For OCALLMETH, the receiver is set in these statements.
	s.stmtList(n.List)

	// Set receiver (for interface calls)
	if rcvr != nil {
		argStart := Ctxt.FixedFrameSize()
		if k != callNormal {
			argStart += int64(2 * Widthptr)
		}
		addr := s.entryNewValue1I(ssa.OpOffPtr, Types[TUINTPTR], argStart, s.sp)
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, int64(Widthptr), addr, rcvr, s.mem())
	}

	// Defer/go args
	if k != callNormal {
		// Write argsize and closure (args to Newproc/Deferproc).
		argsize := s.constInt32(Types[TUINT32], int32(stksize))
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, 4, s.sp, argsize, s.mem())
		addr := s.entryNewValue1I(ssa.OpOffPtr, Ptrto(Types[TUINTPTR]), int64(Widthptr), s.sp)
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, int64(Widthptr), addr, closure, s.mem())
		stksize += 2 * int64(Widthptr)
	}

	// call target
	bNext := s.f.NewBlock(ssa.BlockPlain)
	var call *ssa.Value
	switch {
	case k == callDefer:
		call = s.newValue1(ssa.OpDeferCall, ssa.TypeMem, s.mem())
	case k == callGo:
		call = s.newValue1(ssa.OpGoCall, ssa.TypeMem, s.mem())
	case closure != nil:
		codeptr = s.newValue2(ssa.OpLoad, Types[TUINTPTR], closure, s.mem())
		call = s.newValue3(ssa.OpClosureCall, ssa.TypeMem, codeptr, closure, s.mem())
	case codeptr != nil:
		call = s.newValue2(ssa.OpInterCall, ssa.TypeMem, codeptr, s.mem())
	case sym != nil:
		call = s.newValue1A(ssa.OpStaticCall, ssa.TypeMem, sym, s.mem())
	default:
		Fatalf("bad call type %s %v", opnames[n.Op], n)
	}
	call.AuxInt = stksize // Call operations carry the argsize of the callee along with them

	// Finish call block
	s.vars[&memVar] = call
	b := s.endBlock()
	b.Kind = ssa.BlockCall
	b.Control = call
	b.AddEdgeTo(bNext)

	// Read result from stack at the start of the fallthrough block
	s.startBlock(bNext)
	var titer Iter
	fp := Structfirst(&titer, Getoutarg(n.Left.Type))
	if fp == nil || k != callNormal {
		// call has no return value. Continue with the next statement.
		return nil
	}
	a := s.entryNewValue1I(ssa.OpOffPtr, Ptrto(fp.Type), fp.Width, s.sp)
	return s.newValue2(ssa.OpLoad, fp.Type, a, call)
}

// etypesign returns the signed-ness of e, for integer/pointer etypes.
// -1 means signed, +1 means unsigned, 0 means non-integer/non-pointer.
func etypesign(e EType) int8 {
	switch e {
	case TINT8, TINT16, TINT32, TINT64, TINT:
		return -1
	case TUINT8, TUINT16, TUINT32, TUINT64, TUINT, TUINTPTR, TUNSAFEPTR:
		return +1
	}
	return 0
}

// lookupSymbol is used to retrieve the symbol (Extern, Arg or Auto) used for a particular node.
// This improves the effectiveness of cse by using the same Aux values for the
// same symbols.
func (s *state) lookupSymbol(n *Node, sym interface{}) interface{} {
	switch sym.(type) {
	default:
		s.Fatalf("sym %v is of uknown type %T", sym, sym)
	case *ssa.ExternSymbol, *ssa.ArgSymbol, *ssa.AutoSymbol:
		// these are the only valid types
	}

	if lsym, ok := s.varsyms[n]; ok {
		return lsym
	} else {
		s.varsyms[n] = sym
		return sym
	}
}

// addr converts the address of the expression n to SSA, adds it to s and returns the SSA result.
// The value that the returned Value represents is guaranteed to be non-nil.
// If bounded is true then this address does not require a nil check for its operand
// even if that would otherwise be implied.
func (s *state) addr(n *Node, bounded bool) *ssa.Value {
	t := Ptrto(n.Type)
	switch n.Op {
	case ONAME:
		switch n.Class {
		case PEXTERN:
			// global variable
			aux := s.lookupSymbol(n, &ssa.ExternSymbol{n.Type, n.Sym})
			v := s.entryNewValue1A(ssa.OpAddr, t, aux, s.sb)
			// TODO: Make OpAddr use AuxInt as well as Aux.
			if n.Xoffset != 0 {
				v = s.entryNewValue1I(ssa.OpOffPtr, v.Type, n.Xoffset, v)
			}
			return v
		case PPARAM:
			// parameter slot
			v := s.decladdrs[n]
			if v != nil {
				return v
			}
			if n.String() == ".fp" {
				// Special arg that points to the frame pointer.
				// (Used by the race detector, others?)
				aux := s.lookupSymbol(n, &ssa.ArgSymbol{Typ: n.Type, Node: n})
				return s.entryNewValue1A(ssa.OpAddr, t, aux, s.sp)
			}
			s.Fatalf("addr of undeclared ONAME %v. declared: %v", n, s.decladdrs)
			return nil
		case PAUTO:
			// We need to regenerate the address of autos
			// at every use.  This prevents LEA instructions
			// from occurring before the corresponding VarDef
			// op and confusing the liveness analysis into thinking
			// the variable is live at function entry.
			// TODO: I'm not sure if this really works or we're just
			// getting lucky.  We might need a real dependency edge
			// between vardef and addr ops.
			aux := &ssa.AutoSymbol{Typ: n.Type, Node: n}
			return s.newValue1A(ssa.OpAddr, t, aux, s.sp)
		case PPARAMOUT: // Same as PAUTO -- cannot generate LEA early.
			// ensure that we reuse symbols for out parameters so
			// that cse works on their addresses
			aux := s.lookupSymbol(n, &ssa.ArgSymbol{Typ: n.Type, Node: n})
			return s.newValue1A(ssa.OpAddr, t, aux, s.sp)
		case PAUTO | PHEAP, PPARAM | PHEAP, PPARAMOUT | PHEAP, PPARAMREF:
			return s.expr(n.Name.Heapaddr)
		default:
			s.Unimplementedf("variable address class %v not implemented", n.Class)
			return nil
		}
	case OINDREG:
		// indirect off a register
		// used for storing/loading arguments/returns to/from callees
		if int(n.Reg) != Thearch.REGSP {
			s.Unimplementedf("OINDREG of non-SP register %s in addr: %v", obj.Rconv(int(n.Reg)), n)
			return nil
		}
		return s.entryNewValue1I(ssa.OpOffPtr, t, n.Xoffset, s.sp)
	case OINDEX:
		if n.Left.Type.IsSlice() {
			a := s.expr(n.Left)
			i := s.expr(n.Right)
			i = s.extendIndex(i)
			len := s.newValue1(ssa.OpSliceLen, Types[TINT], a)
			if !n.Bounded {
				s.boundsCheck(i, len)
			}
			p := s.newValue1(ssa.OpSlicePtr, t, a)
			return s.newValue2(ssa.OpPtrIndex, t, p, i)
		} else { // array
			a := s.addr(n.Left, bounded)
			i := s.expr(n.Right)
			i = s.extendIndex(i)
			len := s.constInt(Types[TINT], n.Left.Type.Bound)
			if !n.Bounded {
				s.boundsCheck(i, len)
			}
			return s.newValue2(ssa.OpPtrIndex, Ptrto(n.Left.Type.Type), a, i)
		}
	case OIND:
		p := s.expr(n.Left)
		if !bounded {
			s.nilCheck(p)
		}
		return p
	case ODOT:
		p := s.addr(n.Left, bounded)
		return s.newValue2(ssa.OpAddPtr, t, p, s.constInt(Types[TINT], n.Xoffset))
	case ODOTPTR:
		p := s.expr(n.Left)
		if !bounded {
			s.nilCheck(p)
		}
		return s.newValue2(ssa.OpAddPtr, t, p, s.constInt(Types[TINT], n.Xoffset))
	case OCLOSUREVAR:
		return s.newValue2(ssa.OpAddPtr, t,
			s.entryNewValue0(ssa.OpGetClosurePtr, Ptrto(Types[TUINT8])),
			s.constInt(Types[TINT], n.Xoffset))
	case OPARAM:
		p := n.Left
		if p.Op != ONAME || !(p.Class == PPARAM|PHEAP || p.Class == PPARAMOUT|PHEAP) {
			s.Fatalf("OPARAM not of ONAME,{PPARAM,PPARAMOUT}|PHEAP, instead %s", nodedump(p, 0))
		}

		// Recover original offset to address passed-in param value.
		original_p := *p
		original_p.Xoffset = n.Xoffset
		aux := &ssa.ArgSymbol{Typ: n.Type, Node: &original_p}
		return s.entryNewValue1A(ssa.OpAddr, t, aux, s.sp)
	case OCONVNOP:
		addr := s.addr(n.Left, bounded)
		return s.newValue1(ssa.OpCopy, t, addr) // ensure that addr has the right type

	default:
		s.Unimplementedf("unhandled addr %v", Oconv(int(n.Op), 0))
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
	switch n.Class {
	case PEXTERN, PPARAMOUT, PPARAMREF:
		return false
	}
	if n.Class == PPARAM && n.String() == ".this" {
		// wrappers generated by genwrapper need to update
		// the .this pointer in place.
		return false
	}
	return canSSAType(n.Type)
	// TODO: try to make more variables SSAable?
}

// canSSA reports whether variables of type t are SSA-able.
func canSSAType(t *Type) bool {
	dowidth(t)
	if t.Width > int64(4*Widthptr) {
		// 4*Widthptr is an arbitrary constant.  We want it
		// to be at least 3*Widthptr so slices can be registerized.
		// Too big and we'll introduce too much register pressure.
		return false
	}
	switch t.Etype {
	case TARRAY:
		if Isslice(t) {
			return true
		}
		// We can't do arrays because dynamic indexing is
		// not supported on SSA variables.
		// TODO: maybe allow if length is <=1?  All indexes
		// are constant?  Might be good for the arrays
		// introduced by the compiler for variadic functions.
		return false
	case TSTRUCT:
		if countfield(t) > 4 {
			// 4 is an arbitrary constant.  Same reasoning
			// as above, lots of small fields would waste
			// register space needed by other values.
			return false
		}
		for t1 := t.Type; t1 != nil; t1 = t1.Down {
			if !canSSAType(t1.Type) {
				return false
			}
		}
		return false // until it is implemented
		//return true
	default:
		return true
	}
}

// nilCheck generates nil pointer checking code.
// Starts a new block on return, unless nil checks are disabled.
// Used only for automatically inserted nil checks,
// not for user code like 'x != nil'.
func (s *state) nilCheck(ptr *ssa.Value) {
	if Disable_checknil != 0 {
		return
	}
	chk := s.newValue2(ssa.OpNilCheck, ssa.TypeVoid, ptr, s.mem())
	b := s.endBlock()
	b.Kind = ssa.BlockCheck
	b.Control = chk
	bNext := s.f.NewBlock(ssa.BlockPlain)
	b.AddEdgeTo(bNext)
	s.startBlock(bNext)
}

// boundsCheck generates bounds checking code.  Checks if 0 <= idx < len, branches to exit if not.
// Starts a new block on return.
func (s *state) boundsCheck(idx, len *ssa.Value) {
	if Debug['B'] != 0 {
		return
	}
	// TODO: convert index to full width?
	// TODO: if index is 64-bit and we're compiling to 32-bit, check that high 32 bits are zero.

	// bounds check
	cmp := s.newValue2(ssa.OpIsInBounds, Types[TBOOL], idx, len)
	s.check(cmp, Panicindex)
}

// sliceBoundsCheck generates slice bounds checking code.  Checks if 0 <= idx <= len, branches to exit if not.
// Starts a new block on return.
func (s *state) sliceBoundsCheck(idx, len *ssa.Value) {
	if Debug['B'] != 0 {
		return
	}
	// TODO: convert index to full width?
	// TODO: if index is 64-bit and we're compiling to 32-bit, check that high 32 bits are zero.

	// bounds check
	cmp := s.newValue2(ssa.OpIsSliceInBounds, Types[TBOOL], idx, len)
	s.check(cmp, panicslice)
}

// If cmp (a bool) is true, panic using the given function.
func (s *state) check(cmp *ssa.Value, fn *Node) {
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = cmp
	b.Likely = ssa.BranchLikely
	bNext := s.f.NewBlock(ssa.BlockPlain)
	line := s.peekLine()
	bPanic := s.panics[funcLine{fn, line}]
	if bPanic == nil {
		bPanic = s.f.NewBlock(ssa.BlockPlain)
		s.panics[funcLine{fn, line}] = bPanic
		s.startBlock(bPanic)
		// The panic call takes/returns memory to ensure that the right
		// memory state is observed if the panic happens.
		s.rtcall(fn, false, nil)
	}
	b.AddEdgeTo(bNext)
	b.AddEdgeTo(bPanic)
	s.startBlock(bNext)
}

// rtcall issues a call to the given runtime function fn with the listed args.
// Returns a slice of results of the given result types.
// The call is added to the end of the current block.
// If returns is false, the block is marked as an exit block.
// If returns is true, the block is marked as a call block.  A new block
// is started to load the return values.
func (s *state) rtcall(fn *Node, returns bool, results []*Type, args ...*ssa.Value) []*ssa.Value {
	// Write args to the stack
	var off int64 // TODO: arch-dependent starting offset?
	for _, arg := range args {
		t := arg.Type
		off = Rnd(off, t.Alignment())
		ptr := s.sp
		if off != 0 {
			ptr = s.newValue1I(ssa.OpOffPtr, Types[TUINTPTR], off, s.sp)
		}
		size := t.Size()
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, size, ptr, arg, s.mem())
		off += size
	}
	off = Rnd(off, int64(Widthptr))

	// Issue call
	call := s.newValue1A(ssa.OpStaticCall, ssa.TypeMem, fn.Sym, s.mem())
	s.vars[&memVar] = call

	// Finish block
	b := s.endBlock()
	if !returns {
		b.Kind = ssa.BlockExit
		b.Control = call
		call.AuxInt = off
		if len(results) > 0 {
			Fatalf("panic call can't have results")
		}
		return nil
	}
	b.Kind = ssa.BlockCall
	b.Control = call
	bNext := s.f.NewBlock(ssa.BlockPlain)
	b.AddEdgeTo(bNext)
	s.startBlock(bNext)

	// Load results
	res := make([]*ssa.Value, len(results))
	for i, t := range results {
		off = Rnd(off, t.Alignment())
		ptr := s.sp
		if off != 0 {
			ptr = s.newValue1I(ssa.OpOffPtr, Types[TUINTPTR], off, s.sp)
		}
		res[i] = s.newValue2(ssa.OpLoad, t, ptr, s.mem())
		off += t.Size()
	}
	off = Rnd(off, int64(Widthptr))

	// Remember how much callee stack space we needed.
	call.AuxInt = off

	return res
}

// insertWB inserts a write barrier.  A value of type t has already
// been stored at location p.  Tell the runtime about this write.
// Note: there must be no GC suspension points between the write and
// the call that this function inserts.
func (s *state) insertWB(t *Type, p *ssa.Value, line int32) {
	// if writeBarrier.enabled {
	//   typedmemmove_nostore(&t, p)
	// }
	bThen := s.f.NewBlock(ssa.BlockPlain)

	aux := &ssa.ExternSymbol{Types[TBOOL], syslook("writeBarrier", 0).Sym}
	flagaddr := s.newValue1A(ssa.OpAddr, Ptrto(Types[TBOOL]), aux, s.sb)
	// TODO: select the .enabled field.  It is currently first, so not needed for now.
	flag := s.newValue2(ssa.OpLoad, Types[TBOOL], flagaddr, s.mem())
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Likely = ssa.BranchUnlikely
	b.Control = flag
	b.AddEdgeTo(bThen)

	s.startBlock(bThen)
	// TODO: writebarrierptr_nostore if just one pointer word (or a few?)
	taddr := s.newValue1A(ssa.OpAddr, Types[TUINTPTR], &ssa.ExternSymbol{Types[TUINTPTR], typenamesym(t)}, s.sb)
	s.rtcall(typedmemmove_nostore, true, nil, taddr, p)

	if Debug_wb > 0 {
		Warnl(int(line), "write barrier")
	}

	b.AddEdgeTo(s.curBlock)
}

// slice computes the slice v[i:j:k] and returns ptr, len, and cap of result.
// i,j,k may be nil, in which case they are set to their default value.
// t is a slice, ptr to array, or string type.
func (s *state) slice(t *Type, v, i, j, k *ssa.Value) (p, l, c *ssa.Value) {
	var elemtype *Type
	var ptrtype *Type
	var ptr *ssa.Value
	var len *ssa.Value
	var cap *ssa.Value
	zero := s.constInt(Types[TINT], 0)
	switch {
	case t.IsSlice():
		elemtype = t.Type
		ptrtype = Ptrto(elemtype)
		ptr = s.newValue1(ssa.OpSlicePtr, ptrtype, v)
		len = s.newValue1(ssa.OpSliceLen, Types[TINT], v)
		cap = s.newValue1(ssa.OpSliceCap, Types[TINT], v)
	case t.IsString():
		elemtype = Types[TUINT8]
		ptrtype = Ptrto(elemtype)
		ptr = s.newValue1(ssa.OpStringPtr, ptrtype, v)
		len = s.newValue1(ssa.OpStringLen, Types[TINT], v)
		cap = len
	case t.IsPtr():
		if !t.Type.IsArray() {
			s.Fatalf("bad ptr to array in slice %v\n", t)
		}
		elemtype = t.Type.Type
		ptrtype = Ptrto(elemtype)
		s.nilCheck(v)
		ptr = v
		len = s.constInt(Types[TINT], t.Type.Bound)
		cap = len
	default:
		s.Fatalf("bad type in slice %v\n", t)
	}

	// Set default values
	if i == nil {
		i = zero
	}
	if j == nil {
		j = len
	}
	if k == nil {
		k = cap
	}

	// Panic if slice indices are not in bounds.
	s.sliceBoundsCheck(i, j)
	if j != k {
		s.sliceBoundsCheck(j, k)
	}
	if k != cap {
		s.sliceBoundsCheck(k, cap)
	}

	// Generate the following code assuming that indexes are in bounds.
	// The conditional is to make sure that we don't generate a slice
	// that points to the next object in memory.
	// rlen = (Sub64 j i)
	// rcap = (Sub64 k i)
	// p = ptr
	// if rcap != 0 {
	//    p = (AddPtr ptr (Mul64 low (Const64 size)))
	// }
	// result = (SliceMake p size)
	subOp := s.ssaOp(OSUB, Types[TINT])
	neqOp := s.ssaOp(ONE, Types[TINT])
	mulOp := s.ssaOp(OMUL, Types[TINT])
	rlen := s.newValue2(subOp, Types[TINT], j, i)
	var rcap *ssa.Value
	switch {
	case t.IsString():
		// Capacity of the result is unimportant.  However, we use
		// rcap to test if we've generated a zero-length slice.
		// Use length of strings for that.
		rcap = rlen
	case j == k:
		rcap = rlen
	default:
		rcap = s.newValue2(subOp, Types[TINT], k, i)
	}

	s.vars[&ptrVar] = ptr

	// Generate code to test the resulting slice length.
	cmp := s.newValue2(neqOp, Types[TBOOL], rcap, s.constInt(Types[TINT], 0))

	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Likely = ssa.BranchLikely
	b.Control = cmp

	// Generate code for non-zero length slice case.
	nz := s.f.NewBlock(ssa.BlockPlain)
	b.AddEdgeTo(nz)
	s.startBlock(nz)
	var inc *ssa.Value
	if elemtype.Width == 1 {
		inc = i
	} else {
		inc = s.newValue2(mulOp, Types[TINT], i, s.constInt(Types[TINT], elemtype.Width))
	}
	s.vars[&ptrVar] = s.newValue2(ssa.OpAddPtr, ptrtype, ptr, inc)
	s.endBlock()

	// All done.
	merge := s.f.NewBlock(ssa.BlockPlain)
	b.AddEdgeTo(merge)
	nz.AddEdgeTo(merge)
	s.startBlock(merge)
	rptr := s.variable(&ptrVar, ptrtype)
	delete(s.vars, &ptrVar)
	return rptr, rlen, rcap
}

type u2fcvtTab struct {
	geq, cvt2F, and, rsh, or, add ssa.Op
	one                           func(*state, ssa.Type, int64) *ssa.Value
}

var u64_f64 u2fcvtTab = u2fcvtTab{
	geq:   ssa.OpGeq64,
	cvt2F: ssa.OpCvt64to64F,
	and:   ssa.OpAnd64,
	rsh:   ssa.OpRsh64Ux64,
	or:    ssa.OpOr64,
	add:   ssa.OpAdd64F,
	one:   (*state).constInt64,
}

var u64_f32 u2fcvtTab = u2fcvtTab{
	geq:   ssa.OpGeq64,
	cvt2F: ssa.OpCvt64to32F,
	and:   ssa.OpAnd64,
	rsh:   ssa.OpRsh64Ux64,
	or:    ssa.OpOr64,
	add:   ssa.OpAdd32F,
	one:   (*state).constInt64,
}

// Excess generality on a machine with 64-bit integer registers.
// Not used on AMD64.
var u32_f32 u2fcvtTab = u2fcvtTab{
	geq:   ssa.OpGeq32,
	cvt2F: ssa.OpCvt32to32F,
	and:   ssa.OpAnd32,
	rsh:   ssa.OpRsh32Ux32,
	or:    ssa.OpOr32,
	add:   ssa.OpAdd32F,
	one: func(s *state, t ssa.Type, x int64) *ssa.Value {
		return s.constInt32(t, int32(x))
	},
}

func (s *state) uint64Tofloat64(n *Node, x *ssa.Value, ft, tt *Type) *ssa.Value {
	return s.uintTofloat(&u64_f64, n, x, ft, tt)
}

func (s *state) uint64Tofloat32(n *Node, x *ssa.Value, ft, tt *Type) *ssa.Value {
	return s.uintTofloat(&u64_f32, n, x, ft, tt)
}

func (s *state) uintTofloat(cvttab *u2fcvtTab, n *Node, x *ssa.Value, ft, tt *Type) *ssa.Value {
	// if x >= 0 {
	//    result = (floatY) x
	// } else {
	// 	  y = uintX(x) ; y = x & 1
	// 	  z = uintX(x) ; z = z >> 1
	// 	  z = z >> 1
	// 	  z = z | y
	// 	  result = floatY(z)
	// 	  result = result + result
	// }
	//
	// Code borrowed from old code generator.
	// What's going on: large 64-bit "unsigned" looks like
	// negative number to hardware's integer-to-float
	// conversion.  However, because the mantissa is only
	// 63 bits, we don't need the LSB, so instead we do an
	// unsigned right shift (divide by two), convert, and
	// double.  However, before we do that, we need to be
	// sure that we do not lose a "1" if that made the
	// difference in the resulting rounding.  Therefore, we
	// preserve it, and OR (not ADD) it back in.  The case
	// that matters is when the eleven discarded bits are
	// equal to 10000000001; that rounds up, and the 1 cannot
	// be lost else it would round down if the LSB of the
	// candidate mantissa is 0.
	cmp := s.newValue2(cvttab.geq, Types[TBOOL], x, s.zeroVal(ft))
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = cmp
	b.Likely = ssa.BranchLikely

	bThen := s.f.NewBlock(ssa.BlockPlain)
	bElse := s.f.NewBlock(ssa.BlockPlain)
	bAfter := s.f.NewBlock(ssa.BlockPlain)

	b.AddEdgeTo(bThen)
	s.startBlock(bThen)
	a0 := s.newValue1(cvttab.cvt2F, tt, x)
	s.vars[n] = a0
	s.endBlock()
	bThen.AddEdgeTo(bAfter)

	b.AddEdgeTo(bElse)
	s.startBlock(bElse)
	one := cvttab.one(s, ft, 1)
	y := s.newValue2(cvttab.and, ft, x, one)
	z := s.newValue2(cvttab.rsh, ft, x, one)
	z = s.newValue2(cvttab.or, ft, z, y)
	a := s.newValue1(cvttab.cvt2F, tt, z)
	a1 := s.newValue2(cvttab.add, tt, a, a)
	s.vars[n] = a1
	s.endBlock()
	bElse.AddEdgeTo(bAfter)

	s.startBlock(bAfter)
	return s.variable(n, n.Type)
}

// referenceTypeBuiltin generates code for the len/cap builtins for maps and channels.
func (s *state) referenceTypeBuiltin(n *Node, x *ssa.Value) *ssa.Value {
	if !n.Left.Type.IsMap() && !n.Left.Type.IsChan() {
		s.Fatalf("node must be a map or a channel")
	}
	// if n == nil {
	//   return 0
	// } else {
	//   // len
	//   return *((*int)n)
	//   // cap
	//   return *(((*int)n)+1)
	// }
	lenType := n.Type
	nilValue := s.newValue0(ssa.OpConstNil, Types[TUINTPTR])
	cmp := s.newValue2(ssa.OpEqPtr, Types[TBOOL], x, nilValue)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = cmp
	b.Likely = ssa.BranchUnlikely

	bThen := s.f.NewBlock(ssa.BlockPlain)
	bElse := s.f.NewBlock(ssa.BlockPlain)
	bAfter := s.f.NewBlock(ssa.BlockPlain)

	// length/capacity of a nil map/chan is zero
	b.AddEdgeTo(bThen)
	s.startBlock(bThen)
	s.vars[n] = s.zeroVal(lenType)
	s.endBlock()
	bThen.AddEdgeTo(bAfter)

	b.AddEdgeTo(bElse)
	s.startBlock(bElse)
	if n.Op == OLEN {
		// length is stored in the first word for map/chan
		s.vars[n] = s.newValue2(ssa.OpLoad, lenType, x, s.mem())
	} else if n.Op == OCAP {
		// capacity is stored in the second word for chan
		sw := s.newValue1I(ssa.OpOffPtr, lenType.PtrTo(), lenType.Width, x)
		s.vars[n] = s.newValue2(ssa.OpLoad, lenType, sw, s.mem())
	} else {
		s.Fatalf("op must be OLEN or OCAP")
	}
	s.endBlock()
	bElse.AddEdgeTo(bAfter)

	s.startBlock(bAfter)
	return s.variable(n, lenType)
}

type f2uCvtTab struct {
	ltf, cvt2U, subf ssa.Op
	value            func(*state, ssa.Type, float64) *ssa.Value
}

var f32_u64 f2uCvtTab = f2uCvtTab{
	ltf:   ssa.OpLess32F,
	cvt2U: ssa.OpCvt32Fto64,
	subf:  ssa.OpSub32F,
	value: (*state).constFloat32,
}

var f64_u64 f2uCvtTab = f2uCvtTab{
	ltf:   ssa.OpLess64F,
	cvt2U: ssa.OpCvt64Fto64,
	subf:  ssa.OpSub64F,
	value: (*state).constFloat64,
}

func (s *state) float32ToUint64(n *Node, x *ssa.Value, ft, tt *Type) *ssa.Value {
	return s.floatToUint(&f32_u64, n, x, ft, tt)
}
func (s *state) float64ToUint64(n *Node, x *ssa.Value, ft, tt *Type) *ssa.Value {
	return s.floatToUint(&f64_u64, n, x, ft, tt)
}

func (s *state) floatToUint(cvttab *f2uCvtTab, n *Node, x *ssa.Value, ft, tt *Type) *ssa.Value {
	// if x < 9223372036854775808.0 {
	// 	result = uintY(x)
	// } else {
	// 	y = x - 9223372036854775808.0
	// 	z = uintY(y)
	// 	result = z | -9223372036854775808
	// }
	twoToThe63 := cvttab.value(s, ft, 9223372036854775808.0)
	cmp := s.newValue2(cvttab.ltf, Types[TBOOL], x, twoToThe63)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = cmp
	b.Likely = ssa.BranchLikely

	bThen := s.f.NewBlock(ssa.BlockPlain)
	bElse := s.f.NewBlock(ssa.BlockPlain)
	bAfter := s.f.NewBlock(ssa.BlockPlain)

	b.AddEdgeTo(bThen)
	s.startBlock(bThen)
	a0 := s.newValue1(cvttab.cvt2U, tt, x)
	s.vars[n] = a0
	s.endBlock()
	bThen.AddEdgeTo(bAfter)

	b.AddEdgeTo(bElse)
	s.startBlock(bElse)
	y := s.newValue2(cvttab.subf, ft, x, twoToThe63)
	y = s.newValue1(cvttab.cvt2U, tt, y)
	z := s.constInt64(tt, -9223372036854775808)
	a1 := s.newValue2(ssa.OpOr64, tt, y, z)
	s.vars[n] = a1
	s.endBlock()
	bElse.AddEdgeTo(bAfter)

	s.startBlock(bAfter)
	return s.variable(n, n.Type)
}

// ifaceType returns the value for the word containing the type.
// n is the node for the interface expression.
// v is the corresponding value.
func (s *state) ifaceType(n *Node, v *ssa.Value) *ssa.Value {
	byteptr := Ptrto(Types[TUINT8]) // type used in runtime prototypes for runtime type (*byte)

	if isnilinter(n.Type) {
		// Have *eface. The type is the first word in the struct.
		return s.newValue1(ssa.OpITab, byteptr, v)
	}

	// Have *iface.
	// The first word in the struct is the *itab.
	// If the *itab is nil, return 0.
	// Otherwise, the second word in the *itab is the type.

	tab := s.newValue1(ssa.OpITab, byteptr, v)
	s.vars[&typVar] = tab
	isnonnil := s.newValue2(ssa.OpNeqPtr, Types[TBOOL], tab, s.entryNewValue0(ssa.OpConstNil, byteptr))
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = isnonnil
	b.Likely = ssa.BranchLikely

	bLoad := s.f.NewBlock(ssa.BlockPlain)
	bEnd := s.f.NewBlock(ssa.BlockPlain)

	b.AddEdgeTo(bLoad)
	b.AddEdgeTo(bEnd)
	bLoad.AddEdgeTo(bEnd)

	s.startBlock(bLoad)
	off := s.newValue1I(ssa.OpOffPtr, byteptr, int64(Widthptr), tab)
	s.vars[&typVar] = s.newValue2(ssa.OpLoad, byteptr, off, s.mem())
	s.endBlock()

	s.startBlock(bEnd)
	typ := s.variable(&typVar, byteptr)
	delete(s.vars, &typVar)
	return typ
}

// dottype generates SSA for a type assertion node.
// commaok indicates whether to panic or return a bool.
// If commaok is false, resok will be nil.
func (s *state) dottype(n *Node, commaok bool) (res, resok *ssa.Value) {
	iface := s.expr(n.Left)
	typ := s.ifaceType(n.Left, iface)  // actual concrete type
	target := s.expr(typename(n.Type)) // target type
	if !isdirectiface(n.Type) {
		// walk rewrites ODOTTYPE/OAS2DOTTYPE into runtime calls except for this case.
		Fatalf("dottype needs a direct iface type %s", n.Type)
	}

	if Debug_typeassert > 0 {
		Warnl(int(n.Lineno), "type assertion inlined")
	}

	// TODO:  If we have a nonempty interface and its itab field is nil,
	// then this test is redundant and ifaceType should just branch directly to bFail.
	cond := s.newValue2(ssa.OpEqPtr, Types[TBOOL], typ, target)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Control = cond
	b.Likely = ssa.BranchLikely

	byteptr := Ptrto(Types[TUINT8])

	bOk := s.f.NewBlock(ssa.BlockPlain)
	bFail := s.f.NewBlock(ssa.BlockPlain)
	b.AddEdgeTo(bOk)
	b.AddEdgeTo(bFail)

	if !commaok {
		// on failure, panic by calling panicdottype
		s.startBlock(bFail)
		taddr := s.newValue1A(ssa.OpAddr, byteptr, &ssa.ExternSymbol{byteptr, typenamesym(n.Left.Type)}, s.sb)
		s.rtcall(panicdottype, false, nil, typ, target, taddr)

		// on success, return idata field
		s.startBlock(bOk)
		return s.newValue1(ssa.OpIData, n.Type, iface), nil
	}

	// commaok is the more complicated case because we have
	// a control flow merge point.
	bEnd := s.f.NewBlock(ssa.BlockPlain)

	// type assertion succeeded
	s.startBlock(bOk)
	s.vars[&idataVar] = s.newValue1(ssa.OpIData, n.Type, iface)
	s.vars[&okVar] = s.constBool(true)
	s.endBlock()
	bOk.AddEdgeTo(bEnd)

	// type assertion failed
	s.startBlock(bFail)
	s.vars[&idataVar] = s.entryNewValue0(ssa.OpConstNil, byteptr)
	s.vars[&okVar] = s.constBool(false)
	s.endBlock()
	bFail.AddEdgeTo(bEnd)

	// merge point
	s.startBlock(bEnd)
	res = s.variable(&idataVar, byteptr)
	resok = s.variable(&okVar, Types[TBOOL])
	delete(s.vars, &idataVar)
	delete(s.vars, &okVar)
	return res, resok
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
	return s.variable(&memVar, ssa.TypeMem)
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
		if name == &memVar {
			return s.startmem
		}
		if canSSA(name) {
			v := s.entryNewValue0A(ssa.OpArg, t, name)
			// v starts with AuxInt == 0.
			s.addNamedValue(name, v)
			return v
		}
		// variable is live at the entry block.  Load it.
		addr := s.decladdrs[name]
		if addr == nil {
			// TODO: closure args reach here.
			s.Unimplementedf("unhandled closure arg %s at entry to function %s", name, b.Func.Name)
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
		if name == &memVar {
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
			s.addNamedValue(name, v)
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

func (s *state) addNamedValue(n *Node, v *ssa.Value) {
	if n.Class == Pxxx {
		// Don't track our dummy nodes (&memVar etc.).
		return
	}
	if n.Sym == nil {
		// TODO: What the heck is this?
		return
	}
	if strings.HasPrefix(n.Sym.Name, "autotmp_") {
		// Don't track autotmp_ variables.
		return
	}
	if n.Class == PAUTO && (v.Type.IsString() || v.Type.IsSlice() || v.Type.IsInterface()) {
		// TODO: can't handle auto compound objects with pointers yet.
		// The live variable analysis barfs because we don't put VARDEF
		// pseudos in the right place when we spill to these nodes.
		return
	}
	if n.Class == PAUTO && n.Xoffset != 0 {
		s.Fatalf("AUTO var with offset %s %d", n, n.Xoffset)
	}
	loc := ssa.LocalSlot{N: n, Type: n.Type, Off: 0}
	values, ok := s.f.NamedValues[loc]
	if !ok {
		s.f.Names = append(s.f.Names, loc)
	}
	s.f.NamedValues[loc] = append(values, v)
}

// an unresolved branch
type branch struct {
	p *obj.Prog  // branch instruction
	b *ssa.Block // target
}

type genState struct {
	// branches remembers all the branch instructions we've seen
	// and where they would like to go.
	branches []branch

	// bstart remembers where each block starts (indexed by block ID)
	bstart []*obj.Prog

	// deferBranches remembers all the defer branches we've seen.
	deferBranches []*obj.Prog

	// deferTarget remembers the (last) deferreturn call site.
	deferTarget *obj.Prog
}

// genssa appends entries to ptxt for each instruction in f.
// gcargs and gclocals are filled in with pointer maps for the frame.
func genssa(f *ssa.Func, ptxt *obj.Prog, gcargs, gclocals *Sym) {
	var s genState

	e := f.Config.Frontend().(*ssaExport)
	// We're about to emit a bunch of Progs.
	// Since the only way to get here is to explicitly request it,
	// just fail on unimplemented instead of trying to unwind our mess.
	e.mustImplement = true

	// Remember where each block starts.
	s.bstart = make([]*obj.Prog, f.NumBlocks())

	var valueProgs map[*obj.Prog]*ssa.Value
	var blockProgs map[*obj.Prog]*ssa.Block
	const logProgs = true
	if logProgs {
		valueProgs = make(map[*obj.Prog]*ssa.Value, f.NumValues())
		blockProgs = make(map[*obj.Prog]*ssa.Block, f.NumBlocks())
		f.Logf("genssa %s\n", f.Name)
		blockProgs[Pc] = f.Blocks[0]
	}

	// Emit basic blocks
	for i, b := range f.Blocks {
		s.bstart[b.ID] = Pc
		// Emit values in block
		for _, v := range b.Values {
			x := Pc
			s.genValue(v)
			if logProgs {
				for ; x != Pc; x = x.Link {
					valueProgs[x] = v
				}
			}
		}
		// Emit control flow instructions for block
		var next *ssa.Block
		if i < len(f.Blocks)-1 {
			next = f.Blocks[i+1]
		}
		x := Pc
		s.genBlock(b, next)
		if logProgs {
			for ; x != Pc; x = x.Link {
				blockProgs[x] = b
			}
		}
	}

	// Resolve branches
	for _, br := range s.branches {
		br.p.To.Val = s.bstart[br.b.ID]
	}
	if s.deferBranches != nil && s.deferTarget == nil {
		// This can happen when the function has a defer but
		// no return (because it has an infinite loop).
		s.deferReturn()
		Prog(obj.ARET)
	}
	for _, p := range s.deferBranches {
		p.To.Val = s.deferTarget
	}

	if logProgs {
		for p := ptxt; p != nil; p = p.Link {
			var s string
			if v, ok := valueProgs[p]; ok {
				s = v.String()
			} else if b, ok := blockProgs[p]; ok {
				s = b.String()
			} else {
				s = "   " // most value and branch strings are 2-3 characters long
			}
			f.Logf("%s\t%s\n", s, p)
		}
		if f.Config.HTML != nil {
			saved := ptxt.Ctxt.LineHist.PrintFilenameOnly
			ptxt.Ctxt.LineHist.PrintFilenameOnly = true
			var buf bytes.Buffer
			buf.WriteString("<code>")
			buf.WriteString("<dl class=\"ssa-gen\">")
			for p := ptxt; p != nil; p = p.Link {
				buf.WriteString("<dt class=\"ssa-prog-src\">")
				if v, ok := valueProgs[p]; ok {
					buf.WriteString(v.HTML())
				} else if b, ok := blockProgs[p]; ok {
					buf.WriteString(b.HTML())
				}
				buf.WriteString("</dt>")
				buf.WriteString("<dd class=\"ssa-prog\">")
				buf.WriteString(html.EscapeString(p.String()))
				buf.WriteString("</dd>")
				buf.WriteString("</li>")
			}
			buf.WriteString("</dl>")
			buf.WriteString("</code>")
			f.Config.HTML.WriteColumn("genssa", buf.String())
			ptxt.Ctxt.LineHist.PrintFilenameOnly = saved
		}
	}

	// Emit static data
	if f.StaticData != nil {
		for _, n := range f.StaticData.([]*Node) {
			if !gen_as_init(n, false) {
				Fatalf("non-static data marked as static: %v\n\n", n, f)
			}
		}
	}

	// Allocate stack frame
	allocauto(ptxt)

	// Generate gc bitmaps.
	liveness(Curfn, ptxt, gcargs, gclocals)
	gcsymdup(gcargs)
	gcsymdup(gclocals)

	// Add frame prologue.  Zero ambiguously live variables.
	Thearch.Defframe(ptxt)
	if Debug['f'] != 0 {
		frame(0)
	}

	// Remove leftover instrumentation from the instruction stream.
	removevardef(ptxt)

	f.Config.HTML.Close()
}

// opregreg emits instructions for
//     dest := dest(To) op src(From)
// and also returns the created obj.Prog so it
// may be further adjusted (offset, scale, etc).
func opregreg(op int, dest, src int16) *obj.Prog {
	p := Prog(op)
	p.From.Type = obj.TYPE_REG
	p.To.Type = obj.TYPE_REG
	p.To.Reg = dest
	p.From.Reg = src
	return p
}

func (s *genState) genValue(v *ssa.Value) {
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
	case ssa.OpAMD64ADDB, ssa.OpAMD64ADDSS, ssa.OpAMD64ADDSD,
		ssa.OpAMD64ANDQ, ssa.OpAMD64ANDL, ssa.OpAMD64ANDW, ssa.OpAMD64ANDB,
		ssa.OpAMD64ORQ, ssa.OpAMD64ORL, ssa.OpAMD64ORW, ssa.OpAMD64ORB,
		ssa.OpAMD64XORQ, ssa.OpAMD64XORL, ssa.OpAMD64XORW, ssa.OpAMD64XORB,
		ssa.OpAMD64MULQ, ssa.OpAMD64MULL, ssa.OpAMD64MULW, ssa.OpAMD64MULB,
		ssa.OpAMD64MULSS, ssa.OpAMD64MULSD, ssa.OpAMD64PXOR:
		r := regnum(v)
		x := regnum(v.Args[0])
		y := regnum(v.Args[1])
		if x != r && y != r {
			opregreg(regMoveByTypeAMD64(v.Type), r, x)
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
			opregreg(regMoveByTypeAMD64(v.Type), r, x)
		}
		opregreg(v.Op.Asm(), r, y)

		if neg {
			p := Prog(x86.ANEGQ) // TODO: use correct size?  This is mostly a hack until regalloc does 2-address correctly
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		}
	case ssa.OpAMD64SUBSS, ssa.OpAMD64SUBSD, ssa.OpAMD64DIVSS, ssa.OpAMD64DIVSD:
		r := regnum(v)
		x := regnum(v.Args[0])
		y := regnum(v.Args[1])
		if y == r && x != r {
			// r/y := x op r/y, need to preserve x and rewrite to
			// r/y := r/y op x15
			x15 := int16(x86.REG_X15)
			// register move y to x15
			// register move x to y
			// rename y with x15
			opregreg(regMoveByTypeAMD64(v.Type), x15, y)
			opregreg(regMoveByTypeAMD64(v.Type), r, x)
			y = x15
		} else if x != r {
			opregreg(regMoveByTypeAMD64(v.Type), r, x)
		}
		opregreg(v.Op.Asm(), r, y)

	case ssa.OpAMD64DIVQ, ssa.OpAMD64DIVL, ssa.OpAMD64DIVW,
		ssa.OpAMD64DIVQU, ssa.OpAMD64DIVLU, ssa.OpAMD64DIVWU,
		ssa.OpAMD64MODQ, ssa.OpAMD64MODL, ssa.OpAMD64MODW,
		ssa.OpAMD64MODQU, ssa.OpAMD64MODLU, ssa.OpAMD64MODWU:

		// Arg[0] is already in AX as it's the only register we allow
		// and AX is the only output
		x := regnum(v.Args[1])

		// CPU faults upon signed overflow, which occurs when most
		// negative int is divided by -1.
		var j *obj.Prog
		if v.Op == ssa.OpAMD64DIVQ || v.Op == ssa.OpAMD64DIVL ||
			v.Op == ssa.OpAMD64DIVW || v.Op == ssa.OpAMD64MODQ ||
			v.Op == ssa.OpAMD64MODL || v.Op == ssa.OpAMD64MODW {

			var c *obj.Prog
			switch v.Op {
			case ssa.OpAMD64DIVQ, ssa.OpAMD64MODQ:
				c = Prog(x86.ACMPQ)
				j = Prog(x86.AJEQ)
				// go ahead and sign extend to save doing it later
				Prog(x86.ACQO)

			case ssa.OpAMD64DIVL, ssa.OpAMD64MODL:
				c = Prog(x86.ACMPL)
				j = Prog(x86.AJEQ)
				Prog(x86.ACDQ)

			case ssa.OpAMD64DIVW, ssa.OpAMD64MODW:
				c = Prog(x86.ACMPW)
				j = Prog(x86.AJEQ)
				Prog(x86.ACWD)
			}
			c.From.Type = obj.TYPE_REG
			c.From.Reg = x
			c.To.Type = obj.TYPE_CONST
			c.To.Offset = -1

			j.To.Type = obj.TYPE_BRANCH

		}

		// for unsigned ints, we sign extend by setting DX = 0
		// signed ints were sign extended above
		if v.Op == ssa.OpAMD64DIVQU || v.Op == ssa.OpAMD64MODQU ||
			v.Op == ssa.OpAMD64DIVLU || v.Op == ssa.OpAMD64MODLU ||
			v.Op == ssa.OpAMD64DIVWU || v.Op == ssa.OpAMD64MODWU {
			c := Prog(x86.AXORQ)
			c.From.Type = obj.TYPE_REG
			c.From.Reg = x86.REG_DX
			c.To.Type = obj.TYPE_REG
			c.To.Reg = x86.REG_DX
		}

		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x

		// signed division, rest of the check for -1 case
		if j != nil {
			j2 := Prog(obj.AJMP)
			j2.To.Type = obj.TYPE_BRANCH

			var n *obj.Prog
			if v.Op == ssa.OpAMD64DIVQ || v.Op == ssa.OpAMD64DIVL ||
				v.Op == ssa.OpAMD64DIVW {
				// n * -1 = -n
				n = Prog(x86.ANEGQ)
				n.To.Type = obj.TYPE_REG
				n.To.Reg = x86.REG_AX
			} else {
				// n % -1 == 0
				n = Prog(x86.AXORQ)
				n.From.Type = obj.TYPE_REG
				n.From.Reg = x86.REG_DX
				n.To.Type = obj.TYPE_REG
				n.To.Reg = x86.REG_DX
			}

			j.To.Val = n
			j2.To.Val = Pc
		}

	case ssa.OpAMD64HMULL, ssa.OpAMD64HMULW, ssa.OpAMD64HMULB,
		ssa.OpAMD64HMULLU, ssa.OpAMD64HMULWU, ssa.OpAMD64HMULBU:
		// the frontend rewrites constant division by 8/16/32 bit integers into
		// HMUL by a constant

		// Arg[0] is already in AX as it's the only register we allow
		// and DX is the only output we care about (the high bits)
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[1])

		// IMULB puts the high portion in AH instead of DL,
		// so move it to DL for consistency
		if v.Type.Size() == 1 {
			m := Prog(x86.AMOVB)
			m.From.Type = obj.TYPE_REG
			m.From.Reg = x86.REG_AH
			m.To.Type = obj.TYPE_REG
			m.To.Reg = x86.REG_DX
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
	case ssa.OpAMD64MULQconst, ssa.OpAMD64MULLconst, ssa.OpAMD64MULWconst, ssa.OpAMD64MULBconst:
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
		ssa.OpAMD64SARQconst, ssa.OpAMD64SARLconst, ssa.OpAMD64SARWconst, ssa.OpAMD64SARBconst,
		ssa.OpAMD64ROLQconst, ssa.OpAMD64ROLLconst, ssa.OpAMD64ROLWconst, ssa.OpAMD64ROLBconst:
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
		opregreg(v.Op.Asm(), regnum(v.Args[1]), regnum(v.Args[0]))
	case ssa.OpAMD64UCOMISS, ssa.OpAMD64UCOMISD:
		// Go assembler has swapped operands for UCOMISx relative to CMP,
		// must account for that right here.
		opregreg(v.Op.Asm(), regnum(v.Args[0]), regnum(v.Args[1]))
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
	case ssa.OpAMD64MOVSSconst, ssa.OpAMD64MOVSDconst:
		x := regnum(v)
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssa.OpAMD64MOVQload, ssa.OpAMD64MOVSSload, ssa.OpAMD64MOVSDload, ssa.OpAMD64MOVLload, ssa.OpAMD64MOVWload, ssa.OpAMD64MOVBload, ssa.OpAMD64MOVBQSXload, ssa.OpAMD64MOVBQZXload, ssa.OpAMD64MOVOload:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		addAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64MOVQloadidx8, ssa.OpAMD64MOVSDloadidx8:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		addAux(&p.From, v)
		p.From.Scale = 8
		p.From.Index = regnum(v.Args[1])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64MOVSSloadidx4:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = regnum(v.Args[0])
		addAux(&p.From, v)
		p.From.Scale = 4
		p.From.Index = regnum(v.Args[1])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpAMD64MOVQstore, ssa.OpAMD64MOVSSstore, ssa.OpAMD64MOVSDstore, ssa.OpAMD64MOVLstore, ssa.OpAMD64MOVWstore, ssa.OpAMD64MOVBstore, ssa.OpAMD64MOVOstore:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[1])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = regnum(v.Args[0])
		addAux(&p.To, v)
	case ssa.OpAMD64MOVQstoreidx8, ssa.OpAMD64MOVSDstoreidx8:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[2])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = regnum(v.Args[0])
		p.To.Scale = 8
		p.To.Index = regnum(v.Args[1])
		addAux(&p.To, v)
	case ssa.OpAMD64MOVSSstoreidx4:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[2])
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = regnum(v.Args[0])
		p.To.Scale = 4
		p.To.Index = regnum(v.Args[1])
		addAux(&p.To, v)
	case ssa.OpAMD64MOVQstoreconst, ssa.OpAMD64MOVLstoreconst, ssa.OpAMD64MOVWstoreconst, ssa.OpAMD64MOVBstoreconst:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := ssa.StoreConst(v.AuxInt)
		i := sc.Val()
		switch v.Op {
		case ssa.OpAMD64MOVBstoreconst:
			i = int64(int8(i))
		case ssa.OpAMD64MOVWstoreconst:
			i = int64(int16(i))
		case ssa.OpAMD64MOVLstoreconst:
			i = int64(int32(i))
		case ssa.OpAMD64MOVQstoreconst:
		}
		p.From.Offset = i
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = regnum(v.Args[0])
		addAux2(&p.To, v, sc.Off())
	case ssa.OpAMD64MOVLQSX, ssa.OpAMD64MOVWQSX, ssa.OpAMD64MOVBQSX, ssa.OpAMD64MOVLQZX, ssa.OpAMD64MOVWQZX, ssa.OpAMD64MOVBQZX,
		ssa.OpAMD64CVTSL2SS, ssa.OpAMD64CVTSL2SD, ssa.OpAMD64CVTSQ2SS, ssa.OpAMD64CVTSQ2SD,
		ssa.OpAMD64CVTTSS2SL, ssa.OpAMD64CVTTSD2SL, ssa.OpAMD64CVTTSS2SQ, ssa.OpAMD64CVTTSD2SQ,
		ssa.OpAMD64CVTSS2SD, ssa.OpAMD64CVTSD2SS:
		opregreg(v.Op.Asm(), regnum(v), regnum(v.Args[0]))
	case ssa.OpAMD64DUFFZERO:
		p := Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = Linksym(Pkglookup("duffzero", Runtimepkg))
		p.To.Offset = v.AuxInt
	case ssa.OpAMD64MOVOconst:
		if v.AuxInt != 0 {
			v.Unimplementedf("MOVOconst can only do constant=0")
		}
		r := regnum(v)
		opregreg(x86.AXORPS, r, r)
	case ssa.OpAMD64DUFFCOPY:
		p := Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_ADDR
		p.To.Sym = Linksym(Pkglookup("duffcopy", Runtimepkg))
		p.To.Offset = v.AuxInt

	case ssa.OpCopy, ssa.OpAMD64MOVQconvert: // TODO: lower Copy to MOVQ earlier?
		if v.Type.IsMemory() {
			return
		}
		x := regnum(v.Args[0])
		y := regnum(v)
		if x != y {
			opregreg(regMoveByTypeAMD64(v.Type), y, x)
		}
	case ssa.OpLoadReg:
		if v.Type.IsFlags() {
			v.Unimplementedf("load flags not implemented: %v", v.LongString())
			return
		}
		p := Prog(movSizeByType(v.Type))
		n, off := autoVar(v.Args[0])
		p.From.Type = obj.TYPE_MEM
		p.From.Node = n
		p.From.Sym = Linksym(n.Sym)
		p.From.Offset = off
		if n.Class == PPARAM {
			p.From.Name = obj.NAME_PARAM
			p.From.Offset += n.Xoffset
		} else {
			p.From.Name = obj.NAME_AUTO
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)

	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Unimplementedf("store flags not implemented: %v", v.LongString())
			return
		}
		p := Prog(movSizeByType(v.Type))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[0])
		n, off := autoVar(v)
		p.To.Type = obj.TYPE_MEM
		p.To.Node = n
		p.To.Sym = Linksym(n.Sym)
		p.To.Offset = off
		if n.Class == PPARAM {
			p.To.Name = obj.NAME_PARAM
			p.To.Offset += n.Xoffset
		} else {
			p.To.Name = obj.NAME_AUTO
		}
	case ssa.OpPhi:
		// just check to make sure regalloc and stackalloc did it right
		if v.Type.IsMemory() {
			return
		}
		f := v.Block.Func
		loc := f.RegAlloc[v.ID]
		for _, a := range v.Args {
			if aloc := f.RegAlloc[a.ID]; aloc != loc { // TODO: .Equal() instead?
				v.Fatalf("phi arg at different location than phi: %v @ %v, but arg %v @ %v\n%s\n", v, loc, a, aloc, v.Block.Func)
			}
		}
	case ssa.OpConst8, ssa.OpConst16, ssa.OpConst32, ssa.OpConst64, ssa.OpConstString, ssa.OpConstNil, ssa.OpConstBool,
		ssa.OpConst32F, ssa.OpConst64F:
		if v.Block.Func.RegAlloc[v.ID] != nil {
			v.Fatalf("const value %v shouldn't have a location", v)
		}

	case ssa.OpInitMem:
		// memory arg needs no code
	case ssa.OpArg:
		// input args need no code
	case ssa.OpAMD64LoweredGetClosurePtr:
		// Output is hardwired to DX only,
		// and DX contains the closure pointer on
		// closure entry, and this "instruction"
		// is scheduled to the very beginning
		// of the entry block.
	case ssa.OpAMD64LoweredGetG:
		r := regnum(v)
		// See the comments in cmd/internal/obj/x86/obj6.go
		// near CanUse1InsnTLS for a detailed explanation of these instructions.
		if x86.CanUse1InsnTLS(Ctxt) {
			// MOVQ (TLS), r
			p := Prog(x86.AMOVQ)
			p.From.Type = obj.TYPE_MEM
			p.From.Reg = x86.REG_TLS
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
		} else {
			// MOVQ TLS, r
			// MOVQ (r)(TLS*1), r
			p := Prog(x86.AMOVQ)
			p.From.Type = obj.TYPE_REG
			p.From.Reg = x86.REG_TLS
			p.To.Type = obj.TYPE_REG
			p.To.Reg = r
			q := Prog(x86.AMOVQ)
			q.From.Type = obj.TYPE_MEM
			q.From.Reg = r
			q.From.Index = x86.REG_TLS
			q.From.Scale = 1
			q.To.Type = obj.TYPE_REG
			q.To.Reg = r
		}
	case ssa.OpAMD64CALLstatic:
		p := Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = Linksym(v.Aux.(*Sym))
		if Maxarg < v.AuxInt {
			Maxarg = v.AuxInt
		}
	case ssa.OpAMD64CALLclosure:
		p := Prog(obj.ACALL)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v.Args[0])
		if Maxarg < v.AuxInt {
			Maxarg = v.AuxInt
		}
	case ssa.OpAMD64CALLdefer:
		p := Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = Linksym(Deferproc.Sym)
		if Maxarg < v.AuxInt {
			Maxarg = v.AuxInt
		}
		// defer returns in rax:
		// 0 if we should continue executing
		// 1 if we should jump to deferreturn call
		p = Prog(x86.ATESTL)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x86.REG_AX
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x86.REG_AX
		p = Prog(x86.AJNE)
		p.To.Type = obj.TYPE_BRANCH
		s.deferBranches = append(s.deferBranches, p)
	case ssa.OpAMD64CALLgo:
		p := Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = Linksym(Newproc.Sym)
		if Maxarg < v.AuxInt {
			Maxarg = v.AuxInt
		}
	case ssa.OpAMD64CALLinter:
		p := Prog(obj.ACALL)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v.Args[0])
		if Maxarg < v.AuxInt {
			Maxarg = v.AuxInt
		}
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
	case ssa.OpAMD64SQRTSD:
		p := Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = regnum(v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
	case ssa.OpSP, ssa.OpSB:
		// nothing to do
	case ssa.OpAMD64SETEQ, ssa.OpAMD64SETNE,
		ssa.OpAMD64SETL, ssa.OpAMD64SETLE,
		ssa.OpAMD64SETG, ssa.OpAMD64SETGE,
		ssa.OpAMD64SETGF, ssa.OpAMD64SETGEF,
		ssa.OpAMD64SETB, ssa.OpAMD64SETBE,
		ssa.OpAMD64SETORD, ssa.OpAMD64SETNAN,
		ssa.OpAMD64SETA, ssa.OpAMD64SETAE:
		p := Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)

	case ssa.OpAMD64SETNEF:
		p := Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
		q := Prog(x86.ASETPS)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = x86.REG_AX
		// TODO AORQ copied from old code generator, why not AORB?
		opregreg(x86.AORQ, regnum(v), x86.REG_AX)

	case ssa.OpAMD64SETEQF:
		p := Prog(v.Op.Asm())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = regnum(v)
		q := Prog(x86.ASETPC)
		q.To.Type = obj.TYPE_REG
		q.To.Reg = x86.REG_AX
		// TODO AANDQ copied from old code generator, why not AANDB?
		opregreg(x86.AANDQ, regnum(v), x86.REG_AX)

	case ssa.OpAMD64InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v)
	case ssa.OpAMD64FlagEQ, ssa.OpAMD64FlagLT_ULT, ssa.OpAMD64FlagLT_UGT, ssa.OpAMD64FlagGT_ULT, ssa.OpAMD64FlagGT_UGT:
		v.Fatalf("Flag* ops should never make it to codegen %v", v)
	case ssa.OpAMD64REPSTOSQ:
		Prog(x86.AREP)
		Prog(x86.ASTOSQ)
	case ssa.OpAMD64REPMOVSQ:
		Prog(x86.AREP)
		Prog(x86.AMOVSQ)
	case ssa.OpVarDef:
		Gvardef(v.Aux.(*Node))
	case ssa.OpVarKill:
		gvarkill(v.Aux.(*Node))
	case ssa.OpAMD64LoweredNilCheck:
		// Optimization - if the subsequent block has a load or store
		// at the same address, we don't need to issue this instruction.
		for _, w := range v.Block.Succs[0].Values {
			if len(w.Args) == 0 || !w.Args[len(w.Args)-1].Type.IsMemory() {
				// w doesn't use a store - can't be a memory op.
				continue
			}
			if w.Args[len(w.Args)-1] != v.Args[1] {
				v.Fatalf("wrong store after nilcheck v=%s w=%s", v, w)
			}
			switch w.Op {
			case ssa.OpAMD64MOVQload, ssa.OpAMD64MOVLload, ssa.OpAMD64MOVWload, ssa.OpAMD64MOVBload,
				ssa.OpAMD64MOVQstore, ssa.OpAMD64MOVLstore, ssa.OpAMD64MOVWstore, ssa.OpAMD64MOVBstore:
				if w.Args[0] == v.Args[0] && w.Aux == nil && w.AuxInt >= 0 && w.AuxInt < minZeroPage {
					return
				}
			case ssa.OpAMD64MOVQstoreconst, ssa.OpAMD64MOVLstoreconst, ssa.OpAMD64MOVWstoreconst, ssa.OpAMD64MOVBstoreconst:
				off := ssa.StoreConst(v.AuxInt).Off()
				if w.Args[0] == v.Args[0] && w.Aux == nil && off >= 0 && off < minZeroPage {
					return
				}
			}
			if w.Type.IsMemory() {
				// We can't delay the nil check past the next store.
				break
			}
		}
		// Issue a load which will fault if the input is nil.
		// TODO: We currently use the 2-byte instruction TESTB AX, (reg).
		// Should we use the 3-byte TESTB $0, (reg) instead?  It is larger
		// but it doesn't have false dependency on AX.
		// Or maybe allocate an output register and use MOVL (reg),reg2 ?
		// That trades clobbering flags for clobbering a register.
		p := Prog(x86.ATESTB)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x86.REG_AX
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = regnum(v.Args[0])
		addAux(&p.To, v)
		if Debug_checknil != 0 && v.Line > 1 { // v.Line==1 in generated wrappers
			Warnl(int(v.Line), "generated nil check")
		}
	default:
		v.Unimplementedf("genValue not implemented: %s", v.LongString())
	}
}

// movSizeByType returns the MOV instruction of the given type.
func movSizeByType(t ssa.Type) (asm int) {
	// For x86, there's no difference between reg move opcodes
	// and memory move opcodes.
	asm = regMoveByTypeAMD64(t)
	return
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

var blockJump = [...]struct {
	asm, invasm int
}{
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
	ssa.BlockAMD64ORD: {x86.AJPC, x86.AJPS},
	ssa.BlockAMD64NAN: {x86.AJPS, x86.AJPC},
}

type floatingEQNEJump struct {
	jump, index int
}

var eqfJumps = [2][2]floatingEQNEJump{
	{{x86.AJNE, 1}, {x86.AJPS, 1}}, // next == b.Succs[0]
	{{x86.AJNE, 1}, {x86.AJPC, 0}}, // next == b.Succs[1]
}
var nefJumps = [2][2]floatingEQNEJump{
	{{x86.AJNE, 0}, {x86.AJPC, 1}}, // next == b.Succs[0]
	{{x86.AJNE, 0}, {x86.AJPS, 0}}, // next == b.Succs[1]
}

func oneFPJump(b *ssa.Block, jumps *floatingEQNEJump, likely ssa.BranchPrediction, branches []branch) []branch {
	p := Prog(jumps.jump)
	p.To.Type = obj.TYPE_BRANCH
	to := jumps.index
	branches = append(branches, branch{p, b.Succs[to]})
	if to == 1 {
		likely = -likely
	}
	// liblink reorders the instruction stream as it sees fit.
	// Pass along what we know so liblink can make use of it.
	// TODO: Once we've fully switched to SSA,
	// make liblink leave our output alone.
	switch likely {
	case ssa.BranchUnlikely:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0
	case ssa.BranchLikely:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
	}
	return branches
}

func genFPJump(s *genState, b, next *ssa.Block, jumps *[2][2]floatingEQNEJump) {
	likely := b.Likely
	switch next {
	case b.Succs[0]:
		s.branches = oneFPJump(b, &jumps[0][0], likely, s.branches)
		s.branches = oneFPJump(b, &jumps[0][1], likely, s.branches)
	case b.Succs[1]:
		s.branches = oneFPJump(b, &jumps[1][0], likely, s.branches)
		s.branches = oneFPJump(b, &jumps[1][1], likely, s.branches)
	default:
		s.branches = oneFPJump(b, &jumps[1][0], likely, s.branches)
		s.branches = oneFPJump(b, &jumps[1][1], likely, s.branches)
		q := Prog(obj.AJMP)
		q.To.Type = obj.TYPE_BRANCH
		s.branches = append(s.branches, branch{q, b.Succs[1]})
	}
}

func (s *genState) genBlock(b, next *ssa.Block) {
	lineno = b.Line

	switch b.Kind {
	case ssa.BlockPlain, ssa.BlockCall, ssa.BlockCheck:
		if b.Succs[0] != next {
			p := Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.branches = append(s.branches, branch{p, b.Succs[0]})
		}
	case ssa.BlockExit:
		Prog(obj.AUNDEF) // tell plive.go that we never reach here
	case ssa.BlockRet:
		if hasdefer {
			s.deferReturn()
		}
		Prog(obj.ARET)
	case ssa.BlockRetJmp:
		p := Prog(obj.AJMP)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = Linksym(b.Aux.(*Sym))

	case ssa.BlockAMD64EQF:
		genFPJump(s, b, next, &eqfJumps)

	case ssa.BlockAMD64NEF:
		genFPJump(s, b, next, &nefJumps)

	case ssa.BlockAMD64EQ, ssa.BlockAMD64NE,
		ssa.BlockAMD64LT, ssa.BlockAMD64GE,
		ssa.BlockAMD64LE, ssa.BlockAMD64GT,
		ssa.BlockAMD64ULT, ssa.BlockAMD64UGT,
		ssa.BlockAMD64ULE, ssa.BlockAMD64UGE:
		jmp := blockJump[b.Kind]
		likely := b.Likely
		var p *obj.Prog
		switch next {
		case b.Succs[0]:
			p = Prog(jmp.invasm)
			likely *= -1
			p.To.Type = obj.TYPE_BRANCH
			s.branches = append(s.branches, branch{p, b.Succs[1]})
		case b.Succs[1]:
			p = Prog(jmp.asm)
			p.To.Type = obj.TYPE_BRANCH
			s.branches = append(s.branches, branch{p, b.Succs[0]})
		default:
			p = Prog(jmp.asm)
			p.To.Type = obj.TYPE_BRANCH
			s.branches = append(s.branches, branch{p, b.Succs[0]})
			q := Prog(obj.AJMP)
			q.To.Type = obj.TYPE_BRANCH
			s.branches = append(s.branches, branch{q, b.Succs[1]})
		}

		// liblink reorders the instruction stream as it sees fit.
		// Pass along what we know so liblink can make use of it.
		// TODO: Once we've fully switched to SSA,
		// make liblink leave our output alone.
		switch likely {
		case ssa.BranchUnlikely:
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 0
		case ssa.BranchLikely:
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = 1
		}

	default:
		b.Unimplementedf("branch not implemented: %s. Control: %s", b.LongString(), b.Control.LongString())
	}
}

func (s *genState) deferReturn() {
	// Deferred calls will appear to be returning to
	// the CALL deferreturn(SB) that we are about to emit.
	// However, the stack trace code will show the line
	// of the instruction byte before the return PC.
	// To avoid that being an unrelated instruction,
	// insert an actual hardware NOP that will have the right line number.
	// This is different from obj.ANOP, which is a virtual no-op
	// that doesn't make it into the instruction stream.
	s.deferTarget = Pc
	Thearch.Ginsnop()
	p := Prog(obj.ACALL)
	p.To.Type = obj.TYPE_MEM
	p.To.Name = obj.NAME_EXTERN
	p.To.Sym = Linksym(Deferreturn.Sym)
}

// addAux adds the offset in the aux fields (AuxInt and Aux) of v to a.
func addAux(a *obj.Addr, v *ssa.Value) {
	addAux2(a, v, v.AuxInt)
}
func addAux2(a *obj.Addr, v *ssa.Value, offset int64) {
	if a.Type != obj.TYPE_MEM {
		v.Fatalf("bad addAux addr %s", a)
	}
	// add integer offset
	a.Offset += offset

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
		n := sym.Node.(*Node)
		a.Name = obj.NAME_PARAM
		a.Node = n
		a.Sym = Linksym(n.Orig.Sym)
		a.Offset += n.Xoffset // TODO: why do I have to add this here?  I don't for auto variables.
	case *ssa.AutoSymbol:
		n := sym.Node.(*Node)
		a.Name = obj.NAME_AUTO
		a.Node = n
		a.Sym = Linksym(n.Sym)
	default:
		v.Fatalf("aux in %s not implemented %#v", v, v.Aux)
	}
}

// extendIndex extends v to a full int width.
func (s *state) extendIndex(v *ssa.Value) *ssa.Value {
	size := v.Type.Size()
	if size == s.config.IntSize {
		return v
	}
	if size > s.config.IntSize {
		// TODO: truncate 64-bit indexes on 32-bit pointer archs.  We'd need to test
		// the high word and branch to out-of-bounds failure if it is not 0.
		s.Unimplementedf("64->32 index truncation not implemented")
		return v
	}

	// Extend value to the required size
	var op ssa.Op
	if v.Type.IsSigned() {
		switch 10*size + s.config.IntSize {
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
		switch 10*size + s.config.IntSize {
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
	return s.newValue1(op, Types[TINT], v)
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
		panic("bad int register width")
	}
}

func regMoveByTypeAMD64(t ssa.Type) int {
	width := t.Size()
	if t.IsFloat() {
		switch width {
		case 4:
			return x86.AMOVSS
		case 8:
			return x86.AMOVSD
		default:
			panic("bad float register width")
		}
	} else {
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
			panic("bad int register width")
		}
	}

	panic("bad register type")
}

// regnum returns the register (in cmd/internal/obj numbering) to
// which v has been allocated.  Panics if v is not assigned to a
// register.
// TODO: Make this panic again once it stops happening routinely.
func regnum(v *ssa.Value) int16 {
	reg := v.Block.Func.RegAlloc[v.ID]
	if reg == nil {
		v.Unimplementedf("nil regnum for value: %s\n%s\n", v.LongString(), v.Block.Func)
		return 0
	}
	return ssaRegToReg[reg.(*ssa.Register).Num]
}

// autoVar returns a *Node and int64 representing the auto variable and offset within it
// where v should be spilled.
func autoVar(v *ssa.Value) (*Node, int64) {
	loc := v.Block.Func.RegAlloc[v.ID].(ssa.LocalSlot)
	if v.Type.Size() > loc.Type.Size() {
		v.Fatalf("spill/restore type %s doesn't fit in slot type %s", v.Type, loc.Type)
	}
	return loc.N.(*Node), loc.Off
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
func (s *ssaExport) TypeFloat32() ssa.Type { return Types[TFLOAT32] }
func (s *ssaExport) TypeFloat64() ssa.Type { return Types[TFLOAT64] }
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

func (e *ssaExport) Auto(t ssa.Type) ssa.GCNode {
	n := temp(t.(*Type))   // Note: adds new auto to Curfn.Func.Dcl list
	e.mustImplement = true // This modifies the input to SSA, so we want to make sure we succeed from here!
	return n
}

func (e *ssaExport) CanSSA(t ssa.Type) bool {
	return canSSAType(t.(*Type))
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
		Fatalf(msg, args...)
	}
}

// Unimplemented reports that the function cannot be compiled.
// It will be removed once SSA work is complete.
func (e *ssaExport) Unimplementedf(msg string, args ...interface{}) {
	if e.mustImplement {
		Fatalf(msg, args...)
	}
	const alwaysLog = false // enable to calculate top unimplemented features
	if !e.unimplemented && (e.log || alwaysLog) {
		// first implementation failure, print explanation
		fmt.Printf("SSA unimplemented: "+msg+"\n", args...)
	}
	e.unimplemented = true
}

// Warnl reports a "warning", which is usually flag-triggered
// logging output for the benefit of tests.
func (e *ssaExport) Warnl(line int, fmt_ string, args ...interface{}) {
	Warnl(line, fmt_, args...)
}

func (e *ssaExport) Debug_checknil() bool {
	return Debug_checknil != 0
}

func (n *Node) Typ() ssa.Type {
	return n.Type
}
