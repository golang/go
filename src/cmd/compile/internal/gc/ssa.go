// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"fmt"
	"html"
	"os"
	"strings"

	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"cmd/internal/sys"
)

var ssaEnabled = true

var ssaConfig *ssa.Config
var ssaExp ssaExport

func initssa() *ssa.Config {
	ssaExp.unimplemented = false
	ssaExp.mustImplement = true
	if ssaConfig == nil {
		ssaConfig = ssa.NewConfig(Thearch.LinkArch.Name, &ssaExp, Ctxt, Debug['N'] == 0)
	}
	return ssaConfig
}

func shouldssa(fn *Node) bool {
	switch Thearch.LinkArch.Name {
	default:
		// Only available for testing.
		if os.Getenv("SSATEST") == "" {
			return false
		}
		// Generally available.
	case "amd64":
	}
	if !ssaEnabled {
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

	return initssa().DebugHashMatch("GOSSAHASH", name)
}

// buildssa builds an SSA function.
func buildssa(fn *Node) *ssa.Func {
	name := fn.Func.Nname.Sym.Name
	printssa := name == os.Getenv("GOSSAFUNC")
	if printssa {
		fmt.Println("generating SSA for", name)
		dumplist("buildssa-enter", fn.Func.Enter)
		dumplist("buildssa-body", fn.Nbody)
		dumplist("buildssa-exit", fn.Func.Exit)
	}

	var s state
	s.pushLine(fn.Lineno)
	defer s.popLine()

	if fn.Func.Pragma&CgoUnsafeArgs != 0 {
		s.cgoUnsafeArgs = true
	}
	if fn.Func.Pragma&Nowritebarrier != 0 {
		s.noWB = true
	}
	defer func() {
		if s.WBLineno != 0 {
			fn.Func.WBLineno = s.WBLineno
		}
	}()
	// TODO(khr): build config just once at the start of the compiler binary

	ssaExp.log = printssa

	s.config = initssa()
	s.f = s.config.NewFunc()
	s.f.Name = name
	s.exitCode = fn.Func.Exit
	s.panics = map[funcLine]*ssa.Block{}

	if name == os.Getenv("GOSSAFUNC") {
		// TODO: tempfile? it is handy to have the location
		// of this file be stable, so you can just reload in the browser.
		s.config.HTML = ssa.NewHTMLWriter("ssa.html", s.config, name)
		// TODO: generate and print a mapping from nodes to values and blocks
	}
	defer func() {
		if !printssa {
			s.config.HTML.Close()
		}
	}()

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
	for _, n := range fn.Func.Dcl {
		switch n.Class {
		case PPARAM, PPARAMOUT:
			aux := s.lookupSymbol(n, &ssa.ArgSymbol{Typ: n.Type, Node: n})
			s.decladdrs[n] = s.entryNewValue1A(ssa.OpAddr, Ptrto(n.Type), aux, s.sp)
			if n.Class == PPARAMOUT && s.canSSA(n) {
				// Save ssa-able PPARAMOUT variables so we can
				// store them back to the stack at the end of
				// the function.
				s.returns = append(s.returns, n)
			}
			if n.Class == PPARAM && s.canSSA(n) && n.Type.IsPtrShaped() {
				s.ptrargs = append(s.ptrargs, n)
				n.SetNotLiveAtEnd(true) // SSA takes care of this explicitly
			}
		case PAUTO:
			// processed at each use, to prevent Addr coming
			// before the decl.
		case PAUTOHEAP:
			// moved to heap - already handled by frontend
		case PFUNC:
			// local function - already handled by frontend
		default:
			s.Unimplementedf("local variable with class %s unimplemented", classnames[n.Class])
		}
	}

	// Convert the AST-based IR to the SSA-based IR
	s.stmts(fn.Func.Enter)
	s.stmts(fn.Nbody)

	// fallthrough to exit
	if s.curBlock != nil {
		s.pushLine(fn.Func.Endlineno)
		s.exit()
		s.popLine()
	}

	// Check that we used all labels
	for name, lab := range s.labels {
		if !lab.used() && !lab.reported {
			yyerrorl(lab.defNode.Lineno, "label %v defined and not used", name)
			lab.reported = true
		}
		if lab.used() && !lab.defined() && !lab.reported {
			yyerrorl(lab.useNode.Lineno, "label %v not defined", name)
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
		s.f.Free()
		return nil
	}

	prelinkNumvars := s.f.NumValues()
	sparseDefState := s.locatePotentialPhiFunctions(fn)

	// Link up variable uses to variable definitions
	s.linkForwardReferences(sparseDefState)

	if ssa.BuildStats > 0 {
		s.f.LogStat("build", s.f.NumBlocks(), "blocks", prelinkNumvars, "vars_before",
			s.f.NumValues(), "vars_after", prelinkNumvars*s.f.NumBlocks(), "ssa_phi_loc_cutoff_score")
	}

	// Don't carry reference this around longer than necessary
	s.exitCode = Nodes{}

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
	exitCode Nodes

	// unlabeled break and continue statement tracking
	breakTo    *ssa.Block // current target for plain break statement
	continueTo *ssa.Block // current target for plain continue statement

	// current location where we're interpreting the AST
	curBlock *ssa.Block

	// variable assignments in the current block (map from variable symbol to ssa value)
	// *Node is the unique identifier (an ONAME Node) for the variable.
	vars map[*Node]*ssa.Value

	// all defined variables at the end of each block. Indexed by block ID.
	defvars []map[*Node]*ssa.Value

	// addresses of PPARAM and PPARAMOUT variables.
	decladdrs map[*Node]*ssa.Value

	// symbols for PEXTERN, PAUTO and PPARAMOUT variables so they can be reused.
	varsyms map[*Node]interface{}

	// starting values. Memory, stack pointer, and globals pointer
	startmem *ssa.Value
	sp       *ssa.Value
	sb       *ssa.Value

	// line number stack. The current line number is top of stack
	line []int32

	// list of panic calls by function name and line number.
	// Used to deduplicate panic calls.
	panics map[funcLine]*ssa.Block

	// list of FwdRef values.
	fwdRefs []*ssa.Value

	// list of PPARAMOUT (return) variables.
	returns []*Node

	// list of PPARAM SSA-able pointer-shaped args. We ensure these are live
	// throughout the function to help users avoid premature finalizers.
	ptrargs []*Node

	cgoUnsafeArgs bool
	noWB          bool
	WBLineno      int32 // line number of first write barrier. 0=no write barriers
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

func (s *state) Logf(msg string, args ...interface{})   { s.config.Logf(msg, args...) }
func (s *state) Log() bool                              { return s.config.Log() }
func (s *state) Fatalf(msg string, args ...interface{}) { s.config.Fatalf(s.peekLine(), msg, args...) }
func (s *state) Unimplementedf(msg string, args ...interface{}) {
	s.config.Unimplementedf(s.peekLine(), msg, args...)
}
func (s *state) Warnl(line int32, msg string, args ...interface{}) { s.config.Warnl(line, msg, args...) }
func (s *state) Debug_checknil() bool                              { return s.config.Debug_checknil() }

var (
	// dummy node for the memory variable
	memVar = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "mem"}}

	// dummy nodes for temporary variables
	ptrVar    = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "ptr"}}
	lenVar    = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "len"}}
	newlenVar = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "newlen"}}
	capVar    = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "cap"}}
	typVar    = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "typ"}}
	idataVar  = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "idata"}}
	okVar     = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "ok"}}
	deltaVar  = Node{Op: ONAME, Class: Pxxx, Sym: &Sym{Name: "delta"}}
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
// Returns the (former) current block. Returns nil if there is no current
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
	if line == 0 {
		// the frontend may emit node with line number missing,
		// use the parent line number in this case.
		line = s.peekLine()
		if Debug['K'] != 0 {
			Warn("buildssa: line 0")
		}
	}
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
	yyerrorl(s.peekLine(), msg, args...)
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
func (s *state) constSlice(t ssa.Type) *ssa.Value       { return s.f.ConstSlice(s.peekLine(), t) }
func (s *state) constInterface(t ssa.Type) *ssa.Value   { return s.f.ConstInterface(s.peekLine(), t) }
func (s *state) constNil(t ssa.Type) *ssa.Value         { return s.f.ConstNil(s.peekLine(), t) }
func (s *state) constEmptyString(t ssa.Type) *ssa.Value { return s.f.ConstEmptyString(s.peekLine(), t) }
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

func (s *state) stmts(a Nodes) {
	for _, x := range a.Slice() {
		s.stmt(x)
	}
}

// ssaStmtList converts the statement n to SSA and adds it to s.
func (s *state) stmtList(l Nodes) {
	for _, n := range l.Slice() {
		s.stmt(n)
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
		if n.Op == OCALLFUNC && n.Left.Op == ONAME && n.Left.Class == PFUNC &&
			(compiling_runtime && n.Left.Sym.Name == "throw" ||
				n.Left.Sym.Pkg == Runtimepkg && (n.Left.Sym.Name == "gopanic" || n.Left.Sym.Name == "selectgo" || n.Left.Sym.Name == "block")) {
			m := s.mem()
			b := s.endBlock()
			b.Kind = ssa.BlockExit
			b.SetControl(m)
			// TODO: never rewrite OPANIC to OCALLFUNC in the
			// first place. Need to wait until all backends
			// go through SSA.
		}
	case ODEFER:
		s.call(n.Left, callDefer)
	case OPROC:
		s.call(n.Left, callGo)

	case OAS2DOTTYPE:
		res, resok := s.dottype(n.Rlist.First(), true)
		s.assign(n.List.First(), res, needwritebarrier(n.List.First(), n.Rlist.First()), false, n.Lineno, 0, false)
		s.assign(n.List.Second(), resok, false, false, n.Lineno, 0, false)
		return

	case ODCL:
		if n.Left.Class == PAUTOHEAP {
			Fatalf("DCL %v", n)
		}

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
			s.Error("label %v already defined at %v", sym, linestr(lab.defNode.Lineno))
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

		if n.Left == n.Right && n.Left.Op == ONAME {
			// An x=x assignment. No point in doing anything
			// here. In addition, skipping this assignment
			// prevents generating:
			//   VARDEF x
			//   COPY x -> x
			// which is bad because x is incorrectly considered
			// dead before the vardef. See issue #14904.
			return
		}

		var t *Type
		if n.Right != nil {
			t = n.Right.Type
		} else {
			t = n.Left.Type
		}

		// Evaluate RHS.
		rhs := n.Right
		if rhs != nil {
			switch rhs.Op {
			case OSTRUCTLIT, OARRAYLIT:
				// All literals with nonzero fields have already been
				// rewritten during walk. Any that remain are just T{}
				// or equivalents. Use the zero value.
				if !iszero(rhs) {
					Fatalf("literal with nonzero value in SSA: %v", rhs)
				}
				rhs = nil
			case OAPPEND:
				// If we're writing the result of an append back to the same slice,
				// handle it specially to avoid write barriers on the fast (non-growth) path.
				// If the slice can be SSA'd, it'll be on the stack,
				// so there will be no write barriers,
				// so there's no need to attempt to prevent them.
				if samesafeexpr(n.Left, rhs.List.First()) && !s.canSSA(n.Left) {
					s.append(rhs, true)
					return
				}
			}
		}
		var r *ssa.Value
		var isVolatile bool
		needwb := n.Op == OASWB && rhs != nil
		deref := !canSSAType(t)
		if deref {
			if rhs == nil {
				r = nil // Signal assign to use OpZero.
			} else {
				r, isVolatile = s.addr(rhs, false)
			}
		} else {
			if rhs == nil {
				r = s.zeroVal(t)
			} else {
				r = s.expr(rhs)
			}
		}
		if rhs != nil && rhs.Op == OAPPEND {
			// The frontend gets rid of the write barrier to enable the special OAPPEND
			// handling above, but since this is not a special case, we need it.
			// TODO: just add a ptr graying to the end of growslice?
			// TODO: check whether we need to provide special handling and a write barrier
			// for ODOTTYPE and ORECV also.
			// They get similar wb-removal treatment in walk.go:OAS.
			needwb = true
		}

		var skip skipMask
		if rhs != nil && (rhs.Op == OSLICE || rhs.Op == OSLICE3 || rhs.Op == OSLICESTR) && samesafeexpr(rhs.Left, n.Left) {
			// We're assigning a slicing operation back to its source.
			// Don't write back fields we aren't changing. See issue #14855.
			i, j, k := rhs.SliceBounds()
			if i != nil && (i.Op == OLITERAL && i.Val().Ctype() == CTINT && i.Int64() == 0) {
				// [0:...] is the same as [:...]
				i = nil
			}
			// TODO: detect defaults for len/cap also.
			// Currently doesn't really work because (*p)[:len(*p)] appears here as:
			//    tmp = len(*p)
			//    (*p)[:tmp]
			//if j != nil && (j.Op == OLEN && samesafeexpr(j.Left, n.Left)) {
			//      j = nil
			//}
			//if k != nil && (k.Op == OCAP && samesafeexpr(k.Left, n.Left)) {
			//      k = nil
			//}
			if i == nil {
				skip |= skipPtr
				if j == nil {
					skip |= skipLen
				}
				if k == nil {
					skip |= skipCap
				}
			}
		}

		s.assign(n.Left, r, needwb, deref, n.Lineno, skip, isVolatile)

	case OIF:
		bThen := s.f.NewBlock(ssa.BlockPlain)
		bEnd := s.f.NewBlock(ssa.BlockPlain)
		var bElse *ssa.Block
		if n.Rlist.Len() != 0 {
			bElse = s.f.NewBlock(ssa.BlockPlain)
			s.condBranch(n.Left, bThen, bElse, n.Likely)
		} else {
			s.condBranch(n.Left, bThen, bEnd, n.Likely)
		}

		s.startBlock(bThen)
		s.stmts(n.Nbody)
		if b := s.endBlock(); b != nil {
			b.AddEdgeTo(bEnd)
		}

		if n.Rlist.Len() != 0 {
			s.startBlock(bElse)
			s.stmtList(n.Rlist)
			if b := s.endBlock(); b != nil {
				b.AddEdgeTo(bEnd)
			}
		}
		s.startBlock(bEnd)

	case ORETURN:
		s.stmtList(n.List)
		s.exit()
	case ORETJMP:
		s.stmtList(n.List)
		b := s.exit()
		b.Kind = ssa.BlockRetJmp // override BlockRet
		b.Aux = n.Left.Sym

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
		s.stmts(n.Nbody)

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
		s.stmts(n.Nbody)

		s.breakTo = prevBreak
		if lab != nil {
			lab.breakTarget = nil
		}

		// OSWITCH never falls through (s.curBlock == nil here).
		// OSELECT does not fall through if we're calling selectgo.
		// OSELECT does fall through if we're calling selectnb{send,recv}[2].
		// In those latter cases, go to the code after the select.
		if b := s.endBlock(); b != nil {
			b.AddEdgeTo(bEnd)
		}
		s.startBlock(bEnd)

	case OVARKILL:
		// Insert a varkill op to record that a variable is no longer live.
		// We only care about liveness info at call sites, so putting the
		// varkill in the store chain is enough to keep it correctly ordered
		// with respect to call ops.
		if !s.canSSA(n.Left) {
			s.vars[&memVar] = s.newValue1A(ssa.OpVarKill, ssa.TypeMem, n.Left, s.mem())
		}

	case OVARLIVE:
		// Insert a varlive op to record that a variable is still live.
		if !n.Left.Addrtaken {
			s.Fatalf("VARLIVE variable %s must have Addrtaken set", n.Left)
		}
		s.vars[&memVar] = s.newValue1A(ssa.OpVarLive, ssa.TypeMem, n.Left, s.mem())

	case OCHECKNIL:
		p := s.expr(n.Left)
		s.nilCheck(p)

	default:
		s.Unimplementedf("unhandled stmt %s", n.Op)
	}
}

// exit processes any code that needs to be generated just before returning.
// It returns a BlockRet block that ends the control flow. Its control value
// will be set to the final memory state.
func (s *state) exit() *ssa.Block {
	if hasdefer {
		s.rtcall(Deferreturn, true, nil)
	}

	// Run exit code. Typically, this code copies heap-allocated PPARAMOUT
	// variables back to the stack.
	s.stmts(s.exitCode)

	// Store SSAable PPARAMOUT variables back to stack locations.
	for _, n := range s.returns {
		addr := s.decladdrs[n]
		val := s.variable(n, n.Type)
		s.vars[&memVar] = s.newValue1A(ssa.OpVarDef, ssa.TypeMem, n, s.mem())
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, n.Type.Size(), addr, val, s.mem())
		// TODO: if val is ever spilled, we'd like to use the
		// PPARAMOUT slot for spilling it. That won't happen
		// currently.
	}

	// Keep input pointer args live until the return. This is a bandaid
	// fix for 1.7 for what will become in 1.8 explicit runtime.KeepAlive calls.
	// For <= 1.7 we guarantee that pointer input arguments live to the end of
	// the function to prevent premature (from the user's point of view)
	// execution of finalizers. See issue 15277.
	// TODO: remove for 1.8?
	for _, n := range s.ptrargs {
		s.vars[&memVar] = s.newValue2(ssa.OpKeepAlive, ssa.TypeMem, s.variable(n, n.Type), s.mem())
	}

	// Do actual return.
	m := s.mem()
	b := s.endBlock()
	b.Kind = ssa.BlockRet
	b.SetControl(m)
	return b
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

	opAndType{OEQ, TBOOL}:      ssa.OpEqB,
	opAndType{OEQ, TINT8}:      ssa.OpEq8,
	opAndType{OEQ, TUINT8}:     ssa.OpEq8,
	opAndType{OEQ, TINT16}:     ssa.OpEq16,
	opAndType{OEQ, TUINT16}:    ssa.OpEq16,
	opAndType{OEQ, TINT32}:     ssa.OpEq32,
	opAndType{OEQ, TUINT32}:    ssa.OpEq32,
	opAndType{OEQ, TINT64}:     ssa.OpEq64,
	opAndType{OEQ, TUINT64}:    ssa.OpEq64,
	opAndType{OEQ, TINTER}:     ssa.OpEqInter,
	opAndType{OEQ, TSLICE}:     ssa.OpEqSlice,
	opAndType{OEQ, TFUNC}:      ssa.OpEqPtr,
	opAndType{OEQ, TMAP}:       ssa.OpEqPtr,
	opAndType{OEQ, TCHAN}:      ssa.OpEqPtr,
	opAndType{OEQ, TPTR64}:     ssa.OpEqPtr,
	opAndType{OEQ, TUINTPTR}:   ssa.OpEqPtr,
	opAndType{OEQ, TUNSAFEPTR}: ssa.OpEqPtr,
	opAndType{OEQ, TFLOAT64}:   ssa.OpEq64F,
	opAndType{OEQ, TFLOAT32}:   ssa.OpEq32F,

	opAndType{ONE, TBOOL}:      ssa.OpNeqB,
	opAndType{ONE, TINT8}:      ssa.OpNeq8,
	opAndType{ONE, TUINT8}:     ssa.OpNeq8,
	opAndType{ONE, TINT16}:     ssa.OpNeq16,
	opAndType{ONE, TUINT16}:    ssa.OpNeq16,
	opAndType{ONE, TINT32}:     ssa.OpNeq32,
	opAndType{ONE, TUINT32}:    ssa.OpNeq32,
	opAndType{ONE, TINT64}:     ssa.OpNeq64,
	opAndType{ONE, TUINT64}:    ssa.OpNeq64,
	opAndType{ONE, TINTER}:     ssa.OpNeqInter,
	opAndType{ONE, TSLICE}:     ssa.OpNeqSlice,
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
		s.Unimplementedf("unhandled binary op %s %s", op, etype)
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
		s.Unimplementedf("unhandled shift op %s etype=%s/%s", op, etype1, etype2)
	}
	return x
}

func (s *state) ssaRotateOp(op Op, t *Type) ssa.Op {
	etype1 := s.concreteEtype(t)
	x, ok := opToSSA[opAndType{op, etype1}]
	if !ok {
		s.Unimplementedf("unhandled rotate op %s etype=%s", op, etype1)
	}
	return x
}

// expr converts the expression n to ssa, adds it to s and returns the ssa result.
func (s *state) expr(n *Node) *ssa.Value {
	if !(n.Op == ONAME || n.Op == OLITERAL && n.Sym != nil) {
		// ONAMEs and named OLITERALs have the line number
		// of the decl, not the use. See issue 14742.
		s.pushLine(n.Lineno)
		defer s.popLine()
	}

	s.stmtList(n.Ninit)
	switch n.Op {
	case OCFUNC:
		aux := s.lookupSymbol(n, &ssa.ExternSymbol{Typ: n.Type, Sym: n.Left.Sym})
		return s.entryNewValue1A(ssa.OpAddr, n.Type, aux, s.sb)
	case ONAME:
		if n.Class == PFUNC {
			// "value" of a function is the address of the function's closure
			sym := funcsym(n.Sym)
			aux := &ssa.ExternSymbol{Typ: n.Type, Sym: sym}
			return s.entryNewValue1A(ssa.OpAddr, Ptrto(n.Type), aux, s.sb)
		}
		if s.canSSA(n) {
			return s.variable(n, n.Type)
		}
		addr, _ := s.addr(n, false)
		return s.newValue2(ssa.OpLoad, n.Type, addr, s.mem())
	case OCLOSUREVAR:
		addr, _ := s.addr(n, false)
		return s.newValue2(ssa.OpLoad, n.Type, addr, s.mem())
	case OLITERAL:
		switch u := n.Val().U.(type) {
		case *Mpint:
			i := u.Int64()
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
		case string:
			if u == "" {
				return s.constEmptyString(n.Type)
			}
			return s.entryNewValue0A(ssa.OpConstString, n.Type, u)
		case bool:
			return s.constBool(u)
		case *NilVal:
			t := n.Type
			switch {
			case t.IsSlice():
				return s.constSlice(t)
			case t.IsInterface():
				return s.constInterface(t)
			default:
				return s.constNil(t)
			}
		case *Mpflt:
			switch n.Type.Size() {
			case 4:
				return s.constFloat32(n.Type, u.Float32())
			case 8:
				return s.constFloat64(n.Type, u.Float64())
			default:
				s.Fatalf("bad float size %d", n.Type.Size())
				return nil
			}
		case *Mpcplx:
			r := &u.Real
			i := &u.Imag
			switch n.Type.Size() {
			case 8:
				pt := Types[TFLOAT32]
				return s.newValue2(ssa.OpComplexMake, n.Type,
					s.constFloat32(pt, r.Float32()),
					s.constFloat32(pt, i.Float32()))
			case 16:
				pt := Types[TFLOAT64]
				return s.newValue2(ssa.OpComplexMake, n.Type,
					s.constFloat64(pt, r.Float64()),
					s.constFloat64(pt, i.Float64()))
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
		if to.IsPtrShaped() != from.IsPtrShaped() {
			return s.newValue2(ssa.OpConvert, to, x, s.mem())
		}

		v := s.newValue1(ssa.OpCopy, to, x) // ensure that v has the right type

		// CONVNOP closure
		if to.Etype == TFUNC && from.IsPtrShaped() {
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
			s.Fatalf("CONVNOP sign mismatch %v (%s) -> %v (%s)\n", from, from.Etype, to, to.Etype)
			return nil
		}

		if instrumenting {
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

		s.Unimplementedf("unhandled OCONV %s -> %s", n.Left.Type.Etype, n.Type.Etype)
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
				s.Fatalf("ordered complex compare %s", n.Op)
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
			wt := Types[TFLOAT64]         // Compute in Float64 to minimize cancelation error

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
			wt := Types[TFLOAT64]         // Compute in Float64 to minimize cancelation error

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
		i := n.Right.Int64()
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
		b.SetControl(el)
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
		a, _ := s.addr(n.Left, n.Bounded)
		// Note we know the volatile result is false because you can't write &f() in Go.
		return a

	case OINDREG:
		if int(n.Reg) != Thearch.REGSP {
			s.Unimplementedf("OINDREG of non-SP register %s in expr: %v", obj.Rconv(int(n.Reg)), n)
			return nil
		}
		addr := s.entryNewValue1I(ssa.OpOffPtr, Ptrto(n.Type), n.Xoffset, s.sp)
		return s.newValue2(ssa.OpLoad, n.Type, addr, s.mem())

	case OIND:
		p := s.exprPtr(n.Left, false, n.Lineno)
		return s.newValue2(ssa.OpLoad, n.Type, p, s.mem())

	case ODOT:
		t := n.Left.Type
		if canSSAType(t) {
			v := s.expr(n.Left)
			return s.newValue1I(ssa.OpStructSelect, n.Type, int64(fieldIdx(n)), v)
		}
		p, _ := s.addr(n, false)
		return s.newValue2(ssa.OpLoad, n.Type, p, s.mem())

	case ODOTPTR:
		p := s.exprPtr(n.Left, false, n.Lineno)
		p = s.newValue1I(ssa.OpOffPtr, p.Type, n.Xoffset, p)
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
			if Isconst(n.Right, CTINT) {
				ptr = s.newValue1I(ssa.OpOffPtr, ptrtyp, n.Right.Int64(), ptr)
			} else {
				ptr = s.newValue2(ssa.OpAddPtr, ptrtyp, ptr, i)
			}
			return s.newValue2(ssa.OpLoad, Types[TUINT8], ptr, s.mem())
		case n.Left.Type.IsSlice():
			p, _ := s.addr(n, false)
			return s.newValue2(ssa.OpLoad, n.Left.Type.Elem(), p, s.mem())
		case n.Left.Type.IsArray():
			// TODO: fix when we can SSA arrays of length 1.
			p, _ := s.addr(n, false)
			return s.newValue2(ssa.OpLoad, n.Left.Type.Elem(), p, s.mem())
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
			return s.constInt(Types[TINT], n.Left.Type.NumElem())
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
		// the data portion of an eface. But we don't want struct{*byte}
		// as a register type because (among other reasons) the liveness
		// analysis is confused by the "fat" variables that result from
		// such types being spilled.
		// So here we ensure that we are selecting the underlying pointer
		// when we build an eface.
		// TODO: get rid of this now that structs can be SSA'd?
		for !data.Type.IsPtrShaped() {
			switch {
			case data.Type.IsArray():
				data = s.newValue1I(ssa.OpArrayIndex, data.Type.ElemType(), 0, data)
			case data.Type.IsStruct():
				for i := data.Type.NumFields() - 1; i >= 0; i-- {
					f := data.Type.FieldType(i)
					if f.Size() == 0 {
						// eface type could also be struct{p *byte; q [0]int}
						continue
					}
					data = s.newValue1I(ssa.OpStructSelect, f, int64(i), data)
					break
				}
			default:
				s.Fatalf("type being put into an eface isn't a pointer")
			}
		}
		return s.newValue2(ssa.OpIMake, n.Type, tab, data)

	case OSLICE, OSLICEARR, OSLICE3, OSLICE3ARR:
		v := s.expr(n.Left)
		var i, j, k *ssa.Value
		low, high, max := n.SliceBounds()
		if low != nil {
			i = s.extendIndex(s.expr(low))
		}
		if high != nil {
			j = s.extendIndex(s.expr(high))
		}
		if max != nil {
			k = s.extendIndex(s.expr(max))
		}
		p, l, c := s.slice(n.Left.Type, v, i, j, k)
		return s.newValue3(ssa.OpSliceMake, n.Type, p, l, c)

	case OSLICESTR:
		v := s.expr(n.Left)
		var i, j *ssa.Value
		low, high, _ := n.SliceBounds()
		if low != nil {
			i = s.extendIndex(s.expr(low))
		}
		if high != nil {
			j = s.extendIndex(s.expr(high))
		}
		p, l, _ := s.slice(n.Left.Type, v, i, j, nil)
		return s.newValue2(ssa.OpStringMake, n.Type, p, l)

	case OCALLFUNC:
		if isIntrinsicCall1(n) {
			return s.intrinsicCall1(n)
		}
		fallthrough

	case OCALLINTER, OCALLMETH:
		a := s.call(n, callNormal)
		return s.newValue2(ssa.OpLoad, n.Type, a, s.mem())

	case OGETG:
		return s.newValue1(ssa.OpGetG, n.Type, s.mem())

	case OAPPEND:
		return s.append(n, false)

	default:
		s.Unimplementedf("unhandled expr %s", n.Op)
		return nil
	}
}

// append converts an OAPPEND node to SSA.
// If inplace is false, it converts the OAPPEND expression n to an ssa.Value,
// adds it to s, and returns the Value.
// If inplace is true, it writes the result of the OAPPEND expression n
// back to the slice being appended to, and returns nil.
// inplace MUST be set to false if the slice can be SSA'd.
func (s *state) append(n *Node, inplace bool) *ssa.Value {
	// If inplace is false, process as expression "append(s, e1, e2, e3)":
	//
	// ptr, len, cap := s
	// newlen := len + 3
	// if newlen > cap {
	//     ptr, len, cap = growslice(s, newlen)
	//     newlen = len + 3 // recalculate to avoid a spill
	// }
	// // with write barriers, if needed:
	// *(ptr+len) = e1
	// *(ptr+len+1) = e2
	// *(ptr+len+2) = e3
	// return makeslice(ptr, newlen, cap)
	//
	//
	// If inplace is true, process as statement "s = append(s, e1, e2, e3)":
	//
	// a := &s
	// ptr, len, cap := s
	// newlen := len + 3
	// if newlen > cap {
	//    newptr, len, newcap = growslice(ptr, len, cap, newlen)
	//    vardef(a)       // if necessary, advise liveness we are writing a new a
	//    *a.cap = newcap // write before ptr to avoid a spill
	//    *a.ptr = newptr // with write barrier
	// }
	// newlen = len + 3 // recalculate to avoid a spill
	// *a.len = newlen
	// // with write barriers, if needed:
	// *(ptr+len) = e1
	// *(ptr+len+1) = e2
	// *(ptr+len+2) = e3

	et := n.Type.Elem()
	pt := Ptrto(et)

	// Evaluate slice
	sn := n.List.First() // the slice node is the first in the list

	var slice, addr *ssa.Value
	if inplace {
		addr, _ = s.addr(sn, false)
		slice = s.newValue2(ssa.OpLoad, n.Type, addr, s.mem())
	} else {
		slice = s.expr(sn)
	}

	// Allocate new blocks
	grow := s.f.NewBlock(ssa.BlockPlain)
	assign := s.f.NewBlock(ssa.BlockPlain)

	// Decide if we need to grow
	nargs := int64(n.List.Len() - 1)
	p := s.newValue1(ssa.OpSlicePtr, pt, slice)
	l := s.newValue1(ssa.OpSliceLen, Types[TINT], slice)
	c := s.newValue1(ssa.OpSliceCap, Types[TINT], slice)
	nl := s.newValue2(s.ssaOp(OADD, Types[TINT]), Types[TINT], l, s.constInt(Types[TINT], nargs))

	cmp := s.newValue2(s.ssaOp(OGT, Types[TINT]), Types[TBOOL], nl, c)
	s.vars[&ptrVar] = p

	if !inplace {
		s.vars[&newlenVar] = nl
		s.vars[&capVar] = c
	} else {
		s.vars[&lenVar] = l
	}

	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Likely = ssa.BranchUnlikely
	b.SetControl(cmp)
	b.AddEdgeTo(grow)
	b.AddEdgeTo(assign)

	// Call growslice
	s.startBlock(grow)
	taddr := s.newValue1A(ssa.OpAddr, Types[TUINTPTR], &ssa.ExternSymbol{Typ: Types[TUINTPTR], Sym: typenamesym(n.Type.Elem())}, s.sb)

	r := s.rtcall(growslice, true, []*Type{pt, Types[TINT], Types[TINT]}, taddr, p, l, c, nl)

	if inplace {
		if sn.Op == ONAME {
			// Tell liveness we're about to build a new slice
			s.vars[&memVar] = s.newValue1A(ssa.OpVarDef, ssa.TypeMem, sn, s.mem())
		}
		capaddr := s.newValue1I(ssa.OpOffPtr, pt, int64(Array_cap), addr)
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.IntSize, capaddr, r[2], s.mem())
		s.insertWBstore(pt, addr, r[0], n.Lineno, 0)
		// load the value we just stored to avoid having to spill it
		s.vars[&ptrVar] = s.newValue2(ssa.OpLoad, pt, addr, s.mem())
		s.vars[&lenVar] = r[1] // avoid a spill in the fast path
	} else {
		s.vars[&ptrVar] = r[0]
		s.vars[&newlenVar] = s.newValue2(s.ssaOp(OADD, Types[TINT]), Types[TINT], r[1], s.constInt(Types[TINT], nargs))
		s.vars[&capVar] = r[2]
	}

	b = s.endBlock()
	b.AddEdgeTo(assign)

	// assign new elements to slots
	s.startBlock(assign)

	if inplace {
		l = s.variable(&lenVar, Types[TINT]) // generates phi for len
		nl = s.newValue2(s.ssaOp(OADD, Types[TINT]), Types[TINT], l, s.constInt(Types[TINT], nargs))
		lenaddr := s.newValue1I(ssa.OpOffPtr, pt, int64(Array_nel), addr)
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.IntSize, lenaddr, nl, s.mem())
	}

	// Evaluate args
	type argRec struct {
		// if store is true, we're appending the value v.  If false, we're appending the
		// value at *v.  If store==false, isVolatile reports whether the source
		// is in the outargs section of the stack frame.
		v          *ssa.Value
		store      bool
		isVolatile bool
	}
	args := make([]argRec, 0, nargs)
	for _, n := range n.List.Slice()[1:] {
		if canSSAType(n.Type) {
			args = append(args, argRec{v: s.expr(n), store: true})
		} else {
			v, isVolatile := s.addr(n, false)
			args = append(args, argRec{v: v, isVolatile: isVolatile})
		}
	}

	p = s.variable(&ptrVar, pt) // generates phi for ptr
	if !inplace {
		nl = s.variable(&newlenVar, Types[TINT]) // generates phi for nl
		c = s.variable(&capVar, Types[TINT])     // generates phi for cap
	}
	p2 := s.newValue2(ssa.OpPtrIndex, pt, p, l)
	// TODO: just one write barrier call for all of these writes?
	// TODO: maybe just one writeBarrier.enabled check?
	for i, arg := range args {
		addr := s.newValue2(ssa.OpPtrIndex, pt, p2, s.constInt(Types[TINT], int64(i)))
		if arg.store {
			if haspointers(et) {
				s.insertWBstore(et, addr, arg.v, n.Lineno, 0)
			} else {
				s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, et.Size(), addr, arg.v, s.mem())
			}
		} else {
			if haspointers(et) {
				s.insertWBmove(et, addr, arg.v, n.Lineno, arg.isVolatile)
			} else {
				s.vars[&memVar] = s.newValue3I(ssa.OpMove, ssa.TypeMem, et.Size(), addr, arg.v, s.mem())
			}
		}
	}

	delete(s.vars, &ptrVar)
	if inplace {
		delete(s.vars, &lenVar)
		return nil
	}
	delete(s.vars, &newlenVar)
	delete(s.vars, &capVar)
	// make result
	return s.newValue3(ssa.OpSliceMake, n.Type, p, nl, c)
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
		// whether the first branch is likely or not. So we pass 0 for
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
	b.SetControl(c)
	b.Likely = ssa.BranchPrediction(likely) // gc and ssa both use -1/0/+1 for likeliness
	b.AddEdgeTo(yes)
	b.AddEdgeTo(no)
}

type skipMask uint8

const (
	skipPtr skipMask = 1 << iota
	skipLen
	skipCap
)

// assign does left = right.
// Right has already been evaluated to ssa, left has not.
// If deref is true, then we do left = *right instead (and right has already been nil-checked).
// If deref is true and right == nil, just do left = 0.
// If deref is true, rightIsVolatile reports whether right points to volatile (clobbered by a call) storage.
// Include a write barrier if wb is true.
// skip indicates assignments (at the top level) that can be avoided.
func (s *state) assign(left *Node, right *ssa.Value, wb, deref bool, line int32, skip skipMask, rightIsVolatile bool) {
	if left.Op == ONAME && isblank(left) {
		return
	}
	t := left.Type
	dowidth(t)
	if s.canSSA(left) {
		if deref {
			s.Fatalf("can SSA LHS %s but not RHS %s", left, right)
		}
		if left.Op == ODOT {
			// We're assigning to a field of an ssa-able value.
			// We need to build a new structure with the new value for the
			// field we're assigning and the old values for the other fields.
			// For instance:
			//   type T struct {a, b, c int}
			//   var T x
			//   x.b = 5
			// For the x.b = 5 assignment we want to generate x = T{x.a, 5, x.c}

			// Grab information about the structure type.
			t := left.Left.Type
			nf := t.NumFields()
			idx := fieldIdx(left)

			// Grab old value of structure.
			old := s.expr(left.Left)

			// Make new structure.
			new := s.newValue0(ssa.StructMakeOp(t.NumFields()), t)

			// Add fields as args.
			for i := 0; i < nf; i++ {
				if i == idx {
					new.AddArg(right)
				} else {
					new.AddArg(s.newValue1I(ssa.OpStructSelect, t.FieldType(i), int64(i), old))
				}
			}

			// Recursively assign the new value we've made to the base of the dot op.
			s.assign(left.Left, new, false, false, line, 0, rightIsVolatile)
			// TODO: do we need to update named values here?
			return
		}
		// Update variable assignment.
		s.vars[left] = right
		s.addNamedValue(left, right)
		return
	}
	// Left is not ssa-able. Compute its address.
	addr, _ := s.addr(left, false)
	if left.Op == ONAME && skip == 0 {
		s.vars[&memVar] = s.newValue1A(ssa.OpVarDef, ssa.TypeMem, left, s.mem())
	}
	if deref {
		// Treat as a mem->mem move.
		if right == nil {
			s.vars[&memVar] = s.newValue2I(ssa.OpZero, ssa.TypeMem, t.Size(), addr, s.mem())
			return
		}
		if wb {
			s.insertWBmove(t, addr, right, line, rightIsVolatile)
			return
		}
		s.vars[&memVar] = s.newValue3I(ssa.OpMove, ssa.TypeMem, t.Size(), addr, right, s.mem())
		return
	}
	// Treat as a store.
	if wb {
		if skip&skipPtr != 0 {
			// Special case: if we don't write back the pointers, don't bother
			// doing the write barrier check.
			s.storeTypeScalars(t, addr, right, skip)
			return
		}
		s.insertWBstore(t, addr, right, line, skip)
		return
	}
	if skip != 0 {
		if skip&skipPtr == 0 {
			s.storeTypePtrs(t, addr, right)
		}
		s.storeTypeScalars(t, addr, right, skip)
		return
	}
	s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, t.Size(), addr, right, s.mem())
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
		return s.constEmptyString(t)
	case t.IsPtrShaped():
		return s.constNil(t)
	case t.IsBoolean():
		return s.constBool(false)
	case t.IsInterface():
		return s.constInterface(t)
	case t.IsSlice():
		return s.constSlice(t)
	case t.IsStruct():
		n := t.NumFields()
		v := s.entryNewValue0(ssa.StructMakeOp(t.NumFields()), t)
		for i := 0; i < n; i++ {
			v.AddArg(s.zeroVal(t.FieldType(i).(*Type)))
		}
		return v
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

// isSSAIntrinsic1 returns true if n is a call to a recognized 1-arg intrinsic
// that can be handled by the SSA backend.
// SSA uses this, but so does the front end to see if should not
// inline a function because it is a candidate for intrinsic
// substitution.
func isSSAIntrinsic1(s *Sym) bool {
	// The test below is not quite accurate -- in the event that
	// a function is disabled on a per-function basis, for example
	// because of hash-keyed binary failure search, SSA might be
	// disabled for that function but it would not be noted here,
	// and thus an inlining would not occur (in practice, inlining
	// so far has only been noticed for Bswap32 and the 16-bit count
	// leading/trailing instructions, but heuristics might change
	// in the future or on different architectures).
	if !ssaEnabled || ssa.IntrinsicsDisable || Thearch.LinkArch.Family != sys.AMD64 {
		return false
	}
	if s != nil && s.Pkg != nil && s.Pkg.Path == "runtime/internal/sys" {
		switch s.Name {
		case
			"Ctz64", "Ctz32", "Ctz16",
			"Bswap64", "Bswap32":
			return true
		}
	}
	return false
}

func isIntrinsicCall1(n *Node) bool {
	if n == nil || n.Left == nil {
		return false
	}
	return isSSAIntrinsic1(n.Left.Sym)
}

// intrinsicFirstArg extracts arg from n.List and eval
func (s *state) intrinsicFirstArg(n *Node) *ssa.Value {
	x := n.List.First()
	if x.Op == OAS {
		x = x.Right
	}
	return s.expr(x)
}

// intrinsicCall1 converts a call to a recognized 1-arg intrinsic
// into the intrinsic
func (s *state) intrinsicCall1(n *Node) *ssa.Value {
	var result *ssa.Value
	switch n.Left.Sym.Name {
	case "Ctz64":
		result = s.newValue1(ssa.OpCtz64, Types[TUINT64], s.intrinsicFirstArg(n))
	case "Ctz32":
		result = s.newValue1(ssa.OpCtz32, Types[TUINT32], s.intrinsicFirstArg(n))
	case "Ctz16":
		result = s.newValue1(ssa.OpCtz16, Types[TUINT16], s.intrinsicFirstArg(n))
	case "Bswap64":
		result = s.newValue1(ssa.OpBswap64, Types[TUINT64], s.intrinsicFirstArg(n))
	case "Bswap32":
		result = s.newValue1(ssa.OpBswap32, Types[TUINT32], s.intrinsicFirstArg(n))
	}
	if result == nil {
		Fatalf("Unknown special call: %v", n.Left.Sym)
	}
	if ssa.IntrinsicsDebug > 0 {
		Warnl(n.Lineno, "intrinsic substitution for %v with %s", n.Left.Sym.Name, result.LongString())
	}
	return result
}

// Calls the function n using the specified call type.
// Returns the address of the return value (or nil if none).
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
	case OCALLMETH:
		if fn.Op != ODOTMETH {
			Fatalf("OCALLMETH: n.Left not an ODOTMETH: %v", fn)
		}
		if k == callNormal {
			sym = fn.Sym
			break
		}
		n2 := newname(fn.Sym)
		n2.Class = PFUNC
		n2.Lineno = fn.Lineno
		closure = s.expr(n2)
		// Note: receiver is already assigned in n.List, so we don't
		// want to set it here.
	case OCALLINTER:
		if fn.Op != ODOTINTER {
			Fatalf("OCALLINTER: n.Left not an ODOTINTER: %v", fn.Op)
		}
		i := s.expr(fn.Left)
		itab := s.newValue1(ssa.OpITab, Types[TUINTPTR], i)
		if k != callNormal {
			s.nilCheck(itab)
		}
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
	stksize := fn.Type.ArgWidth() // includes receiver

	// Run all argument assignments. The arg slots have already
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
		Fatalf("bad call type %s %v", n.Op, n)
	}
	call.AuxInt = stksize // Call operations carry the argsize of the callee along with them

	// Finish call block
	s.vars[&memVar] = call
	b := s.endBlock()
	b.Kind = ssa.BlockCall
	b.SetControl(call)
	b.AddEdgeTo(bNext)
	if k == callDefer {
		// Add recover edge to exit code.
		b.Kind = ssa.BlockDefer
		r := s.f.NewBlock(ssa.BlockPlain)
		s.startBlock(r)
		s.exit()
		b.AddEdgeTo(r)
		b.Likely = ssa.BranchLikely
	}

	// Start exit block, find address of result.
	s.startBlock(bNext)
	// Keep input pointer args live across calls.  This is a bandaid until 1.8.
	for _, n := range s.ptrargs {
		s.vars[&memVar] = s.newValue2(ssa.OpKeepAlive, ssa.TypeMem, s.variable(n, n.Type), s.mem())
	}
	res := n.Left.Type.Results()
	if res.NumFields() == 0 || k != callNormal {
		// call has no return value. Continue with the next statement.
		return nil
	}
	fp := res.Field(0)
	return s.entryNewValue1I(ssa.OpOffPtr, Ptrto(fp.Type), fp.Offset+Ctxt.FixedFrameSize(), s.sp)
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
// Also returns a bool reporting whether the returned value is "volatile", that is it
// points to the outargs section and thus the referent will be clobbered by any call.
// The value that the returned Value represents is guaranteed to be non-nil.
// If bounded is true then this address does not require a nil check for its operand
// even if that would otherwise be implied.
func (s *state) addr(n *Node, bounded bool) (*ssa.Value, bool) {
	t := Ptrto(n.Type)
	switch n.Op {
	case ONAME:
		switch n.Class {
		case PEXTERN:
			// global variable
			aux := s.lookupSymbol(n, &ssa.ExternSymbol{Typ: n.Type, Sym: n.Sym})
			v := s.entryNewValue1A(ssa.OpAddr, t, aux, s.sb)
			// TODO: Make OpAddr use AuxInt as well as Aux.
			if n.Xoffset != 0 {
				v = s.entryNewValue1I(ssa.OpOffPtr, v.Type, n.Xoffset, v)
			}
			return v, false
		case PPARAM:
			// parameter slot
			v := s.decladdrs[n]
			if v != nil {
				return v, false
			}
			if n.String() == ".fp" {
				// Special arg that points to the frame pointer.
				// (Used by the race detector, others?)
				aux := s.lookupSymbol(n, &ssa.ArgSymbol{Typ: n.Type, Node: n})
				return s.entryNewValue1A(ssa.OpAddr, t, aux, s.sp), false
			}
			s.Fatalf("addr of undeclared ONAME %v. declared: %v", n, s.decladdrs)
			return nil, false
		case PAUTO:
			aux := s.lookupSymbol(n, &ssa.AutoSymbol{Typ: n.Type, Node: n})
			return s.newValue1A(ssa.OpAddr, t, aux, s.sp), false
		case PPARAMOUT: // Same as PAUTO -- cannot generate LEA early.
			// ensure that we reuse symbols for out parameters so
			// that cse works on their addresses
			aux := s.lookupSymbol(n, &ssa.ArgSymbol{Typ: n.Type, Node: n})
			return s.newValue1A(ssa.OpAddr, t, aux, s.sp), false
		default:
			s.Unimplementedf("variable address class %v not implemented", classnames[n.Class])
			return nil, false
		}
	case OINDREG:
		// indirect off a register
		// used for storing/loading arguments/returns to/from callees
		if int(n.Reg) != Thearch.REGSP {
			s.Unimplementedf("OINDREG of non-SP register %s in addr: %v", obj.Rconv(int(n.Reg)), n)
			return nil, false
		}
		return s.entryNewValue1I(ssa.OpOffPtr, t, n.Xoffset, s.sp), true
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
			return s.newValue2(ssa.OpPtrIndex, t, p, i), false
		} else { // array
			a, isVolatile := s.addr(n.Left, bounded)
			i := s.expr(n.Right)
			i = s.extendIndex(i)
			len := s.constInt(Types[TINT], n.Left.Type.NumElem())
			if !n.Bounded {
				s.boundsCheck(i, len)
			}
			return s.newValue2(ssa.OpPtrIndex, Ptrto(n.Left.Type.Elem()), a, i), isVolatile
		}
	case OIND:
		return s.exprPtr(n.Left, bounded, n.Lineno), false
	case ODOT:
		p, isVolatile := s.addr(n.Left, bounded)
		return s.newValue1I(ssa.OpOffPtr, t, n.Xoffset, p), isVolatile
	case ODOTPTR:
		p := s.exprPtr(n.Left, bounded, n.Lineno)
		return s.newValue1I(ssa.OpOffPtr, t, n.Xoffset, p), false
	case OCLOSUREVAR:
		return s.newValue1I(ssa.OpOffPtr, t, n.Xoffset,
			s.entryNewValue0(ssa.OpGetClosurePtr, Ptrto(Types[TUINT8]))), false
	case OCONVNOP:
		addr, isVolatile := s.addr(n.Left, bounded)
		return s.newValue1(ssa.OpCopy, t, addr), isVolatile // ensure that addr has the right type
	case OCALLFUNC, OCALLINTER, OCALLMETH:
		return s.call(n, callNormal), true

	default:
		s.Unimplementedf("unhandled addr %v", n.Op)
		return nil, false
	}
}

// canSSA reports whether n is SSA-able.
// n must be an ONAME (or an ODOT sequence with an ONAME base).
func (s *state) canSSA(n *Node) bool {
	if Debug['N'] != 0 {
		return false
	}
	for n.Op == ODOT {
		n = n.Left
	}
	if n.Op != ONAME {
		return false
	}
	if n.Addrtaken {
		return false
	}
	if n.isParamHeapCopy() {
		return false
	}
	if n.Class == PAUTOHEAP {
		Fatalf("canSSA of PAUTOHEAP %v", n)
	}
	switch n.Class {
	case PEXTERN:
		return false
	case PPARAMOUT:
		if hasdefer {
			// TODO: handle this case?  Named return values must be
			// in memory so that the deferred function can see them.
			// Maybe do: if !strings.HasPrefix(n.String(), "~") { return false }
			return false
		}
		if s.cgoUnsafeArgs {
			// Cgo effectively takes the address of all result args,
			// but the compiler can't see that.
			return false
		}
	}
	if n.Class == PPARAM && n.String() == ".this" {
		// wrappers generated by genwrapper need to update
		// the .this pointer in place.
		// TODO: treat as a PPARMOUT?
		return false
	}
	return canSSAType(n.Type)
	// TODO: try to make more variables SSAable?
}

// canSSA reports whether variables of type t are SSA-able.
func canSSAType(t *Type) bool {
	dowidth(t)
	if t.Width > int64(4*Widthptr) {
		// 4*Widthptr is an arbitrary constant. We want it
		// to be at least 3*Widthptr so slices can be registerized.
		// Too big and we'll introduce too much register pressure.
		return false
	}
	switch t.Etype {
	case TARRAY:
		// We can't do arrays because dynamic indexing is
		// not supported on SSA variables.
		// TODO: maybe allow if length is <=1?  All indexes
		// are constant?  Might be good for the arrays
		// introduced by the compiler for variadic functions.
		return false
	case TSTRUCT:
		if t.NumFields() > ssa.MaxStruct {
			return false
		}
		for _, t1 := range t.Fields().Slice() {
			if !canSSAType(t1.Type) {
				return false
			}
		}
		return true
	default:
		return true
	}
}

// exprPtr evaluates n to a pointer and nil-checks it.
func (s *state) exprPtr(n *Node, bounded bool, lineno int32) *ssa.Value {
	p := s.expr(n)
	if bounded || n.NonNil {
		if s.f.Config.Debug_checknil() && lineno > 1 {
			s.f.Config.Warnl(lineno, "removed nil check")
		}
		return p
	}
	s.nilCheck(p)
	return p
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
	b.SetControl(chk)
	bNext := s.f.NewBlock(ssa.BlockPlain)
	b.AddEdgeTo(bNext)
	s.startBlock(bNext)
}

// boundsCheck generates bounds checking code. Checks if 0 <= idx < len, branches to exit if not.
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

// sliceBoundsCheck generates slice bounds checking code. Checks if 0 <= idx <= len, branches to exit if not.
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
	b.SetControl(cmp)
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
// If returns is true, the block is marked as a call block. A new block
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
		b.SetControl(call)
		call.AuxInt = off
		if len(results) > 0 {
			Fatalf("panic call can't have results")
		}
		return nil
	}
	b.Kind = ssa.BlockCall
	b.SetControl(call)
	bNext := s.f.NewBlock(ssa.BlockPlain)
	b.AddEdgeTo(bNext)
	s.startBlock(bNext)

	// Keep input pointer args live across calls.  This is a bandaid until 1.8.
	for _, n := range s.ptrargs {
		s.vars[&memVar] = s.newValue2(ssa.OpKeepAlive, ssa.TypeMem, s.variable(n, n.Type), s.mem())
	}

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

// insertWBmove inserts the assignment *left = *right including a write barrier.
// t is the type being assigned.
func (s *state) insertWBmove(t *Type, left, right *ssa.Value, line int32, rightIsVolatile bool) {
	// if writeBarrier.enabled {
	//   typedmemmove(&t, left, right)
	// } else {
	//   *left = *right
	// }

	if s.noWB {
		s.Fatalf("write barrier prohibited")
	}
	if s.WBLineno == 0 {
		s.WBLineno = left.Line
	}
	bThen := s.f.NewBlock(ssa.BlockPlain)
	bElse := s.f.NewBlock(ssa.BlockPlain)
	bEnd := s.f.NewBlock(ssa.BlockPlain)

	aux := &ssa.ExternSymbol{Typ: Types[TBOOL], Sym: syslook("writeBarrier").Sym}
	flagaddr := s.newValue1A(ssa.OpAddr, Ptrto(Types[TUINT32]), aux, s.sb)
	// TODO: select the .enabled field. It is currently first, so not needed for now.
	// Load word, test byte, avoiding partial register write from load byte.
	flag := s.newValue2(ssa.OpLoad, Types[TUINT32], flagaddr, s.mem())
	flag = s.newValue1(ssa.OpTrunc64to8, Types[TBOOL], flag)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Likely = ssa.BranchUnlikely
	b.SetControl(flag)
	b.AddEdgeTo(bThen)
	b.AddEdgeTo(bElse)

	s.startBlock(bThen)

	if !rightIsVolatile {
		// Issue typedmemmove call.
		taddr := s.newValue1A(ssa.OpAddr, Types[TUINTPTR], &ssa.ExternSymbol{Typ: Types[TUINTPTR], Sym: typenamesym(t)}, s.sb)
		s.rtcall(typedmemmove, true, nil, taddr, left, right)
	} else {
		// Copy to temp location if the source is volatile (will be clobbered by
		// a function call).  Marshaling the args to typedmemmove might clobber the
		// value we're trying to move.
		tmp := temp(t)
		s.vars[&memVar] = s.newValue1A(ssa.OpVarDef, ssa.TypeMem, tmp, s.mem())
		tmpaddr, _ := s.addr(tmp, true)
		s.vars[&memVar] = s.newValue3I(ssa.OpMove, ssa.TypeMem, t.Size(), tmpaddr, right, s.mem())
		// Issue typedmemmove call.
		taddr := s.newValue1A(ssa.OpAddr, Types[TUINTPTR], &ssa.ExternSymbol{Typ: Types[TUINTPTR], Sym: typenamesym(t)}, s.sb)
		s.rtcall(typedmemmove, true, nil, taddr, left, tmpaddr)
		// Mark temp as dead.
		s.vars[&memVar] = s.newValue1A(ssa.OpVarKill, ssa.TypeMem, tmp, s.mem())
	}
	s.endBlock().AddEdgeTo(bEnd)

	s.startBlock(bElse)
	s.vars[&memVar] = s.newValue3I(ssa.OpMove, ssa.TypeMem, t.Size(), left, right, s.mem())
	s.endBlock().AddEdgeTo(bEnd)

	s.startBlock(bEnd)

	if Debug_wb > 0 {
		Warnl(line, "write barrier")
	}
}

// insertWBstore inserts the assignment *left = right including a write barrier.
// t is the type being assigned.
func (s *state) insertWBstore(t *Type, left, right *ssa.Value, line int32, skip skipMask) {
	// store scalar fields
	// if writeBarrier.enabled {
	//   writebarrierptr for pointer fields
	// } else {
	//   store pointer fields
	// }

	if s.noWB {
		s.Fatalf("write barrier prohibited")
	}
	if s.WBLineno == 0 {
		s.WBLineno = left.Line
	}
	s.storeTypeScalars(t, left, right, skip)

	bThen := s.f.NewBlock(ssa.BlockPlain)
	bElse := s.f.NewBlock(ssa.BlockPlain)
	bEnd := s.f.NewBlock(ssa.BlockPlain)

	aux := &ssa.ExternSymbol{Typ: Types[TBOOL], Sym: syslook("writeBarrier").Sym}
	flagaddr := s.newValue1A(ssa.OpAddr, Ptrto(Types[TUINT32]), aux, s.sb)
	// TODO: select the .enabled field. It is currently first, so not needed for now.
	// Load word, test byte, avoiding partial register write from load byte.
	flag := s.newValue2(ssa.OpLoad, Types[TUINT32], flagaddr, s.mem())
	flag = s.newValue1(ssa.OpTrunc64to8, Types[TBOOL], flag)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.Likely = ssa.BranchUnlikely
	b.SetControl(flag)
	b.AddEdgeTo(bThen)
	b.AddEdgeTo(bElse)

	// Issue write barriers for pointer writes.
	s.startBlock(bThen)
	s.storeTypePtrsWB(t, left, right)
	s.endBlock().AddEdgeTo(bEnd)

	// Issue regular stores for pointer writes.
	s.startBlock(bElse)
	s.storeTypePtrs(t, left, right)
	s.endBlock().AddEdgeTo(bEnd)

	s.startBlock(bEnd)

	if Debug_wb > 0 {
		Warnl(line, "write barrier")
	}
}

// do *left = right for all scalar (non-pointer) parts of t.
func (s *state) storeTypeScalars(t *Type, left, right *ssa.Value, skip skipMask) {
	switch {
	case t.IsBoolean() || t.IsInteger() || t.IsFloat() || t.IsComplex():
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, t.Size(), left, right, s.mem())
	case t.IsPtrShaped():
		// no scalar fields.
	case t.IsString():
		if skip&skipLen != 0 {
			return
		}
		len := s.newValue1(ssa.OpStringLen, Types[TINT], right)
		lenAddr := s.newValue1I(ssa.OpOffPtr, Ptrto(Types[TINT]), s.config.IntSize, left)
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.IntSize, lenAddr, len, s.mem())
	case t.IsSlice():
		if skip&skipLen == 0 {
			len := s.newValue1(ssa.OpSliceLen, Types[TINT], right)
			lenAddr := s.newValue1I(ssa.OpOffPtr, Ptrto(Types[TINT]), s.config.IntSize, left)
			s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.IntSize, lenAddr, len, s.mem())
		}
		if skip&skipCap == 0 {
			cap := s.newValue1(ssa.OpSliceCap, Types[TINT], right)
			capAddr := s.newValue1I(ssa.OpOffPtr, Ptrto(Types[TINT]), 2*s.config.IntSize, left)
			s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.IntSize, capAddr, cap, s.mem())
		}
	case t.IsInterface():
		// itab field doesn't need a write barrier (even though it is a pointer).
		itab := s.newValue1(ssa.OpITab, Ptrto(Types[TUINT8]), right)
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.IntSize, left, itab, s.mem())
	case t.IsStruct():
		n := t.NumFields()
		for i := 0; i < n; i++ {
			ft := t.FieldType(i)
			addr := s.newValue1I(ssa.OpOffPtr, ft.PtrTo(), t.FieldOff(i), left)
			val := s.newValue1I(ssa.OpStructSelect, ft, int64(i), right)
			s.storeTypeScalars(ft.(*Type), addr, val, 0)
		}
	default:
		s.Fatalf("bad write barrier type %s", t)
	}
}

// do *left = right for all pointer parts of t.
func (s *state) storeTypePtrs(t *Type, left, right *ssa.Value) {
	switch {
	case t.IsPtrShaped():
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.PtrSize, left, right, s.mem())
	case t.IsString():
		ptr := s.newValue1(ssa.OpStringPtr, Ptrto(Types[TUINT8]), right)
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.PtrSize, left, ptr, s.mem())
	case t.IsSlice():
		ptr := s.newValue1(ssa.OpSlicePtr, Ptrto(Types[TUINT8]), right)
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.PtrSize, left, ptr, s.mem())
	case t.IsInterface():
		// itab field is treated as a scalar.
		idata := s.newValue1(ssa.OpIData, Ptrto(Types[TUINT8]), right)
		idataAddr := s.newValue1I(ssa.OpOffPtr, Ptrto(Types[TUINT8]), s.config.PtrSize, left)
		s.vars[&memVar] = s.newValue3I(ssa.OpStore, ssa.TypeMem, s.config.PtrSize, idataAddr, idata, s.mem())
	case t.IsStruct():
		n := t.NumFields()
		for i := 0; i < n; i++ {
			ft := t.FieldType(i)
			if !haspointers(ft.(*Type)) {
				continue
			}
			addr := s.newValue1I(ssa.OpOffPtr, ft.PtrTo(), t.FieldOff(i), left)
			val := s.newValue1I(ssa.OpStructSelect, ft, int64(i), right)
			s.storeTypePtrs(ft.(*Type), addr, val)
		}
	default:
		s.Fatalf("bad write barrier type %s", t)
	}
}

// do *left = right with a write barrier for all pointer parts of t.
func (s *state) storeTypePtrsWB(t *Type, left, right *ssa.Value) {
	switch {
	case t.IsPtrShaped():
		s.rtcall(writebarrierptr, true, nil, left, right)
	case t.IsString():
		ptr := s.newValue1(ssa.OpStringPtr, Ptrto(Types[TUINT8]), right)
		s.rtcall(writebarrierptr, true, nil, left, ptr)
	case t.IsSlice():
		ptr := s.newValue1(ssa.OpSlicePtr, Ptrto(Types[TUINT8]), right)
		s.rtcall(writebarrierptr, true, nil, left, ptr)
	case t.IsInterface():
		idata := s.newValue1(ssa.OpIData, Ptrto(Types[TUINT8]), right)
		idataAddr := s.newValue1I(ssa.OpOffPtr, Ptrto(Types[TUINT8]), s.config.PtrSize, left)
		s.rtcall(writebarrierptr, true, nil, idataAddr, idata)
	case t.IsStruct():
		n := t.NumFields()
		for i := 0; i < n; i++ {
			ft := t.FieldType(i)
			if !haspointers(ft.(*Type)) {
				continue
			}
			addr := s.newValue1I(ssa.OpOffPtr, ft.PtrTo(), t.FieldOff(i), left)
			val := s.newValue1I(ssa.OpStructSelect, ft, int64(i), right)
			s.storeTypePtrsWB(ft.(*Type), addr, val)
		}
	default:
		s.Fatalf("bad write barrier type %s", t)
	}
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
		elemtype = t.Elem()
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
		if !t.Elem().IsArray() {
			s.Fatalf("bad ptr to array in slice %v\n", t)
		}
		elemtype = t.Elem().Elem()
		ptrtype = Ptrto(elemtype)
		s.nilCheck(v)
		ptr = v
		len = s.constInt(Types[TINT], t.Elem().NumElem())
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
	// rlen = j-i
	// rcap = k-i
	// delta = i*elemsize
	// if rcap == 0 {
	//    delta = 0
	// }
	// rptr = p+delta
	// result = (SliceMake rptr rlen rcap)
	subOp := s.ssaOp(OSUB, Types[TINT])
	eqOp := s.ssaOp(OEQ, Types[TINT])
	mulOp := s.ssaOp(OMUL, Types[TINT])
	rlen := s.newValue2(subOp, Types[TINT], j, i)
	var rcap *ssa.Value
	switch {
	case t.IsString():
		// Capacity of the result is unimportant. However, we use
		// rcap to test if we've generated a zero-length slice.
		// Use length of strings for that.
		rcap = rlen
	case j == k:
		rcap = rlen
	default:
		rcap = s.newValue2(subOp, Types[TINT], k, i)
	}

	// delta = # of elements to offset pointer by.
	s.vars[&deltaVar] = i

	// Generate code to set delta=0 if the resulting capacity is zero.
	if !((i.Op == ssa.OpConst64 && i.AuxInt == 0) ||
		(i.Op == ssa.OpConst32 && int32(i.AuxInt) == 0)) {
		cmp := s.newValue2(eqOp, Types[TBOOL], rcap, zero)

		b := s.endBlock()
		b.Kind = ssa.BlockIf
		b.Likely = ssa.BranchUnlikely
		b.SetControl(cmp)

		// Generate block which zeros the delta variable.
		nz := s.f.NewBlock(ssa.BlockPlain)
		b.AddEdgeTo(nz)
		s.startBlock(nz)
		s.vars[&deltaVar] = zero
		s.endBlock()

		// All done.
		merge := s.f.NewBlock(ssa.BlockPlain)
		b.AddEdgeTo(merge)
		nz.AddEdgeTo(merge)
		s.startBlock(merge)

		// TODO: use conditional moves somehow?
	}

	// Compute rptr = ptr + delta * elemsize
	rptr := s.newValue2(ssa.OpAddPtr, ptrtype, ptr, s.newValue2(mulOp, Types[TINT], s.variable(&deltaVar, Types[TINT]), s.constInt(Types[TINT], elemtype.Width)))
	delete(s.vars, &deltaVar)
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
	// conversion. However, because the mantissa is only
	// 63 bits, we don't need the LSB, so instead we do an
	// unsigned right shift (divide by two), convert, and
	// double. However, before we do that, we need to be
	// sure that we do not lose a "1" if that made the
	// difference in the resulting rounding. Therefore, we
	// preserve it, and OR (not ADD) it back in. The case
	// that matters is when the eleven discarded bits are
	// equal to 10000000001; that rounds up, and the 1 cannot
	// be lost else it would round down if the LSB of the
	// candidate mantissa is 0.
	cmp := s.newValue2(cvttab.geq, Types[TBOOL], x, s.zeroVal(ft))
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.SetControl(cmp)
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
	nilValue := s.constNil(Types[TUINTPTR])
	cmp := s.newValue2(ssa.OpEqPtr, Types[TBOOL], x, nilValue)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.SetControl(cmp)
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
	b.SetControl(cmp)
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

	if n.Type.IsEmptyInterface() {
		// Have *eface. The type is the first word in the struct.
		return s.newValue1(ssa.OpITab, byteptr, v)
	}

	// Have *iface.
	// The first word in the struct is the *itab.
	// If the *itab is nil, return 0.
	// Otherwise, the second word in the *itab is the type.

	tab := s.newValue1(ssa.OpITab, byteptr, v)
	s.vars[&typVar] = tab
	isnonnil := s.newValue2(ssa.OpNeqPtr, Types[TBOOL], tab, s.constNil(byteptr))
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.SetControl(isnonnil)
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
		Warnl(n.Lineno, "type assertion inlined")
	}

	// TODO:  If we have a nonempty interface and its itab field is nil,
	// then this test is redundant and ifaceType should just branch directly to bFail.
	cond := s.newValue2(ssa.OpEqPtr, Types[TBOOL], typ, target)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.SetControl(cond)
	b.Likely = ssa.BranchLikely

	byteptr := Ptrto(Types[TUINT8])

	bOk := s.f.NewBlock(ssa.BlockPlain)
	bFail := s.f.NewBlock(ssa.BlockPlain)
	b.AddEdgeTo(bOk)
	b.AddEdgeTo(bFail)

	if !commaok {
		// on failure, panic by calling panicdottype
		s.startBlock(bFail)
		taddr := s.newValue1A(ssa.OpAddr, byteptr, &ssa.ExternSymbol{Typ: byteptr, Sym: typenamesym(n.Left.Type)}, s.sb)
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
	s.vars[&idataVar] = s.constNil(byteptr)
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

		lno := from.Left.Lineno
		if block != nil {
			yyerrorl(lno, "goto %v jumps into block starting at %v", from.Left.Sym, linestr(block.Lastlineno))
		} else {
			yyerrorl(lno, "goto %v jumps over declaration of %v at %v", from.Left.Sym, dcl, linestr(dcl.Lastlineno))
		}
	}
}

// variable returns the value of a variable at the current location.
func (s *state) variable(name *Node, t ssa.Type) *ssa.Value {
	v := s.vars[name]
	if v == nil {
		v = s.newValue0A(ssa.OpFwdRef, t, name)
		s.fwdRefs = append(s.fwdRefs, v)
		s.vars[name] = v
		s.addNamedValue(name, v)
	}
	return v
}

func (s *state) mem() *ssa.Value {
	return s.variable(&memVar, ssa.TypeMem)
}

func (s *state) linkForwardReferences(dm *sparseDefState) {

	// Build SSA graph. Each variable on its first use in a basic block
	// leaves a FwdRef in that block representing the incoming value
	// of that variable. This function links that ref up with possible definitions,
	// inserting Phi values as needed. This is essentially the algorithm
	// described by Braun, Buchwald, Hack, Leißa, Mallon, and Zwinkau:
	// http://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
	// Differences:
	//   - We use FwdRef nodes to postpone phi building until the CFG is
	//     completely built. That way we can avoid the notion of "sealed"
	//     blocks.
	//   - Phi optimization is a separate pass (in ../ssa/phielim.go).
	for len(s.fwdRefs) > 0 {
		v := s.fwdRefs[len(s.fwdRefs)-1]
		s.fwdRefs = s.fwdRefs[:len(s.fwdRefs)-1]
		s.resolveFwdRef(v, dm)
	}
}

// resolveFwdRef modifies v to be the variable's value at the start of its block.
// v must be a FwdRef op.
func (s *state) resolveFwdRef(v *ssa.Value, dm *sparseDefState) {
	b := v.Block
	name := v.Aux.(*Node)
	v.Aux = nil
	if b == s.f.Entry {
		// Live variable at start of function.
		if s.canSSA(name) {
			if strings.HasPrefix(name.Sym.Name, "autotmp_") {
				// It's likely that this is an uninitialized variable in the entry block.
				s.Fatalf("Treating auto as if it were arg, func %s, node %v, value %v", b.Func.Name, name, v)
			}
			v.Op = ssa.OpArg
			v.Aux = name
			return
		}
		// Not SSAable. Load it.
		addr := s.decladdrs[name]
		if addr == nil {
			// TODO: closure args reach here.
			s.Unimplementedf("unhandled closure arg %s at entry to function %s", name, b.Func.Name)
		}
		if _, ok := addr.Aux.(*ssa.ArgSymbol); !ok {
			s.Fatalf("variable live at start of function %s is not an argument %s", b.Func.Name, name)
		}
		v.Op = ssa.OpLoad
		v.AddArgs(addr, s.startmem)
		return
	}
	if len(b.Preds) == 0 {
		// This block is dead; we have no predecessors and we're not the entry block.
		// It doesn't matter what we use here as long as it is well-formed.
		v.Op = ssa.OpUnknown
		return
	}
	// Find variable value on each predecessor.
	var argstore [4]*ssa.Value
	args := argstore[:0]
	for _, e := range b.Preds {
		p := e.Block()
		p = dm.FindBetterDefiningBlock(name, p) // try sparse improvement on p
		args = append(args, s.lookupVarOutgoing(p, v.Type, name, v.Line))
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
			v.Op = ssa.OpPhi
			v.AddArgs(args...)
			return
		}
		w = a // save witness
	}
	if w == nil {
		s.Fatalf("no witness for reachable phi %s", v)
	}
	// One witness. Make v a copy of w.
	v.Op = ssa.OpCopy
	v.AddArg(w)
}

// lookupVarOutgoing finds the variable's value at the end of block b.
func (s *state) lookupVarOutgoing(b *ssa.Block, t ssa.Type, name *Node, line int32) *ssa.Value {
	for {
		if v, ok := s.defvars[b.ID][name]; ok {
			return v
		}
		// The variable is not defined by b and we haven't looked it up yet.
		// If b has exactly one predecessor, loop to look it up there.
		// Otherwise, give up and insert a new FwdRef and resolve it later.
		if len(b.Preds) != 1 {
			break
		}
		b = b.Preds[0].Block()
	}
	// Generate a FwdRef for the variable and return that.
	v := b.NewValue0A(line, ssa.OpFwdRef, t, name)
	s.fwdRefs = append(s.fwdRefs, v)
	s.defvars[b.ID][name] = v
	s.addNamedValue(name, v)
	return v
}

func (s *state) addNamedValue(n *Node, v *ssa.Value) {
	if n.Class == Pxxx {
		// Don't track our dummy nodes (&memVar etc.).
		return
	}
	if strings.HasPrefix(n.Sym.Name, "autotmp_") {
		// Don't track autotmp_ variables.
		return
	}
	if n.Class == PPARAMOUT {
		// Don't track named output values.  This prevents return values
		// from being assigned too early. See #14591 and #14762. TODO: allow this.
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

// Branch is an unresolved branch.
type Branch struct {
	P *obj.Prog  // branch instruction
	B *ssa.Block // target
}

// SSAGenState contains state needed during Prog generation.
type SSAGenState struct {
	// Branches remembers all the branch instructions we've seen
	// and where they would like to go.
	Branches []Branch

	// bstart remembers where each block starts (indexed by block ID)
	bstart []*obj.Prog
}

// Pc returns the current Prog.
func (s *SSAGenState) Pc() *obj.Prog {
	return Pc
}

// SetLineno sets the current source line number.
func (s *SSAGenState) SetLineno(l int32) {
	lineno = l
}

// genssa appends entries to ptxt for each instruction in f.
// gcargs and gclocals are filled in with pointer maps for the frame.
func genssa(f *ssa.Func, ptxt *obj.Prog, gcargs, gclocals *Sym) {
	var s SSAGenState

	e := f.Config.Frontend().(*ssaExport)
	// We're about to emit a bunch of Progs.
	// Since the only way to get here is to explicitly request it,
	// just fail on unimplemented instead of trying to unwind our mess.
	e.mustImplement = true

	// Remember where each block starts.
	s.bstart = make([]*obj.Prog, f.NumBlocks())

	var valueProgs map[*obj.Prog]*ssa.Value
	var blockProgs map[*obj.Prog]*ssa.Block
	var logProgs = e.log
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
		Thearch.SSAMarkMoves(&s, b)
		for _, v := range b.Values {
			x := Pc
			Thearch.SSAGenValue(&s, v)
			if logProgs {
				for ; x != Pc; x = x.Link {
					valueProgs[x] = v
				}
			}
		}
		// Emit control flow instructions for block
		var next *ssa.Block
		if i < len(f.Blocks)-1 && (Debug['N'] == 0 || b.Kind == ssa.BlockCall) {
			// If -N, leave next==nil so every block with successors
			// ends in a JMP (except call blocks - plive doesn't like
			// select{send,recv} followed by a JMP call).  Helps keep
			// line numbers for otherwise empty blocks.
			next = f.Blocks[i+1]
		}
		x := Pc
		Thearch.SSAGenBlock(&s, b, next)
		if logProgs {
			for ; x != Pc; x = x.Link {
				blockProgs[x] = b
			}
		}
	}

	// Resolve branches
	for _, br := range s.Branches {
		br.P.To.Val = s.bstart[br.B.ID]
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
				Fatalf("non-static data marked as static: %v\n\n", n)
			}
		}
	}

	// Allocate stack frame
	allocauto(ptxt)

	// Generate gc bitmaps.
	liveness(Curfn, ptxt, gcargs, gclocals)

	// Add frame prologue. Zero ambiguously live variables.
	Thearch.Defframe(ptxt)
	if Debug['f'] != 0 {
		frame(0)
	}

	// Remove leftover instrumentation from the instruction stream.
	removevardef(ptxt)

	f.Config.HTML.Close()
}

// movZero generates a register indirect move with a 0 immediate and keeps track of bytes left and next offset
func movZero(as obj.As, width int64, nbytes int64, offset int64, regnum int16) (nleft int64, noff int64) {
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

type FloatingEQNEJump struct {
	Jump  obj.As
	Index int
}

func oneFPJump(b *ssa.Block, jumps *FloatingEQNEJump, likely ssa.BranchPrediction, branches []Branch) []Branch {
	p := Prog(jumps.Jump)
	p.To.Type = obj.TYPE_BRANCH
	to := jumps.Index
	branches = append(branches, Branch{p, b.Succs[to].Block()})
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

func SSAGenFPJump(s *SSAGenState, b, next *ssa.Block, jumps *[2][2]FloatingEQNEJump) {
	likely := b.Likely
	switch next {
	case b.Succs[0].Block():
		s.Branches = oneFPJump(b, &jumps[0][0], likely, s.Branches)
		s.Branches = oneFPJump(b, &jumps[0][1], likely, s.Branches)
	case b.Succs[1].Block():
		s.Branches = oneFPJump(b, &jumps[1][0], likely, s.Branches)
		s.Branches = oneFPJump(b, &jumps[1][1], likely, s.Branches)
	default:
		s.Branches = oneFPJump(b, &jumps[1][0], likely, s.Branches)
		s.Branches = oneFPJump(b, &jumps[1][1], likely, s.Branches)
		q := Prog(obj.AJMP)
		q.To.Type = obj.TYPE_BRANCH
		s.Branches = append(s.Branches, Branch{q, b.Succs[1].Block()})
	}
}

// AddAux adds the offset in the aux fields (AuxInt and Aux) of v to a.
func AddAux(a *obj.Addr, v *ssa.Value) {
	AddAux2(a, v, v.AuxInt)
}
func AddAux2(a *obj.Addr, v *ssa.Value, offset int64) {
	if a.Type != obj.TYPE_MEM {
		v.Fatalf("bad AddAux addr %v", a)
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
		switch s := sym.Sym.(type) {
		case *Sym:
			a.Sym = Linksym(s)
		case *obj.LSym:
			a.Sym = s
		default:
			v.Fatalf("ExternSymbol.Sym is %T", s)
		}
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
		// TODO: truncate 64-bit indexes on 32-bit pointer archs. We'd need to test
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

// SSARegNum returns the register (in cmd/internal/obj numbering) to
// which v has been allocated. Panics if v is not assigned to a
// register.
// TODO: Make this panic again once it stops happening routinely.
func SSARegNum(v *ssa.Value) int16 {
	reg := v.Block.Func.RegAlloc[v.ID]
	if reg == nil {
		v.Unimplementedf("nil regnum for value: %s\n%s\n", v.LongString(), v.Block.Func)
		return 0
	}
	return Thearch.SSARegToReg[reg.(*ssa.Register).Num]
}

// AutoVar returns a *Node and int64 representing the auto variable and offset within it
// where v should be spilled.
func AutoVar(v *ssa.Value) (*Node, int64) {
	loc := v.Block.Func.RegAlloc[v.ID].(ssa.LocalSlot)
	if v.Type.Size() > loc.Type.Size() {
		v.Fatalf("spill/restore type %s doesn't fit in slot type %s", v.Type, loc.Type)
	}
	return loc.N.(*Node), loc.Off
}

// fieldIdx finds the index of the field referred to by the ODOT node n.
func fieldIdx(n *Node) int {
	t := n.Left.Type
	f := n.Sym
	if !t.IsStruct() {
		panic("ODOT's LHS is not a struct")
	}

	var i int
	for _, t1 := range t.Fields().Slice() {
		if t1.Sym != f {
			i++
			continue
		}
		if t1.Offset != n.Xoffset {
			panic("field offset doesn't match")
		}
		return i
	}
	panic(fmt.Sprintf("can't find field in expr %s\n", n))

	// TODO: keep the result of this function somewhere in the ODOT Node
	// so we don't have to recompute it each time we need it.
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

func (e *ssaExport) SplitString(name ssa.LocalSlot) (ssa.LocalSlot, ssa.LocalSlot) {
	n := name.N.(*Node)
	ptrType := Ptrto(Types[TUINT8])
	lenType := Types[TINT]
	if n.Class == PAUTO && !n.Addrtaken {
		// Split this string up into two separate variables.
		p := e.namedAuto(n.Sym.Name+".ptr", ptrType)
		l := e.namedAuto(n.Sym.Name+".len", lenType)
		return ssa.LocalSlot{N: p, Type: ptrType, Off: 0}, ssa.LocalSlot{N: l, Type: lenType, Off: 0}
	}
	// Return the two parts of the larger variable.
	return ssa.LocalSlot{N: n, Type: ptrType, Off: name.Off}, ssa.LocalSlot{N: n, Type: lenType, Off: name.Off + int64(Widthptr)}
}

func (e *ssaExport) SplitInterface(name ssa.LocalSlot) (ssa.LocalSlot, ssa.LocalSlot) {
	n := name.N.(*Node)
	t := Ptrto(Types[TUINT8])
	if n.Class == PAUTO && !n.Addrtaken {
		// Split this interface up into two separate variables.
		f := ".itab"
		if n.Type.IsEmptyInterface() {
			f = ".type"
		}
		c := e.namedAuto(n.Sym.Name+f, t)
		d := e.namedAuto(n.Sym.Name+".data", t)
		return ssa.LocalSlot{N: c, Type: t, Off: 0}, ssa.LocalSlot{N: d, Type: t, Off: 0}
	}
	// Return the two parts of the larger variable.
	return ssa.LocalSlot{N: n, Type: t, Off: name.Off}, ssa.LocalSlot{N: n, Type: t, Off: name.Off + int64(Widthptr)}
}

func (e *ssaExport) SplitSlice(name ssa.LocalSlot) (ssa.LocalSlot, ssa.LocalSlot, ssa.LocalSlot) {
	n := name.N.(*Node)
	ptrType := Ptrto(name.Type.ElemType().(*Type))
	lenType := Types[TINT]
	if n.Class == PAUTO && !n.Addrtaken {
		// Split this slice up into three separate variables.
		p := e.namedAuto(n.Sym.Name+".ptr", ptrType)
		l := e.namedAuto(n.Sym.Name+".len", lenType)
		c := e.namedAuto(n.Sym.Name+".cap", lenType)
		return ssa.LocalSlot{N: p, Type: ptrType, Off: 0}, ssa.LocalSlot{N: l, Type: lenType, Off: 0}, ssa.LocalSlot{N: c, Type: lenType, Off: 0}
	}
	// Return the three parts of the larger variable.
	return ssa.LocalSlot{N: n, Type: ptrType, Off: name.Off},
		ssa.LocalSlot{N: n, Type: lenType, Off: name.Off + int64(Widthptr)},
		ssa.LocalSlot{N: n, Type: lenType, Off: name.Off + int64(2*Widthptr)}
}

func (e *ssaExport) SplitComplex(name ssa.LocalSlot) (ssa.LocalSlot, ssa.LocalSlot) {
	n := name.N.(*Node)
	s := name.Type.Size() / 2
	var t *Type
	if s == 8 {
		t = Types[TFLOAT64]
	} else {
		t = Types[TFLOAT32]
	}
	if n.Class == PAUTO && !n.Addrtaken {
		// Split this complex up into two separate variables.
		c := e.namedAuto(n.Sym.Name+".real", t)
		d := e.namedAuto(n.Sym.Name+".imag", t)
		return ssa.LocalSlot{N: c, Type: t, Off: 0}, ssa.LocalSlot{N: d, Type: t, Off: 0}
	}
	// Return the two parts of the larger variable.
	return ssa.LocalSlot{N: n, Type: t, Off: name.Off}, ssa.LocalSlot{N: n, Type: t, Off: name.Off + s}
}

func (e *ssaExport) SplitStruct(name ssa.LocalSlot, i int) ssa.LocalSlot {
	n := name.N.(*Node)
	st := name.Type
	ft := st.FieldType(i)
	if n.Class == PAUTO && !n.Addrtaken {
		// Note: the _ field may appear several times.  But
		// have no fear, identically-named but distinct Autos are
		// ok, albeit maybe confusing for a debugger.
		x := e.namedAuto(n.Sym.Name+"."+st.FieldName(i), ft)
		return ssa.LocalSlot{N: x, Type: ft, Off: 0}
	}
	return ssa.LocalSlot{N: n, Type: ft, Off: name.Off + st.FieldOff(i)}
}

// namedAuto returns a new AUTO variable with the given name and type.
func (e *ssaExport) namedAuto(name string, typ ssa.Type) ssa.GCNode {
	t := typ.(*Type)
	s := &Sym{Name: name, Pkg: autopkg}
	n := Nod(ONAME, nil, nil)
	s.Def = n
	s.Def.Used = true
	n.Sym = s
	n.Type = t
	n.Class = PAUTO
	n.Addable = true
	n.Ullman = 1
	n.Esc = EscNever
	n.Xoffset = 0
	n.Name.Curfn = Curfn
	Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)

	dowidth(t)
	e.mustImplement = true

	return n
}

func (e *ssaExport) CanSSA(t ssa.Type) bool {
	return canSSAType(t.(*Type))
}

func (e *ssaExport) Line(line int32) string {
	return linestr(line)
}

// Log logs a message from the compiler.
func (e *ssaExport) Logf(msg string, args ...interface{}) {
	// If e was marked as unimplemented, anything could happen. Ignore.
	if e.log && !e.unimplemented {
		fmt.Printf(msg, args...)
	}
}

func (e *ssaExport) Log() bool {
	return e.log
}

// Fatal reports a compiler error and exits.
func (e *ssaExport) Fatalf(line int32, msg string, args ...interface{}) {
	// If e was marked as unimplemented, anything could happen. Ignore.
	if !e.unimplemented {
		lineno = line
		Fatalf(msg, args...)
	}
}

// Unimplemented reports that the function cannot be compiled.
// It will be removed once SSA work is complete.
func (e *ssaExport) Unimplementedf(line int32, msg string, args ...interface{}) {
	if e.mustImplement {
		lineno = line
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
func (e *ssaExport) Warnl(line int32, fmt_ string, args ...interface{}) {
	Warnl(line, fmt_, args...)
}

func (e *ssaExport) Debug_checknil() bool {
	return Debug_checknil != 0
}

func (n *Node) Typ() ssa.Type {
	return n.Type
}
