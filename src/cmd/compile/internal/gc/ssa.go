// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"html"
	"os"
	"sort"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
)

var ssaConfig *ssa.Config
var ssaCaches []ssa.Cache

var ssaDump string     // early copy of $GOSSAFUNC; the func name to dump output for
var ssaDumpStdout bool // whether to dump to stdout
const ssaDumpFile = "ssa.html"

// ssaDumpInlined holds all inlined functions when ssaDump contains a function name.
var ssaDumpInlined []*Node

func initssaconfig() {
	types_ := ssa.NewTypes()

	if thearch.SoftFloat {
		softfloatInit()
	}

	// Generate a few pointer types that are uncommon in the frontend but common in the backend.
	// Caching is disabled in the backend, so generating these here avoids allocations.
	_ = types.NewPtr(types.Types[TINTER])                             // *interface{}
	_ = types.NewPtr(types.NewPtr(types.Types[TSTRING]))              // **string
	_ = types.NewPtr(types.NewPtr(types.Idealstring))                 // **string
	_ = types.NewPtr(types.NewSlice(types.Types[TINTER]))             // *[]interface{}
	_ = types.NewPtr(types.NewPtr(types.Bytetype))                    // **byte
	_ = types.NewPtr(types.NewSlice(types.Bytetype))                  // *[]byte
	_ = types.NewPtr(types.NewSlice(types.Types[TSTRING]))            // *[]string
	_ = types.NewPtr(types.NewSlice(types.Idealstring))               // *[]string
	_ = types.NewPtr(types.NewPtr(types.NewPtr(types.Types[TUINT8]))) // ***uint8
	_ = types.NewPtr(types.Types[TINT16])                             // *int16
	_ = types.NewPtr(types.Types[TINT64])                             // *int64
	_ = types.NewPtr(types.Errortype)                                 // *error
	types.NewPtrCacheEnabled = false
	ssaConfig = ssa.NewConfig(thearch.LinkArch.Name, *types_, Ctxt, Debug['N'] == 0)
	if thearch.LinkArch.Name == "386" {
		ssaConfig.Set387(thearch.Use387)
	}
	ssaConfig.SoftFloat = thearch.SoftFloat
	ssaConfig.Race = flag_race
	ssaCaches = make([]ssa.Cache, nBackendWorkers)

	// Set up some runtime functions we'll need to call.
	assertE2I = sysfunc("assertE2I")
	assertE2I2 = sysfunc("assertE2I2")
	assertI2I = sysfunc("assertI2I")
	assertI2I2 = sysfunc("assertI2I2")
	deferproc = sysfunc("deferproc")
	Deferreturn = sysfunc("deferreturn")
	Duffcopy = sysvar("duffcopy")             // asm func with special ABI
	Duffzero = sysvar("duffzero")             // asm func with special ABI
	gcWriteBarrier = sysvar("gcWriteBarrier") // asm func with special ABI
	goschedguarded = sysfunc("goschedguarded")
	growslice = sysfunc("growslice")
	msanread = sysfunc("msanread")
	msanwrite = sysfunc("msanwrite")
	newproc = sysfunc("newproc")
	panicdivide = sysfunc("panicdivide")
	panicdottypeE = sysfunc("panicdottypeE")
	panicdottypeI = sysfunc("panicdottypeI")
	panicindex = sysfunc("panicindex")
	panicnildottype = sysfunc("panicnildottype")
	panicslice = sysfunc("panicslice")
	raceread = sysfunc("raceread")
	racereadrange = sysfunc("racereadrange")
	racewrite = sysfunc("racewrite")
	racewriterange = sysfunc("racewriterange")
	x86HasPOPCNT = sysvar("x86HasPOPCNT")       // bool
	x86HasSSE41 = sysvar("x86HasSSE41")         // bool
	arm64HasATOMICS = sysvar("arm64HasATOMICS") // bool
	typedmemclr = sysfunc("typedmemclr")
	typedmemmove = sysfunc("typedmemmove")
	Udiv = sysvar("udiv")                 // asm func with special ABI
	writeBarrier = sysvar("writeBarrier") // struct { bool; ... }

	// GO386=387 runtime definitions
	ControlWord64trunc = sysvar("controlWord64trunc") // uint16
	ControlWord32 = sysvar("controlWord32")           // uint16

	// Wasm (all asm funcs with special ABIs)
	WasmMove = sysvar("wasmMove")
	WasmZero = sysvar("wasmZero")
	WasmDiv = sysvar("wasmDiv")
	WasmTruncS = sysvar("wasmTruncS")
	WasmTruncU = sysvar("wasmTruncU")
	SigPanic = sysvar("sigpanic")
}

// buildssa builds an SSA function for fn.
// worker indicates which of the backend workers is doing the processing.
func buildssa(fn *Node, worker int) *ssa.Func {
	name := fn.funcname()
	printssa := name == ssaDump
	var astBuf *bytes.Buffer
	if printssa {
		astBuf = &bytes.Buffer{}
		fdumplist(astBuf, "buildssa-enter", fn.Func.Enter)
		fdumplist(astBuf, "buildssa-body", fn.Nbody)
		fdumplist(astBuf, "buildssa-exit", fn.Func.Exit)
		if ssaDumpStdout {
			fmt.Println("generating SSA for", name)
			fmt.Print(astBuf.String())
		}
	}

	var s state
	s.pushLine(fn.Pos)
	defer s.popLine()

	s.hasdefer = fn.Func.HasDefer()
	if fn.Func.Pragma&CgoUnsafeArgs != 0 {
		s.cgoUnsafeArgs = true
	}

	fe := ssafn{
		curfn: fn,
		log:   printssa && ssaDumpStdout,
	}
	s.curfn = fn

	s.f = ssa.NewFunc(&fe)
	s.config = ssaConfig
	s.f.Type = fn.Type
	s.f.Config = ssaConfig
	s.f.Cache = &ssaCaches[worker]
	s.f.Cache.Reset()
	s.f.DebugTest = s.f.DebugHashMatch("GOSSAHASH", name)
	s.f.Name = name
	s.f.PrintOrHtmlSSA = printssa
	if fn.Func.Pragma&Nosplit != 0 {
		s.f.NoSplit = true
	}
	s.panics = map[funcLine]*ssa.Block{}
	s.softFloat = s.config.SoftFloat

	if printssa {
		s.f.HTMLWriter = ssa.NewHTMLWriter(ssaDumpFile, s.f.Frontend(), name)
		// TODO: generate and print a mapping from nodes to values and blocks
		dumpSourcesColumn(s.f.HTMLWriter, fn)
		s.f.HTMLWriter.WriteAST("AST", astBuf)
	}

	// Allocate starting block
	s.f.Entry = s.f.NewBlock(ssa.BlockPlain)

	// Allocate starting values
	s.labels = map[string]*ssaLabel{}
	s.labeledNodes = map[*Node]*ssaLabel{}
	s.fwdVars = map[*Node]*ssa.Value{}
	s.startmem = s.entryNewValue0(ssa.OpInitMem, types.TypeMem)
	s.sp = s.entryNewValue0(ssa.OpSP, types.Types[TUINTPTR]) // TODO: use generic pointer type (unsafe.Pointer?) instead
	s.sb = s.entryNewValue0(ssa.OpSB, types.Types[TUINTPTR])

	s.startBlock(s.f.Entry)
	s.vars[&memVar] = s.startmem

	// Generate addresses of local declarations
	s.decladdrs = map[*Node]*ssa.Value{}
	for _, n := range fn.Func.Dcl {
		switch n.Class() {
		case PPARAM, PPARAMOUT:
			s.decladdrs[n] = s.entryNewValue2A(ssa.OpLocalAddr, types.NewPtr(n.Type), n, s.sp, s.startmem)
			if n.Class() == PPARAMOUT && s.canSSA(n) {
				// Save ssa-able PPARAMOUT variables so we can
				// store them back to the stack at the end of
				// the function.
				s.returns = append(s.returns, n)
			}
		case PAUTO:
			// processed at each use, to prevent Addr coming
			// before the decl.
		case PAUTOHEAP:
			// moved to heap - already handled by frontend
		case PFUNC:
			// local function - already handled by frontend
		default:
			s.Fatalf("local variable with class %v unimplemented", n.Class())
		}
	}

	// Populate SSAable arguments.
	for _, n := range fn.Func.Dcl {
		if n.Class() == PPARAM && s.canSSA(n) {
			v := s.newValue0A(ssa.OpArg, n.Type, n)
			s.vars[n] = v
			s.addNamedValue(n, v) // This helps with debugging information, not needed for compilation itself.
		}
	}

	// Convert the AST-based IR to the SSA-based IR
	s.stmtList(fn.Func.Enter)
	s.stmtList(fn.Nbody)

	// fallthrough to exit
	if s.curBlock != nil {
		s.pushLine(fn.Func.Endlineno)
		s.exit()
		s.popLine()
	}

	for _, b := range s.f.Blocks {
		if b.Pos != src.NoXPos {
			s.updateUnsetPredPos(b)
		}
	}

	s.insertPhis()

	// Main call to ssa package to compile function
	ssa.Compile(s.f)
	return s.f
}

func dumpSourcesColumn(writer *ssa.HTMLWriter, fn *Node) {
	// Read sources of target function fn.
	fname := Ctxt.PosTable.Pos(fn.Pos).Filename()
	targetFn, err := readFuncLines(fname, fn.Pos.Line(), fn.Func.Endlineno.Line())
	if err != nil {
		writer.Logger.Logf("cannot read sources for function %v: %v", fn, err)
	}

	// Read sources of inlined functions.
	var inlFns []*ssa.FuncLines
	for _, fi := range ssaDumpInlined {
		var elno src.XPos
		if fi.Name.Defn == nil {
			// Endlineno is filled from exported data.
			elno = fi.Func.Endlineno
		} else {
			elno = fi.Name.Defn.Func.Endlineno
		}
		fname := Ctxt.PosTable.Pos(fi.Pos).Filename()
		fnLines, err := readFuncLines(fname, fi.Pos.Line(), elno.Line())
		if err != nil {
			writer.Logger.Logf("cannot read sources for function %v: %v", fi, err)
			continue
		}
		inlFns = append(inlFns, fnLines)
	}

	sort.Sort(ssa.ByTopo(inlFns))
	if targetFn != nil {
		inlFns = append([]*ssa.FuncLines{targetFn}, inlFns...)
	}

	writer.WriteSources("sources", inlFns)
}

func readFuncLines(file string, start, end uint) (*ssa.FuncLines, error) {
	f, err := os.Open(os.ExpandEnv(file))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lines []string
	ln := uint(1)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() && ln <= end {
		if ln >= start {
			lines = append(lines, scanner.Text())
		}
		ln++
	}
	return &ssa.FuncLines{Filename: file, StartLineno: start, Lines: lines}, nil
}

// updateUnsetPredPos propagates the earliest-value position information for b
// towards all of b's predecessors that need a position, and recurs on that
// predecessor if its position is updated. B should have a non-empty position.
func (s *state) updateUnsetPredPos(b *ssa.Block) {
	if b.Pos == src.NoXPos {
		s.Fatalf("Block %s should have a position", b)
	}
	bestPos := src.NoXPos
	for _, e := range b.Preds {
		p := e.Block()
		if !p.LackingPos() {
			continue
		}
		if bestPos == src.NoXPos {
			bestPos = b.Pos
			for _, v := range b.Values {
				if v.LackingPos() {
					continue
				}
				if v.Pos != src.NoXPos {
					// Assume values are still in roughly textual order;
					// TODO: could also seek minimum position?
					bestPos = v.Pos
					break
				}
			}
		}
		p.Pos = bestPos
		s.updateUnsetPredPos(p) // We do not expect long chains of these, thus recursion is okay.
	}
}

type state struct {
	// configuration (arch) information
	config *ssa.Config

	// function we're building
	f *ssa.Func

	// Node for function
	curfn *Node

	// labels and labeled control flow nodes (OFOR, OFORUNTIL, OSWITCH, OSELECT) in f
	labels       map[string]*ssaLabel
	labeledNodes map[*Node]*ssaLabel

	// unlabeled break and continue statement tracking
	breakTo    *ssa.Block // current target for plain break statement
	continueTo *ssa.Block // current target for plain continue statement

	// current location where we're interpreting the AST
	curBlock *ssa.Block

	// variable assignments in the current block (map from variable symbol to ssa value)
	// *Node is the unique identifier (an ONAME Node) for the variable.
	// TODO: keep a single varnum map, then make all of these maps slices instead?
	vars map[*Node]*ssa.Value

	// fwdVars are variables that are used before they are defined in the current block.
	// This map exists just to coalesce multiple references into a single FwdRef op.
	// *Node is the unique identifier (an ONAME Node) for the variable.
	fwdVars map[*Node]*ssa.Value

	// all defined variables at the end of each block. Indexed by block ID.
	defvars []map[*Node]*ssa.Value

	// addresses of PPARAM and PPARAMOUT variables.
	decladdrs map[*Node]*ssa.Value

	// starting values. Memory, stack pointer, and globals pointer
	startmem *ssa.Value
	sp       *ssa.Value
	sb       *ssa.Value

	// line number stack. The current line number is top of stack
	line []src.XPos
	// the last line number processed; it may have been popped
	lastPos src.XPos

	// list of panic calls by function name and line number.
	// Used to deduplicate panic calls.
	panics map[funcLine]*ssa.Block

	// list of PPARAMOUT (return) variables.
	returns []*Node

	cgoUnsafeArgs bool
	hasdefer      bool // whether the function contains a defer statement
	softFloat     bool
}

type funcLine struct {
	f    *obj.LSym
	base *src.PosBase
	line uint
}

type ssaLabel struct {
	target         *ssa.Block // block identified by this label
	breakTarget    *ssa.Block // block to break to in control flow node identified by this label
	continueTarget *ssa.Block // block to continue to in control flow node identified by this label
}

// label returns the label associated with sym, creating it if necessary.
func (s *state) label(sym *types.Sym) *ssaLabel {
	lab := s.labels[sym.Name]
	if lab == nil {
		lab = new(ssaLabel)
		s.labels[sym.Name] = lab
	}
	return lab
}

func (s *state) Logf(msg string, args ...interface{}) { s.f.Logf(msg, args...) }
func (s *state) Log() bool                            { return s.f.Log() }
func (s *state) Fatalf(msg string, args ...interface{}) {
	s.f.Frontend().Fatalf(s.peekPos(), msg, args...)
}
func (s *state) Warnl(pos src.XPos, msg string, args ...interface{}) { s.f.Warnl(pos, msg, args...) }
func (s *state) Debug_checknil() bool                                { return s.f.Frontend().Debug_checknil() }

var (
	// dummy node for the memory variable
	memVar = Node{Op: ONAME, Sym: &types.Sym{Name: "mem"}}

	// dummy nodes for temporary variables
	ptrVar    = Node{Op: ONAME, Sym: &types.Sym{Name: "ptr"}}
	lenVar    = Node{Op: ONAME, Sym: &types.Sym{Name: "len"}}
	newlenVar = Node{Op: ONAME, Sym: &types.Sym{Name: "newlen"}}
	capVar    = Node{Op: ONAME, Sym: &types.Sym{Name: "cap"}}
	typVar    = Node{Op: ONAME, Sym: &types.Sym{Name: "typ"}}
	okVar     = Node{Op: ONAME, Sym: &types.Sym{Name: "ok"}}
)

// startBlock sets the current block we're generating code in to b.
func (s *state) startBlock(b *ssa.Block) {
	if s.curBlock != nil {
		s.Fatalf("starting block %v when block %v has not ended", b, s.curBlock)
	}
	s.curBlock = b
	s.vars = map[*Node]*ssa.Value{}
	for n := range s.fwdVars {
		delete(s.fwdVars, n)
	}
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
	if b.LackingPos() {
		// Empty plain blocks get the line of their successor (handled after all blocks created),
		// except for increment blocks in For statements (handled in ssa conversion of OFOR),
		// and for blocks ending in GOTO/BREAK/CONTINUE.
		b.Pos = src.NoXPos
	} else {
		b.Pos = s.lastPos
	}
	return b
}

// pushLine pushes a line number on the line number stack.
func (s *state) pushLine(line src.XPos) {
	if !line.IsKnown() {
		// the frontend may emit node with line number missing,
		// use the parent line number in this case.
		line = s.peekPos()
		if Debug['K'] != 0 {
			Warn("buildssa: unknown position (line 0)")
		}
	} else {
		s.lastPos = line
	}

	s.line = append(s.line, line)
}

// popLine pops the top of the line number stack.
func (s *state) popLine() {
	s.line = s.line[:len(s.line)-1]
}

// peekPos peeks the top of the line number stack.
func (s *state) peekPos() src.XPos {
	return s.line[len(s.line)-1]
}

// newValue0 adds a new value with no arguments to the current block.
func (s *state) newValue0(op ssa.Op, t *types.Type) *ssa.Value {
	return s.curBlock.NewValue0(s.peekPos(), op, t)
}

// newValue0A adds a new value with no arguments and an aux value to the current block.
func (s *state) newValue0A(op ssa.Op, t *types.Type, aux interface{}) *ssa.Value {
	return s.curBlock.NewValue0A(s.peekPos(), op, t, aux)
}

// newValue0I adds a new value with no arguments and an auxint value to the current block.
func (s *state) newValue0I(op ssa.Op, t *types.Type, auxint int64) *ssa.Value {
	return s.curBlock.NewValue0I(s.peekPos(), op, t, auxint)
}

// newValue1 adds a new value with one argument to the current block.
func (s *state) newValue1(op ssa.Op, t *types.Type, arg *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue1(s.peekPos(), op, t, arg)
}

// newValue1A adds a new value with one argument and an aux value to the current block.
func (s *state) newValue1A(op ssa.Op, t *types.Type, aux interface{}, arg *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue1A(s.peekPos(), op, t, aux, arg)
}

// newValue1Apos adds a new value with one argument and an aux value to the current block.
// isStmt determines whether the created values may be a statement or not
// (i.e., false means never, yes means maybe).
func (s *state) newValue1Apos(op ssa.Op, t *types.Type, aux interface{}, arg *ssa.Value, isStmt bool) *ssa.Value {
	if isStmt {
		return s.curBlock.NewValue1A(s.peekPos(), op, t, aux, arg)
	}
	return s.curBlock.NewValue1A(s.peekPos().WithNotStmt(), op, t, aux, arg)
}

// newValue1I adds a new value with one argument and an auxint value to the current block.
func (s *state) newValue1I(op ssa.Op, t *types.Type, aux int64, arg *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue1I(s.peekPos(), op, t, aux, arg)
}

// newValue2 adds a new value with two arguments to the current block.
func (s *state) newValue2(op ssa.Op, t *types.Type, arg0, arg1 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue2(s.peekPos(), op, t, arg0, arg1)
}

// newValue2Apos adds a new value with two arguments and an aux value to the current block.
// isStmt determines whether the created values may be a statement or not
// (i.e., false means never, yes means maybe).
func (s *state) newValue2Apos(op ssa.Op, t *types.Type, aux interface{}, arg0, arg1 *ssa.Value, isStmt bool) *ssa.Value {
	if isStmt {
		return s.curBlock.NewValue2A(s.peekPos(), op, t, aux, arg0, arg1)
	}
	return s.curBlock.NewValue2A(s.peekPos().WithNotStmt(), op, t, aux, arg0, arg1)
}

// newValue2I adds a new value with two arguments and an auxint value to the current block.
func (s *state) newValue2I(op ssa.Op, t *types.Type, aux int64, arg0, arg1 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue2I(s.peekPos(), op, t, aux, arg0, arg1)
}

// newValue3 adds a new value with three arguments to the current block.
func (s *state) newValue3(op ssa.Op, t *types.Type, arg0, arg1, arg2 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue3(s.peekPos(), op, t, arg0, arg1, arg2)
}

// newValue3I adds a new value with three arguments and an auxint value to the current block.
func (s *state) newValue3I(op ssa.Op, t *types.Type, aux int64, arg0, arg1, arg2 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue3I(s.peekPos(), op, t, aux, arg0, arg1, arg2)
}

// newValue3A adds a new value with three arguments and an aux value to the current block.
func (s *state) newValue3A(op ssa.Op, t *types.Type, aux interface{}, arg0, arg1, arg2 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue3A(s.peekPos(), op, t, aux, arg0, arg1, arg2)
}

// newValue3Apos adds a new value with three arguments and an aux value to the current block.
// isStmt determines whether the created values may be a statement or not
// (i.e., false means never, yes means maybe).
func (s *state) newValue3Apos(op ssa.Op, t *types.Type, aux interface{}, arg0, arg1, arg2 *ssa.Value, isStmt bool) *ssa.Value {
	if isStmt {
		return s.curBlock.NewValue3A(s.peekPos(), op, t, aux, arg0, arg1, arg2)
	}
	return s.curBlock.NewValue3A(s.peekPos().WithNotStmt(), op, t, aux, arg0, arg1, arg2)
}

// newValue4 adds a new value with four arguments to the current block.
func (s *state) newValue4(op ssa.Op, t *types.Type, arg0, arg1, arg2, arg3 *ssa.Value) *ssa.Value {
	return s.curBlock.NewValue4(s.peekPos(), op, t, arg0, arg1, arg2, arg3)
}

// entryNewValue0 adds a new value with no arguments to the entry block.
func (s *state) entryNewValue0(op ssa.Op, t *types.Type) *ssa.Value {
	return s.f.Entry.NewValue0(src.NoXPos, op, t)
}

// entryNewValue0A adds a new value with no arguments and an aux value to the entry block.
func (s *state) entryNewValue0A(op ssa.Op, t *types.Type, aux interface{}) *ssa.Value {
	return s.f.Entry.NewValue0A(src.NoXPos, op, t, aux)
}

// entryNewValue1 adds a new value with one argument to the entry block.
func (s *state) entryNewValue1(op ssa.Op, t *types.Type, arg *ssa.Value) *ssa.Value {
	return s.f.Entry.NewValue1(src.NoXPos, op, t, arg)
}

// entryNewValue1 adds a new value with one argument and an auxint value to the entry block.
func (s *state) entryNewValue1I(op ssa.Op, t *types.Type, auxint int64, arg *ssa.Value) *ssa.Value {
	return s.f.Entry.NewValue1I(src.NoXPos, op, t, auxint, arg)
}

// entryNewValue1A adds a new value with one argument and an aux value to the entry block.
func (s *state) entryNewValue1A(op ssa.Op, t *types.Type, aux interface{}, arg *ssa.Value) *ssa.Value {
	return s.f.Entry.NewValue1A(src.NoXPos, op, t, aux, arg)
}

// entryNewValue2 adds a new value with two arguments to the entry block.
func (s *state) entryNewValue2(op ssa.Op, t *types.Type, arg0, arg1 *ssa.Value) *ssa.Value {
	return s.f.Entry.NewValue2(src.NoXPos, op, t, arg0, arg1)
}

// entryNewValue2A adds a new value with two arguments and an aux value to the entry block.
func (s *state) entryNewValue2A(op ssa.Op, t *types.Type, aux interface{}, arg0, arg1 *ssa.Value) *ssa.Value {
	return s.f.Entry.NewValue2A(src.NoXPos, op, t, aux, arg0, arg1)
}

// const* routines add a new const value to the entry block.
func (s *state) constSlice(t *types.Type) *ssa.Value {
	return s.f.ConstSlice(t)
}
func (s *state) constInterface(t *types.Type) *ssa.Value {
	return s.f.ConstInterface(t)
}
func (s *state) constNil(t *types.Type) *ssa.Value { return s.f.ConstNil(t) }
func (s *state) constEmptyString(t *types.Type) *ssa.Value {
	return s.f.ConstEmptyString(t)
}
func (s *state) constBool(c bool) *ssa.Value {
	return s.f.ConstBool(types.Types[TBOOL], c)
}
func (s *state) constInt8(t *types.Type, c int8) *ssa.Value {
	return s.f.ConstInt8(t, c)
}
func (s *state) constInt16(t *types.Type, c int16) *ssa.Value {
	return s.f.ConstInt16(t, c)
}
func (s *state) constInt32(t *types.Type, c int32) *ssa.Value {
	return s.f.ConstInt32(t, c)
}
func (s *state) constInt64(t *types.Type, c int64) *ssa.Value {
	return s.f.ConstInt64(t, c)
}
func (s *state) constFloat32(t *types.Type, c float64) *ssa.Value {
	return s.f.ConstFloat32(t, c)
}
func (s *state) constFloat64(t *types.Type, c float64) *ssa.Value {
	return s.f.ConstFloat64(t, c)
}
func (s *state) constInt(t *types.Type, c int64) *ssa.Value {
	if s.config.PtrSize == 8 {
		return s.constInt64(t, c)
	}
	if int64(int32(c)) != c {
		s.Fatalf("integer constant too big %d", c)
	}
	return s.constInt32(t, int32(c))
}
func (s *state) constOffPtrSP(t *types.Type, c int64) *ssa.Value {
	return s.f.ConstOffPtrSP(t, c, s.sp)
}

// newValueOrSfCall* are wrappers around newValue*, which may create a call to a
// soft-float runtime function instead (when emitting soft-float code).
func (s *state) newValueOrSfCall1(op ssa.Op, t *types.Type, arg *ssa.Value) *ssa.Value {
	if s.softFloat {
		if c, ok := s.sfcall(op, arg); ok {
			return c
		}
	}
	return s.newValue1(op, t, arg)
}
func (s *state) newValueOrSfCall2(op ssa.Op, t *types.Type, arg0, arg1 *ssa.Value) *ssa.Value {
	if s.softFloat {
		if c, ok := s.sfcall(op, arg0, arg1); ok {
			return c
		}
	}
	return s.newValue2(op, t, arg0, arg1)
}

func (s *state) instrument(t *types.Type, addr *ssa.Value, wr bool) {
	if !s.curfn.Func.InstrumentBody() {
		return
	}

	w := t.Size()
	if w == 0 {
		return // can't race on zero-sized things
	}

	if ssa.IsSanitizerSafeAddr(addr) {
		return
	}

	var fn *obj.LSym
	needWidth := false

	if flag_msan {
		fn = msanread
		if wr {
			fn = msanwrite
		}
		needWidth = true
	} else if flag_race && t.NumComponents(types.CountBlankFields) > 1 {
		// for composite objects we have to write every address
		// because a write might happen to any subobject.
		// composites with only one element don't have subobjects, though.
		fn = racereadrange
		if wr {
			fn = racewriterange
		}
		needWidth = true
	} else if flag_race {
		// for non-composite objects we can write just the start
		// address, as any write must write the first byte.
		fn = raceread
		if wr {
			fn = racewrite
		}
	} else {
		panic("unreachable")
	}

	args := []*ssa.Value{addr}
	if needWidth {
		args = append(args, s.constInt(types.Types[TUINTPTR], w))
	}
	s.rtcall(fn, true, nil, args...)
}

func (s *state) load(t *types.Type, src *ssa.Value) *ssa.Value {
	s.instrument(t, src, false)
	return s.rawLoad(t, src)
}

func (s *state) rawLoad(t *types.Type, src *ssa.Value) *ssa.Value {
	return s.newValue2(ssa.OpLoad, t, src, s.mem())
}

func (s *state) store(t *types.Type, dst, val *ssa.Value) {
	s.vars[&memVar] = s.newValue3A(ssa.OpStore, types.TypeMem, t, dst, val, s.mem())
}

func (s *state) zero(t *types.Type, dst *ssa.Value) {
	s.instrument(t, dst, true)
	store := s.newValue2I(ssa.OpZero, types.TypeMem, t.Size(), dst, s.mem())
	store.Aux = t
	s.vars[&memVar] = store
}

func (s *state) move(t *types.Type, dst, src *ssa.Value) {
	s.instrument(t, src, false)
	s.instrument(t, dst, true)
	store := s.newValue3I(ssa.OpMove, types.TypeMem, t.Size(), dst, src, s.mem())
	store.Aux = t
	s.vars[&memVar] = store
}

// stmtList converts the statement list n to SSA and adds it to s.
func (s *state) stmtList(l Nodes) {
	for _, n := range l.Slice() {
		s.stmt(n)
	}
}

// stmt converts the statement n to SSA and adds it to s.
func (s *state) stmt(n *Node) {
	if !(n.Op == OVARKILL || n.Op == OVARLIVE || n.Op == OVARDEF) {
		// OVARKILL, OVARLIVE, and OVARDEF are invisible to the programmer, so we don't use their line numbers to avoid confusion in debugging.
		s.pushLine(n.Pos)
		defer s.popLine()
	}

	// If s.curBlock is nil, and n isn't a label (which might have an associated goto somewhere),
	// then this code is dead. Stop here.
	if s.curBlock == nil && n.Op != OLABEL {
		return
	}

	s.stmtList(n.Ninit)
	switch n.Op {

	case OBLOCK:
		s.stmtList(n.List)

	// No-ops
	case OEMPTY, ODCLCONST, ODCLTYPE, OFALL:

	// Expression statements
	case OCALLFUNC:
		if isIntrinsicCall(n) {
			s.intrinsicCall(n)
			return
		}
		fallthrough

	case OCALLMETH, OCALLINTER:
		s.call(n, callNormal)
		if n.Op == OCALLFUNC && n.Left.Op == ONAME && n.Left.Class() == PFUNC {
			if fn := n.Left.Sym.Name; compiling_runtime && fn == "throw" ||
				n.Left.Sym.Pkg == Runtimepkg && (fn == "throwinit" || fn == "gopanic" || fn == "panicwrap" || fn == "block" || fn == "panicmakeslicelen" || fn == "panicmakeslicecap") {
				m := s.mem()
				b := s.endBlock()
				b.Kind = ssa.BlockExit
				b.SetControl(m)
				// TODO: never rewrite OPANIC to OCALLFUNC in the
				// first place. Need to wait until all backends
				// go through SSA.
			}
		}
	case ODEFER:
		s.call(n.Left, callDefer)
	case OGO:
		s.call(n.Left, callGo)

	case OAS2DOTTYPE:
		res, resok := s.dottype(n.Rlist.First(), true)
		deref := false
		if !canSSAType(n.Rlist.First().Type) {
			if res.Op != ssa.OpLoad {
				s.Fatalf("dottype of non-load")
			}
			mem := s.mem()
			if mem.Op == ssa.OpVarKill {
				mem = mem.Args[0]
			}
			if res.Args[1] != mem {
				s.Fatalf("memory no longer live from 2-result dottype load")
			}
			deref = true
			res = res.Args[0]
		}
		s.assign(n.List.First(), res, deref, 0)
		s.assign(n.List.Second(), resok, false, 0)
		return

	case OAS2FUNC:
		// We come here only when it is an intrinsic call returning two values.
		if !isIntrinsicCall(n.Rlist.First()) {
			s.Fatalf("non-intrinsic AS2FUNC not expanded %v", n.Rlist.First())
		}
		v := s.intrinsicCall(n.Rlist.First())
		v1 := s.newValue1(ssa.OpSelect0, n.List.First().Type, v)
		v2 := s.newValue1(ssa.OpSelect1, n.List.Second().Type, v)
		s.assign(n.List.First(), v1, false, 0)
		s.assign(n.List.Second(), v2, false, 0)
		return

	case ODCL:
		if n.Left.Class() == PAUTOHEAP {
			Fatalf("DCL %v", n)
		}

	case OLABEL:
		sym := n.Sym
		lab := s.label(sym)

		// Associate label with its control flow node, if any
		if ctl := n.labeledControl(); ctl != nil {
			s.labeledNodes[ctl] = lab
		}

		// The label might already have a target block via a goto.
		if lab.target == nil {
			lab.target = s.f.NewBlock(ssa.BlockPlain)
		}

		// Go to that label.
		// (We pretend "label:" is preceded by "goto label", unless the predecessor is unreachable.)
		if s.curBlock != nil {
			b := s.endBlock()
			b.AddEdgeTo(lab.target)
		}
		s.startBlock(lab.target)

	case OGOTO:
		sym := n.Sym

		lab := s.label(sym)
		if lab.target == nil {
			lab.target = s.f.NewBlock(ssa.BlockPlain)
		}

		b := s.endBlock()
		b.Pos = s.lastPos.WithIsStmt() // Do this even if b is an empty block.
		b.AddEdgeTo(lab.target)

	case OAS:
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

		// Evaluate RHS.
		rhs := n.Right
		if rhs != nil {
			switch rhs.Op {
			case OSTRUCTLIT, OARRAYLIT, OSLICELIT:
				// All literals with nonzero fields have already been
				// rewritten during walk. Any that remain are just T{}
				// or equivalents. Use the zero value.
				if !isZero(rhs) {
					Fatalf("literal with nonzero value in SSA: %v", rhs)
				}
				rhs = nil
			case OAPPEND:
				// Check whether we're writing the result of an append back to the same slice.
				// If so, we handle it specially to avoid write barriers on the fast
				// (non-growth) path.
				if !samesafeexpr(n.Left, rhs.List.First()) || Debug['N'] != 0 {
					break
				}
				// If the slice can be SSA'd, it'll be on the stack,
				// so there will be no write barriers,
				// so there's no need to attempt to prevent them.
				if s.canSSA(n.Left) {
					if Debug_append > 0 { // replicating old diagnostic message
						Warnl(n.Pos, "append: len-only update (in local slice)")
					}
					break
				}
				if Debug_append > 0 {
					Warnl(n.Pos, "append: len-only update")
				}
				s.append(rhs, true)
				return
			}
		}

		if n.Left.isBlank() {
			// _ = rhs
			// Just evaluate rhs for side-effects.
			if rhs != nil {
				s.expr(rhs)
			}
			return
		}

		var t *types.Type
		if n.Right != nil {
			t = n.Right.Type
		} else {
			t = n.Left.Type
		}

		var r *ssa.Value
		deref := !canSSAType(t)
		if deref {
			if rhs == nil {
				r = nil // Signal assign to use OpZero.
			} else {
				r = s.addr(rhs, false)
			}
		} else {
			if rhs == nil {
				r = s.zeroVal(t)
			} else {
				r = s.expr(rhs)
			}
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

		s.assign(n.Left, r, deref, skip)

	case OIF:
		bThen := s.f.NewBlock(ssa.BlockPlain)
		bEnd := s.f.NewBlock(ssa.BlockPlain)
		var bElse *ssa.Block
		var likely int8
		if n.Likely() {
			likely = 1
		}
		if n.Rlist.Len() != 0 {
			bElse = s.f.NewBlock(ssa.BlockPlain)
			s.condBranch(n.Left, bThen, bElse, likely)
		} else {
			s.condBranch(n.Left, bThen, bEnd, likely)
		}

		s.startBlock(bThen)
		s.stmtList(n.Nbody)
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
		b := s.exit()
		b.Pos = s.lastPos.WithIsStmt()

	case ORETJMP:
		s.stmtList(n.List)
		b := s.exit()
		b.Kind = ssa.BlockRetJmp // override BlockRet
		b.Aux = n.Sym.Linksym()

	case OCONTINUE, OBREAK:
		var to *ssa.Block
		if n.Sym == nil {
			// plain break/continue
			switch n.Op {
			case OCONTINUE:
				to = s.continueTo
			case OBREAK:
				to = s.breakTo
			}
		} else {
			// labeled break/continue; look up the target
			sym := n.Sym
			lab := s.label(sym)
			switch n.Op {
			case OCONTINUE:
				to = lab.continueTarget
			case OBREAK:
				to = lab.breakTarget
			}
		}

		b := s.endBlock()
		b.Pos = s.lastPos.WithIsStmt() // Do this even if b is an empty block.
		b.AddEdgeTo(to)

	case OFOR, OFORUNTIL:
		// OFOR: for Ninit; Left; Right { Nbody }
		// cond (Left); body (Nbody); incr (Right)
		//
		// OFORUNTIL: for Ninit; Left; Right; List { Nbody }
		// => body: { Nbody }; incr: Right; if Left { lateincr: List; goto body }; end:
		bCond := s.f.NewBlock(ssa.BlockPlain)
		bBody := s.f.NewBlock(ssa.BlockPlain)
		bIncr := s.f.NewBlock(ssa.BlockPlain)
		bEnd := s.f.NewBlock(ssa.BlockPlain)

		// first, jump to condition test (OFOR) or body (OFORUNTIL)
		b := s.endBlock()
		if n.Op == OFOR {
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

		} else {
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

		// generate incr (and, for OFORUNTIL, condition)
		s.startBlock(bIncr)
		if n.Right != nil {
			s.stmt(n.Right)
		}
		if n.Op == OFOR {
			if b := s.endBlock(); b != nil {
				b.AddEdgeTo(bCond)
				// It can happen that bIncr ends in a block containing only VARKILL,
				// and that muddles the debugging experience.
				if n.Op != OFORUNTIL && b.Pos == src.NoXPos {
					b.Pos = bCond.Pos
				}
			}
		} else {
			// bCond is unused in OFORUNTIL, so repurpose it.
			bLateIncr := bCond
			// test condition
			s.condBranch(n.Left, bLateIncr, bEnd, 1)
			// generate late increment
			s.startBlock(bLateIncr)
			s.stmtList(n.List)
			s.endBlock().AddEdgeTo(bBody)
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

		// walk adds explicit OBREAK nodes to the end of all reachable code paths.
		// If we still have a current block here, then mark it unreachable.
		if s.curBlock != nil {
			m := s.mem()
			b := s.endBlock()
			b.Kind = ssa.BlockExit
			b.SetControl(m)
		}
		s.startBlock(bEnd)

	case OVARDEF:
		if !s.canSSA(n.Left) {
			s.vars[&memVar] = s.newValue1Apos(ssa.OpVarDef, types.TypeMem, n.Left, s.mem(), false)
		}
	case OVARKILL:
		// Insert a varkill op to record that a variable is no longer live.
		// We only care about liveness info at call sites, so putting the
		// varkill in the store chain is enough to keep it correctly ordered
		// with respect to call ops.
		if !s.canSSA(n.Left) {
			s.vars[&memVar] = s.newValue1Apos(ssa.OpVarKill, types.TypeMem, n.Left, s.mem(), false)
		}

	case OVARLIVE:
		// Insert a varlive op to record that a variable is still live.
		if !n.Left.Addrtaken() {
			s.Fatalf("VARLIVE variable %v must have Addrtaken set", n.Left)
		}
		switch n.Left.Class() {
		case PAUTO, PPARAM, PPARAMOUT:
		default:
			s.Fatalf("VARLIVE variable %v must be Auto or Arg", n.Left)
		}
		s.vars[&memVar] = s.newValue1A(ssa.OpVarLive, types.TypeMem, n.Left, s.mem())

	case OCHECKNIL:
		p := s.expr(n.Left)
		s.nilCheck(p)

	default:
		s.Fatalf("unhandled stmt %v", n.Op)
	}
}

// exit processes any code that needs to be generated just before returning.
// It returns a BlockRet block that ends the control flow. Its control value
// will be set to the final memory state.
func (s *state) exit() *ssa.Block {
	if s.hasdefer {
		s.rtcall(Deferreturn, true, nil)
	}

	// Run exit code. Typically, this code copies heap-allocated PPARAMOUT
	// variables back to the stack.
	s.stmtList(s.curfn.Func.Exit)

	// Store SSAable PPARAMOUT variables back to stack locations.
	for _, n := range s.returns {
		addr := s.decladdrs[n]
		val := s.variable(n, n.Type)
		s.vars[&memVar] = s.newValue1A(ssa.OpVarDef, types.TypeMem, n, s.mem())
		s.store(n.Type, addr, val)
		// TODO: if val is ever spilled, we'd like to use the
		// PPARAMOUT slot for spilling it. That won't happen
		// currently.
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
	etype types.EType
}

var opToSSA = map[opAndType]ssa.Op{
	opAndType{OADD, TINT8}:    ssa.OpAdd8,
	opAndType{OADD, TUINT8}:   ssa.OpAdd8,
	opAndType{OADD, TINT16}:   ssa.OpAdd16,
	opAndType{OADD, TUINT16}:  ssa.OpAdd16,
	opAndType{OADD, TINT32}:   ssa.OpAdd32,
	opAndType{OADD, TUINT32}:  ssa.OpAdd32,
	opAndType{OADD, TINT64}:   ssa.OpAdd64,
	opAndType{OADD, TUINT64}:  ssa.OpAdd64,
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

	opAndType{ONEG, TINT8}:    ssa.OpNeg8,
	opAndType{ONEG, TUINT8}:   ssa.OpNeg8,
	opAndType{ONEG, TINT16}:   ssa.OpNeg16,
	opAndType{ONEG, TUINT16}:  ssa.OpNeg16,
	opAndType{ONEG, TINT32}:   ssa.OpNeg32,
	opAndType{ONEG, TUINT32}:  ssa.OpNeg32,
	opAndType{ONEG, TINT64}:   ssa.OpNeg64,
	opAndType{ONEG, TUINT64}:  ssa.OpNeg64,
	opAndType{ONEG, TFLOAT32}: ssa.OpNeg32F,
	opAndType{ONEG, TFLOAT64}: ssa.OpNeg64F,

	opAndType{OBITNOT, TINT8}:   ssa.OpCom8,
	opAndType{OBITNOT, TUINT8}:  ssa.OpCom8,
	opAndType{OBITNOT, TINT16}:  ssa.OpCom16,
	opAndType{OBITNOT, TUINT16}: ssa.OpCom16,
	opAndType{OBITNOT, TINT32}:  ssa.OpCom32,
	opAndType{OBITNOT, TUINT32}: ssa.OpCom32,
	opAndType{OBITNOT, TINT64}:  ssa.OpCom64,
	opAndType{OBITNOT, TUINT64}: ssa.OpCom64,

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
	opAndType{OEQ, TPTR}:       ssa.OpEqPtr,
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
	opAndType{ONE, TPTR}:       ssa.OpNeqPtr,
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
}

func (s *state) concreteEtype(t *types.Type) types.EType {
	e := t.Etype
	switch e {
	default:
		return e
	case TINT:
		if s.config.PtrSize == 8 {
			return TINT64
		}
		return TINT32
	case TUINT:
		if s.config.PtrSize == 8 {
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

func (s *state) ssaOp(op Op, t *types.Type) ssa.Op {
	etype := s.concreteEtype(t)
	x, ok := opToSSA[opAndType{op, etype}]
	if !ok {
		s.Fatalf("unhandled binary op %v %s", op, etype)
	}
	return x
}

func floatForComplex(t *types.Type) *types.Type {
	if t.Size() == 8 {
		return types.Types[TFLOAT32]
	} else {
		return types.Types[TFLOAT64]
	}
}

type opAndTwoTypes struct {
	op     Op
	etype1 types.EType
	etype2 types.EType
}

type twoTypes struct {
	etype1 types.EType
	etype2 types.EType
}

type twoOpsAndType struct {
	op1              ssa.Op
	op2              ssa.Op
	intermediateType types.EType
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
	twoTypes{TFLOAT64, TFLOAT64}: twoOpsAndType{ssa.OpRound64F, ssa.OpCopy, TFLOAT64},
	twoTypes{TFLOAT32, TFLOAT32}: twoOpsAndType{ssa.OpRound32F, ssa.OpCopy, TFLOAT32},
	twoTypes{TFLOAT32, TFLOAT64}: twoOpsAndType{ssa.OpCvt32Fto64F, ssa.OpCopy, TFLOAT64},
}

// this map is used only for 32-bit arch, and only includes the difference
// on 32-bit arch, don't use int64<->float conversion for uint32
var fpConvOpToSSA32 = map[twoTypes]twoOpsAndType{
	twoTypes{TUINT32, TFLOAT32}: twoOpsAndType{ssa.OpCopy, ssa.OpCvt32Uto32F, TUINT32},
	twoTypes{TUINT32, TFLOAT64}: twoOpsAndType{ssa.OpCopy, ssa.OpCvt32Uto64F, TUINT32},
	twoTypes{TFLOAT32, TUINT32}: twoOpsAndType{ssa.OpCvt32Fto32U, ssa.OpCopy, TUINT32},
	twoTypes{TFLOAT64, TUINT32}: twoOpsAndType{ssa.OpCvt64Fto32U, ssa.OpCopy, TUINT32},
}

// uint64<->float conversions, only on machines that have intructions for that
var uint64fpConvOpToSSA = map[twoTypes]twoOpsAndType{
	twoTypes{TUINT64, TFLOAT32}: twoOpsAndType{ssa.OpCopy, ssa.OpCvt64Uto32F, TUINT64},
	twoTypes{TUINT64, TFLOAT64}: twoOpsAndType{ssa.OpCopy, ssa.OpCvt64Uto64F, TUINT64},
	twoTypes{TFLOAT32, TUINT64}: twoOpsAndType{ssa.OpCvt32Fto64U, ssa.OpCopy, TUINT64},
	twoTypes{TFLOAT64, TUINT64}: twoOpsAndType{ssa.OpCvt64Fto64U, ssa.OpCopy, TUINT64},
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

func (s *state) ssaShiftOp(op Op, t *types.Type, u *types.Type) ssa.Op {
	etype1 := s.concreteEtype(t)
	etype2 := s.concreteEtype(u)
	x, ok := shiftOpToSSA[opAndTwoTypes{op, etype1, etype2}]
	if !ok {
		s.Fatalf("unhandled shift op %v etype=%s/%s", op, etype1, etype2)
	}
	return x
}

// expr converts the expression n to ssa, adds it to s and returns the ssa result.
func (s *state) expr(n *Node) *ssa.Value {
	if !(n.Op == ONAME || n.Op == OLITERAL && n.Sym != nil) {
		// ONAMEs and named OLITERALs have the line number
		// of the decl, not the use. See issue 14742.
		s.pushLine(n.Pos)
		defer s.popLine()
	}

	s.stmtList(n.Ninit)
	switch n.Op {
	case OBYTES2STRTMP:
		slice := s.expr(n.Left)
		ptr := s.newValue1(ssa.OpSlicePtr, s.f.Config.Types.BytePtr, slice)
		len := s.newValue1(ssa.OpSliceLen, types.Types[TINT], slice)
		return s.newValue2(ssa.OpStringMake, n.Type, ptr, len)
	case OSTR2BYTESTMP:
		str := s.expr(n.Left)
		ptr := s.newValue1(ssa.OpStringPtr, s.f.Config.Types.BytePtr, str)
		len := s.newValue1(ssa.OpStringLen, types.Types[TINT], str)
		return s.newValue3(ssa.OpSliceMake, n.Type, ptr, len, len)
	case OCFUNC:
		aux := n.Left.Sym.Linksym()
		return s.entryNewValue1A(ssa.OpAddr, n.Type, aux, s.sb)
	case ONAME:
		if n.Class() == PFUNC {
			// "value" of a function is the address of the function's closure
			sym := funcsym(n.Sym).Linksym()
			return s.entryNewValue1A(ssa.OpAddr, types.NewPtr(n.Type), sym, s.sb)
		}
		if s.canSSA(n) {
			return s.variable(n, n.Type)
		}
		addr := s.addr(n, false)
		return s.load(n.Type, addr)
	case OCLOSUREVAR:
		addr := s.addr(n, false)
		return s.load(n.Type, addr)
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
				pt := types.Types[TFLOAT32]
				return s.newValue2(ssa.OpComplexMake, n.Type,
					s.constFloat32(pt, r.Float32()),
					s.constFloat32(pt, i.Float32()))
			case 16:
				pt := types.Types[TFLOAT64]
				return s.newValue2(ssa.OpComplexMake, n.Type,
					s.constFloat64(pt, r.Float64()),
					s.constFloat64(pt, i.Float64()))
			default:
				s.Fatalf("bad float size %d", n.Type.Size())
				return nil
			}

		default:
			s.Fatalf("unhandled OLITERAL %v", n.Val().Ctype())
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
		if to.Etype == TUNSAFEPTR && from.IsPtrShaped() || from.Etype == TUNSAFEPTR && to.IsPtrShaped() {
			return v
		}

		// map <--> *hmap
		if to.Etype == TMAP && from.IsPtr() &&
			to.MapType().Hmap == from.Elem() {
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
		if ft.IsBoolean() && tt.IsKind(TUINT8) {
			// Bool -> uint8 is generated internally when indexing into runtime.staticbyte.
			return s.newValue1(ssa.OpCopy, n.Type, x)
		}
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
					s.Fatalf("weird integer truncation %v -> %v", ft, tt)
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
					s.Fatalf("bad integer sign extension %v -> %v", ft, tt)
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
					s.Fatalf("weird integer sign extension %v -> %v", ft, tt)
				}
			}
			return s.newValue1(op, n.Type, x)
		}

		if ft.IsFloat() || tt.IsFloat() {
			conv, ok := fpConvOpToSSA[twoTypes{s.concreteEtype(ft), s.concreteEtype(tt)}]
			if s.config.RegSize == 4 && thearch.LinkArch.Family != sys.MIPS && !s.softFloat {
				if conv1, ok1 := fpConvOpToSSA32[twoTypes{s.concreteEtype(ft), s.concreteEtype(tt)}]; ok1 {
					conv = conv1
				}
			}
			if thearch.LinkArch.Family == sys.ARM64 || thearch.LinkArch.Family == sys.Wasm || s.softFloat {
				if conv1, ok1 := uint64fpConvOpToSSA[twoTypes{s.concreteEtype(ft), s.concreteEtype(tt)}]; ok1 {
					conv = conv1
				}
			}

			if thearch.LinkArch.Family == sys.MIPS && !s.softFloat {
				if ft.Size() == 4 && ft.IsInteger() && !ft.IsSigned() {
					// tt is float32 or float64, and ft is also unsigned
					if tt.Size() == 4 {
						return s.uint32Tofloat32(n, x, ft, tt)
					}
					if tt.Size() == 8 {
						return s.uint32Tofloat64(n, x, ft, tt)
					}
				} else if tt.Size() == 4 && tt.IsInteger() && !tt.IsSigned() {
					// ft is float32 or float64, and tt is unsigned integer
					if ft.Size() == 4 {
						return s.float32ToUint32(n, x, ft, tt)
					}
					if ft.Size() == 8 {
						return s.float64ToUint32(n, x, ft, tt)
					}
				}
			}

			if !ok {
				s.Fatalf("weird float conversion %v -> %v", ft, tt)
			}
			op1, op2, it := conv.op1, conv.op2, conv.intermediateType

			if op1 != ssa.OpInvalid && op2 != ssa.OpInvalid {
				// normal case, not tripping over unsigned 64
				if op1 == ssa.OpCopy {
					if op2 == ssa.OpCopy {
						return x
					}
					return s.newValueOrSfCall1(op2, n.Type, x)
				}
				if op2 == ssa.OpCopy {
					return s.newValueOrSfCall1(op1, n.Type, x)
				}
				return s.newValueOrSfCall1(op2, n.Type, s.newValueOrSfCall1(op1, types.Types[it], x))
			}
			// Tricky 64-bit unsigned cases.
			if ft.IsInteger() {
				// tt is float32 or float64, and ft is also unsigned
				if tt.Size() == 4 {
					return s.uint64Tofloat32(n, x, ft, tt)
				}
				if tt.Size() == 8 {
					return s.uint64Tofloat64(n, x, ft, tt)
				}
				s.Fatalf("weird unsigned integer to float conversion %v -> %v", ft, tt)
			}
			// ft is float32 or float64, and tt is unsigned integer
			if ft.Size() == 4 {
				return s.float32ToUint64(n, x, ft, tt)
			}
			if ft.Size() == 8 {
				return s.float64ToUint64(n, x, ft, tt)
			}
			s.Fatalf("weird float to unsigned integer conversion %v -> %v", ft, tt)
			return nil
		}

		if ft.IsComplex() && tt.IsComplex() {
			var op ssa.Op
			if ft.Size() == tt.Size() {
				switch ft.Size() {
				case 8:
					op = ssa.OpRound32F
				case 16:
					op = ssa.OpRound64F
				default:
					s.Fatalf("weird complex conversion %v -> %v", ft, tt)
				}
			} else if ft.Size() == 8 && tt.Size() == 16 {
				op = ssa.OpCvt32Fto64F
			} else if ft.Size() == 16 && tt.Size() == 8 {
				op = ssa.OpCvt64Fto32F
			} else {
				s.Fatalf("weird complex conversion %v -> %v", ft, tt)
			}
			ftp := floatForComplex(ft)
			ttp := floatForComplex(tt)
			return s.newValue2(ssa.OpComplexMake, tt,
				s.newValueOrSfCall1(op, ttp, s.newValue1(ssa.OpComplexReal, ftp, x)),
				s.newValueOrSfCall1(op, ttp, s.newValue1(ssa.OpComplexImag, ftp, x)))
		}

		s.Fatalf("unhandled OCONV %s -> %s", n.Left.Type.Etype, n.Type.Etype)
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
			r := s.newValueOrSfCall2(op, types.Types[TBOOL], s.newValue1(ssa.OpComplexReal, pt, a), s.newValue1(ssa.OpComplexReal, pt, b))
			i := s.newValueOrSfCall2(op, types.Types[TBOOL], s.newValue1(ssa.OpComplexImag, pt, a), s.newValue1(ssa.OpComplexImag, pt, b))
			c := s.newValue2(ssa.OpAndB, types.Types[TBOOL], r, i)
			switch n.Op {
			case OEQ:
				return c
			case ONE:
				return s.newValue1(ssa.OpNot, types.Types[TBOOL], c)
			default:
				s.Fatalf("ordered complex compare %v", n.Op)
			}
		}
		if n.Left.Type.IsFloat() {
			return s.newValueOrSfCall2(s.ssaOp(n.Op, n.Left.Type), types.Types[TBOOL], a, b)
		}
		return s.newValue2(s.ssaOp(n.Op, n.Left.Type), types.Types[TBOOL], a, b)
	case OMUL:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		if n.Type.IsComplex() {
			mulop := ssa.OpMul64F
			addop := ssa.OpAdd64F
			subop := ssa.OpSub64F
			pt := floatForComplex(n.Type) // Could be Float32 or Float64
			wt := types.Types[TFLOAT64]   // Compute in Float64 to minimize cancelation error

			areal := s.newValue1(ssa.OpComplexReal, pt, a)
			breal := s.newValue1(ssa.OpComplexReal, pt, b)
			aimag := s.newValue1(ssa.OpComplexImag, pt, a)
			bimag := s.newValue1(ssa.OpComplexImag, pt, b)

			if pt != wt { // Widen for calculation
				areal = s.newValueOrSfCall1(ssa.OpCvt32Fto64F, wt, areal)
				breal = s.newValueOrSfCall1(ssa.OpCvt32Fto64F, wt, breal)
				aimag = s.newValueOrSfCall1(ssa.OpCvt32Fto64F, wt, aimag)
				bimag = s.newValueOrSfCall1(ssa.OpCvt32Fto64F, wt, bimag)
			}

			xreal := s.newValueOrSfCall2(subop, wt, s.newValueOrSfCall2(mulop, wt, areal, breal), s.newValueOrSfCall2(mulop, wt, aimag, bimag))
			ximag := s.newValueOrSfCall2(addop, wt, s.newValueOrSfCall2(mulop, wt, areal, bimag), s.newValueOrSfCall2(mulop, wt, aimag, breal))

			if pt != wt { // Narrow to store back
				xreal = s.newValueOrSfCall1(ssa.OpCvt64Fto32F, pt, xreal)
				ximag = s.newValueOrSfCall1(ssa.OpCvt64Fto32F, pt, ximag)
			}

			return s.newValue2(ssa.OpComplexMake, n.Type, xreal, ximag)
		}

		if n.Type.IsFloat() {
			return s.newValueOrSfCall2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
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
			wt := types.Types[TFLOAT64]   // Compute in Float64 to minimize cancelation error

			areal := s.newValue1(ssa.OpComplexReal, pt, a)
			breal := s.newValue1(ssa.OpComplexReal, pt, b)
			aimag := s.newValue1(ssa.OpComplexImag, pt, a)
			bimag := s.newValue1(ssa.OpComplexImag, pt, b)

			if pt != wt { // Widen for calculation
				areal = s.newValueOrSfCall1(ssa.OpCvt32Fto64F, wt, areal)
				breal = s.newValueOrSfCall1(ssa.OpCvt32Fto64F, wt, breal)
				aimag = s.newValueOrSfCall1(ssa.OpCvt32Fto64F, wt, aimag)
				bimag = s.newValueOrSfCall1(ssa.OpCvt32Fto64F, wt, bimag)
			}

			denom := s.newValueOrSfCall2(addop, wt, s.newValueOrSfCall2(mulop, wt, breal, breal), s.newValueOrSfCall2(mulop, wt, bimag, bimag))
			xreal := s.newValueOrSfCall2(addop, wt, s.newValueOrSfCall2(mulop, wt, areal, breal), s.newValueOrSfCall2(mulop, wt, aimag, bimag))
			ximag := s.newValueOrSfCall2(subop, wt, s.newValueOrSfCall2(mulop, wt, aimag, breal), s.newValueOrSfCall2(mulop, wt, areal, bimag))

			// TODO not sure if this is best done in wide precision or narrow
			// Double-rounding might be an issue.
			// Note that the pre-SSA implementation does the entire calculation
			// in wide format, so wide is compatible.
			xreal = s.newValueOrSfCall2(divop, wt, xreal, denom)
			ximag = s.newValueOrSfCall2(divop, wt, ximag, denom)

			if pt != wt { // Narrow to store back
				xreal = s.newValueOrSfCall1(ssa.OpCvt64Fto32F, pt, xreal)
				ximag = s.newValueOrSfCall1(ssa.OpCvt64Fto32F, pt, ximag)
			}
			return s.newValue2(ssa.OpComplexMake, n.Type, xreal, ximag)
		}
		if n.Type.IsFloat() {
			return s.newValueOrSfCall2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
		}
		return s.intDivide(n, a, b)
	case OMOD:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		return s.intDivide(n, a, b)
	case OADD, OSUB:
		a := s.expr(n.Left)
		b := s.expr(n.Right)
		if n.Type.IsComplex() {
			pt := floatForComplex(n.Type)
			op := s.ssaOp(n.Op, pt)
			return s.newValue2(ssa.OpComplexMake, n.Type,
				s.newValueOrSfCall2(op, pt, s.newValue1(ssa.OpComplexReal, pt, a), s.newValue1(ssa.OpComplexReal, pt, b)),
				s.newValueOrSfCall2(op, pt, s.newValue1(ssa.OpComplexImag, pt, a), s.newValue1(ssa.OpComplexImag, pt, b)))
		}
		if n.Type.IsFloat() {
			return s.newValueOrSfCall2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
		}
		return s.newValue2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
	case OAND, OOR, OXOR:
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
		return s.variable(n, types.Types[TBOOL])
	case OCOMPLEX:
		r := s.expr(n.Left)
		i := s.expr(n.Right)
		return s.newValue2(ssa.OpComplexMake, n.Type, r, i)

	// unary ops
	case ONEG:
		a := s.expr(n.Left)
		if n.Type.IsComplex() {
			tp := floatForComplex(n.Type)
			negop := s.ssaOp(n.Op, tp)
			return s.newValue2(ssa.OpComplexMake, n.Type,
				s.newValue1(negop, tp, s.newValue1(ssa.OpComplexReal, tp, a)),
				s.newValue1(negop, tp, s.newValue1(ssa.OpComplexImag, tp, a)))
		}
		return s.newValue1(s.ssaOp(n.Op, n.Type), a.Type, a)
	case ONOT, OBITNOT:
		a := s.expr(n.Left)
		return s.newValue1(s.ssaOp(n.Op, n.Type), a.Type, a)
	case OIMAG, OREAL:
		a := s.expr(n.Left)
		return s.newValue1(s.ssaOp(n.Op, n.Left.Type), n.Type, a)
	case OPLUS:
		return s.expr(n.Left)

	case OADDR:
		return s.addr(n.Left, n.Bounded())

	case OINDREGSP:
		addr := s.constOffPtrSP(types.NewPtr(n.Type), n.Xoffset)
		return s.load(n.Type, addr)

	case ODEREF:
		p := s.exprPtr(n.Left, false, n.Pos)
		return s.load(n.Type, p)

	case ODOT:
		if n.Left.Op == OSTRUCTLIT {
			// All literals with nonzero fields have already been
			// rewritten during walk. Any that remain are just T{}
			// or equivalents. Use the zero value.
			if !isZero(n.Left) {
				Fatalf("literal with nonzero value in SSA: %v", n.Left)
			}
			return s.zeroVal(n.Type)
		}
		// If n is addressable and can't be represented in
		// SSA, then load just the selected field. This
		// prevents false memory dependencies in race/msan
		// instrumentation.
		if islvalue(n) && !s.canSSA(n) {
			p := s.addr(n, false)
			return s.load(n.Type, p)
		}
		v := s.expr(n.Left)
		return s.newValue1I(ssa.OpStructSelect, n.Type, int64(fieldIdx(n)), v)

	case ODOTPTR:
		p := s.exprPtr(n.Left, false, n.Pos)
		p = s.newValue1I(ssa.OpOffPtr, types.NewPtr(n.Type), n.Xoffset, p)
		return s.load(n.Type, p)

	case OINDEX:
		switch {
		case n.Left.Type.IsString():
			if n.Bounded() && Isconst(n.Left, CTSTR) && Isconst(n.Right, CTINT) {
				// Replace "abc"[1] with 'b'.
				// Delayed until now because "abc"[1] is not an ideal constant.
				// See test/fixedbugs/issue11370.go.
				return s.newValue0I(ssa.OpConst8, types.Types[TUINT8], int64(int8(n.Left.Val().U.(string)[n.Right.Int64()])))
			}
			a := s.expr(n.Left)
			i := s.expr(n.Right)
			i = s.extendIndex(i, panicindex)
			if !n.Bounded() {
				len := s.newValue1(ssa.OpStringLen, types.Types[TINT], a)
				s.boundsCheck(i, len)
			}
			ptrtyp := s.f.Config.Types.BytePtr
			ptr := s.newValue1(ssa.OpStringPtr, ptrtyp, a)
			if Isconst(n.Right, CTINT) {
				ptr = s.newValue1I(ssa.OpOffPtr, ptrtyp, n.Right.Int64(), ptr)
			} else {
				ptr = s.newValue2(ssa.OpAddPtr, ptrtyp, ptr, i)
			}
			return s.load(types.Types[TUINT8], ptr)
		case n.Left.Type.IsSlice():
			p := s.addr(n, false)
			return s.load(n.Left.Type.Elem(), p)
		case n.Left.Type.IsArray():
			if canSSAType(n.Left.Type) {
				// SSA can handle arrays of length at most 1.
				bound := n.Left.Type.NumElem()
				a := s.expr(n.Left)
				i := s.expr(n.Right)
				if bound == 0 {
					// Bounds check will never succeed.  Might as well
					// use constants for the bounds check.
					z := s.constInt(types.Types[TINT], 0)
					s.boundsCheck(z, z)
					// The return value won't be live, return junk.
					return s.newValue0(ssa.OpUnknown, n.Type)
				}
				i = s.extendIndex(i, panicindex)
				if !n.Bounded() {
					s.boundsCheck(i, s.constInt(types.Types[TINT], bound))
				}
				return s.newValue1I(ssa.OpArraySelect, n.Type, 0, a)
			}
			p := s.addr(n, false)
			return s.load(n.Left.Type.Elem(), p)
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
			return s.newValue1(op, types.Types[TINT], s.expr(n.Left))
		case n.Left.Type.IsString(): // string; not reachable for OCAP
			return s.newValue1(ssa.OpStringLen, types.Types[TINT], s.expr(n.Left))
		case n.Left.Type.IsMap(), n.Left.Type.IsChan():
			return s.referenceTypeBuiltin(n, s.expr(n.Left))
		default: // array
			return s.constInt(types.Types[TINT], n.Left.Type.NumElem())
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

	case OIDATA:
		a := s.expr(n.Left)
		return s.newValue1(ssa.OpIData, n.Type, a)

	case OEFACE:
		tab := s.expr(n.Left)
		data := s.expr(n.Right)
		return s.newValue2(ssa.OpIMake, n.Type, tab, data)

	case OSLICEHEADER:
		p := s.expr(n.Left)
		l := s.expr(n.List.First())
		c := s.expr(n.List.Second())
		return s.newValue3(ssa.OpSliceMake, n.Type, p, l, c)

	case OSLICE, OSLICEARR, OSLICE3, OSLICE3ARR:
		v := s.expr(n.Left)
		var i, j, k *ssa.Value
		low, high, max := n.SliceBounds()
		if low != nil {
			i = s.extendIndex(s.expr(low), panicslice)
		}
		if high != nil {
			j = s.extendIndex(s.expr(high), panicslice)
		}
		if max != nil {
			k = s.extendIndex(s.expr(max), panicslice)
		}
		p, l, c := s.slice(n.Left.Type, v, i, j, k, n.Bounded())
		return s.newValue3(ssa.OpSliceMake, n.Type, p, l, c)

	case OSLICESTR:
		v := s.expr(n.Left)
		var i, j *ssa.Value
		low, high, _ := n.SliceBounds()
		if low != nil {
			i = s.extendIndex(s.expr(low), panicslice)
		}
		if high != nil {
			j = s.extendIndex(s.expr(high), panicslice)
		}
		p, l, _ := s.slice(n.Left.Type, v, i, j, nil, n.Bounded())
		return s.newValue2(ssa.OpStringMake, n.Type, p, l)

	case OCALLFUNC:
		if isIntrinsicCall(n) {
			return s.intrinsicCall(n)
		}
		fallthrough

	case OCALLINTER, OCALLMETH:
		a := s.call(n, callNormal)
		return s.load(n.Type, a)

	case OGETG:
		return s.newValue1(ssa.OpGetG, n.Type, s.mem())

	case OAPPEND:
		return s.append(n, false)

	case OSTRUCTLIT, OARRAYLIT:
		// All literals with nonzero fields have already been
		// rewritten during walk. Any that remain are just T{}
		// or equivalents. Use the zero value.
		if !isZero(n) {
			Fatalf("literal with nonzero value in SSA: %v", n)
		}
		return s.zeroVal(n.Type)

	default:
		s.Fatalf("unhandled expr %v", n.Op)
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
	pt := types.NewPtr(et)

	// Evaluate slice
	sn := n.List.First() // the slice node is the first in the list

	var slice, addr *ssa.Value
	if inplace {
		addr = s.addr(sn, false)
		slice = s.load(n.Type, addr)
	} else {
		slice = s.expr(sn)
	}

	// Allocate new blocks
	grow := s.f.NewBlock(ssa.BlockPlain)
	assign := s.f.NewBlock(ssa.BlockPlain)

	// Decide if we need to grow
	nargs := int64(n.List.Len() - 1)
	p := s.newValue1(ssa.OpSlicePtr, pt, slice)
	l := s.newValue1(ssa.OpSliceLen, types.Types[TINT], slice)
	c := s.newValue1(ssa.OpSliceCap, types.Types[TINT], slice)
	nl := s.newValue2(s.ssaOp(OADD, types.Types[TINT]), types.Types[TINT], l, s.constInt(types.Types[TINT], nargs))

	cmp := s.newValue2(s.ssaOp(OGT, types.Types[TINT]), types.Types[TBOOL], nl, c)
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
	taddr := s.expr(n.Left)
	r := s.rtcall(growslice, true, []*types.Type{pt, types.Types[TINT], types.Types[TINT]}, taddr, p, l, c, nl)

	if inplace {
		if sn.Op == ONAME && sn.Class() != PEXTERN {
			// Tell liveness we're about to build a new slice
			s.vars[&memVar] = s.newValue1A(ssa.OpVarDef, types.TypeMem, sn, s.mem())
		}
		capaddr := s.newValue1I(ssa.OpOffPtr, s.f.Config.Types.IntPtr, int64(array_cap), addr)
		s.store(types.Types[TINT], capaddr, r[2])
		s.store(pt, addr, r[0])
		// load the value we just stored to avoid having to spill it
		s.vars[&ptrVar] = s.load(pt, addr)
		s.vars[&lenVar] = r[1] // avoid a spill in the fast path
	} else {
		s.vars[&ptrVar] = r[0]
		s.vars[&newlenVar] = s.newValue2(s.ssaOp(OADD, types.Types[TINT]), types.Types[TINT], r[1], s.constInt(types.Types[TINT], nargs))
		s.vars[&capVar] = r[2]
	}

	b = s.endBlock()
	b.AddEdgeTo(assign)

	// assign new elements to slots
	s.startBlock(assign)

	if inplace {
		l = s.variable(&lenVar, types.Types[TINT]) // generates phi for len
		nl = s.newValue2(s.ssaOp(OADD, types.Types[TINT]), types.Types[TINT], l, s.constInt(types.Types[TINT], nargs))
		lenaddr := s.newValue1I(ssa.OpOffPtr, s.f.Config.Types.IntPtr, int64(array_nel), addr)
		s.store(types.Types[TINT], lenaddr, nl)
	}

	// Evaluate args
	type argRec struct {
		// if store is true, we're appending the value v.  If false, we're appending the
		// value at *v.
		v     *ssa.Value
		store bool
	}
	args := make([]argRec, 0, nargs)
	for _, n := range n.List.Slice()[1:] {
		if canSSAType(n.Type) {
			args = append(args, argRec{v: s.expr(n), store: true})
		} else {
			v := s.addr(n, false)
			args = append(args, argRec{v: v})
		}
	}

	p = s.variable(&ptrVar, pt) // generates phi for ptr
	if !inplace {
		nl = s.variable(&newlenVar, types.Types[TINT]) // generates phi for nl
		c = s.variable(&capVar, types.Types[TINT])     // generates phi for cap
	}
	p2 := s.newValue2(ssa.OpPtrIndex, pt, p, l)
	for i, arg := range args {
		addr := s.newValue2(ssa.OpPtrIndex, pt, p2, s.constInt(types.Types[TINT], int64(i)))
		if arg.store {
			s.storeType(et, addr, arg.v, 0, true)
		} else {
			s.move(et, addr, arg.v)
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
	switch cond.Op {
	case OANDAND:
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
	case OOROR:
		mid := s.f.NewBlock(ssa.BlockPlain)
		s.stmtList(cond.Ninit)
		s.condBranch(cond.Left, yes, mid, min8(likely, 0))
		s.startBlock(mid)
		s.condBranch(cond.Right, yes, no, likely)
		return
		// Note: if likely==-1, then both recursive calls pass -1.
		// If likely==1, then we don't have enough info to decide
		// the likelihood of the first branch.
	case ONOT:
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
// skip indicates assignments (at the top level) that can be avoided.
func (s *state) assign(left *Node, right *ssa.Value, deref bool, skip skipMask) {
	if left.Op == ONAME && left.isBlank() {
		return
	}
	t := left.Type
	dowidth(t)
	if s.canSSA(left) {
		if deref {
			s.Fatalf("can SSA LHS %v but not RHS %s", left, right)
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
			s.assign(left.Left, new, false, 0)
			// TODO: do we need to update named values here?
			return
		}
		if left.Op == OINDEX && left.Left.Type.IsArray() {
			// We're assigning to an element of an ssa-able array.
			// a[i] = v
			t := left.Left.Type
			n := t.NumElem()

			i := s.expr(left.Right) // index
			if n == 0 {
				// The bounds check must fail.  Might as well
				// ignore the actual index and just use zeros.
				z := s.constInt(types.Types[TINT], 0)
				s.boundsCheck(z, z)
				return
			}
			if n != 1 {
				s.Fatalf("assigning to non-1-length array")
			}
			// Rewrite to a = [1]{v}
			i = s.extendIndex(i, panicindex)
			s.boundsCheck(i, s.constInt(types.Types[TINT], 1))
			v := s.newValue1(ssa.OpArrayMake1, t, right)
			s.assign(left.Left, v, false, 0)
			return
		}
		// Update variable assignment.
		s.vars[left] = right
		s.addNamedValue(left, right)
		return
	}
	// Left is not ssa-able. Compute its address.
	if left.Op == ONAME && left.Class() != PEXTERN && skip == 0 {
		s.vars[&memVar] = s.newValue1Apos(ssa.OpVarDef, types.TypeMem, left, s.mem(), !left.IsAutoTmp())
	}
	addr := s.addr(left, false)
	if isReflectHeaderDataField(left) {
		// Package unsafe's documentation says storing pointers into
		// reflect.SliceHeader and reflect.StringHeader's Data fields
		// is valid, even though they have type uintptr (#19168).
		// Mark it pointer type to signal the writebarrier pass to
		// insert a write barrier.
		t = types.Types[TUNSAFEPTR]
	}
	if deref {
		// Treat as a mem->mem move.
		if right == nil {
			s.zero(t, addr)
		} else {
			s.move(t, addr, right)
		}
		return
	}
	// Treat as a store.
	s.storeType(t, addr, right, skip, !left.IsAutoTmp())
}

// zeroVal returns the zero value for type t.
func (s *state) zeroVal(t *types.Type) *ssa.Value {
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
			s.Fatalf("bad sized integer type %v", t)
		}
	case t.IsFloat():
		switch t.Size() {
		case 4:
			return s.constFloat32(t, 0)
		case 8:
			return s.constFloat64(t, 0)
		default:
			s.Fatalf("bad sized float type %v", t)
		}
	case t.IsComplex():
		switch t.Size() {
		case 8:
			z := s.constFloat32(types.Types[TFLOAT32], 0)
			return s.entryNewValue2(ssa.OpComplexMake, t, z, z)
		case 16:
			z := s.constFloat64(types.Types[TFLOAT64], 0)
			return s.entryNewValue2(ssa.OpComplexMake, t, z, z)
		default:
			s.Fatalf("bad sized complex type %v", t)
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
			v.AddArg(s.zeroVal(t.FieldType(i)))
		}
		return v
	case t.IsArray():
		switch t.NumElem() {
		case 0:
			return s.entryNewValue0(ssa.OpArrayMake0, t)
		case 1:
			return s.entryNewValue1(ssa.OpArrayMake1, t, s.zeroVal(t.Elem()))
		}
	}
	s.Fatalf("zero for type %v not implemented", t)
	return nil
}

type callKind int8

const (
	callNormal callKind = iota
	callDefer
	callGo
)

type sfRtCallDef struct {
	rtfn  *obj.LSym
	rtype types.EType
}

var softFloatOps map[ssa.Op]sfRtCallDef

func softfloatInit() {
	// Some of these operations get transformed by sfcall.
	softFloatOps = map[ssa.Op]sfRtCallDef{
		ssa.OpAdd32F: sfRtCallDef{sysfunc("fadd32"), TFLOAT32},
		ssa.OpAdd64F: sfRtCallDef{sysfunc("fadd64"), TFLOAT64},
		ssa.OpSub32F: sfRtCallDef{sysfunc("fadd32"), TFLOAT32},
		ssa.OpSub64F: sfRtCallDef{sysfunc("fadd64"), TFLOAT64},
		ssa.OpMul32F: sfRtCallDef{sysfunc("fmul32"), TFLOAT32},
		ssa.OpMul64F: sfRtCallDef{sysfunc("fmul64"), TFLOAT64},
		ssa.OpDiv32F: sfRtCallDef{sysfunc("fdiv32"), TFLOAT32},
		ssa.OpDiv64F: sfRtCallDef{sysfunc("fdiv64"), TFLOAT64},

		ssa.OpEq64F:      sfRtCallDef{sysfunc("feq64"), TBOOL},
		ssa.OpEq32F:      sfRtCallDef{sysfunc("feq32"), TBOOL},
		ssa.OpNeq64F:     sfRtCallDef{sysfunc("feq64"), TBOOL},
		ssa.OpNeq32F:     sfRtCallDef{sysfunc("feq32"), TBOOL},
		ssa.OpLess64F:    sfRtCallDef{sysfunc("fgt64"), TBOOL},
		ssa.OpLess32F:    sfRtCallDef{sysfunc("fgt32"), TBOOL},
		ssa.OpGreater64F: sfRtCallDef{sysfunc("fgt64"), TBOOL},
		ssa.OpGreater32F: sfRtCallDef{sysfunc("fgt32"), TBOOL},
		ssa.OpLeq64F:     sfRtCallDef{sysfunc("fge64"), TBOOL},
		ssa.OpLeq32F:     sfRtCallDef{sysfunc("fge32"), TBOOL},
		ssa.OpGeq64F:     sfRtCallDef{sysfunc("fge64"), TBOOL},
		ssa.OpGeq32F:     sfRtCallDef{sysfunc("fge32"), TBOOL},

		ssa.OpCvt32to32F:  sfRtCallDef{sysfunc("fint32to32"), TFLOAT32},
		ssa.OpCvt32Fto32:  sfRtCallDef{sysfunc("f32toint32"), TINT32},
		ssa.OpCvt64to32F:  sfRtCallDef{sysfunc("fint64to32"), TFLOAT32},
		ssa.OpCvt32Fto64:  sfRtCallDef{sysfunc("f32toint64"), TINT64},
		ssa.OpCvt64Uto32F: sfRtCallDef{sysfunc("fuint64to32"), TFLOAT32},
		ssa.OpCvt32Fto64U: sfRtCallDef{sysfunc("f32touint64"), TUINT64},
		ssa.OpCvt32to64F:  sfRtCallDef{sysfunc("fint32to64"), TFLOAT64},
		ssa.OpCvt64Fto32:  sfRtCallDef{sysfunc("f64toint32"), TINT32},
		ssa.OpCvt64to64F:  sfRtCallDef{sysfunc("fint64to64"), TFLOAT64},
		ssa.OpCvt64Fto64:  sfRtCallDef{sysfunc("f64toint64"), TINT64},
		ssa.OpCvt64Uto64F: sfRtCallDef{sysfunc("fuint64to64"), TFLOAT64},
		ssa.OpCvt64Fto64U: sfRtCallDef{sysfunc("f64touint64"), TUINT64},
		ssa.OpCvt32Fto64F: sfRtCallDef{sysfunc("f32to64"), TFLOAT64},
		ssa.OpCvt64Fto32F: sfRtCallDef{sysfunc("f64to32"), TFLOAT32},
	}
}

// TODO: do not emit sfcall if operation can be optimized to constant in later
// opt phase
func (s *state) sfcall(op ssa.Op, args ...*ssa.Value) (*ssa.Value, bool) {
	if callDef, ok := softFloatOps[op]; ok {
		switch op {
		case ssa.OpLess32F,
			ssa.OpLess64F,
			ssa.OpLeq32F,
			ssa.OpLeq64F:
			args[0], args[1] = args[1], args[0]
		case ssa.OpSub32F,
			ssa.OpSub64F:
			args[1] = s.newValue1(s.ssaOp(ONEG, types.Types[callDef.rtype]), args[1].Type, args[1])
		}

		result := s.rtcall(callDef.rtfn, true, []*types.Type{types.Types[callDef.rtype]}, args...)[0]
		if op == ssa.OpNeq32F || op == ssa.OpNeq64F {
			result = s.newValue1(ssa.OpNot, result.Type, result)
		}
		return result, true
	}
	return nil, false
}

var intrinsics map[intrinsicKey]intrinsicBuilder

// An intrinsicBuilder converts a call node n into an ssa value that
// implements that call as an intrinsic. args is a list of arguments to the func.
type intrinsicBuilder func(s *state, n *Node, args []*ssa.Value) *ssa.Value

type intrinsicKey struct {
	arch *sys.Arch
	pkg  string
	fn   string
}

func init() {
	intrinsics = map[intrinsicKey]intrinsicBuilder{}

	var all []*sys.Arch
	var p4 []*sys.Arch
	var p8 []*sys.Arch
	var lwatomics []*sys.Arch
	for _, a := range sys.Archs {
		all = append(all, a)
		if a.PtrSize == 4 {
			p4 = append(p4, a)
		} else {
			p8 = append(p8, a)
		}
		if a.Family != sys.PPC64 {
			lwatomics = append(lwatomics, a)
		}
	}

	// add adds the intrinsic b for pkg.fn for the given list of architectures.
	add := func(pkg, fn string, b intrinsicBuilder, archs ...*sys.Arch) {
		for _, a := range archs {
			intrinsics[intrinsicKey{a, pkg, fn}] = b
		}
	}
	// addF does the same as add but operates on architecture families.
	addF := func(pkg, fn string, b intrinsicBuilder, archFamilies ...sys.ArchFamily) {
		m := 0
		for _, f := range archFamilies {
			if f >= 32 {
				panic("too many architecture families")
			}
			m |= 1 << uint(f)
		}
		for _, a := range all {
			if m>>uint(a.Family)&1 != 0 {
				intrinsics[intrinsicKey{a, pkg, fn}] = b
			}
		}
	}
	// alias defines pkg.fn = pkg2.fn2 for all architectures in archs for which pkg2.fn2 exists.
	alias := func(pkg, fn, pkg2, fn2 string, archs ...*sys.Arch) {
		for _, a := range archs {
			if b, ok := intrinsics[intrinsicKey{a, pkg2, fn2}]; ok {
				intrinsics[intrinsicKey{a, pkg, fn}] = b
			}
		}
	}

	/******** runtime ********/
	if !instrumenting {
		add("runtime", "slicebytetostringtmp",
			func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
				// Compiler frontend optimizations emit OBYTES2STRTMP nodes
				// for the backend instead of slicebytetostringtmp calls
				// when not instrumenting.
				slice := args[0]
				ptr := s.newValue1(ssa.OpSlicePtr, s.f.Config.Types.BytePtr, slice)
				len := s.newValue1(ssa.OpSliceLen, types.Types[TINT], slice)
				return s.newValue2(ssa.OpStringMake, n.Type, ptr, len)
			},
			all...)
	}
	addF("runtime/internal/math", "MulUintptr",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			if s.config.PtrSize == 4 {
				return s.newValue2(ssa.OpMul32uover, types.NewTuple(types.Types[TUINT], types.Types[TUINT]), args[0], args[1])
			}
			return s.newValue2(ssa.OpMul64uover, types.NewTuple(types.Types[TUINT], types.Types[TUINT]), args[0], args[1])
		},
		sys.AMD64, sys.I386)
	add("runtime", "KeepAlive",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			data := s.newValue1(ssa.OpIData, s.f.Config.Types.BytePtr, args[0])
			s.vars[&memVar] = s.newValue2(ssa.OpKeepAlive, types.TypeMem, data, s.mem())
			return nil
		},
		all...)
	add("runtime", "getclosureptr",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue0(ssa.OpGetClosurePtr, s.f.Config.Types.Uintptr)
		},
		all...)

	add("runtime", "getcallerpc",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue0(ssa.OpGetCallerPC, s.f.Config.Types.Uintptr)
		},
		all...)

	add("runtime", "getcallersp",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue0(ssa.OpGetCallerSP, s.f.Config.Types.Uintptr)
		},
		all...)

	/******** runtime/internal/sys ********/
	addF("runtime/internal/sys", "Ctz32",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz32, types.Types[TINT], args[0])
		},
		sys.AMD64, sys.ARM64, sys.ARM, sys.S390X, sys.MIPS, sys.PPC64)
	addF("runtime/internal/sys", "Ctz64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz64, types.Types[TINT], args[0])
		},
		sys.AMD64, sys.ARM64, sys.ARM, sys.S390X, sys.MIPS, sys.PPC64)
	addF("runtime/internal/sys", "Bswap32",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBswap32, types.Types[TUINT32], args[0])
		},
		sys.AMD64, sys.ARM64, sys.ARM, sys.S390X)
	addF("runtime/internal/sys", "Bswap64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBswap64, types.Types[TUINT64], args[0])
		},
		sys.AMD64, sys.ARM64, sys.ARM, sys.S390X)

	/******** runtime/internal/atomic ********/
	addF("runtime/internal/atomic", "Load",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoad32, types.NewTuple(types.Types[TUINT32], types.TypeMem), args[0], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TUINT32], v)
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS, sys.MIPS64, sys.PPC64)
	addF("runtime/internal/atomic", "Load64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoad64, types.NewTuple(types.Types[TUINT64], types.TypeMem), args[0], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TUINT64], v)
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS64, sys.PPC64)
	addF("runtime/internal/atomic", "LoadAcq",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoadAcq32, types.NewTuple(types.Types[TUINT32], types.TypeMem), args[0], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TUINT32], v)
		},
		sys.PPC64)
	addF("runtime/internal/atomic", "Loadp",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue2(ssa.OpAtomicLoadPtr, types.NewTuple(s.f.Config.Types.BytePtr, types.TypeMem), args[0], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, s.f.Config.Types.BytePtr, v)
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS, sys.MIPS64, sys.PPC64)

	addF("runtime/internal/atomic", "Store",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			s.vars[&memVar] = s.newValue3(ssa.OpAtomicStore32, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS, sys.MIPS64, sys.PPC64)
	addF("runtime/internal/atomic", "Store64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			s.vars[&memVar] = s.newValue3(ssa.OpAtomicStore64, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS64, sys.PPC64)
	addF("runtime/internal/atomic", "StorepNoWB",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			s.vars[&memVar] = s.newValue3(ssa.OpAtomicStorePtrNoWB, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS, sys.MIPS64)
	addF("runtime/internal/atomic", "StoreRel",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			s.vars[&memVar] = s.newValue3(ssa.OpAtomicStoreRel32, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.PPC64)

	addF("runtime/internal/atomic", "Xchg",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicExchange32, types.NewTuple(types.Types[TUINT32], types.TypeMem), args[0], args[1], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TUINT32], v)
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS, sys.MIPS64, sys.PPC64)
	addF("runtime/internal/atomic", "Xchg64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicExchange64, types.NewTuple(types.Types[TUINT64], types.TypeMem), args[0], args[1], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TUINT64], v)
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS64, sys.PPC64)

	addF("runtime/internal/atomic", "Xadd",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicAdd32, types.NewTuple(types.Types[TUINT32], types.TypeMem), args[0], args[1], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TUINT32], v)
		},
		sys.AMD64, sys.S390X, sys.MIPS, sys.MIPS64, sys.PPC64)
	addF("runtime/internal/atomic", "Xadd64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue3(ssa.OpAtomicAdd64, types.NewTuple(types.Types[TUINT64], types.TypeMem), args[0], args[1], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TUINT64], v)
		},
		sys.AMD64, sys.S390X, sys.MIPS64, sys.PPC64)

	makeXaddARM64 := func(op0 ssa.Op, op1 ssa.Op, ty types.EType) func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			// Target Atomic feature is identified by dynamic detection
			addr := s.entryNewValue1A(ssa.OpAddr, types.Types[TBOOL].PtrTo(), arm64HasATOMICS, s.sb)
			v := s.load(types.Types[TBOOL], addr)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchUnlikely // most machines don't have Atomics nowadays

			// We have atomic instructions - use it directly.
			s.startBlock(bTrue)
			v0 := s.newValue3(op1, types.NewTuple(types.Types[ty], types.TypeMem), args[0], args[1], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v0)
			s.vars[n] = s.newValue1(ssa.OpSelect0, types.Types[ty], v0)
			s.endBlock().AddEdgeTo(bEnd)

			// Use original instruction sequence.
			s.startBlock(bFalse)
			v1 := s.newValue3(op0, types.NewTuple(types.Types[ty], types.TypeMem), args[0], args[1], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v1)
			s.vars[n] = s.newValue1(ssa.OpSelect0, types.Types[ty], v1)
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[ty])
		}
	}

	addF("runtime/internal/atomic", "Xadd",
		makeXaddARM64(ssa.OpAtomicAdd32, ssa.OpAtomicAdd32Variant, TUINT32),
		sys.ARM64)
	addF("runtime/internal/atomic", "Xadd64",
		makeXaddARM64(ssa.OpAtomicAdd64, ssa.OpAtomicAdd64Variant, TUINT64),
		sys.ARM64)

	addF("runtime/internal/atomic", "Cas",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue4(ssa.OpAtomicCompareAndSwap32, types.NewTuple(types.Types[TBOOL], types.TypeMem), args[0], args[1], args[2], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TBOOL], v)
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS, sys.MIPS64, sys.PPC64)
	addF("runtime/internal/atomic", "Cas64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue4(ssa.OpAtomicCompareAndSwap64, types.NewTuple(types.Types[TBOOL], types.TypeMem), args[0], args[1], args[2], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TBOOL], v)
		},
		sys.AMD64, sys.ARM64, sys.S390X, sys.MIPS64, sys.PPC64)
	addF("runtime/internal/atomic", "CasRel",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			v := s.newValue4(ssa.OpAtomicCompareAndSwap32, types.NewTuple(types.Types[TBOOL], types.TypeMem), args[0], args[1], args[2], s.mem())
			s.vars[&memVar] = s.newValue1(ssa.OpSelect1, types.TypeMem, v)
			return s.newValue1(ssa.OpSelect0, types.Types[TBOOL], v)
		},
		sys.PPC64)

	addF("runtime/internal/atomic", "And8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			s.vars[&memVar] = s.newValue3(ssa.OpAtomicAnd8, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.ARM64, sys.MIPS, sys.PPC64)
	addF("runtime/internal/atomic", "Or8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			s.vars[&memVar] = s.newValue3(ssa.OpAtomicOr8, types.TypeMem, args[0], args[1], s.mem())
			return nil
		},
		sys.AMD64, sys.ARM64, sys.MIPS, sys.PPC64)

	alias("runtime/internal/atomic", "Loadint64", "runtime/internal/atomic", "Load64", all...)
	alias("runtime/internal/atomic", "Xaddint64", "runtime/internal/atomic", "Xadd64", all...)
	alias("runtime/internal/atomic", "Loaduint", "runtime/internal/atomic", "Load", p4...)
	alias("runtime/internal/atomic", "Loaduint", "runtime/internal/atomic", "Load64", p8...)
	alias("runtime/internal/atomic", "Loaduintptr", "runtime/internal/atomic", "Load", p4...)
	alias("runtime/internal/atomic", "Loaduintptr", "runtime/internal/atomic", "Load64", p8...)
	alias("runtime/internal/atomic", "LoadAcq", "runtime/internal/atomic", "Load", lwatomics...)
	alias("runtime/internal/atomic", "Storeuintptr", "runtime/internal/atomic", "Store", p4...)
	alias("runtime/internal/atomic", "Storeuintptr", "runtime/internal/atomic", "Store64", p8...)
	alias("runtime/internal/atomic", "StoreRel", "runtime/internal/atomic", "Store", lwatomics...)
	alias("runtime/internal/atomic", "Xchguintptr", "runtime/internal/atomic", "Xchg", p4...)
	alias("runtime/internal/atomic", "Xchguintptr", "runtime/internal/atomic", "Xchg64", p8...)
	alias("runtime/internal/atomic", "Xadduintptr", "runtime/internal/atomic", "Xadd", p4...)
	alias("runtime/internal/atomic", "Xadduintptr", "runtime/internal/atomic", "Xadd64", p8...)
	alias("runtime/internal/atomic", "Casuintptr", "runtime/internal/atomic", "Cas", p4...)
	alias("runtime/internal/atomic", "Casuintptr", "runtime/internal/atomic", "Cas64", p8...)
	alias("runtime/internal/atomic", "Casp1", "runtime/internal/atomic", "Cas", p4...)
	alias("runtime/internal/atomic", "Casp1", "runtime/internal/atomic", "Cas64", p8...)
	alias("runtime/internal/atomic", "CasRel", "runtime/internal/atomic", "Cas", lwatomics...)

	alias("runtime/internal/sys", "Ctz8", "math/bits", "TrailingZeros8", all...)

	/******** math ********/
	addF("math", "Sqrt",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpSqrt, types.Types[TFLOAT64], args[0])
		},
		sys.I386, sys.AMD64, sys.ARM, sys.ARM64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.S390X)
	addF("math", "Trunc",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpTrunc, types.Types[TFLOAT64], args[0])
		},
		sys.ARM64, sys.PPC64, sys.S390X)
	addF("math", "Ceil",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCeil, types.Types[TFLOAT64], args[0])
		},
		sys.ARM64, sys.PPC64, sys.S390X)
	addF("math", "Floor",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpFloor, types.Types[TFLOAT64], args[0])
		},
		sys.ARM64, sys.PPC64, sys.S390X)
	addF("math", "Round",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpRound, types.Types[TFLOAT64], args[0])
		},
		sys.ARM64, sys.PPC64, sys.S390X)
	addF("math", "RoundToEven",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpRoundToEven, types.Types[TFLOAT64], args[0])
		},
		sys.ARM64, sys.S390X)
	addF("math", "Abs",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpAbs, types.Types[TFLOAT64], args[0])
		},
		sys.ARM64, sys.PPC64)
	addF("math", "Copysign",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpCopysign, types.Types[TFLOAT64], args[0], args[1])
		},
		sys.PPC64)

	makeRoundAMD64 := func(op ssa.Op) func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			addr := s.entryNewValue1A(ssa.OpAddr, types.Types[TBOOL].PtrTo(), x86HasSSE41, s.sb)
			v := s.load(types.Types[TBOOL], addr)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely // most machines have sse4.1 nowadays

			// We have the intrinsic - use it directly.
			s.startBlock(bTrue)
			s.vars[n] = s.newValue1(op, types.Types[TFLOAT64], args[0])
			s.endBlock().AddEdgeTo(bEnd)

			// Call the pure Go version.
			s.startBlock(bFalse)
			a := s.call(n, callNormal)
			s.vars[n] = s.load(types.Types[TFLOAT64], a)
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[TFLOAT64])
		}
	}
	addF("math", "RoundToEven",
		makeRoundAMD64(ssa.OpRoundToEven),
		sys.AMD64)
	addF("math", "Floor",
		makeRoundAMD64(ssa.OpFloor),
		sys.AMD64)
	addF("math", "Ceil",
		makeRoundAMD64(ssa.OpCeil),
		sys.AMD64)
	addF("math", "Trunc",
		makeRoundAMD64(ssa.OpTrunc),
		sys.AMD64)

	/******** math/bits ********/
	addF("math/bits", "TrailingZeros64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz64, types.Types[TINT], args[0])
		},
		sys.AMD64, sys.ARM64, sys.ARM, sys.S390X, sys.MIPS, sys.PPC64)
	addF("math/bits", "TrailingZeros32",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz32, types.Types[TINT], args[0])
		},
		sys.AMD64, sys.ARM64, sys.ARM, sys.S390X, sys.MIPS, sys.PPC64)
	addF("math/bits", "TrailingZeros16",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			x := s.newValue1(ssa.OpZeroExt16to32, types.Types[TUINT32], args[0])
			c := s.constInt32(types.Types[TUINT32], 1<<16)
			y := s.newValue2(ssa.OpOr32, types.Types[TUINT32], x, c)
			return s.newValue1(ssa.OpCtz32, types.Types[TINT], y)
		},
		sys.ARM, sys.MIPS)
	addF("math/bits", "TrailingZeros16",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz16, types.Types[TINT], args[0])
		},
		sys.AMD64)
	addF("math/bits", "TrailingZeros16",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			x := s.newValue1(ssa.OpZeroExt16to64, types.Types[TUINT64], args[0])
			c := s.constInt64(types.Types[TUINT64], 1<<16)
			y := s.newValue2(ssa.OpOr64, types.Types[TUINT64], x, c)
			return s.newValue1(ssa.OpCtz64, types.Types[TINT], y)
		},
		sys.ARM64, sys.S390X, sys.PPC64)
	addF("math/bits", "TrailingZeros8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			x := s.newValue1(ssa.OpZeroExt8to32, types.Types[TUINT32], args[0])
			c := s.constInt32(types.Types[TUINT32], 1<<8)
			y := s.newValue2(ssa.OpOr32, types.Types[TUINT32], x, c)
			return s.newValue1(ssa.OpCtz32, types.Types[TINT], y)
		},
		sys.ARM, sys.MIPS)
	addF("math/bits", "TrailingZeros8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpCtz8, types.Types[TINT], args[0])
		},
		sys.AMD64)
	addF("math/bits", "TrailingZeros8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			x := s.newValue1(ssa.OpZeroExt8to64, types.Types[TUINT64], args[0])
			c := s.constInt64(types.Types[TUINT64], 1<<8)
			y := s.newValue2(ssa.OpOr64, types.Types[TUINT64], x, c)
			return s.newValue1(ssa.OpCtz64, types.Types[TINT], y)
		},
		sys.ARM64, sys.S390X)
	alias("math/bits", "ReverseBytes64", "runtime/internal/sys", "Bswap64", all...)
	alias("math/bits", "ReverseBytes32", "runtime/internal/sys", "Bswap32", all...)
	// ReverseBytes inlines correctly, no need to intrinsify it.
	// ReverseBytes16 lowers to a rotate, no need for anything special here.
	addF("math/bits", "Len64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitLen64, types.Types[TINT], args[0])
		},
		sys.AMD64, sys.ARM64, sys.ARM, sys.S390X, sys.MIPS, sys.PPC64)
	addF("math/bits", "Len32",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitLen32, types.Types[TINT], args[0])
		},
		sys.AMD64)
	addF("math/bits", "Len32",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			if s.config.PtrSize == 4 {
				return s.newValue1(ssa.OpBitLen32, types.Types[TINT], args[0])
			}
			x := s.newValue1(ssa.OpZeroExt32to64, types.Types[TUINT64], args[0])
			return s.newValue1(ssa.OpBitLen64, types.Types[TINT], x)
		},
		sys.ARM64, sys.ARM, sys.S390X, sys.MIPS, sys.PPC64)
	addF("math/bits", "Len16",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			if s.config.PtrSize == 4 {
				x := s.newValue1(ssa.OpZeroExt16to32, types.Types[TUINT32], args[0])
				return s.newValue1(ssa.OpBitLen32, types.Types[TINT], x)
			}
			x := s.newValue1(ssa.OpZeroExt16to64, types.Types[TUINT64], args[0])
			return s.newValue1(ssa.OpBitLen64, types.Types[TINT], x)
		},
		sys.ARM64, sys.ARM, sys.S390X, sys.MIPS, sys.PPC64)
	addF("math/bits", "Len16",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitLen16, types.Types[TINT], args[0])
		},
		sys.AMD64)
	addF("math/bits", "Len8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			if s.config.PtrSize == 4 {
				x := s.newValue1(ssa.OpZeroExt8to32, types.Types[TUINT32], args[0])
				return s.newValue1(ssa.OpBitLen32, types.Types[TINT], x)
			}
			x := s.newValue1(ssa.OpZeroExt8to64, types.Types[TUINT64], args[0])
			return s.newValue1(ssa.OpBitLen64, types.Types[TINT], x)
		},
		sys.ARM64, sys.ARM, sys.S390X, sys.MIPS, sys.PPC64)
	addF("math/bits", "Len8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitLen8, types.Types[TINT], args[0])
		},
		sys.AMD64)
	addF("math/bits", "Len",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			if s.config.PtrSize == 4 {
				return s.newValue1(ssa.OpBitLen32, types.Types[TINT], args[0])
			}
			return s.newValue1(ssa.OpBitLen64, types.Types[TINT], args[0])
		},
		sys.AMD64, sys.ARM64, sys.ARM, sys.S390X, sys.MIPS, sys.PPC64)
	// LeadingZeros is handled because it trivially calls Len.
	addF("math/bits", "Reverse64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitRev64, types.Types[TINT], args[0])
		},
		sys.ARM64)
	addF("math/bits", "Reverse32",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitRev32, types.Types[TINT], args[0])
		},
		sys.ARM64)
	addF("math/bits", "Reverse16",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitRev16, types.Types[TINT], args[0])
		},
		sys.ARM64)
	addF("math/bits", "Reverse8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpBitRev8, types.Types[TINT], args[0])
		},
		sys.ARM64)
	addF("math/bits", "Reverse",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			if s.config.PtrSize == 4 {
				return s.newValue1(ssa.OpBitRev32, types.Types[TINT], args[0])
			}
			return s.newValue1(ssa.OpBitRev64, types.Types[TINT], args[0])
		},
		sys.ARM64)
	addF("math/bits", "RotateLeft8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpRotateLeft8, types.Types[TUINT8], args[0], args[1])
		},
		sys.AMD64)
	addF("math/bits", "RotateLeft16",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpRotateLeft16, types.Types[TUINT16], args[0], args[1])
		},
		sys.AMD64)
	addF("math/bits", "RotateLeft32",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpRotateLeft32, types.Types[TUINT32], args[0], args[1])
		},
		sys.AMD64, sys.ARM64, sys.S390X)
	addF("math/bits", "RotateLeft64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpRotateLeft64, types.Types[TUINT64], args[0], args[1])
		},
		sys.AMD64, sys.ARM64, sys.S390X)
	alias("math/bits", "RotateLeft", "math/bits", "RotateLeft64", p8...)

	makeOnesCountAMD64 := func(op64 ssa.Op, op32 ssa.Op) func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			addr := s.entryNewValue1A(ssa.OpAddr, types.Types[TBOOL].PtrTo(), x86HasPOPCNT, s.sb)
			v := s.load(types.Types[TBOOL], addr)
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(v)
			bTrue := s.f.NewBlock(ssa.BlockPlain)
			bFalse := s.f.NewBlock(ssa.BlockPlain)
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bTrue)
			b.AddEdgeTo(bFalse)
			b.Likely = ssa.BranchLikely // most machines have popcnt nowadays

			// We have the intrinsic - use it directly.
			s.startBlock(bTrue)
			op := op64
			if s.config.PtrSize == 4 {
				op = op32
			}
			s.vars[n] = s.newValue1(op, types.Types[TINT], args[0])
			s.endBlock().AddEdgeTo(bEnd)

			// Call the pure Go version.
			s.startBlock(bFalse)
			a := s.call(n, callNormal)
			s.vars[n] = s.load(types.Types[TINT], a)
			s.endBlock().AddEdgeTo(bEnd)

			// Merge results.
			s.startBlock(bEnd)
			return s.variable(n, types.Types[TINT])
		}
	}
	addF("math/bits", "OnesCount64",
		makeOnesCountAMD64(ssa.OpPopCount64, ssa.OpPopCount64),
		sys.AMD64)
	addF("math/bits", "OnesCount64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpPopCount64, types.Types[TINT], args[0])
		},
		sys.PPC64, sys.ARM64, sys.S390X)
	addF("math/bits", "OnesCount32",
		makeOnesCountAMD64(ssa.OpPopCount32, ssa.OpPopCount32),
		sys.AMD64)
	addF("math/bits", "OnesCount32",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpPopCount32, types.Types[TINT], args[0])
		},
		sys.PPC64, sys.ARM64, sys.S390X)
	addF("math/bits", "OnesCount16",
		makeOnesCountAMD64(ssa.OpPopCount16, ssa.OpPopCount16),
		sys.AMD64)
	addF("math/bits", "OnesCount16",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpPopCount16, types.Types[TINT], args[0])
		},
		sys.ARM64, sys.S390X, sys.PPC64)
	addF("math/bits", "OnesCount8",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue1(ssa.OpPopCount8, types.Types[TINT], args[0])
		},
		sys.S390X, sys.PPC64)
	addF("math/bits", "OnesCount",
		makeOnesCountAMD64(ssa.OpPopCount64, ssa.OpPopCount32),
		sys.AMD64)
	addF("math/bits", "Mul64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpMul64uhilo, types.NewTuple(types.Types[TUINT64], types.Types[TUINT64]), args[0], args[1])
		},
		sys.AMD64, sys.ARM64, sys.PPC64)
	alias("math/bits", "Mul", "math/bits", "Mul64", sys.ArchAMD64, sys.ArchARM64, sys.ArchPPC64)

	addF("math/bits", "Add64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue3(ssa.OpAdd64carry, types.NewTuple(types.Types[TUINT64], types.Types[TUINT64]), args[0], args[1], args[2])
		},
		sys.AMD64)
	alias("math/bits", "Add", "math/bits", "Add64", sys.ArchAMD64)

	addF("math/bits", "Sub64",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue3(ssa.OpSub64borrow, types.NewTuple(types.Types[TUINT64], types.Types[TUINT64]), args[0], args[1], args[2])
		},
		sys.AMD64)
	alias("math/bits", "Sub", "math/bits", "Sub64", sys.ArchAMD64)

	/******** sync/atomic ********/

	// Note: these are disabled by flag_race in findIntrinsic below.
	alias("sync/atomic", "LoadInt32", "runtime/internal/atomic", "Load", all...)
	alias("sync/atomic", "LoadInt64", "runtime/internal/atomic", "Load64", all...)
	alias("sync/atomic", "LoadPointer", "runtime/internal/atomic", "Loadp", all...)
	alias("sync/atomic", "LoadUint32", "runtime/internal/atomic", "Load", all...)
	alias("sync/atomic", "LoadUint64", "runtime/internal/atomic", "Load64", all...)
	alias("sync/atomic", "LoadUintptr", "runtime/internal/atomic", "Load", p4...)
	alias("sync/atomic", "LoadUintptr", "runtime/internal/atomic", "Load64", p8...)

	alias("sync/atomic", "StoreInt32", "runtime/internal/atomic", "Store", all...)
	alias("sync/atomic", "StoreInt64", "runtime/internal/atomic", "Store64", all...)
	// Note: not StorePointer, that needs a write barrier.  Same below for {CompareAnd}Swap.
	alias("sync/atomic", "StoreUint32", "runtime/internal/atomic", "Store", all...)
	alias("sync/atomic", "StoreUint64", "runtime/internal/atomic", "Store64", all...)
	alias("sync/atomic", "StoreUintptr", "runtime/internal/atomic", "Store", p4...)
	alias("sync/atomic", "StoreUintptr", "runtime/internal/atomic", "Store64", p8...)

	alias("sync/atomic", "SwapInt32", "runtime/internal/atomic", "Xchg", all...)
	alias("sync/atomic", "SwapInt64", "runtime/internal/atomic", "Xchg64", all...)
	alias("sync/atomic", "SwapUint32", "runtime/internal/atomic", "Xchg", all...)
	alias("sync/atomic", "SwapUint64", "runtime/internal/atomic", "Xchg64", all...)
	alias("sync/atomic", "SwapUintptr", "runtime/internal/atomic", "Xchg", p4...)
	alias("sync/atomic", "SwapUintptr", "runtime/internal/atomic", "Xchg64", p8...)

	alias("sync/atomic", "CompareAndSwapInt32", "runtime/internal/atomic", "Cas", all...)
	alias("sync/atomic", "CompareAndSwapInt64", "runtime/internal/atomic", "Cas64", all...)
	alias("sync/atomic", "CompareAndSwapUint32", "runtime/internal/atomic", "Cas", all...)
	alias("sync/atomic", "CompareAndSwapUint64", "runtime/internal/atomic", "Cas64", all...)
	alias("sync/atomic", "CompareAndSwapUintptr", "runtime/internal/atomic", "Cas", p4...)
	alias("sync/atomic", "CompareAndSwapUintptr", "runtime/internal/atomic", "Cas64", p8...)

	alias("sync/atomic", "AddInt32", "runtime/internal/atomic", "Xadd", all...)
	alias("sync/atomic", "AddInt64", "runtime/internal/atomic", "Xadd64", all...)
	alias("sync/atomic", "AddUint32", "runtime/internal/atomic", "Xadd", all...)
	alias("sync/atomic", "AddUint64", "runtime/internal/atomic", "Xadd64", all...)
	alias("sync/atomic", "AddUintptr", "runtime/internal/atomic", "Xadd", p4...)
	alias("sync/atomic", "AddUintptr", "runtime/internal/atomic", "Xadd64", p8...)

	/******** math/big ********/
	add("math/big", "mulWW",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue2(ssa.OpMul64uhilo, types.NewTuple(types.Types[TUINT64], types.Types[TUINT64]), args[0], args[1])
		},
		sys.ArchAMD64, sys.ArchARM64, sys.ArchPPC64LE, sys.ArchPPC64)
	add("math/big", "divWW",
		func(s *state, n *Node, args []*ssa.Value) *ssa.Value {
			return s.newValue3(ssa.OpDiv128u, types.NewTuple(types.Types[TUINT64], types.Types[TUINT64]), args[0], args[1], args[2])
		},
		sys.ArchAMD64)
}

// findIntrinsic returns a function which builds the SSA equivalent of the
// function identified by the symbol sym.  If sym is not an intrinsic call, returns nil.
func findIntrinsic(sym *types.Sym) intrinsicBuilder {
	if ssa.IntrinsicsDisable {
		return nil
	}
	if sym == nil || sym.Pkg == nil {
		return nil
	}
	pkg := sym.Pkg.Path
	if sym.Pkg == localpkg {
		pkg = myimportpath
	}
	if flag_race && pkg == "sync/atomic" {
		// The race detector needs to be able to intercept these calls.
		// We can't intrinsify them.
		return nil
	}
	// Skip intrinsifying math functions (which may contain hard-float
	// instructions) when soft-float
	if thearch.SoftFloat && pkg == "math" {
		return nil
	}

	fn := sym.Name
	return intrinsics[intrinsicKey{thearch.LinkArch.Arch, pkg, fn}]
}

func isIntrinsicCall(n *Node) bool {
	if n == nil || n.Left == nil {
		return false
	}
	return findIntrinsic(n.Left.Sym) != nil
}

// intrinsicCall converts a call to a recognized intrinsic function into the intrinsic SSA operation.
func (s *state) intrinsicCall(n *Node) *ssa.Value {
	v := findIntrinsic(n.Left.Sym)(s, n, s.intrinsicArgs(n))
	if ssa.IntrinsicsDebug > 0 {
		x := v
		if x == nil {
			x = s.mem()
		}
		if x.Op == ssa.OpSelect0 || x.Op == ssa.OpSelect1 {
			x = x.Args[0]
		}
		Warnl(n.Pos, "intrinsic substitution for %v with %s", n.Left.Sym.Name, x.LongString())
	}
	return v
}

// intrinsicArgs extracts args from n, evaluates them to SSA values, and returns them.
func (s *state) intrinsicArgs(n *Node) []*ssa.Value {
	// Construct map of temps; see comments in s.call about the structure of n.
	temps := map[*Node]*ssa.Value{}
	for _, a := range n.List.Slice() {
		if a.Op != OAS {
			s.Fatalf("non-assignment as a temp function argument %v", a.Op)
		}
		l, r := a.Left, a.Right
		if l.Op != ONAME {
			s.Fatalf("non-ONAME temp function argument %v", a.Op)
		}
		// Evaluate and store to "temporary".
		// Walk ensures these temporaries are dead outside of n.
		temps[l] = s.expr(r)
	}
	args := make([]*ssa.Value, n.Rlist.Len())
	for i, n := range n.Rlist.Slice() {
		// Store a value to an argument slot.
		if x, ok := temps[n]; ok {
			// This is a previously computed temporary.
			args[i] = x
			continue
		}
		// This is an explicit value; evaluate it.
		args[i] = s.expr(n)
	}
	return args
}

// Calls the function n using the specified call type.
// Returns the address of the return value (or nil if none).
func (s *state) call(n *Node, k callKind) *ssa.Value {
	var sym *types.Sym     // target symbol (if static)
	var closure *ssa.Value // ptr to closure to run (if dynamic)
	var codeptr *ssa.Value // ptr to target code (if dynamic)
	var rcvr *ssa.Value    // receiver to set
	fn := n.Left
	switch n.Op {
	case OCALLFUNC:
		if k == callNormal && fn.Op == ONAME && fn.Class() == PFUNC {
			sym = fn.Sym
			break
		}
		closure = s.expr(fn)
		if thearch.LinkArch.Family == sys.Wasm {
			// TODO(neelance): On other architectures this should be eliminated by the optimization steps
			s.nilCheck(closure)
		}
	case OCALLMETH:
		if fn.Op != ODOTMETH {
			Fatalf("OCALLMETH: n.Left not an ODOTMETH: %v", fn)
		}
		if k == callNormal {
			sym = fn.Sym
			break
		}
		// Make a name n2 for the function.
		// fn.Sym might be sync.(*Mutex).Unlock.
		// Make a PFUNC node out of that, then evaluate it.
		// We get back an SSA value representing &sync.(*Mutex).Unlock·f.
		// We can then pass that to defer or go.
		n2 := newnamel(fn.Pos, fn.Sym)
		n2.Name.Curfn = s.curfn
		n2.SetClass(PFUNC)
		// n2.Sym already existed, so it's already marked as a function.
		n2.Pos = fn.Pos
		n2.Type = types.Types[TUINT8] // dummy type for a static closure. Could use runtime.funcval if we had it.
		closure = s.expr(n2)
		// Note: receiver is already present in n.Rlist, so we don't
		// want to set it here.
	case OCALLINTER:
		if fn.Op != ODOTINTER {
			Fatalf("OCALLINTER: n.Left not an ODOTINTER: %v", fn.Op)
		}
		i := s.expr(fn.Left)
		itab := s.newValue1(ssa.OpITab, types.Types[TUINTPTR], i)
		s.nilCheck(itab)
		itabidx := fn.Xoffset + 2*int64(Widthptr) + 8 // offset of fun field in runtime.itab
		itab = s.newValue1I(ssa.OpOffPtr, s.f.Config.Types.UintptrPtr, itabidx, itab)
		if k == callNormal {
			codeptr = s.load(types.Types[TUINTPTR], itab)
		} else {
			closure = itab
		}
		rcvr = s.newValue1(ssa.OpIData, types.Types[TUINTPTR], i)
	}
	dowidth(fn.Type)
	stksize := fn.Type.ArgWidth() // includes receiver

	// Run all assignments of temps.
	// The temps are introduced to avoid overwriting argument
	// slots when arguments themselves require function calls.
	s.stmtList(n.List)

	// Store arguments to stack, including defer/go arguments and receiver for method calls.
	// These are written in SP-offset order.
	argStart := Ctxt.FixedFrameSize()
	// Defer/go args.
	if k != callNormal {
		// Write argsize and closure (args to newproc/deferproc).
		argsize := s.constInt32(types.Types[TUINT32], int32(stksize))
		addr := s.constOffPtrSP(s.f.Config.Types.UInt32Ptr, argStart)
		s.store(types.Types[TUINT32], addr, argsize)
		addr = s.constOffPtrSP(s.f.Config.Types.UintptrPtr, argStart+int64(Widthptr))
		s.store(types.Types[TUINTPTR], addr, closure)
		stksize += 2 * int64(Widthptr)
		argStart += 2 * int64(Widthptr)
	}

	// Set receiver (for interface calls).
	if rcvr != nil {
		addr := s.constOffPtrSP(s.f.Config.Types.UintptrPtr, argStart)
		s.store(types.Types[TUINTPTR], addr, rcvr)
	}

	// Write args.
	t := n.Left.Type
	args := n.Rlist.Slice()
	if n.Op == OCALLMETH {
		f := t.Recv()
		s.storeArg(args[0], f.Type, argStart+f.Offset)
		args = args[1:]
	}
	for i, n := range args {
		f := t.Params().Field(i)
		s.storeArg(n, f.Type, argStart+f.Offset)
	}

	// call target
	var call *ssa.Value
	switch {
	case k == callDefer:
		call = s.newValue1A(ssa.OpStaticCall, types.TypeMem, deferproc, s.mem())
	case k == callGo:
		call = s.newValue1A(ssa.OpStaticCall, types.TypeMem, newproc, s.mem())
	case closure != nil:
		// rawLoad because loading the code pointer from a
		// closure is always safe, but IsSanitizerSafeAddr
		// can't always figure that out currently, and it's
		// critical that we not clobber any arguments already
		// stored onto the stack.
		codeptr = s.rawLoad(types.Types[TUINTPTR], closure)
		call = s.newValue3(ssa.OpClosureCall, types.TypeMem, codeptr, closure, s.mem())
	case codeptr != nil:
		call = s.newValue2(ssa.OpInterCall, types.TypeMem, codeptr, s.mem())
	case sym != nil:
		call = s.newValue1A(ssa.OpStaticCall, types.TypeMem, sym.Linksym(), s.mem())
	default:
		Fatalf("bad call type %v %v", n.Op, n)
	}
	call.AuxInt = stksize // Call operations carry the argsize of the callee along with them
	s.vars[&memVar] = call

	// Finish block for defers
	if k == callDefer {
		b := s.endBlock()
		b.Kind = ssa.BlockDefer
		b.SetControl(call)
		bNext := s.f.NewBlock(ssa.BlockPlain)
		b.AddEdgeTo(bNext)
		// Add recover edge to exit code.
		r := s.f.NewBlock(ssa.BlockPlain)
		s.startBlock(r)
		s.exit()
		b.AddEdgeTo(r)
		b.Likely = ssa.BranchLikely
		s.startBlock(bNext)
	}

	res := n.Left.Type.Results()
	if res.NumFields() == 0 || k != callNormal {
		// call has no return value. Continue with the next statement.
		return nil
	}
	fp := res.Field(0)
	return s.constOffPtrSP(types.NewPtr(fp.Type), fp.Offset+Ctxt.FixedFrameSize())
}

// etypesign returns the signed-ness of e, for integer/pointer etypes.
// -1 means signed, +1 means unsigned, 0 means non-integer/non-pointer.
func etypesign(e types.EType) int8 {
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
// If bounded is true then this address does not require a nil check for its operand
// even if that would otherwise be implied.
func (s *state) addr(n *Node, bounded bool) *ssa.Value {
	t := types.NewPtr(n.Type)
	switch n.Op {
	case ONAME:
		switch n.Class() {
		case PEXTERN:
			// global variable
			v := s.entryNewValue1A(ssa.OpAddr, t, n.Sym.Linksym(), s.sb)
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
			if n == nodfp {
				// Special arg that points to the frame pointer (Used by ORECOVER).
				return s.entryNewValue2A(ssa.OpLocalAddr, t, n, s.sp, s.startmem)
			}
			s.Fatalf("addr of undeclared ONAME %v. declared: %v", n, s.decladdrs)
			return nil
		case PAUTO:
			return s.newValue2Apos(ssa.OpLocalAddr, t, n, s.sp, s.mem(), !n.IsAutoTmp())

		case PPARAMOUT: // Same as PAUTO -- cannot generate LEA early.
			// ensure that we reuse symbols for out parameters so
			// that cse works on their addresses
			return s.newValue2Apos(ssa.OpLocalAddr, t, n, s.sp, s.mem(), true)
		default:
			s.Fatalf("variable address class %v not implemented", n.Class())
			return nil
		}
	case OINDREGSP:
		// indirect off REGSP
		// used for storing/loading arguments/returns to/from callees
		return s.constOffPtrSP(t, n.Xoffset)
	case OINDEX:
		if n.Left.Type.IsSlice() {
			a := s.expr(n.Left)
			i := s.expr(n.Right)
			i = s.extendIndex(i, panicindex)
			len := s.newValue1(ssa.OpSliceLen, types.Types[TINT], a)
			if !n.Bounded() {
				s.boundsCheck(i, len)
			}
			p := s.newValue1(ssa.OpSlicePtr, t, a)
			return s.newValue2(ssa.OpPtrIndex, t, p, i)
		} else { // array
			a := s.addr(n.Left, bounded)
			i := s.expr(n.Right)
			i = s.extendIndex(i, panicindex)
			len := s.constInt(types.Types[TINT], n.Left.Type.NumElem())
			if !n.Bounded() {
				s.boundsCheck(i, len)
			}
			return s.newValue2(ssa.OpPtrIndex, types.NewPtr(n.Left.Type.Elem()), a, i)
		}
	case ODEREF:
		return s.exprPtr(n.Left, bounded, n.Pos)
	case ODOT:
		p := s.addr(n.Left, bounded)
		return s.newValue1I(ssa.OpOffPtr, t, n.Xoffset, p)
	case ODOTPTR:
		p := s.exprPtr(n.Left, bounded, n.Pos)
		return s.newValue1I(ssa.OpOffPtr, t, n.Xoffset, p)
	case OCLOSUREVAR:
		return s.newValue1I(ssa.OpOffPtr, t, n.Xoffset,
			s.entryNewValue0(ssa.OpGetClosurePtr, s.f.Config.Types.BytePtr))
	case OCONVNOP:
		addr := s.addr(n.Left, bounded)
		return s.newValue1(ssa.OpCopy, t, addr) // ensure that addr has the right type
	case OCALLFUNC, OCALLINTER, OCALLMETH:
		return s.call(n, callNormal)
	case ODOTTYPE:
		v, _ := s.dottype(n, false)
		if v.Op != ssa.OpLoad {
			s.Fatalf("dottype of non-load")
		}
		if v.Args[1] != s.mem() {
			s.Fatalf("memory no longer live from dottype load")
		}
		return v.Args[0]
	default:
		s.Fatalf("unhandled addr %v", n.Op)
		return nil
	}
}

// canSSA reports whether n is SSA-able.
// n must be an ONAME (or an ODOT sequence with an ONAME base).
func (s *state) canSSA(n *Node) bool {
	if Debug['N'] != 0 {
		return false
	}
	for n.Op == ODOT || (n.Op == OINDEX && n.Left.Type.IsArray()) {
		n = n.Left
	}
	if n.Op != ONAME {
		return false
	}
	if n.Addrtaken() {
		return false
	}
	if n.isParamHeapCopy() {
		return false
	}
	if n.Class() == PAUTOHEAP {
		Fatalf("canSSA of PAUTOHEAP %v", n)
	}
	switch n.Class() {
	case PEXTERN:
		return false
	case PPARAMOUT:
		if s.hasdefer {
			// TODO: handle this case? Named return values must be
			// in memory so that the deferred function can see them.
			// Maybe do: if !strings.HasPrefix(n.String(), "~") { return false }
			// Or maybe not, see issue 18860.  Even unnamed return values
			// must be written back so if a defer recovers, the caller can see them.
			return false
		}
		if s.cgoUnsafeArgs {
			// Cgo effectively takes the address of all result args,
			// but the compiler can't see that.
			return false
		}
	}
	if n.Class() == PPARAM && n.Sym != nil && n.Sym.Name == ".this" {
		// wrappers generated by genwrapper need to update
		// the .this pointer in place.
		// TODO: treat as a PPARMOUT?
		return false
	}
	return canSSAType(n.Type)
	// TODO: try to make more variables SSAable?
}

// canSSA reports whether variables of type t are SSA-able.
func canSSAType(t *types.Type) bool {
	dowidth(t)
	if t.Width > int64(4*Widthptr) {
		// 4*Widthptr is an arbitrary constant. We want it
		// to be at least 3*Widthptr so slices can be registerized.
		// Too big and we'll introduce too much register pressure.
		return false
	}
	switch t.Etype {
	case TARRAY:
		// We can't do larger arrays because dynamic indexing is
		// not supported on SSA variables.
		// TODO: allow if all indexes are constant.
		if t.NumElem() <= 1 {
			return canSSAType(t.Elem())
		}
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
func (s *state) exprPtr(n *Node, bounded bool, lineno src.XPos) *ssa.Value {
	p := s.expr(n)
	if bounded || n.NonNil() {
		if s.f.Frontend().Debug_checknil() && lineno.Line() > 1 {
			s.f.Warnl(lineno, "removed nil check")
		}
		return p
	}
	s.nilCheck(p)
	return p
}

// nilCheck generates nil pointer checking code.
// Used only for automatically inserted nil checks,
// not for user code like 'x != nil'.
func (s *state) nilCheck(ptr *ssa.Value) {
	if disable_checknil != 0 || s.curfn.Func.NilCheckDisabled() {
		return
	}
	s.newValue2(ssa.OpNilCheck, types.TypeVoid, ptr, s.mem())
}

// boundsCheck generates bounds checking code. Checks if 0 <= idx < len, branches to exit if not.
// Starts a new block on return.
// idx is already converted to full int width.
func (s *state) boundsCheck(idx, len *ssa.Value) {
	if Debug['B'] != 0 {
		return
	}

	// bounds check
	cmp := s.newValue2(ssa.OpIsInBounds, types.Types[TBOOL], idx, len)
	s.check(cmp, panicindex)
}

// sliceBoundsCheck generates slice bounds checking code. Checks if 0 <= idx <= len, branches to exit if not.
// Starts a new block on return.
// idx and len are already converted to full int width.
func (s *state) sliceBoundsCheck(idx, len *ssa.Value) {
	if Debug['B'] != 0 {
		return
	}

	// bounds check
	cmp := s.newValue2(ssa.OpIsSliceInBounds, types.Types[TBOOL], idx, len)
	s.check(cmp, panicslice)
}

// If cmp (a bool) is false, panic using the given function.
func (s *state) check(cmp *ssa.Value, fn *obj.LSym) {
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.SetControl(cmp)
	b.Likely = ssa.BranchLikely
	bNext := s.f.NewBlock(ssa.BlockPlain)
	line := s.peekPos()
	pos := Ctxt.PosTable.Pos(line)
	fl := funcLine{f: fn, base: pos.Base(), line: pos.Line()}
	bPanic := s.panics[fl]
	if bPanic == nil {
		bPanic = s.f.NewBlock(ssa.BlockPlain)
		s.panics[fl] = bPanic
		s.startBlock(bPanic)
		// The panic call takes/returns memory to ensure that the right
		// memory state is observed if the panic happens.
		s.rtcall(fn, false, nil)
	}
	b.AddEdgeTo(bNext)
	b.AddEdgeTo(bPanic)
	s.startBlock(bNext)
}

func (s *state) intDivide(n *Node, a, b *ssa.Value) *ssa.Value {
	needcheck := true
	switch b.Op {
	case ssa.OpConst8, ssa.OpConst16, ssa.OpConst32, ssa.OpConst64:
		if b.AuxInt != 0 {
			needcheck = false
		}
	}
	if needcheck {
		// do a size-appropriate check for zero
		cmp := s.newValue2(s.ssaOp(ONE, n.Type), types.Types[TBOOL], b, s.zeroVal(n.Type))
		s.check(cmp, panicdivide)
	}
	return s.newValue2(s.ssaOp(n.Op, n.Type), a.Type, a, b)
}

// rtcall issues a call to the given runtime function fn with the listed args.
// Returns a slice of results of the given result types.
// The call is added to the end of the current block.
// If returns is false, the block is marked as an exit block.
func (s *state) rtcall(fn *obj.LSym, returns bool, results []*types.Type, args ...*ssa.Value) []*ssa.Value {
	// Write args to the stack
	off := Ctxt.FixedFrameSize()
	for _, arg := range args {
		t := arg.Type
		off = Rnd(off, t.Alignment())
		ptr := s.constOffPtrSP(t.PtrTo(), off)
		size := t.Size()
		s.store(t, ptr, arg)
		off += size
	}
	off = Rnd(off, int64(Widthreg))

	// Issue call
	call := s.newValue1A(ssa.OpStaticCall, types.TypeMem, fn, s.mem())
	s.vars[&memVar] = call

	if !returns {
		// Finish block
		b := s.endBlock()
		b.Kind = ssa.BlockExit
		b.SetControl(call)
		call.AuxInt = off - Ctxt.FixedFrameSize()
		if len(results) > 0 {
			Fatalf("panic call can't have results")
		}
		return nil
	}

	// Load results
	res := make([]*ssa.Value, len(results))
	for i, t := range results {
		off = Rnd(off, t.Alignment())
		ptr := s.constOffPtrSP(types.NewPtr(t), off)
		res[i] = s.load(t, ptr)
		off += t.Size()
	}
	off = Rnd(off, int64(Widthptr))

	// Remember how much callee stack space we needed.
	call.AuxInt = off

	return res
}

// do *left = right for type t.
func (s *state) storeType(t *types.Type, left, right *ssa.Value, skip skipMask, leftIsStmt bool) {
	s.instrument(t, left, true)

	if skip == 0 && (!types.Haspointers(t) || ssa.IsStackAddr(left)) {
		// Known to not have write barrier. Store the whole type.
		s.vars[&memVar] = s.newValue3Apos(ssa.OpStore, types.TypeMem, t, left, right, s.mem(), leftIsStmt)
		return
	}

	// store scalar fields first, so write barrier stores for
	// pointer fields can be grouped together, and scalar values
	// don't need to be live across the write barrier call.
	// TODO: if the writebarrier pass knows how to reorder stores,
	// we can do a single store here as long as skip==0.
	s.storeTypeScalars(t, left, right, skip)
	if skip&skipPtr == 0 && types.Haspointers(t) {
		s.storeTypePtrs(t, left, right)
	}
}

// do *left = right for all scalar (non-pointer) parts of t.
func (s *state) storeTypeScalars(t *types.Type, left, right *ssa.Value, skip skipMask) {
	switch {
	case t.IsBoolean() || t.IsInteger() || t.IsFloat() || t.IsComplex():
		s.store(t, left, right)
	case t.IsPtrShaped():
		// no scalar fields.
	case t.IsString():
		if skip&skipLen != 0 {
			return
		}
		len := s.newValue1(ssa.OpStringLen, types.Types[TINT], right)
		lenAddr := s.newValue1I(ssa.OpOffPtr, s.f.Config.Types.IntPtr, s.config.PtrSize, left)
		s.store(types.Types[TINT], lenAddr, len)
	case t.IsSlice():
		if skip&skipLen == 0 {
			len := s.newValue1(ssa.OpSliceLen, types.Types[TINT], right)
			lenAddr := s.newValue1I(ssa.OpOffPtr, s.f.Config.Types.IntPtr, s.config.PtrSize, left)
			s.store(types.Types[TINT], lenAddr, len)
		}
		if skip&skipCap == 0 {
			cap := s.newValue1(ssa.OpSliceCap, types.Types[TINT], right)
			capAddr := s.newValue1I(ssa.OpOffPtr, s.f.Config.Types.IntPtr, 2*s.config.PtrSize, left)
			s.store(types.Types[TINT], capAddr, cap)
		}
	case t.IsInterface():
		// itab field doesn't need a write barrier (even though it is a pointer).
		itab := s.newValue1(ssa.OpITab, s.f.Config.Types.BytePtr, right)
		s.store(types.Types[TUINTPTR], left, itab)
	case t.IsStruct():
		n := t.NumFields()
		for i := 0; i < n; i++ {
			ft := t.FieldType(i)
			addr := s.newValue1I(ssa.OpOffPtr, ft.PtrTo(), t.FieldOff(i), left)
			val := s.newValue1I(ssa.OpStructSelect, ft, int64(i), right)
			s.storeTypeScalars(ft, addr, val, 0)
		}
	case t.IsArray() && t.NumElem() == 0:
		// nothing
	case t.IsArray() && t.NumElem() == 1:
		s.storeTypeScalars(t.Elem(), left, s.newValue1I(ssa.OpArraySelect, t.Elem(), 0, right), 0)
	default:
		s.Fatalf("bad write barrier type %v", t)
	}
}

// do *left = right for all pointer parts of t.
func (s *state) storeTypePtrs(t *types.Type, left, right *ssa.Value) {
	switch {
	case t.IsPtrShaped():
		s.store(t, left, right)
	case t.IsString():
		ptr := s.newValue1(ssa.OpStringPtr, s.f.Config.Types.BytePtr, right)
		s.store(s.f.Config.Types.BytePtr, left, ptr)
	case t.IsSlice():
		elType := types.NewPtr(t.Elem())
		ptr := s.newValue1(ssa.OpSlicePtr, elType, right)
		s.store(elType, left, ptr)
	case t.IsInterface():
		// itab field is treated as a scalar.
		idata := s.newValue1(ssa.OpIData, s.f.Config.Types.BytePtr, right)
		idataAddr := s.newValue1I(ssa.OpOffPtr, s.f.Config.Types.BytePtrPtr, s.config.PtrSize, left)
		s.store(s.f.Config.Types.BytePtr, idataAddr, idata)
	case t.IsStruct():
		n := t.NumFields()
		for i := 0; i < n; i++ {
			ft := t.FieldType(i)
			if !types.Haspointers(ft) {
				continue
			}
			addr := s.newValue1I(ssa.OpOffPtr, ft.PtrTo(), t.FieldOff(i), left)
			val := s.newValue1I(ssa.OpStructSelect, ft, int64(i), right)
			s.storeTypePtrs(ft, addr, val)
		}
	case t.IsArray() && t.NumElem() == 0:
		// nothing
	case t.IsArray() && t.NumElem() == 1:
		s.storeTypePtrs(t.Elem(), left, s.newValue1I(ssa.OpArraySelect, t.Elem(), 0, right))
	default:
		s.Fatalf("bad write barrier type %v", t)
	}
}

func (s *state) storeArg(n *Node, t *types.Type, off int64) {
	pt := types.NewPtr(t)
	sp := s.constOffPtrSP(pt, off)

	if !canSSAType(t) {
		a := s.addr(n, false)
		s.move(t, sp, a)
		return
	}

	a := s.expr(n)
	s.storeType(t, sp, a, 0, false)
}

// slice computes the slice v[i:j:k] and returns ptr, len, and cap of result.
// i,j,k may be nil, in which case they are set to their default value.
// t is a slice, ptr to array, or string type.
func (s *state) slice(t *types.Type, v, i, j, k *ssa.Value, bounded bool) (p, l, c *ssa.Value) {
	var elemtype *types.Type
	var ptrtype *types.Type
	var ptr *ssa.Value
	var len *ssa.Value
	var cap *ssa.Value
	zero := s.constInt(types.Types[TINT], 0)
	switch {
	case t.IsSlice():
		elemtype = t.Elem()
		ptrtype = types.NewPtr(elemtype)
		ptr = s.newValue1(ssa.OpSlicePtr, ptrtype, v)
		len = s.newValue1(ssa.OpSliceLen, types.Types[TINT], v)
		cap = s.newValue1(ssa.OpSliceCap, types.Types[TINT], v)
	case t.IsString():
		elemtype = types.Types[TUINT8]
		ptrtype = types.NewPtr(elemtype)
		ptr = s.newValue1(ssa.OpStringPtr, ptrtype, v)
		len = s.newValue1(ssa.OpStringLen, types.Types[TINT], v)
		cap = len
	case t.IsPtr():
		if !t.Elem().IsArray() {
			s.Fatalf("bad ptr to array in slice %v\n", t)
		}
		elemtype = t.Elem().Elem()
		ptrtype = types.NewPtr(elemtype)
		s.nilCheck(v)
		ptr = v
		len = s.constInt(types.Types[TINT], t.Elem().NumElem())
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

	if !bounded {
		// Panic if slice indices are not in bounds.
		s.sliceBoundsCheck(i, j)
		if j != k {
			s.sliceBoundsCheck(j, k)
		}
		if k != cap {
			s.sliceBoundsCheck(k, cap)
		}
	}

	// Generate the following code assuming that indexes are in bounds.
	// The masking is to make sure that we don't generate a slice
	// that points to the next object in memory.
	// rlen = j - i
	// rcap = k - i
	// delta = i * elemsize
	// rptr = p + delta&mask(rcap)
	// result = (SliceMake rptr rlen rcap)
	// where mask(x) is 0 if x==0 and -1 if x>0.
	subOp := s.ssaOp(OSUB, types.Types[TINT])
	mulOp := s.ssaOp(OMUL, types.Types[TINT])
	andOp := s.ssaOp(OAND, types.Types[TINT])
	rlen := s.newValue2(subOp, types.Types[TINT], j, i)
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
		rcap = s.newValue2(subOp, types.Types[TINT], k, i)
	}

	var rptr *ssa.Value
	if (i.Op == ssa.OpConst64 || i.Op == ssa.OpConst32) && i.AuxInt == 0 {
		// No pointer arithmetic necessary.
		rptr = ptr
	} else {
		// delta = # of bytes to offset pointer by.
		delta := s.newValue2(mulOp, types.Types[TINT], i, s.constInt(types.Types[TINT], elemtype.Width))
		// If we're slicing to the point where the capacity is zero,
		// zero out the delta.
		mask := s.newValue1(ssa.OpSlicemask, types.Types[TINT], rcap)
		delta = s.newValue2(andOp, types.Types[TINT], delta, mask)
		// Compute rptr = ptr + delta
		rptr = s.newValue2(ssa.OpAddPtr, ptrtype, ptr, delta)
	}

	return rptr, rlen, rcap
}

type u642fcvtTab struct {
	geq, cvt2F, and, rsh, or, add ssa.Op
	one                           func(*state, *types.Type, int64) *ssa.Value
}

var u64_f64 = u642fcvtTab{
	geq:   ssa.OpGeq64,
	cvt2F: ssa.OpCvt64to64F,
	and:   ssa.OpAnd64,
	rsh:   ssa.OpRsh64Ux64,
	or:    ssa.OpOr64,
	add:   ssa.OpAdd64F,
	one:   (*state).constInt64,
}

var u64_f32 = u642fcvtTab{
	geq:   ssa.OpGeq64,
	cvt2F: ssa.OpCvt64to32F,
	and:   ssa.OpAnd64,
	rsh:   ssa.OpRsh64Ux64,
	or:    ssa.OpOr64,
	add:   ssa.OpAdd32F,
	one:   (*state).constInt64,
}

func (s *state) uint64Tofloat64(n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	return s.uint64Tofloat(&u64_f64, n, x, ft, tt)
}

func (s *state) uint64Tofloat32(n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	return s.uint64Tofloat(&u64_f32, n, x, ft, tt)
}

func (s *state) uint64Tofloat(cvttab *u642fcvtTab, n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
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
	cmp := s.newValue2(cvttab.geq, types.Types[TBOOL], x, s.zeroVal(ft))
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

type u322fcvtTab struct {
	cvtI2F, cvtF2F ssa.Op
}

var u32_f64 = u322fcvtTab{
	cvtI2F: ssa.OpCvt32to64F,
	cvtF2F: ssa.OpCopy,
}

var u32_f32 = u322fcvtTab{
	cvtI2F: ssa.OpCvt32to32F,
	cvtF2F: ssa.OpCvt64Fto32F,
}

func (s *state) uint32Tofloat64(n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	return s.uint32Tofloat(&u32_f64, n, x, ft, tt)
}

func (s *state) uint32Tofloat32(n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	return s.uint32Tofloat(&u32_f32, n, x, ft, tt)
}

func (s *state) uint32Tofloat(cvttab *u322fcvtTab, n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	// if x >= 0 {
	// 	result = floatY(x)
	// } else {
	// 	result = floatY(float64(x) + (1<<32))
	// }
	cmp := s.newValue2(ssa.OpGeq32, types.Types[TBOOL], x, s.zeroVal(ft))
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.SetControl(cmp)
	b.Likely = ssa.BranchLikely

	bThen := s.f.NewBlock(ssa.BlockPlain)
	bElse := s.f.NewBlock(ssa.BlockPlain)
	bAfter := s.f.NewBlock(ssa.BlockPlain)

	b.AddEdgeTo(bThen)
	s.startBlock(bThen)
	a0 := s.newValue1(cvttab.cvtI2F, tt, x)
	s.vars[n] = a0
	s.endBlock()
	bThen.AddEdgeTo(bAfter)

	b.AddEdgeTo(bElse)
	s.startBlock(bElse)
	a1 := s.newValue1(ssa.OpCvt32to64F, types.Types[TFLOAT64], x)
	twoToThe32 := s.constFloat64(types.Types[TFLOAT64], float64(1<<32))
	a2 := s.newValue2(ssa.OpAdd64F, types.Types[TFLOAT64], a1, twoToThe32)
	a3 := s.newValue1(cvttab.cvtF2F, tt, a2)

	s.vars[n] = a3
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
	nilValue := s.constNil(types.Types[TUINTPTR])
	cmp := s.newValue2(ssa.OpEqPtr, types.Types[TBOOL], x, nilValue)
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
	switch n.Op {
	case OLEN:
		// length is stored in the first word for map/chan
		s.vars[n] = s.load(lenType, x)
	case OCAP:
		// capacity is stored in the second word for chan
		sw := s.newValue1I(ssa.OpOffPtr, lenType.PtrTo(), lenType.Width, x)
		s.vars[n] = s.load(lenType, sw)
	default:
		s.Fatalf("op must be OLEN or OCAP")
	}
	s.endBlock()
	bElse.AddEdgeTo(bAfter)

	s.startBlock(bAfter)
	return s.variable(n, lenType)
}

type f2uCvtTab struct {
	ltf, cvt2U, subf, or ssa.Op
	floatValue           func(*state, *types.Type, float64) *ssa.Value
	intValue             func(*state, *types.Type, int64) *ssa.Value
	cutoff               uint64
}

var f32_u64 = f2uCvtTab{
	ltf:        ssa.OpLess32F,
	cvt2U:      ssa.OpCvt32Fto64,
	subf:       ssa.OpSub32F,
	or:         ssa.OpOr64,
	floatValue: (*state).constFloat32,
	intValue:   (*state).constInt64,
	cutoff:     9223372036854775808,
}

var f64_u64 = f2uCvtTab{
	ltf:        ssa.OpLess64F,
	cvt2U:      ssa.OpCvt64Fto64,
	subf:       ssa.OpSub64F,
	or:         ssa.OpOr64,
	floatValue: (*state).constFloat64,
	intValue:   (*state).constInt64,
	cutoff:     9223372036854775808,
}

var f32_u32 = f2uCvtTab{
	ltf:        ssa.OpLess32F,
	cvt2U:      ssa.OpCvt32Fto32,
	subf:       ssa.OpSub32F,
	or:         ssa.OpOr32,
	floatValue: (*state).constFloat32,
	intValue:   func(s *state, t *types.Type, v int64) *ssa.Value { return s.constInt32(t, int32(v)) },
	cutoff:     2147483648,
}

var f64_u32 = f2uCvtTab{
	ltf:        ssa.OpLess64F,
	cvt2U:      ssa.OpCvt64Fto32,
	subf:       ssa.OpSub64F,
	or:         ssa.OpOr32,
	floatValue: (*state).constFloat64,
	intValue:   func(s *state, t *types.Type, v int64) *ssa.Value { return s.constInt32(t, int32(v)) },
	cutoff:     2147483648,
}

func (s *state) float32ToUint64(n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	return s.floatToUint(&f32_u64, n, x, ft, tt)
}
func (s *state) float64ToUint64(n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	return s.floatToUint(&f64_u64, n, x, ft, tt)
}

func (s *state) float32ToUint32(n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	return s.floatToUint(&f32_u32, n, x, ft, tt)
}

func (s *state) float64ToUint32(n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	return s.floatToUint(&f64_u32, n, x, ft, tt)
}

func (s *state) floatToUint(cvttab *f2uCvtTab, n *Node, x *ssa.Value, ft, tt *types.Type) *ssa.Value {
	// cutoff:=1<<(intY_Size-1)
	// if x < floatX(cutoff) {
	// 	result = uintY(x)
	// } else {
	// 	y = x - floatX(cutoff)
	// 	z = uintY(y)
	// 	result = z | -(cutoff)
	// }
	cutoff := cvttab.floatValue(s, ft, float64(cvttab.cutoff))
	cmp := s.newValue2(cvttab.ltf, types.Types[TBOOL], x, cutoff)
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
	y := s.newValue2(cvttab.subf, ft, x, cutoff)
	y = s.newValue1(cvttab.cvt2U, tt, y)
	z := cvttab.intValue(s, tt, int64(-cvttab.cutoff))
	a1 := s.newValue2(cvttab.or, tt, y, z)
	s.vars[n] = a1
	s.endBlock()
	bElse.AddEdgeTo(bAfter)

	s.startBlock(bAfter)
	return s.variable(n, n.Type)
}

// dottype generates SSA for a type assertion node.
// commaok indicates whether to panic or return a bool.
// If commaok is false, resok will be nil.
func (s *state) dottype(n *Node, commaok bool) (res, resok *ssa.Value) {
	iface := s.expr(n.Left)   // input interface
	target := s.expr(n.Right) // target type
	byteptr := s.f.Config.Types.BytePtr

	if n.Type.IsInterface() {
		if n.Type.IsEmptyInterface() {
			// Converting to an empty interface.
			// Input could be an empty or nonempty interface.
			if Debug_typeassert > 0 {
				Warnl(n.Pos, "type assertion inlined")
			}

			// Get itab/type field from input.
			itab := s.newValue1(ssa.OpITab, byteptr, iface)
			// Conversion succeeds iff that field is not nil.
			cond := s.newValue2(ssa.OpNeqPtr, types.Types[TBOOL], itab, s.constNil(byteptr))

			if n.Left.Type.IsEmptyInterface() && commaok {
				// Converting empty interface to empty interface with ,ok is just a nil check.
				return iface, cond
			}

			// Branch on nilness.
			b := s.endBlock()
			b.Kind = ssa.BlockIf
			b.SetControl(cond)
			b.Likely = ssa.BranchLikely
			bOk := s.f.NewBlock(ssa.BlockPlain)
			bFail := s.f.NewBlock(ssa.BlockPlain)
			b.AddEdgeTo(bOk)
			b.AddEdgeTo(bFail)

			if !commaok {
				// On failure, panic by calling panicnildottype.
				s.startBlock(bFail)
				s.rtcall(panicnildottype, false, nil, target)

				// On success, return (perhaps modified) input interface.
				s.startBlock(bOk)
				if n.Left.Type.IsEmptyInterface() {
					res = iface // Use input interface unchanged.
					return
				}
				// Load type out of itab, build interface with existing idata.
				off := s.newValue1I(ssa.OpOffPtr, byteptr, int64(Widthptr), itab)
				typ := s.load(byteptr, off)
				idata := s.newValue1(ssa.OpIData, n.Type, iface)
				res = s.newValue2(ssa.OpIMake, n.Type, typ, idata)
				return
			}

			s.startBlock(bOk)
			// nonempty -> empty
			// Need to load type from itab
			off := s.newValue1I(ssa.OpOffPtr, byteptr, int64(Widthptr), itab)
			s.vars[&typVar] = s.load(byteptr, off)
			s.endBlock()

			// itab is nil, might as well use that as the nil result.
			s.startBlock(bFail)
			s.vars[&typVar] = itab
			s.endBlock()

			// Merge point.
			bEnd := s.f.NewBlock(ssa.BlockPlain)
			bOk.AddEdgeTo(bEnd)
			bFail.AddEdgeTo(bEnd)
			s.startBlock(bEnd)
			idata := s.newValue1(ssa.OpIData, n.Type, iface)
			res = s.newValue2(ssa.OpIMake, n.Type, s.variable(&typVar, byteptr), idata)
			resok = cond
			delete(s.vars, &typVar)
			return
		}
		// converting to a nonempty interface needs a runtime call.
		if Debug_typeassert > 0 {
			Warnl(n.Pos, "type assertion not inlined")
		}
		if n.Left.Type.IsEmptyInterface() {
			if commaok {
				call := s.rtcall(assertE2I2, true, []*types.Type{n.Type, types.Types[TBOOL]}, target, iface)
				return call[0], call[1]
			}
			return s.rtcall(assertE2I, true, []*types.Type{n.Type}, target, iface)[0], nil
		}
		if commaok {
			call := s.rtcall(assertI2I2, true, []*types.Type{n.Type, types.Types[TBOOL]}, target, iface)
			return call[0], call[1]
		}
		return s.rtcall(assertI2I, true, []*types.Type{n.Type}, target, iface)[0], nil
	}

	if Debug_typeassert > 0 {
		Warnl(n.Pos, "type assertion inlined")
	}

	// Converting to a concrete type.
	direct := isdirectiface(n.Type)
	itab := s.newValue1(ssa.OpITab, byteptr, iface) // type word of interface
	if Debug_typeassert > 0 {
		Warnl(n.Pos, "type assertion inlined")
	}
	var targetITab *ssa.Value
	if n.Left.Type.IsEmptyInterface() {
		// Looking for pointer to target type.
		targetITab = target
	} else {
		// Looking for pointer to itab for target type and source interface.
		targetITab = s.expr(n.List.First())
	}

	var tmp *Node       // temporary for use with large types
	var addr *ssa.Value // address of tmp
	if commaok && !canSSAType(n.Type) {
		// unSSAable type, use temporary.
		// TODO: get rid of some of these temporaries.
		tmp = tempAt(n.Pos, s.curfn, n.Type)
		s.vars[&memVar] = s.newValue1A(ssa.OpVarDef, types.TypeMem, tmp, s.mem())
		addr = s.addr(tmp, false)
	}

	cond := s.newValue2(ssa.OpEqPtr, types.Types[TBOOL], itab, targetITab)
	b := s.endBlock()
	b.Kind = ssa.BlockIf
	b.SetControl(cond)
	b.Likely = ssa.BranchLikely

	bOk := s.f.NewBlock(ssa.BlockPlain)
	bFail := s.f.NewBlock(ssa.BlockPlain)
	b.AddEdgeTo(bOk)
	b.AddEdgeTo(bFail)

	if !commaok {
		// on failure, panic by calling panicdottype
		s.startBlock(bFail)
		taddr := s.expr(n.Right.Right)
		if n.Left.Type.IsEmptyInterface() {
			s.rtcall(panicdottypeE, false, nil, itab, target, taddr)
		} else {
			s.rtcall(panicdottypeI, false, nil, itab, target, taddr)
		}

		// on success, return data from interface
		s.startBlock(bOk)
		if direct {
			return s.newValue1(ssa.OpIData, n.Type, iface), nil
		}
		p := s.newValue1(ssa.OpIData, types.NewPtr(n.Type), iface)
		return s.load(n.Type, p), nil
	}

	// commaok is the more complicated case because we have
	// a control flow merge point.
	bEnd := s.f.NewBlock(ssa.BlockPlain)
	// Note that we need a new valVar each time (unlike okVar where we can
	// reuse the variable) because it might have a different type every time.
	valVar := &Node{Op: ONAME, Sym: &types.Sym{Name: "val"}}

	// type assertion succeeded
	s.startBlock(bOk)
	if tmp == nil {
		if direct {
			s.vars[valVar] = s.newValue1(ssa.OpIData, n.Type, iface)
		} else {
			p := s.newValue1(ssa.OpIData, types.NewPtr(n.Type), iface)
			s.vars[valVar] = s.load(n.Type, p)
		}
	} else {
		p := s.newValue1(ssa.OpIData, types.NewPtr(n.Type), iface)
		s.move(n.Type, addr, p)
	}
	s.vars[&okVar] = s.constBool(true)
	s.endBlock()
	bOk.AddEdgeTo(bEnd)

	// type assertion failed
	s.startBlock(bFail)
	if tmp == nil {
		s.vars[valVar] = s.zeroVal(n.Type)
	} else {
		s.zero(n.Type, addr)
	}
	s.vars[&okVar] = s.constBool(false)
	s.endBlock()
	bFail.AddEdgeTo(bEnd)

	// merge point
	s.startBlock(bEnd)
	if tmp == nil {
		res = s.variable(valVar, n.Type)
		delete(s.vars, valVar)
	} else {
		res = s.load(n.Type, addr)
		s.vars[&memVar] = s.newValue1A(ssa.OpVarKill, types.TypeMem, tmp, s.mem())
	}
	resok = s.variable(&okVar, types.Types[TBOOL])
	delete(s.vars, &okVar)
	return res, resok
}

// variable returns the value of a variable at the current location.
func (s *state) variable(name *Node, t *types.Type) *ssa.Value {
	v := s.vars[name]
	if v != nil {
		return v
	}
	v = s.fwdVars[name]
	if v != nil {
		return v
	}

	if s.curBlock == s.f.Entry {
		// No variable should be live at entry.
		s.Fatalf("Value live at entry. It shouldn't be. func %s, node %v, value %v", s.f.Name, name, v)
	}
	// Make a FwdRef, which records a value that's live on block input.
	// We'll find the matching definition as part of insertPhis.
	v = s.newValue0A(ssa.OpFwdRef, t, name)
	s.fwdVars[name] = v
	s.addNamedValue(name, v)
	return v
}

func (s *state) mem() *ssa.Value {
	return s.variable(&memVar, types.TypeMem)
}

func (s *state) addNamedValue(n *Node, v *ssa.Value) {
	if n.Class() == Pxxx {
		// Don't track our dummy nodes (&memVar etc.).
		return
	}
	if n.IsAutoTmp() {
		// Don't track temporary variables.
		return
	}
	if n.Class() == PPARAMOUT {
		// Don't track named output values.  This prevents return values
		// from being assigned too early. See #14591 and #14762. TODO: allow this.
		return
	}
	if n.Class() == PAUTO && n.Xoffset != 0 {
		s.Fatalf("AUTO var with offset %v %d", n, n.Xoffset)
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
	pp *Progs

	// Branches remembers all the branch instructions we've seen
	// and where they would like to go.
	Branches []Branch

	// bstart remembers where each block starts (indexed by block ID)
	bstart []*obj.Prog

	// 387 port: maps from SSE registers (REG_X?) to 387 registers (REG_F?)
	SSEto387 map[int16]int16
	// Some architectures require a 64-bit temporary for FP-related register shuffling. Examples include x86-387, PPC, and Sparc V8.
	ScratchFpMem *Node

	maxarg int64 // largest frame size for arguments to calls made by the function

	// Map from GC safe points to liveness index, generated by
	// liveness analysis.
	livenessMap LivenessMap

	// lineRunStart records the beginning of the current run of instructions
	// within a single block sharing the same line number
	// Used to move statement marks to the beginning of such runs.
	lineRunStart *obj.Prog

	// wasm: The number of values on the WebAssembly stack. This is only used as a safeguard.
	OnWasmStackSkipped int
}

// Prog appends a new Prog.
func (s *SSAGenState) Prog(as obj.As) *obj.Prog {
	p := s.pp.Prog(as)
	if ssa.LosesStmtMark(as) {
		return p
	}
	// Float a statement start to the beginning of any same-line run.
	// lineRunStart is reset at block boundaries, which appears to work well.
	if s.lineRunStart == nil || s.lineRunStart.Pos.Line() != p.Pos.Line() {
		s.lineRunStart = p
	} else if p.Pos.IsStmt() == src.PosIsStmt {
		s.lineRunStart.Pos = s.lineRunStart.Pos.WithIsStmt()
		p.Pos = p.Pos.WithNotStmt()
	}
	return p
}

// Pc returns the current Prog.
func (s *SSAGenState) Pc() *obj.Prog {
	return s.pp.next
}

// SetPos sets the current source position.
func (s *SSAGenState) SetPos(pos src.XPos) {
	s.pp.pos = pos
}

// Br emits a single branch instruction and returns the instruction.
// Not all architectures need the returned instruction, but otherwise
// the boilerplate is common to all.
func (s *SSAGenState) Br(op obj.As, target *ssa.Block) *obj.Prog {
	p := s.Prog(op)
	p.To.Type = obj.TYPE_BRANCH
	s.Branches = append(s.Branches, Branch{P: p, B: target})
	return p
}

// DebugFriendlySetPos adjusts Pos.IsStmt subject to heuristics
// that reduce "jumpy" line number churn when debugging.
// Spill/fill/copy instructions from the register allocator,
// phi functions, and instructions with a no-pos position
// are examples of instructions that can cause churn.
func (s *SSAGenState) DebugFriendlySetPosFrom(v *ssa.Value) {
	switch v.Op {
	case ssa.OpPhi, ssa.OpCopy, ssa.OpLoadReg, ssa.OpStoreReg:
		// These are not statements
		s.SetPos(v.Pos.WithNotStmt())
	default:
		p := v.Pos
		if p != src.NoXPos {
			// If the position is defined, update the position.
			// Also convert default IsStmt to NotStmt; only
			// explicit statement boundaries should appear
			// in the generated code.
			if p.IsStmt() != src.PosIsStmt {
				p = p.WithNotStmt()
			}
			s.SetPos(p)
		}
	}
}

// byXoffset implements sort.Interface for []*Node using Xoffset as the ordering.
type byXoffset []*Node

func (s byXoffset) Len() int           { return len(s) }
func (s byXoffset) Less(i, j int) bool { return s[i].Xoffset < s[j].Xoffset }
func (s byXoffset) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func emitStackObjects(e *ssafn, pp *Progs) {
	var vars []*Node
	for _, n := range e.curfn.Func.Dcl {
		if livenessShouldTrack(n) && n.Addrtaken() {
			vars = append(vars, n)
		}
	}
	if len(vars) == 0 {
		return
	}

	// Sort variables from lowest to highest address.
	sort.Sort(byXoffset(vars))

	// Populate the stack object data.
	// Format must match runtime/stack.go:stackObjectRecord.
	x := e.curfn.Func.lsym.Func.StackObjects
	off := 0
	off = duintptr(x, off, uint64(len(vars)))
	for _, v := range vars {
		// Note: arguments and return values have non-negative Xoffset,
		// in which case the offset is relative to argp.
		// Locals have a negative Xoffset, in which case the offset is relative to varp.
		off = duintptr(x, off, uint64(v.Xoffset))
		if !typesym(v.Type).Siggen() {
			Fatalf("stack object's type symbol not generated for type %s", v.Type)
		}
		off = dsymptr(x, off, dtypesym(v.Type), 0)
	}

	// Emit a funcdata pointing at the stack object data.
	p := pp.Prog(obj.AFUNCDATA)
	Addrconst(&p.From, objabi.FUNCDATA_StackObjects)
	p.To.Type = obj.TYPE_MEM
	p.To.Name = obj.NAME_EXTERN
	p.To.Sym = x

	if debuglive != 0 {
		for _, v := range vars {
			Warnl(v.Pos, "stack object %v %s", v, v.Type.String())
		}
	}
}

// genssa appends entries to pp for each instruction in f.
func genssa(f *ssa.Func, pp *Progs) {
	var s SSAGenState

	e := f.Frontend().(*ssafn)

	s.livenessMap = liveness(e, f, pp)
	emitStackObjects(e, pp)

	// Remember where each block starts.
	s.bstart = make([]*obj.Prog, f.NumBlocks())
	s.pp = pp
	var progToValue map[*obj.Prog]*ssa.Value
	var progToBlock map[*obj.Prog]*ssa.Block
	var valueToProgAfter []*obj.Prog // The first Prog following computation of a value v; v is visible at this point.
	if f.PrintOrHtmlSSA {
		progToValue = make(map[*obj.Prog]*ssa.Value, f.NumValues())
		progToBlock = make(map[*obj.Prog]*ssa.Block, f.NumBlocks())
		f.Logf("genssa %s\n", f.Name)
		progToBlock[s.pp.next] = f.Blocks[0]
	}

	if thearch.Use387 {
		s.SSEto387 = map[int16]int16{}
	}

	s.ScratchFpMem = e.scratchFpMem

	if Ctxt.Flag_locationlists {
		if cap(f.Cache.ValueToProgAfter) < f.NumValues() {
			f.Cache.ValueToProgAfter = make([]*obj.Prog, f.NumValues())
		}
		valueToProgAfter = f.Cache.ValueToProgAfter[:f.NumValues()]
		for i := range valueToProgAfter {
			valueToProgAfter[i] = nil
		}
	}

	// If the very first instruction is not tagged as a statement,
	// debuggers may attribute it to previous function in program.
	firstPos := src.NoXPos
	for _, v := range f.Entry.Values {
		if v.Pos.IsStmt() == src.PosIsStmt {
			firstPos = v.Pos
			v.Pos = firstPos.WithDefaultStmt()
			break
		}
	}

	// Emit basic blocks
	for i, b := range f.Blocks {
		s.bstart[b.ID] = s.pp.next
		s.pp.nextLive = LivenessInvalid
		s.lineRunStart = nil

		// Emit values in block
		thearch.SSAMarkMoves(&s, b)
		for _, v := range b.Values {
			x := s.pp.next
			s.DebugFriendlySetPosFrom(v)
			// Attach this safe point to the next
			// instruction.
			s.pp.nextLive = s.livenessMap.Get(v)
			switch v.Op {
			case ssa.OpInitMem:
				// memory arg needs no code
			case ssa.OpArg:
				// input args need no code
			case ssa.OpSP, ssa.OpSB:
				// nothing to do
			case ssa.OpSelect0, ssa.OpSelect1:
				// nothing to do
			case ssa.OpGetG:
				// nothing to do when there's a g register,
				// and checkLower complains if there's not
			case ssa.OpVarDef, ssa.OpVarLive, ssa.OpKeepAlive, ssa.OpVarKill:
				// nothing to do; already used by liveness
			case ssa.OpPhi:
				CheckLoweredPhi(v)
			case ssa.OpConvert:
				// nothing to do; no-op conversion for liveness
				if v.Args[0].Reg() != v.Reg() {
					v.Fatalf("OpConvert should be a no-op: %s; %s", v.Args[0].LongString(), v.LongString())
				}
			default:
				// let the backend handle it
				// Special case for first line in function; move it to the start.
				if firstPos != src.NoXPos {
					s.SetPos(firstPos)
					firstPos = src.NoXPos
				}
				thearch.SSAGenValue(&s, v)
			}

			if Ctxt.Flag_locationlists {
				valueToProgAfter[v.ID] = s.pp.next
			}

			if f.PrintOrHtmlSSA {
				for ; x != s.pp.next; x = x.Link {
					progToValue[x] = v
				}
			}
		}
		// Emit control flow instructions for block
		var next *ssa.Block
		if i < len(f.Blocks)-1 && Debug['N'] == 0 {
			// If -N, leave next==nil so every block with successors
			// ends in a JMP (except call blocks - plive doesn't like
			// select{send,recv} followed by a JMP call).  Helps keep
			// line numbers for otherwise empty blocks.
			next = f.Blocks[i+1]
		}
		x := s.pp.next
		s.SetPos(b.Pos)
		thearch.SSAGenBlock(&s, b, next)
		if f.PrintOrHtmlSSA {
			for ; x != s.pp.next; x = x.Link {
				progToBlock[x] = b
			}
		}
	}

	if Ctxt.Flag_locationlists {
		e.curfn.Func.DebugInfo = ssa.BuildFuncDebug(Ctxt, f, Debug_locationlist > 1, stackOffset)
		bstart := s.bstart
		// Note that at this moment, Prog.Pc is a sequence number; it's
		// not a real PC until after assembly, so this mapping has to
		// be done later.
		e.curfn.Func.DebugInfo.GetPC = func(b, v ssa.ID) int64 {
			switch v {
			case ssa.BlockStart.ID:
				return bstart[b].Pc
			case ssa.BlockEnd.ID:
				return e.curfn.Func.lsym.Size
			default:
				return valueToProgAfter[v].Pc
			}
		}
	}

	// Resolove branchers, and relax DefaultStmt into NotStmt
	for _, br := range s.Branches {
		br.P.To.Val = s.bstart[br.B.ID]
		if br.P.Pos.IsStmt() != src.PosIsStmt {
			br.P.Pos = br.P.Pos.WithNotStmt()
		}
	}

	if e.log { // spew to stdout
		filename := ""
		for p := pp.Text; p != nil; p = p.Link {
			if p.Pos.IsKnown() && p.InnermostFilename() != filename {
				filename = p.InnermostFilename()
				f.Logf("# %s\n", filename)
			}

			var s string
			if v, ok := progToValue[p]; ok {
				s = v.String()
			} else if b, ok := progToBlock[p]; ok {
				s = b.String()
			} else {
				s = "   " // most value and branch strings are 2-3 characters long
			}
			f.Logf(" %-6s\t%.5d (%s)\t%s\n", s, p.Pc, p.InnermostLineNumber(), p.InstructionString())
		}
	}
	if f.HTMLWriter != nil { // spew to ssa.html
		var buf bytes.Buffer
		buf.WriteString("<code>")
		buf.WriteString("<dl class=\"ssa-gen\">")
		filename := ""
		for p := pp.Text; p != nil; p = p.Link {
			// Don't spam every line with the file name, which is often huge.
			// Only print changes, and "unknown" is not a change.
			if p.Pos.IsKnown() && p.InnermostFilename() != filename {
				filename = p.InnermostFilename()
				buf.WriteString("<dt class=\"ssa-prog-src\"></dt><dd class=\"ssa-prog\">")
				buf.WriteString(html.EscapeString("# " + filename))
				buf.WriteString("</dd>")
			}

			buf.WriteString("<dt class=\"ssa-prog-src\">")
			if v, ok := progToValue[p]; ok {
				buf.WriteString(v.HTML())
			} else if b, ok := progToBlock[p]; ok {
				buf.WriteString("<b>" + b.HTML() + "</b>")
			}
			buf.WriteString("</dt>")
			buf.WriteString("<dd class=\"ssa-prog\">")
			buf.WriteString(fmt.Sprintf("%.5d <span class=\"l%v line-number\">(%s)</span> %s", p.Pc, p.InnermostLineNumber(), p.InnermostLineNumberHTML(), html.EscapeString(p.InstructionString())))
			buf.WriteString("</dd>")
		}
		buf.WriteString("</dl>")
		buf.WriteString("</code>")
		f.HTMLWriter.WriteColumn("genssa", "genssa", "ssa-prog", buf.String())
	}

	defframe(&s, e)

	f.HTMLWriter.Close()
	f.HTMLWriter = nil
}

func defframe(s *SSAGenState, e *ssafn) {
	pp := s.pp

	frame := Rnd(s.maxarg+e.stksize, int64(Widthreg))
	if thearch.PadFrame != nil {
		frame = thearch.PadFrame(frame)
	}

	// Fill in argument and frame size.
	pp.Text.To.Type = obj.TYPE_TEXTSIZE
	pp.Text.To.Val = int32(Rnd(e.curfn.Type.ArgWidth(), int64(Widthreg)))
	pp.Text.To.Offset = frame

	// Insert code to zero ambiguously live variables so that the
	// garbage collector only sees initialized values when it
	// looks for pointers.
	p := pp.Text
	var lo, hi int64

	// Opaque state for backend to use. Current backends use it to
	// keep track of which helper registers have been zeroed.
	var state uint32

	// Iterate through declarations. They are sorted in decreasing Xoffset order.
	for _, n := range e.curfn.Func.Dcl {
		if !n.Name.Needzero() {
			continue
		}
		if n.Class() != PAUTO {
			Fatalf("needzero class %d", n.Class())
		}
		if n.Type.Size()%int64(Widthptr) != 0 || n.Xoffset%int64(Widthptr) != 0 || n.Type.Size() == 0 {
			Fatalf("var %L has size %d offset %d", n, n.Type.Size(), n.Xoffset)
		}

		if lo != hi && n.Xoffset+n.Type.Size() >= lo-int64(2*Widthreg) {
			// Merge with range we already have.
			lo = n.Xoffset
			continue
		}

		// Zero old range
		p = thearch.ZeroRange(pp, p, frame+lo, hi-lo, &state)

		// Set new range.
		lo = n.Xoffset
		hi = lo + n.Type.Size()
	}

	// Zero final range.
	thearch.ZeroRange(pp, p, frame+lo, hi-lo, &state)
}

type FloatingEQNEJump struct {
	Jump  obj.As
	Index int
}

func (s *SSAGenState) oneFPJump(b *ssa.Block, jumps *FloatingEQNEJump) {
	p := s.Prog(jumps.Jump)
	p.To.Type = obj.TYPE_BRANCH
	p.Pos = b.Pos
	to := jumps.Index
	s.Branches = append(s.Branches, Branch{p, b.Succs[to].Block()})
}

func (s *SSAGenState) FPJump(b, next *ssa.Block, jumps *[2][2]FloatingEQNEJump) {
	switch next {
	case b.Succs[0].Block():
		s.oneFPJump(b, &jumps[0][0])
		s.oneFPJump(b, &jumps[0][1])
	case b.Succs[1].Block():
		s.oneFPJump(b, &jumps[1][0])
		s.oneFPJump(b, &jumps[1][1])
	default:
		s.oneFPJump(b, &jumps[1][0])
		s.oneFPJump(b, &jumps[1][1])
		q := s.Prog(obj.AJMP)
		q.Pos = b.Pos
		q.To.Type = obj.TYPE_BRANCH
		s.Branches = append(s.Branches, Branch{q, b.Succs[1].Block()})
	}
}

func AuxOffset(v *ssa.Value) (offset int64) {
	if v.Aux == nil {
		return 0
	}
	n, ok := v.Aux.(*Node)
	if !ok {
		v.Fatalf("bad aux type in %s\n", v.LongString())
	}
	if n.Class() == PAUTO {
		return n.Xoffset
	}
	return 0
}

// AddAux adds the offset in the aux fields (AuxInt and Aux) of v to a.
func AddAux(a *obj.Addr, v *ssa.Value) {
	AddAux2(a, v, v.AuxInt)
}
func AddAux2(a *obj.Addr, v *ssa.Value, offset int64) {
	if a.Type != obj.TYPE_MEM && a.Type != obj.TYPE_ADDR {
		v.Fatalf("bad AddAux addr %v", a)
	}
	// add integer offset
	a.Offset += offset

	// If no additional symbol offset, we're done.
	if v.Aux == nil {
		return
	}
	// Add symbol's offset from its base register.
	switch n := v.Aux.(type) {
	case *obj.LSym:
		a.Name = obj.NAME_EXTERN
		a.Sym = n
	case *Node:
		if n.Class() == PPARAM || n.Class() == PPARAMOUT {
			a.Name = obj.NAME_PARAM
			a.Sym = n.Orig.Sym.Linksym()
			a.Offset += n.Xoffset
			break
		}
		a.Name = obj.NAME_AUTO
		a.Sym = n.Sym.Linksym()
		a.Offset += n.Xoffset
	default:
		v.Fatalf("aux in %s not implemented %#v", v, v.Aux)
	}
}

// extendIndex extends v to a full int width.
// panic using the given function if v does not fit in an int (only on 32-bit archs).
func (s *state) extendIndex(v *ssa.Value, panicfn *obj.LSym) *ssa.Value {
	size := v.Type.Size()
	if size == s.config.PtrSize {
		return v
	}
	if size > s.config.PtrSize {
		// truncate 64-bit indexes on 32-bit pointer archs. Test the
		// high word and branch to out-of-bounds failure if it is not 0.
		if Debug['B'] == 0 {
			hi := s.newValue1(ssa.OpInt64Hi, types.Types[TUINT32], v)
			cmp := s.newValue2(ssa.OpEq32, types.Types[TBOOL], hi, s.constInt32(types.Types[TUINT32], 0))
			s.check(cmp, panicfn)
		}
		return s.newValue1(ssa.OpTrunc64to32, types.Types[TINT], v)
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
	return s.newValue1(op, types.Types[TINT], v)
}

// CheckLoweredPhi checks that regalloc and stackalloc correctly handled phi values.
// Called during ssaGenValue.
func CheckLoweredPhi(v *ssa.Value) {
	if v.Op != ssa.OpPhi {
		v.Fatalf("CheckLoweredPhi called with non-phi value: %v", v.LongString())
	}
	if v.Type.IsMemory() {
		return
	}
	f := v.Block.Func
	loc := f.RegAlloc[v.ID]
	for _, a := range v.Args {
		if aloc := f.RegAlloc[a.ID]; aloc != loc { // TODO: .Equal() instead?
			v.Fatalf("phi arg at different location than phi: %v @ %s, but arg %v @ %s\n%s\n", v, loc, a, aloc, v.Block.Func)
		}
	}
}

// CheckLoweredGetClosurePtr checks that v is the first instruction in the function's entry block.
// The output of LoweredGetClosurePtr is generally hardwired to the correct register.
// That register contains the closure pointer on closure entry.
func CheckLoweredGetClosurePtr(v *ssa.Value) {
	entry := v.Block.Func.Entry
	if entry != v.Block || entry.Values[0] != v {
		Fatalf("in %s, badly placed LoweredGetClosurePtr: %v %v", v.Block.Func.Name, v.Block, v)
	}
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

func AddrAuto(a *obj.Addr, v *ssa.Value) {
	n, off := AutoVar(v)
	a.Type = obj.TYPE_MEM
	a.Sym = n.Sym.Linksym()
	a.Reg = int16(thearch.REGSP)
	a.Offset = n.Xoffset + off
	if n.Class() == PPARAM || n.Class() == PPARAMOUT {
		a.Name = obj.NAME_PARAM
	} else {
		a.Name = obj.NAME_AUTO
	}
}

func (s *SSAGenState) AddrScratch(a *obj.Addr) {
	if s.ScratchFpMem == nil {
		panic("no scratch memory available; forgot to declare usesScratch for Op?")
	}
	a.Type = obj.TYPE_MEM
	a.Name = obj.NAME_AUTO
	a.Sym = s.ScratchFpMem.Sym.Linksym()
	a.Reg = int16(thearch.REGSP)
	a.Offset = s.ScratchFpMem.Xoffset
}

// Call returns a new CALL instruction for the SSA value v.
// It uses PrepareCall to prepare the call.
func (s *SSAGenState) Call(v *ssa.Value) *obj.Prog {
	s.PrepareCall(v)

	p := s.Prog(obj.ACALL)
	if sym, ok := v.Aux.(*obj.LSym); ok {
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = sym
	} else {
		// TODO(mdempsky): Can these differences be eliminated?
		switch thearch.LinkArch.Family {
		case sys.AMD64, sys.I386, sys.PPC64, sys.S390X, sys.Wasm:
			p.To.Type = obj.TYPE_REG
		case sys.ARM, sys.ARM64, sys.MIPS, sys.MIPS64:
			p.To.Type = obj.TYPE_MEM
		default:
			Fatalf("unknown indirect call family")
		}
		p.To.Reg = v.Args[0].Reg()
	}
	return p
}

// PrepareCall prepares to emit a CALL instruction for v and does call-related bookkeeping.
// It must be called immediately before emitting the actual CALL instruction,
// since it emits PCDATA for the stack map at the call (calls are safe points).
func (s *SSAGenState) PrepareCall(v *ssa.Value) {
	idx := s.livenessMap.Get(v)
	if !idx.Valid() {
		// typedmemclr and typedmemmove are write barriers and
		// deeply non-preemptible. They are unsafe points and
		// hence should not have liveness maps.
		if sym, _ := v.Aux.(*obj.LSym); !(sym == typedmemclr || sym == typedmemmove) {
			Fatalf("missing stack map index for %v", v.LongString())
		}
	}

	if sym, _ := v.Aux.(*obj.LSym); sym == Deferreturn {
		// Deferred calls will appear to be returning to
		// the CALL deferreturn(SB) that we are about to emit.
		// However, the stack trace code will show the line
		// of the instruction byte before the return PC.
		// To avoid that being an unrelated instruction,
		// insert an actual hardware NOP that will have the right line number.
		// This is different from obj.ANOP, which is a virtual no-op
		// that doesn't make it into the instruction stream.
		thearch.Ginsnop(s.pp)
	}

	if sym, ok := v.Aux.(*obj.LSym); ok {
		// Record call graph information for nowritebarrierrec
		// analysis.
		if nowritebarrierrecCheck != nil {
			nowritebarrierrecCheck.recordCall(s.pp.curfn, sym, v.Pos)
		}
	}

	if s.maxarg < v.AuxInt {
		s.maxarg = v.AuxInt
	}
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
	panic(fmt.Sprintf("can't find field in expr %v\n", n))

	// TODO: keep the result of this function somewhere in the ODOT Node
	// so we don't have to recompute it each time we need it.
}

// ssafn holds frontend information about a function that the backend is processing.
// It also exports a bunch of compiler services for the ssa backend.
type ssafn struct {
	curfn        *Node
	strings      map[string]interface{} // map from constant string to data symbols
	scratchFpMem *Node                  // temp for floating point register / memory moves on some architectures
	stksize      int64                  // stack size for current frame
	stkptrsize   int64                  // prefix of stack containing pointers
	log          bool                   // print ssa debug to the stdout
}

// StringData returns a symbol (a *types.Sym wrapped in an interface) which
// is the data component of a global string constant containing s.
func (e *ssafn) StringData(s string) interface{} {
	if aux, ok := e.strings[s]; ok {
		return aux
	}
	if e.strings == nil {
		e.strings = make(map[string]interface{})
	}
	data := stringsym(e.curfn.Pos, s)
	e.strings[s] = data
	return data
}

func (e *ssafn) Auto(pos src.XPos, t *types.Type) ssa.GCNode {
	n := tempAt(pos, e.curfn, t) // Note: adds new auto to e.curfn.Func.Dcl list
	return n
}

func (e *ssafn) SplitString(name ssa.LocalSlot) (ssa.LocalSlot, ssa.LocalSlot) {
	n := name.N.(*Node)
	ptrType := types.NewPtr(types.Types[TUINT8])
	lenType := types.Types[TINT]
	if n.Class() == PAUTO && !n.Addrtaken() {
		// Split this string up into two separate variables.
		p := e.splitSlot(&name, ".ptr", 0, ptrType)
		l := e.splitSlot(&name, ".len", ptrType.Size(), lenType)
		return p, l
	}
	// Return the two parts of the larger variable.
	return ssa.LocalSlot{N: n, Type: ptrType, Off: name.Off}, ssa.LocalSlot{N: n, Type: lenType, Off: name.Off + int64(Widthptr)}
}

func (e *ssafn) SplitInterface(name ssa.LocalSlot) (ssa.LocalSlot, ssa.LocalSlot) {
	n := name.N.(*Node)
	u := types.Types[TUINTPTR]
	t := types.NewPtr(types.Types[TUINT8])
	if n.Class() == PAUTO && !n.Addrtaken() {
		// Split this interface up into two separate variables.
		f := ".itab"
		if n.Type.IsEmptyInterface() {
			f = ".type"
		}
		c := e.splitSlot(&name, f, 0, u) // see comment in plive.go:onebitwalktype1.
		d := e.splitSlot(&name, ".data", u.Size(), t)
		return c, d
	}
	// Return the two parts of the larger variable.
	return ssa.LocalSlot{N: n, Type: u, Off: name.Off}, ssa.LocalSlot{N: n, Type: t, Off: name.Off + int64(Widthptr)}
}

func (e *ssafn) SplitSlice(name ssa.LocalSlot) (ssa.LocalSlot, ssa.LocalSlot, ssa.LocalSlot) {
	n := name.N.(*Node)
	ptrType := types.NewPtr(name.Type.Elem())
	lenType := types.Types[TINT]
	if n.Class() == PAUTO && !n.Addrtaken() {
		// Split this slice up into three separate variables.
		p := e.splitSlot(&name, ".ptr", 0, ptrType)
		l := e.splitSlot(&name, ".len", ptrType.Size(), lenType)
		c := e.splitSlot(&name, ".cap", ptrType.Size()+lenType.Size(), lenType)
		return p, l, c
	}
	// Return the three parts of the larger variable.
	return ssa.LocalSlot{N: n, Type: ptrType, Off: name.Off},
		ssa.LocalSlot{N: n, Type: lenType, Off: name.Off + int64(Widthptr)},
		ssa.LocalSlot{N: n, Type: lenType, Off: name.Off + int64(2*Widthptr)}
}

func (e *ssafn) SplitComplex(name ssa.LocalSlot) (ssa.LocalSlot, ssa.LocalSlot) {
	n := name.N.(*Node)
	s := name.Type.Size() / 2
	var t *types.Type
	if s == 8 {
		t = types.Types[TFLOAT64]
	} else {
		t = types.Types[TFLOAT32]
	}
	if n.Class() == PAUTO && !n.Addrtaken() {
		// Split this complex up into two separate variables.
		r := e.splitSlot(&name, ".real", 0, t)
		i := e.splitSlot(&name, ".imag", t.Size(), t)
		return r, i
	}
	// Return the two parts of the larger variable.
	return ssa.LocalSlot{N: n, Type: t, Off: name.Off}, ssa.LocalSlot{N: n, Type: t, Off: name.Off + s}
}

func (e *ssafn) SplitInt64(name ssa.LocalSlot) (ssa.LocalSlot, ssa.LocalSlot) {
	n := name.N.(*Node)
	var t *types.Type
	if name.Type.IsSigned() {
		t = types.Types[TINT32]
	} else {
		t = types.Types[TUINT32]
	}
	if n.Class() == PAUTO && !n.Addrtaken() {
		// Split this int64 up into two separate variables.
		if thearch.LinkArch.ByteOrder == binary.BigEndian {
			return e.splitSlot(&name, ".hi", 0, t), e.splitSlot(&name, ".lo", t.Size(), types.Types[TUINT32])
		}
		return e.splitSlot(&name, ".hi", t.Size(), t), e.splitSlot(&name, ".lo", 0, types.Types[TUINT32])
	}
	// Return the two parts of the larger variable.
	if thearch.LinkArch.ByteOrder == binary.BigEndian {
		return ssa.LocalSlot{N: n, Type: t, Off: name.Off}, ssa.LocalSlot{N: n, Type: types.Types[TUINT32], Off: name.Off + 4}
	}
	return ssa.LocalSlot{N: n, Type: t, Off: name.Off + 4}, ssa.LocalSlot{N: n, Type: types.Types[TUINT32], Off: name.Off}
}

func (e *ssafn) SplitStruct(name ssa.LocalSlot, i int) ssa.LocalSlot {
	n := name.N.(*Node)
	st := name.Type
	ft := st.FieldType(i)
	var offset int64
	for f := 0; f < i; f++ {
		offset += st.FieldType(f).Size()
	}
	if n.Class() == PAUTO && !n.Addrtaken() {
		// Note: the _ field may appear several times.  But
		// have no fear, identically-named but distinct Autos are
		// ok, albeit maybe confusing for a debugger.
		return e.splitSlot(&name, "."+st.FieldName(i), offset, ft)
	}
	return ssa.LocalSlot{N: n, Type: ft, Off: name.Off + st.FieldOff(i)}
}

func (e *ssafn) SplitArray(name ssa.LocalSlot) ssa.LocalSlot {
	n := name.N.(*Node)
	at := name.Type
	if at.NumElem() != 1 {
		Fatalf("bad array size")
	}
	et := at.Elem()
	if n.Class() == PAUTO && !n.Addrtaken() {
		return e.splitSlot(&name, "[0]", 0, et)
	}
	return ssa.LocalSlot{N: n, Type: et, Off: name.Off}
}

func (e *ssafn) DerefItab(it *obj.LSym, offset int64) *obj.LSym {
	return itabsym(it, offset)
}

// splitSlot returns a slot representing the data of parent starting at offset.
func (e *ssafn) splitSlot(parent *ssa.LocalSlot, suffix string, offset int64, t *types.Type) ssa.LocalSlot {
	s := &types.Sym{Name: parent.N.(*Node).Sym.Name + suffix, Pkg: localpkg}

	n := &Node{
		Name: new(Name),
		Op:   ONAME,
		Pos:  parent.N.(*Node).Pos,
	}
	n.Orig = n

	s.Def = asTypesNode(n)
	asNode(s.Def).Name.SetUsed(true)
	n.Sym = s
	n.Type = t
	n.SetClass(PAUTO)
	n.SetAddable(true)
	n.Esc = EscNever
	n.Name.Curfn = e.curfn
	e.curfn.Func.Dcl = append(e.curfn.Func.Dcl, n)
	dowidth(t)
	return ssa.LocalSlot{N: n, Type: t, Off: 0, SplitOf: parent, SplitOffset: offset}
}

func (e *ssafn) CanSSA(t *types.Type) bool {
	return canSSAType(t)
}

func (e *ssafn) Line(pos src.XPos) string {
	return linestr(pos)
}

// Log logs a message from the compiler.
func (e *ssafn) Logf(msg string, args ...interface{}) {
	if e.log {
		fmt.Printf(msg, args...)
	}
}

func (e *ssafn) Log() bool {
	return e.log
}

// Fatal reports a compiler error and exits.
func (e *ssafn) Fatalf(pos src.XPos, msg string, args ...interface{}) {
	lineno = pos
	nargs := append([]interface{}{e.curfn.funcname()}, args...)
	Fatalf("'%s': "+msg, nargs...)
}

// Warnl reports a "warning", which is usually flag-triggered
// logging output for the benefit of tests.
func (e *ssafn) Warnl(pos src.XPos, fmt_ string, args ...interface{}) {
	Warnl(pos, fmt_, args...)
}

func (e *ssafn) Debug_checknil() bool {
	return Debug_checknil != 0
}

func (e *ssafn) UseWriteBarrier() bool {
	return use_writebarrier
}

func (e *ssafn) Syslook(name string) *obj.LSym {
	switch name {
	case "goschedguarded":
		return goschedguarded
	case "writeBarrier":
		return writeBarrier
	case "gcWriteBarrier":
		return gcWriteBarrier
	case "typedmemmove":
		return typedmemmove
	case "typedmemclr":
		return typedmemclr
	}
	Fatalf("unknown Syslook func %v", name)
	return nil
}

func (e *ssafn) SetWBPos(pos src.XPos) {
	e.curfn.Func.setWBPos(pos)
}

func (n *Node) Typ() *types.Type {
	return n.Type
}
func (n *Node) StorageClass() ssa.StorageClass {
	switch n.Class() {
	case PPARAM:
		return ssa.ClassParam
	case PPARAMOUT:
		return ssa.ClassParamOut
	case PAUTO:
		return ssa.ClassAuto
	default:
		Fatalf("untranslatable storage class for %v: %s", n, n.Class())
		return 0
	}
}
