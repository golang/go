// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run mkbuiltin.go

package gc

import (
	"bufio"
	"bytes"
	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/noder"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/staticdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"runtime"
	"sort"
	"strings"
)

func hidePanic() {
	if base.Debug.Panic == 0 && base.Errors() > 0 {
		// If we've already complained about things
		// in the program, don't bother complaining
		// about a panic too; let the user clean up
		// the code and try again.
		if err := recover(); err != nil {
			if err == "-h" {
				panic(err)
			}
			base.ErrorExit()
		}
	}
}

// Main parses flags and Go source files specified in the command-line
// arguments, type-checks the parsed Go package, compiles functions to machine
// code, and finally writes the compiled package definition to disk.
func Main(archInit func(*Arch)) {
	base.Timer.Start("fe", "init")

	defer hidePanic()

	archInit(&thearch)

	base.Ctxt = obj.Linknew(thearch.LinkArch)
	base.Ctxt.DiagFunc = base.Errorf
	base.Ctxt.DiagFlush = base.FlushErrors
	base.Ctxt.Bso = bufio.NewWriter(os.Stdout)

	// UseBASEntries is preferred because it shaves about 2% off build time, but LLDB, dsymutil, and dwarfdump
	// on Darwin don't support it properly, especially since macOS 10.14 (Mojave).  This is exposed as a flag
	// to allow testing with LLVM tools on Linux, and to help with reporting this bug to the LLVM project.
	// See bugs 31188 and 21945 (CLs 170638, 98075, 72371).
	base.Ctxt.UseBASEntries = base.Ctxt.Headtype != objabi.Hdarwin

	types.LocalPkg = types.NewPkg("", "")
	types.LocalPkg.Prefix = "\"\""

	// We won't know localpkg's height until after import
	// processing. In the mean time, set to MaxPkgHeight to ensure
	// height comparisons at least work until then.
	types.LocalPkg.Height = types.MaxPkgHeight

	// pseudo-package, for scoping
	types.BuiltinPkg = types.NewPkg("go.builtin", "") // TODO(gri) name this package go.builtin?
	types.BuiltinPkg.Prefix = "go.builtin"            // not go%2ebuiltin

	// pseudo-package, accessed by import "unsafe"
	ir.Pkgs.Unsafe = types.NewPkg("unsafe", "unsafe")

	// Pseudo-package that contains the compiler's builtin
	// declarations for package runtime. These are declared in a
	// separate package to avoid conflicts with package runtime's
	// actual declarations, which may differ intentionally but
	// insignificantly.
	ir.Pkgs.Runtime = types.NewPkg("go.runtime", "runtime")
	ir.Pkgs.Runtime.Prefix = "runtime"

	// pseudo-packages used in symbol tables
	ir.Pkgs.Itab = types.NewPkg("go.itab", "go.itab")
	ir.Pkgs.Itab.Prefix = "go.itab" // not go%2eitab

	ir.Pkgs.Itablink = types.NewPkg("go.itablink", "go.itablink")
	ir.Pkgs.Itablink.Prefix = "go.itablink" // not go%2eitablink

	ir.Pkgs.Track = types.NewPkg("go.track", "go.track")
	ir.Pkgs.Track.Prefix = "go.track" // not go%2etrack

	// pseudo-package used for map zero values
	ir.Pkgs.Map = types.NewPkg("go.map", "go.map")
	ir.Pkgs.Map.Prefix = "go.map"

	// pseudo-package used for methods with anonymous receivers
	ir.Pkgs.Go = types.NewPkg("go", "")

	base.DebugSSA = ssa.PhaseOption
	base.ParseFlags()

	// Record flags that affect the build result. (And don't
	// record flags that don't, since that would cause spurious
	// changes in the binary.)
	recordFlags("B", "N", "l", "msan", "race", "shared", "dynlink", "dwarflocationlists", "dwarfbasentries", "smallframes", "spectre")

	if !base.EnableTrace && base.Flag.LowerT {
		log.Fatalf("compiler not built with support for -t")
	}

	// Enable inlining (after recordFlags, to avoid recording the rewritten -l).  For now:
	//	default: inlining on.  (Flag.LowerL == 1)
	//	-l: inlining off  (Flag.LowerL == 0)
	//	-l=2, -l=3: inlining on again, with extra debugging (Flag.LowerL > 1)
	if base.Flag.LowerL <= 1 {
		base.Flag.LowerL = 1 - base.Flag.LowerL
	}

	if base.Flag.SmallFrames {
		ir.MaxStackVarSize = 128 * 1024
		ir.MaxImplicitStackVarSize = 16 * 1024
	}

	if base.Flag.Dwarf {
		base.Ctxt.DebugInfo = debuginfo
		base.Ctxt.GenAbstractFunc = genAbstractFunc
		base.Ctxt.DwFixups = obj.NewDwarfFixupTable(base.Ctxt)
	} else {
		// turn off inline generation if no dwarf at all
		base.Flag.GenDwarfInl = 0
		base.Ctxt.Flag_locationlists = false
	}
	if base.Ctxt.Flag_locationlists && len(base.Ctxt.Arch.DWARFRegisters) == 0 {
		log.Fatalf("location lists requested but register mapping not available on %v", base.Ctxt.Arch.Name)
	}

	types.ParseLangFlag()

	if base.Flag.SymABIs != "" {
		readSymABIs(base.Flag.SymABIs, base.Ctxt.Pkgpath)
	}

	if base.Compiling(base.NoInstrumentPkgs) {
		base.Flag.Race = false
		base.Flag.MSan = false
	}

	thearch.LinkArch.Init(base.Ctxt)
	startProfile()
	if base.Flag.Race {
		ir.Pkgs.Race = types.NewPkg("runtime/race", "")
	}
	if base.Flag.MSan {
		ir.Pkgs.Msan = types.NewPkg("runtime/msan", "")
	}
	if base.Flag.Race || base.Flag.MSan {
		base.Flag.Cfg.Instrumenting = true
	}
	if base.Flag.Dwarf {
		dwarf.EnableLogging(base.Debug.DwarfInl != 0)
	}
	if base.Debug.SoftFloat != 0 {
		thearch.SoftFloat = true
	}

	if base.Flag.JSON != "" { // parse version,destination from json logging optimization.
		logopt.LogJsonOption(base.Flag.JSON)
	}

	ir.EscFmt = escape.Fmt
	ir.IsIntrinsicCall = isIntrinsicCall
	inline.SSADumpInline = ssaDumpInline
	initSSAEnv()
	initSSATables()

	types.PtrSize = thearch.LinkArch.PtrSize
	types.RegSize = thearch.LinkArch.RegSize
	types.MaxWidth = thearch.MAXWIDTH
	types.TypeLinkSym = func(t *types.Type) *obj.LSym {
		return typenamesym(t).Linksym()
	}

	typecheck.Target = new(ir.Package)

	typecheck.NeedFuncSym = staticdata.NeedFuncSym
	typecheck.NeedITab = func(t, iface *types.Type) { itabname(t, iface) }
	typecheck.NeedRuntimeType = addsignat // TODO(rsc): typenamesym for lock?

	base.AutogeneratedPos = makePos(src.NewFileBase("<autogenerated>", "<autogenerated>"), 1, 0)

	types.TypeLinkSym = func(t *types.Type) *obj.LSym {
		return typenamesym(t).Linksym()
	}
	typecheck.Init()

	// Parse input.
	base.Timer.Start("fe", "parse")
	lines := noder.ParseFiles(flag.Args())
	cgoSymABIs()
	base.Timer.Stop()
	base.Timer.AddEvent(int64(lines), "lines")
	recordPackageName()

	// Typecheck.
	typecheck.Package()

	// With all user code typechecked, it's now safe to verify unused dot imports.
	noder.CheckDotImports()
	base.ExitIfErrors()

	// Build init task.
	if initTask := fninit(); initTask != nil {
		typecheck.Export(initTask)
	}

	// Inlining
	base.Timer.Start("fe", "inlining")
	if base.Flag.LowerL != 0 {
		inline.InlinePackage()
	}

	// Devirtualize.
	for _, n := range typecheck.Target.Decls {
		if n.Op() == ir.ODCLFUNC {
			inline.Devirtualize(n.(*ir.Func))
		}
	}
	ir.CurFunc = nil

	// Escape analysis.
	// Required for moving heap allocations onto stack,
	// which in turn is required by the closure implementation,
	// which stores the addresses of stack variables into the closure.
	// If the closure does not escape, it needs to be on the stack
	// or else the stack copier will not update it.
	// Large values are also moved off stack in escape analysis;
	// because large values may contain pointers, it must happen early.
	base.Timer.Start("fe", "escapes")
	escape.Funcs(typecheck.Target.Decls)

	// Collect information for go:nowritebarrierrec
	// checking. This must happen before transformclosure.
	// We'll do the final check after write barriers are
	// inserted.
	if base.Flag.CompilingRuntime {
		EnableNoWriteBarrierRecCheck()
	}

	// Transform closure bodies to properly reference captured variables.
	// This needs to happen before walk, because closures must be transformed
	// before walk reaches a call of a closure.
	base.Timer.Start("fe", "xclosures")
	for _, n := range typecheck.Target.Decls {
		if n.Op() == ir.ODCLFUNC {
			n := n.(*ir.Func)
			if n.OClosure != nil {
				ir.CurFunc = n
				transformclosure(n)
			}
		}
	}

	// Prepare for SSA compilation.
	// This must be before peekitabs, because peekitabs
	// can trigger function compilation.
	initssaconfig()

	// Just before compilation, compile itabs found on
	// the right side of OCONVIFACE so that methods
	// can be de-virtualized during compilation.
	ir.CurFunc = nil
	peekitabs()

	// Compile top level functions.
	// Don't use range--walk can add functions to Target.Decls.
	base.Timer.Start("be", "compilefuncs")
	fcount := int64(0)
	for i := 0; i < len(typecheck.Target.Decls); i++ {
		n := typecheck.Target.Decls[i]
		if n.Op() == ir.ODCLFUNC {
			funccompile(n.(*ir.Func))
			fcount++
		}
	}
	base.Timer.AddEvent(fcount, "funcs")

	compileFunctions()

	if base.Flag.CompilingRuntime {
		// Write barriers are now known. Check the call graph.
		NoWriteBarrierRecCheck()
	}

	// Finalize DWARF inline routine DIEs, then explicitly turn off
	// DWARF inlining gen so as to avoid problems with generated
	// method wrappers.
	if base.Ctxt.DwFixups != nil {
		base.Ctxt.DwFixups.Finalize(base.Ctxt.Pkgpath, base.Debug.DwarfInl != 0)
		base.Ctxt.DwFixups = nil
		base.Flag.GenDwarfInl = 0
	}

	// Write object data to disk.
	base.Timer.Start("be", "dumpobj")
	dumpdata()
	base.Ctxt.NumberSyms()
	dumpobj()
	if base.Flag.AsmHdr != "" {
		dumpasmhdr()
	}

	CheckLargeStacks()
	typecheck.CheckFuncStack()

	if len(compilequeue) != 0 {
		base.Fatalf("%d uncompiled functions", len(compilequeue))
	}

	logopt.FlushLoggedOpts(base.Ctxt, base.Ctxt.Pkgpath)
	base.ExitIfErrors()

	base.FlushErrors()
	base.Timer.Stop()

	if base.Flag.Bench != "" {
		if err := writebench(base.Flag.Bench); err != nil {
			log.Fatalf("cannot write benchmark data: %v", err)
		}
	}
}

func CheckLargeStacks() {
	// Check whether any of the functions we have compiled have gigantic stack frames.
	sort.Slice(largeStackFrames, func(i, j int) bool {
		return largeStackFrames[i].pos.Before(largeStackFrames[j].pos)
	})
	for _, large := range largeStackFrames {
		if large.callee != 0 {
			base.ErrorfAt(large.pos, "stack frame too large (>1GB): %d MB locals + %d MB args + %d MB callee", large.locals>>20, large.args>>20, large.callee>>20)
		} else {
			base.ErrorfAt(large.pos, "stack frame too large (>1GB): %d MB locals + %d MB args", large.locals>>20, large.args>>20)
		}
	}
}

func cgoSymABIs() {
	// The linker expects an ABI0 wrapper for all cgo-exported
	// functions.
	for _, prag := range typecheck.Target.CgoPragmas {
		switch prag[0] {
		case "cgo_export_static", "cgo_export_dynamic":
			if symabiRefs == nil {
				symabiRefs = make(map[string]obj.ABI)
			}
			symabiRefs[prag[1]] = obj.ABI0
		}
	}
}

func writebench(filename string) error {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		return err
	}

	var buf bytes.Buffer
	fmt.Fprintln(&buf, "commit:", objabi.Version)
	fmt.Fprintln(&buf, "goos:", runtime.GOOS)
	fmt.Fprintln(&buf, "goarch:", runtime.GOARCH)
	base.Timer.Write(&buf, "BenchmarkCompile:"+base.Ctxt.Pkgpath+":")

	n, err := f.Write(buf.Bytes())
	if err != nil {
		return err
	}
	if n != buf.Len() {
		panic("bad writer")
	}

	return f.Close()
}

// symabiDefs and symabiRefs record the defined and referenced ABIs of
// symbols required by non-Go code. These are keyed by link symbol
// name, where the local package prefix is always `"".`
var symabiDefs, symabiRefs map[string]obj.ABI

// readSymABIs reads a symabis file that specifies definitions and
// references of text symbols by ABI.
//
// The symabis format is a set of lines, where each line is a sequence
// of whitespace-separated fields. The first field is a verb and is
// either "def" for defining a symbol ABI or "ref" for referencing a
// symbol using an ABI. For both "def" and "ref", the second field is
// the symbol name and the third field is the ABI name, as one of the
// named cmd/internal/obj.ABI constants.
func readSymABIs(file, myimportpath string) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("-symabis: %v", err)
	}

	symabiDefs = make(map[string]obj.ABI)
	symabiRefs = make(map[string]obj.ABI)

	localPrefix := ""
	if myimportpath != "" {
		// Symbols in this package may be written either as
		// "".X or with the package's import path already in
		// the symbol.
		localPrefix = objabi.PathToPrefix(myimportpath) + "."
	}

	for lineNum, line := range strings.Split(string(data), "\n") {
		lineNum++ // 1-based
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		switch parts[0] {
		case "def", "ref":
			// Parse line.
			if len(parts) != 3 {
				log.Fatalf(`%s:%d: invalid symabi: syntax is "%s sym abi"`, file, lineNum, parts[0])
			}
			sym, abistr := parts[1], parts[2]
			abi, valid := obj.ParseABI(abistr)
			if !valid {
				log.Fatalf(`%s:%d: invalid symabi: unknown abi "%s"`, file, lineNum, abistr)
			}

			// If the symbol is already prefixed with
			// myimportpath, rewrite it to start with ""
			// so it matches the compiler's internal
			// symbol names.
			if localPrefix != "" && strings.HasPrefix(sym, localPrefix) {
				sym = `"".` + sym[len(localPrefix):]
			}

			// Record for later.
			if parts[0] == "def" {
				symabiDefs[sym] = abi
			} else {
				symabiRefs[sym] = abi
			}
		default:
			log.Fatalf(`%s:%d: invalid symabi type "%s"`, file, lineNum, parts[0])
		}
	}
}

// recordFlags records the specified command-line flags to be placed
// in the DWARF info.
func recordFlags(flags ...string) {
	if base.Ctxt.Pkgpath == "" {
		// We can't record the flags if we don't know what the
		// package name is.
		return
	}

	type BoolFlag interface {
		IsBoolFlag() bool
	}
	type CountFlag interface {
		IsCountFlag() bool
	}
	var cmd bytes.Buffer
	for _, name := range flags {
		f := flag.Lookup(name)
		if f == nil {
			continue
		}
		getter := f.Value.(flag.Getter)
		if getter.String() == f.DefValue {
			// Flag has default value, so omit it.
			continue
		}
		if bf, ok := f.Value.(BoolFlag); ok && bf.IsBoolFlag() {
			val, ok := getter.Get().(bool)
			if ok && val {
				fmt.Fprintf(&cmd, " -%s", f.Name)
				continue
			}
		}
		if cf, ok := f.Value.(CountFlag); ok && cf.IsCountFlag() {
			val, ok := getter.Get().(int)
			if ok && val == 1 {
				fmt.Fprintf(&cmd, " -%s", f.Name)
				continue
			}
		}
		fmt.Fprintf(&cmd, " -%s=%v", f.Name, getter.Get())
	}

	if cmd.Len() == 0 {
		return
	}
	s := base.Ctxt.Lookup(dwarf.CUInfoPrefix + "producer." + base.Ctxt.Pkgpath)
	s.Type = objabi.SDWARFCUINFO
	// Sometimes (for example when building tests) we can link
	// together two package main archives. So allow dups.
	s.Set(obj.AttrDuplicateOK, true)
	base.Ctxt.Data = append(base.Ctxt.Data, s)
	s.P = cmd.Bytes()[1:]
}

// recordPackageName records the name of the package being
// compiled, so that the linker can save it in the compile unit's DIE.
func recordPackageName() {
	s := base.Ctxt.Lookup(dwarf.CUInfoPrefix + "packagename." + base.Ctxt.Pkgpath)
	s.Type = objabi.SDWARFCUINFO
	// Sometimes (for example when building tests) we can link
	// together two package main archives. So allow dups.
	s.Set(obj.AttrDuplicateOK, true)
	base.Ctxt.Data = append(base.Ctxt.Data, s)
	s.P = []byte(types.LocalPkg.Name)
}

// useNewABIWrapGen returns TRUE if the compiler should generate an
// ABI wrapper for the function 'f'.
func useABIWrapGen(f *ir.Func) bool {
	if !base.Flag.ABIWrap {
		return false
	}

	// Support limit option for bisecting.
	if base.Flag.ABIWrapLimit == 1 {
		return false
	}
	if base.Flag.ABIWrapLimit < 1 {
		return true
	}
	base.Flag.ABIWrapLimit--
	if base.Debug.ABIWrap != 0 && base.Flag.ABIWrapLimit == 1 {
		fmt.Fprintf(os.Stderr, "=-= limit reached after new wrapper for %s\n",
			f.LSym.Name)
	}

	return true
}

func makePos(b *src.PosBase, line, col uint) src.XPos {
	return base.Ctxt.PosTable.XPos(src.MakePos(b, line, col))
}
