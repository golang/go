// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bufio"
	"bytes"
	"cmd/compile/internal/base"
	"cmd/compile/internal/deadcode"
	"cmd/compile/internal/devirtualize"
	"cmd/compile/internal/dwarfgen"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/inline"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/noder"
	"cmd/compile/internal/pkginit"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"flag"
	"fmt"
	"internal/buildcfg"
	"log"
	"os"
	"runtime"
)

// handlePanic ensures that we print out an "internal compiler error" for any panic
// or runtime exception during front-end compiler processing (unless there have
// already been some compiler errors). It may also be invoked from the explicit panic in
// hcrash(), in which case, we pass the panic on through.
func handlePanic() {
	if err := recover(); err != nil {
		if err == "-h" {
			// Force real panic now with -h option (hcrash) - the error
			// information will have already been printed.
			panic(err)
		}
		base.Fatalf("panic: %v", err)
	}
}

// Main parses flags and Go source files specified in the command-line
// arguments, type-checks the parsed Go package, compiles functions to machine
// code, and finally writes the compiled package definition to disk.
func Main(archInit func(*ssagen.ArchInfo)) {
	base.Timer.Start("fe", "init")

	defer handlePanic()

	archInit(&ssagen.Arch)

	base.Ctxt = obj.Linknew(ssagen.Arch.LinkArch)
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
	types.UnsafePkg = types.NewPkg("unsafe", "unsafe")

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

	// pseudo-package used for methods with anonymous receivers
	ir.Pkgs.Go = types.NewPkg("go", "")

	base.DebugSSA = ssa.PhaseOption
	base.ParseFlags()

	// Record flags that affect the build result. (And don't
	// record flags that don't, since that would cause spurious
	// changes in the binary.)
	dwarfgen.RecordFlags("B", "N", "l", "msan", "race", "asan", "shared", "dynlink", "dwarf", "dwarflocationlists", "dwarfbasentries", "smallframes", "spectre")

	if !base.EnableTrace && base.Flag.LowerT {
		log.Fatalf("compiler not built with support for -t")
	}

	// Enable inlining (after RecordFlags, to avoid recording the rewritten -l).  For now:
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
		base.Ctxt.DebugInfo = dwarfgen.Info
		base.Ctxt.GenAbstractFunc = dwarfgen.AbstractFunc
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

	symABIs := ssagen.NewSymABIs(base.Ctxt.Pkgpath)
	if base.Flag.SymABIs != "" {
		symABIs.ReadSymABIs(base.Flag.SymABIs)
	}

	if base.Compiling(base.NoInstrumentPkgs) {
		base.Flag.Race = false
		base.Flag.MSan = false
		base.Flag.ASan = false
	}

	ssagen.Arch.LinkArch.Init(base.Ctxt)
	startProfile()
	if base.Flag.Race || base.Flag.MSan || base.Flag.ASan {
		base.Flag.Cfg.Instrumenting = true
	}
	if base.Flag.Dwarf {
		dwarf.EnableLogging(base.Debug.DwarfInl != 0)
	}
	if base.Debug.SoftFloat != 0 {
		ssagen.Arch.SoftFloat = true
	}

	if base.Flag.JSON != "" { // parse version,destination from json logging optimization.
		logopt.LogJsonOption(base.Flag.JSON)
	}

	ir.EscFmt = escape.Fmt
	ir.IsIntrinsicCall = ssagen.IsIntrinsicCall
	inline.SSADumpInline = ssagen.DumpInline
	ssagen.InitEnv()
	ssagen.InitTables()

	types.PtrSize = ssagen.Arch.LinkArch.PtrSize
	types.RegSize = ssagen.Arch.LinkArch.RegSize
	types.MaxWidth = ssagen.Arch.MAXWIDTH

	typecheck.Target = new(ir.Package)

	typecheck.NeedRuntimeType = reflectdata.NeedRuntimeType // TODO(rsc): TypeSym for lock?

	base.AutogeneratedPos = makePos(src.NewFileBase("<autogenerated>", "<autogenerated>"), 1, 0)

	typecheck.InitUniverse()
	typecheck.InitRuntime()

	// Parse and typecheck input.
	noder.LoadPackage(flag.Args())
	if base.Ctxt.Pkgpath == obj.UnlinkablePkg && types.LocalPkg.Name == "main" {
		base.Ctxt.Pkgpath = "main"
	}

	dwarfgen.RecordPackageName()

	// Prepare for backend processing. This must happen before pkginit,
	// because it generates itabs for initializing global variables.
	ssagen.InitConfig()

	// Create "init" function for package-scope variable initialization
	// statements, if any.
	//
	// Note: This needs to happen early, before any optimizations. The
	// Go spec defines a precise order than initialization should be
	// carried out in, and even mundane optimizations like dead code
	// removal can skew the results (e.g., #43444).
	pkginit.MakeInit()

	// Eliminate some obviously dead code.
	// Must happen after typechecking.
	for _, n := range typecheck.Target.Decls {
		if n.Op() == ir.ODCLFUNC {
			deadcode.Func(n.(*ir.Func))
		}
	}

	// Compute Addrtaken for names.
	// We need to wait until typechecking is done so that when we see &x[i]
	// we know that x has its address taken if x is an array, but not if x is a slice.
	// We compute Addrtaken in bulk here.
	// After this phase, we maintain Addrtaken incrementally.
	if typecheck.DirtyAddrtaken {
		typecheck.ComputeAddrtaken(typecheck.Target.Decls)
		typecheck.DirtyAddrtaken = false
	}
	typecheck.IncrementalAddrtaken = true

	if base.Debug.TypecheckInl != 0 {
		// Typecheck imported function bodies if Debug.l > 1,
		// otherwise lazily when used or re-exported.
		typecheck.AllImportedBodies()
	}

	// Inlining
	base.Timer.Start("fe", "inlining")
	if base.Flag.LowerL != 0 {
		inline.InlinePackage()
	}
	noder.MakeWrappers(typecheck.Target) // must happen after inlining

	// Devirtualize.
	for _, n := range typecheck.Target.Decls {
		if n.Op() == ir.ODCLFUNC {
			devirtualize.Func(n.(*ir.Func))
		}
	}
	ir.CurFunc = nil

	// Build init task, if needed.
	if initTask := pkginit.Task(); initTask != nil {
		typecheck.Export(initTask)
	}

	// Generate ABI wrappers. Must happen before escape analysis
	// and doesn't benefit from dead-coding or inlining.
	symABIs.GenABIWrappers()

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

	// TODO(mdempsky): This is a hack. We need a proper, global work
	// queue for scheduling function compilation so components don't
	// need to adjust their behavior depending on when they're called.
	reflectdata.AfterGlobalEscapeAnalysis = true

	// Collect information for go:nowritebarrierrec
	// checking. This must happen before transforming closures during Walk
	// We'll do the final check after write barriers are
	// inserted.
	if base.Flag.CompilingRuntime {
		ssagen.EnableNoWriteBarrierRecCheck()
	}

	ir.CurFunc = nil

	// Compile top level functions.
	// Don't use range--walk can add functions to Target.Decls.
	base.Timer.Start("be", "compilefuncs")
	fcount := int64(0)
	for i := 0; i < len(typecheck.Target.Decls); i++ {
		if fn, ok := typecheck.Target.Decls[i].(*ir.Func); ok {
			// Don't try compiling dead hidden closure.
			if fn.IsDeadcodeClosure() {
				continue
			}
			enqueueFunc(fn)
			fcount++
		}
	}
	base.Timer.AddEvent(fcount, "funcs")

	compileFunctions()

	if base.Flag.CompilingRuntime {
		// Write barriers are now known. Check the call graph.
		ssagen.NoWriteBarrierRecCheck()
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

	ssagen.CheckLargeStacks()
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

func writebench(filename string) error {
	f, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		return err
	}

	var buf bytes.Buffer
	fmt.Fprintln(&buf, "commit:", buildcfg.Version)
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

func makePos(b *src.PosBase, line, col uint) src.XPos {
	return base.Ctxt.PosTable.XPos(src.MakePos(b, line, col))
}
