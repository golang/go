// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run mkbuiltin.go

package gc

import (
	"bufio"
	"bytes"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/bio"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"flag"
	"fmt"
	"internal/goversion"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

var imported_unsafe bool

var (
	buildid      string
	spectre      string
	spectreIndex bool
)

var (
	Debug_append       int
	Debug_checkptr     int
	Debug_closure      int
	Debug_compilelater int
	debug_dclstack     int
	Debug_libfuzzer    int
	Debug_panic        int
	Debug_slice        int
	Debug_vlog         bool
	Debug_wb           int
	Debug_pctab        string
	Debug_locationlist int
	Debug_typecheckinl int
	Debug_gendwarfinl  int
	Debug_softfloat    int
	Debug_defer        int
)

// Debug arguments.
// These can be specified with the -d flag, as in "-d nil"
// to set the debug_checknil variable.
// Multiple options can be comma-separated.
// Each option accepts an optional argument, as in "gcprog=2"
var debugtab = []struct {
	name string
	help string
	val  interface{} // must be *int or *string
}{
	{"append", "print information about append compilation", &Debug_append},
	{"checkptr", "instrument unsafe pointer conversions", &Debug_checkptr},
	{"closure", "print information about closure compilation", &Debug_closure},
	{"compilelater", "compile functions as late as possible", &Debug_compilelater},
	{"disablenil", "disable nil checks", &disable_checknil},
	{"dclstack", "run internal dclstack check", &debug_dclstack},
	{"gcprog", "print dump of GC programs", &Debug_gcprog},
	{"libfuzzer", "coverage instrumentation for libfuzzer", &Debug_libfuzzer},
	{"nil", "print information about nil checks", &Debug_checknil},
	{"panic", "do not hide any compiler panic", &Debug_panic},
	{"slice", "print information about slice compilation", &Debug_slice},
	{"typeassert", "print information about type assertion inlining", &Debug_typeassert},
	{"wb", "print information about write barriers", &Debug_wb},
	{"export", "print export data", &Debug_export},
	{"pctab", "print named pc-value table", &Debug_pctab},
	{"locationlists", "print information about DWARF location list creation", &Debug_locationlist},
	{"typecheckinl", "eager typechecking of inline function bodies", &Debug_typecheckinl},
	{"dwarfinl", "print information about DWARF inlined function creation", &Debug_gendwarfinl},
	{"softfloat", "force compiler to emit soft-float code", &Debug_softfloat},
	{"defer", "print information about defer compilation", &Debug_defer},
}

const debugHelpHeader = `usage: -d arg[,arg]* and arg is <key>[=<value>]

<key> is one of:

`

const debugHelpFooter = `
<value> is key-specific.

Key "checkptr" supports values:
	"0": instrumentation disabled
	"1": conversions involving unsafe.Pointer are instrumented
	"2": conversions to unsafe.Pointer force heap allocation

Key "pctab" supports values:
	"pctospadj", "pctofile", "pctoline", "pctoinline", "pctopcdata"
`

func usage() {
	fmt.Fprintf(os.Stderr, "usage: compile [options] file.go...\n")
	objabi.Flagprint(os.Stderr)
	Exit(2)
}

func hidePanic() {
	if Debug_panic == 0 && nsavederrors+nerrors > 0 {
		// If we've already complained about things
		// in the program, don't bother complaining
		// about a panic too; let the user clean up
		// the code and try again.
		if err := recover(); err != nil {
			errorexit()
		}
	}
}

// supportsDynlink reports whether or not the code generator for the given
// architecture supports the -shared and -dynlink flags.
func supportsDynlink(arch *sys.Arch) bool {
	return arch.InFamily(sys.AMD64, sys.ARM, sys.ARM64, sys.I386, sys.PPC64, sys.S390X)
}

// timing data for compiler phases
var timings Timings
var benchfile string

var nowritebarrierrecCheck *nowritebarrierrecChecker

// Main parses flags and Go source files specified in the command-line
// arguments, type-checks the parsed Go package, compiles functions to machine
// code, and finally writes the compiled package definition to disk.
func Main(archInit func(*Arch)) {
	timings.Start("fe", "init")

	defer hidePanic()

	archInit(&thearch)

	Ctxt = obj.Linknew(thearch.LinkArch)
	Ctxt.DiagFunc = yyerror
	Ctxt.DiagFlush = flusherrors
	Ctxt.Bso = bufio.NewWriter(os.Stdout)

	// UseBASEntries is preferred because it shaves about 2% off build time, but LLDB, dsymutil, and dwarfdump
	// on Darwin don't support it properly, especially since macOS 10.14 (Mojave).  This is exposed as a flag
	// to allow testing with LLVM tools on Linux, and to help with reporting this bug to the LLVM project.
	// See bugs 31188 and 21945 (CLs 170638, 98075, 72371).
	Ctxt.UseBASEntries = Ctxt.Headtype != objabi.Hdarwin

	localpkg = types.NewPkg("", "")
	localpkg.Prefix = "\"\""

	// We won't know localpkg's height until after import
	// processing. In the mean time, set to MaxPkgHeight to ensure
	// height comparisons at least work until then.
	localpkg.Height = types.MaxPkgHeight

	// pseudo-package, for scoping
	builtinpkg = types.NewPkg("go.builtin", "") // TODO(gri) name this package go.builtin?
	builtinpkg.Prefix = "go.builtin"            // not go%2ebuiltin

	// pseudo-package, accessed by import "unsafe"
	unsafepkg = types.NewPkg("unsafe", "unsafe")

	// Pseudo-package that contains the compiler's builtin
	// declarations for package runtime. These are declared in a
	// separate package to avoid conflicts with package runtime's
	// actual declarations, which may differ intentionally but
	// insignificantly.
	Runtimepkg = types.NewPkg("go.runtime", "runtime")
	Runtimepkg.Prefix = "runtime"

	// pseudo-packages used in symbol tables
	itabpkg = types.NewPkg("go.itab", "go.itab")
	itabpkg.Prefix = "go.itab" // not go%2eitab

	itablinkpkg = types.NewPkg("go.itablink", "go.itablink")
	itablinkpkg.Prefix = "go.itablink" // not go%2eitablink

	trackpkg = types.NewPkg("go.track", "go.track")
	trackpkg.Prefix = "go.track" // not go%2etrack

	// pseudo-package used for map zero values
	mappkg = types.NewPkg("go.map", "go.map")
	mappkg.Prefix = "go.map"

	// pseudo-package used for methods with anonymous receivers
	gopkg = types.NewPkg("go", "")

	Wasm := objabi.GOARCH == "wasm"

	// Whether the limit for stack-allocated objects is much smaller than normal.
	// This can be helpful for diagnosing certain causes of GC latency. See #27732.
	smallFrames := false
	jsonLogOpt := ""

	flag.BoolVar(&compiling_runtime, "+", false, "compiling runtime")
	flag.BoolVar(&compiling_std, "std", false, "compiling standard library")
	objabi.Flagcount("%", "debug non-static initializers", &Debug['%'])
	objabi.Flagcount("B", "disable bounds checking", &Debug['B'])
	objabi.Flagcount("C", "disable printing of columns in error messages", &Debug['C']) // TODO(gri) remove eventually
	flag.StringVar(&localimport, "D", "", "set relative `path` for local imports")
	objabi.Flagcount("E", "debug symbol export", &Debug['E'])
	objabi.Flagfn1("I", "add `directory` to import search path", addidir)
	objabi.Flagcount("K", "debug missing line numbers", &Debug['K'])
	objabi.Flagcount("L", "show full file names in error messages", &Debug['L'])
	objabi.Flagcount("N", "disable optimizations", &Debug['N'])
	objabi.Flagcount("S", "print assembly listing", &Debug['S'])
	objabi.AddVersionFlag() // -V
	objabi.Flagcount("W", "debug parse tree after type checking", &Debug['W'])
	flag.StringVar(&asmhdr, "asmhdr", "", "write assembly header to `file`")
	flag.StringVar(&buildid, "buildid", "", "record `id` as the build id in the export metadata")
	flag.IntVar(&nBackendWorkers, "c", 1, "concurrency during compilation, 1 means no concurrency")
	flag.BoolVar(&pure_go, "complete", false, "compiling complete package (no C or assembly)")
	flag.StringVar(&debugstr, "d", "", "print debug information about items in `list`; try -d help")
	flag.BoolVar(&flagDWARF, "dwarf", !Wasm, "generate DWARF symbols")
	flag.BoolVar(&Ctxt.Flag_locationlists, "dwarflocationlists", true, "add location lists to DWARF in optimized mode")
	flag.IntVar(&genDwarfInline, "gendwarfinl", 2, "generate DWARF inline info records")
	objabi.Flagcount("e", "no limit on number of errors reported", &Debug['e'])
	objabi.Flagcount("h", "halt on error", &Debug['h'])
	objabi.Flagfn1("importmap", "add `definition` of the form source=actual to import map", addImportMap)
	objabi.Flagfn1("importcfg", "read import configuration from `file`", readImportCfg)
	flag.StringVar(&flag_installsuffix, "installsuffix", "", "set pkg directory `suffix`")
	objabi.Flagcount("j", "debug runtime-initialized variables", &Debug['j'])
	objabi.Flagcount("l", "disable inlining", &Debug['l'])
	flag.StringVar(&flag_lang, "lang", "", "release to compile for")
	flag.StringVar(&linkobj, "linkobj", "", "write linker-specific object to `file`")
	objabi.Flagcount("live", "debug liveness analysis", &debuglive)
	objabi.Flagcount("m", "print optimization decisions", &Debug['m'])
	if sys.MSanSupported(objabi.GOOS, objabi.GOARCH) {
		flag.BoolVar(&flag_msan, "msan", false, "build code compatible with C/C++ memory sanitizer")
	}
	flag.BoolVar(&nolocalimports, "nolocalimports", false, "reject local (relative) imports")
	flag.StringVar(&outfile, "o", "", "write output to `file`")
	flag.StringVar(&myimportpath, "p", "", "set expected package import `path`")
	flag.BoolVar(&writearchive, "pack", false, "write to file.a instead of file.o")
	objabi.Flagcount("r", "debug generated wrappers", &Debug['r'])
	if sys.RaceDetectorSupported(objabi.GOOS, objabi.GOARCH) {
		flag.BoolVar(&flag_race, "race", false, "enable race detector")
	}
	flag.StringVar(&spectre, "spectre", spectre, "enable spectre mitigations in `list` (all, index, ret)")
	if enableTrace {
		flag.BoolVar(&trace, "t", false, "trace type-checking")
	}
	flag.StringVar(&pathPrefix, "trimpath", "", "remove `prefix` from recorded source file paths")
	flag.BoolVar(&Debug_vlog, "v", false, "increase debug verbosity")
	objabi.Flagcount("w", "debug type checking", &Debug['w'])
	flag.BoolVar(&use_writebarrier, "wb", true, "enable write barrier")
	var flag_shared bool
	var flag_dynlink bool
	if supportsDynlink(thearch.LinkArch.Arch) {
		flag.BoolVar(&flag_shared, "shared", false, "generate code that can be linked into a shared library")
		flag.BoolVar(&flag_dynlink, "dynlink", false, "support references to Go symbols defined in other shared libraries")
		flag.BoolVar(&Ctxt.Flag_linkshared, "linkshared", false, "generate code that will be linked against Go shared libraries")
	}
	flag.StringVar(&cpuprofile, "cpuprofile", "", "write cpu profile to `file`")
	flag.StringVar(&memprofile, "memprofile", "", "write memory profile to `file`")
	flag.Int64Var(&memprofilerate, "memprofilerate", 0, "set runtime.MemProfileRate to `rate`")
	var goversion string
	flag.StringVar(&goversion, "goversion", "", "required version of the runtime")
	var symabisPath string
	flag.StringVar(&symabisPath, "symabis", "", "read symbol ABIs from `file`")
	flag.StringVar(&traceprofile, "traceprofile", "", "write an execution trace to `file`")
	flag.StringVar(&blockprofile, "blockprofile", "", "write block profile to `file`")
	flag.StringVar(&mutexprofile, "mutexprofile", "", "write mutex profile to `file`")
	flag.StringVar(&benchfile, "bench", "", "append benchmark times to `file`")
	flag.BoolVar(&smallFrames, "smallframes", false, "reduce the size limit for stack allocated objects")
	flag.BoolVar(&Ctxt.UseBASEntries, "dwarfbasentries", Ctxt.UseBASEntries, "use base address selection entries in DWARF")
	flag.BoolVar(&Ctxt.Flag_newobj, "newobj", false, "use new object file format")
	flag.StringVar(&jsonLogOpt, "json", "", "version,destination for JSON compiler/optimizer logging")

	objabi.Flagparse(usage)

	for _, f := range strings.Split(spectre, ",") {
		f = strings.TrimSpace(f)
		switch f {
		default:
			log.Fatalf("unknown setting -spectre=%s", f)
		case "":
			// nothing
		case "all":
			spectreIndex = true
			Ctxt.Retpoline = true
		case "index":
			spectreIndex = true
		case "ret":
			Ctxt.Retpoline = true
		}
	}

	if spectreIndex {
		switch objabi.GOARCH {
		case "amd64":
			// ok
		default:
			log.Fatalf("GOARCH=%s does not support -spectre=index", objabi.GOARCH)
		}
	}

	// Record flags that affect the build result. (And don't
	// record flags that don't, since that would cause spurious
	// changes in the binary.)
	recordFlags("B", "N", "l", "msan", "race", "shared", "dynlink", "dwarflocationlists", "dwarfbasentries", "smallframes", "spectre", "newobj")

	if smallFrames {
		maxStackVarSize = 128 * 1024
		maxImplicitStackVarSize = 16 * 1024
	}

	Ctxt.Flag_shared = flag_dynlink || flag_shared
	Ctxt.Flag_dynlink = flag_dynlink
	Ctxt.Flag_optimize = Debug['N'] == 0

	Ctxt.Debugasm = Debug['S']
	Ctxt.Debugvlog = Debug_vlog
	if flagDWARF {
		Ctxt.DebugInfo = debuginfo
		Ctxt.GenAbstractFunc = genAbstractFunc
		Ctxt.DwFixups = obj.NewDwarfFixupTable(Ctxt)
	} else {
		// turn off inline generation if no dwarf at all
		genDwarfInline = 0
		Ctxt.Flag_locationlists = false
	}

	if flag.NArg() < 1 && debugstr != "help" && debugstr != "ssa/help" {
		usage()
	}

	if goversion != "" && goversion != runtime.Version() {
		fmt.Printf("compile: version %q does not match go tool version %q\n", runtime.Version(), goversion)
		Exit(2)
	}

	checkLang()

	if symabisPath != "" {
		readSymABIs(symabisPath, myimportpath)
	}

	thearch.LinkArch.Init(Ctxt)

	if outfile == "" {
		p := flag.Arg(0)
		if i := strings.LastIndex(p, "/"); i >= 0 {
			p = p[i+1:]
		}
		if runtime.GOOS == "windows" {
			if i := strings.LastIndex(p, `\`); i >= 0 {
				p = p[i+1:]
			}
		}
		if i := strings.LastIndex(p, "."); i >= 0 {
			p = p[:i]
		}
		suffix := ".o"
		if writearchive {
			suffix = ".a"
		}
		outfile = p + suffix
	}

	startProfile()

	if flag_race && flag_msan {
		log.Fatal("cannot use both -race and -msan")
	}
	if (flag_race || flag_msan) && objabi.GOOS != "windows" {
		// -race and -msan imply -d=checkptr for now (except on windows).
		// TODO(mdempsky): Re-evaluate before Go 1.14. See #34964.
		Debug_checkptr = 1
	}
	if ispkgin(omit_pkgs) {
		flag_race = false
		flag_msan = false
	}
	if flag_race {
		racepkg = types.NewPkg("runtime/race", "")
	}
	if flag_msan {
		msanpkg = types.NewPkg("runtime/msan", "")
	}
	if flag_race || flag_msan {
		instrumenting = true
	}

	if compiling_runtime && Debug['N'] != 0 {
		log.Fatal("cannot disable optimizations while compiling runtime")
	}
	if nBackendWorkers < 1 {
		log.Fatalf("-c must be at least 1, got %d", nBackendWorkers)
	}
	if nBackendWorkers > 1 && !concurrentBackendAllowed() {
		log.Fatalf("cannot use concurrent backend compilation with provided flags; invoked as %v", os.Args)
	}
	if Ctxt.Flag_locationlists && len(Ctxt.Arch.DWARFRegisters) == 0 {
		log.Fatalf("location lists requested but register mapping not available on %v", Ctxt.Arch.Name)
	}

	// parse -d argument
	if debugstr != "" {
	Split:
		for _, name := range strings.Split(debugstr, ",") {
			if name == "" {
				continue
			}
			// display help about the -d option itself and quit
			if name == "help" {
				fmt.Print(debugHelpHeader)
				maxLen := len("ssa/help")
				for _, t := range debugtab {
					if len(t.name) > maxLen {
						maxLen = len(t.name)
					}
				}
				for _, t := range debugtab {
					fmt.Printf("\t%-*s\t%s\n", maxLen, t.name, t.help)
				}
				// ssa options have their own help
				fmt.Printf("\t%-*s\t%s\n", maxLen, "ssa/help", "print help about SSA debugging")
				fmt.Print(debugHelpFooter)
				os.Exit(0)
			}
			val, valstring, haveInt := 1, "", true
			if i := strings.IndexAny(name, "=:"); i >= 0 {
				var err error
				name, valstring = name[:i], name[i+1:]
				val, err = strconv.Atoi(valstring)
				if err != nil {
					val, haveInt = 1, false
				}
			}
			for _, t := range debugtab {
				if t.name != name {
					continue
				}
				switch vp := t.val.(type) {
				case nil:
					// Ignore
				case *string:
					*vp = valstring
				case *int:
					if !haveInt {
						log.Fatalf("invalid debug value %v", name)
					}
					*vp = val
				default:
					panic("bad debugtab type")
				}
				continue Split
			}
			// special case for ssa for now
			if strings.HasPrefix(name, "ssa/") {
				// expect form ssa/phase/flag
				// e.g. -d=ssa/generic_cse/time
				// _ in phase name also matches space
				phase := name[4:]
				flag := "debug" // default flag is debug
				if i := strings.Index(phase, "/"); i >= 0 {
					flag = phase[i+1:]
					phase = phase[:i]
				}
				err := ssa.PhaseOption(phase, flag, val, valstring)
				if err != "" {
					log.Fatalf(err)
				}
				continue Split
			}
			log.Fatalf("unknown debug key -d %s\n", name)
		}
	}

	if compiling_runtime {
		// Runtime can't use -d=checkptr, at least not yet.
		Debug_checkptr = 0

		// Fuzzing the runtime isn't interesting either.
		Debug_libfuzzer = 0
	}

	// set via a -d flag
	Ctxt.Debugpcln = Debug_pctab
	if flagDWARF {
		dwarf.EnableLogging(Debug_gendwarfinl != 0)
	}

	if Debug_softfloat != 0 {
		thearch.SoftFloat = true
	}

	// enable inlining.  for now:
	//	default: inlining on.  (debug['l'] == 1)
	//	-l: inlining off  (debug['l'] == 0)
	//	-l=2, -l=3: inlining on again, with extra debugging (debug['l'] > 1)
	if Debug['l'] <= 1 {
		Debug['l'] = 1 - Debug['l']
	}

	if jsonLogOpt != "" { // parse version,destination from json logging optimization.
		logopt.LogJsonOption(jsonLogOpt)
	}

	ssaDump = os.Getenv("GOSSAFUNC")
	if ssaDump != "" {
		if strings.HasSuffix(ssaDump, "+") {
			ssaDump = ssaDump[:len(ssaDump)-1]
			ssaDumpStdout = true
		}
		spl := strings.Split(ssaDump, ":")
		if len(spl) > 1 {
			ssaDump = spl[0]
			ssaDumpCFG = spl[1]
		}
	}

	trackScopes = flagDWARF

	Widthptr = thearch.LinkArch.PtrSize
	Widthreg = thearch.LinkArch.RegSize

	// initialize types package
	// (we need to do this to break dependencies that otherwise
	// would lead to import cycles)
	types.Widthptr = Widthptr
	types.Dowidth = dowidth
	types.Fatalf = Fatalf
	types.Sconv = func(s *types.Sym, flag, mode int) string {
		return sconv(s, FmtFlag(flag), fmtMode(mode))
	}
	types.Tconv = func(t *types.Type, flag, mode int) string {
		return tconv(t, FmtFlag(flag), fmtMode(mode))
	}
	types.FormatSym = func(sym *types.Sym, s fmt.State, verb rune, mode int) {
		symFormat(sym, s, verb, fmtMode(mode))
	}
	types.FormatType = func(t *types.Type, s fmt.State, verb rune, mode int) {
		typeFormat(t, s, verb, fmtMode(mode))
	}
	types.TypeLinkSym = func(t *types.Type) *obj.LSym {
		return typenamesym(t).Linksym()
	}
	types.FmtLeft = int(FmtLeft)
	types.FmtUnsigned = int(FmtUnsigned)
	types.FErr = int(FErr)
	types.Ctxt = Ctxt

	initUniverse()

	dclcontext = PEXTERN
	nerrors = 0

	autogeneratedPos = makePos(src.NewFileBase("<autogenerated>", "<autogenerated>"), 1, 0)

	timings.Start("fe", "loadsys")
	loadsys()

	timings.Start("fe", "parse")
	lines := parseFiles(flag.Args())
	timings.Stop()
	timings.AddEvent(int64(lines), "lines")

	finishUniverse()

	recordPackageName()

	typecheckok = true

	// Process top-level declarations in phases.

	// Phase 1: const, type, and names and types of funcs.
	//   This will gather all the information about types
	//   and methods but doesn't depend on any of it.
	//
	//   We also defer type alias declarations until phase 2
	//   to avoid cycles like #18640.
	//   TODO(gri) Remove this again once we have a fix for #25838.

	// Don't use range--typecheck can add closures to xtop.
	timings.Start("fe", "typecheck", "top1")
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if op := n.Op; op != ODCL && op != OAS && op != OAS2 && (op != ODCLTYPE || !n.Left.Name.Param.Alias) {
			xtop[i] = typecheck(n, ctxStmt)
		}
	}

	// Phase 2: Variable assignments.
	//   To check interface assignments, depends on phase 1.

	// Don't use range--typecheck can add closures to xtop.
	timings.Start("fe", "typecheck", "top2")
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if op := n.Op; op == ODCL || op == OAS || op == OAS2 || op == ODCLTYPE && n.Left.Name.Param.Alias {
			xtop[i] = typecheck(n, ctxStmt)
		}
	}

	// Phase 3: Type check function bodies.
	// Don't use range--typecheck can add closures to xtop.
	timings.Start("fe", "typecheck", "func")
	var fcount int64
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if op := n.Op; op == ODCLFUNC || op == OCLOSURE {
			Curfn = n
			decldepth = 1
			saveerrors()
			typecheckslice(Curfn.Nbody.Slice(), ctxStmt)
			checkreturn(Curfn)
			if nerrors != 0 {
				Curfn.Nbody.Set(nil) // type errors; do not compile
			}
			// Now that we've checked whether n terminates,
			// we can eliminate some obviously dead code.
			deadcode(Curfn)
			fcount++
		}
	}
	// With all types checked, it's now safe to verify map keys. One single
	// check past phase 9 isn't sufficient, as we may exit with other errors
	// before then, thus skipping map key errors.
	checkMapKeys()
	timings.AddEvent(fcount, "funcs")

	if nsavederrors+nerrors != 0 {
		errorexit()
	}

	// Phase 4: Decide how to capture closed variables.
	// This needs to run before escape analysis,
	// because variables captured by value do not escape.
	timings.Start("fe", "capturevars")
	for _, n := range xtop {
		if n.Op == ODCLFUNC && n.Func.Closure != nil {
			Curfn = n
			capturevars(n)
		}
	}
	capturevarscomplete = true

	Curfn = nil

	if nsavederrors+nerrors != 0 {
		errorexit()
	}

	// Phase 5: Inlining
	timings.Start("fe", "inlining")
	if Debug_typecheckinl != 0 {
		// Typecheck imported function bodies if debug['l'] > 1,
		// otherwise lazily when used or re-exported.
		for _, n := range importlist {
			if n.Func.Inl != nil {
				saveerrors()
				typecheckinl(n)
			}
		}

		if nsavederrors+nerrors != 0 {
			errorexit()
		}
	}

	if Debug['l'] != 0 {
		// Find functions that can be inlined and clone them before walk expands them.
		visitBottomUp(xtop, func(list []*Node, recursive bool) {
			for _, n := range list {
				if !recursive {
					caninl(n)
				} else {
					if Debug['m'] > 1 {
						fmt.Printf("%v: cannot inline %v: recursive\n", n.Line(), n.Func.Nname)
					}
				}
				inlcalls(n)
			}
		})
	}

	// Phase 6: Escape analysis.
	// Required for moving heap allocations onto stack,
	// which in turn is required by the closure implementation,
	// which stores the addresses of stack variables into the closure.
	// If the closure does not escape, it needs to be on the stack
	// or else the stack copier will not update it.
	// Large values are also moved off stack in escape analysis;
	// because large values may contain pointers, it must happen early.
	timings.Start("fe", "escapes")
	escapes(xtop)

	// Collect information for go:nowritebarrierrec
	// checking. This must happen before transformclosure.
	// We'll do the final check after write barriers are
	// inserted.
	if compiling_runtime {
		nowritebarrierrecCheck = newNowritebarrierrecChecker()
	}

	// Phase 7: Transform closure bodies to properly reference captured variables.
	// This needs to happen before walk, because closures must be transformed
	// before walk reaches a call of a closure.
	timings.Start("fe", "xclosures")
	for _, n := range xtop {
		if n.Op == ODCLFUNC && n.Func.Closure != nil {
			Curfn = n
			transformclosure(n)
		}
	}

	// Prepare for SSA compilation.
	// This must be before peekitabs, because peekitabs
	// can trigger function compilation.
	initssaconfig()

	// Just before compilation, compile itabs found on
	// the right side of OCONVIFACE so that methods
	// can be de-virtualized during compilation.
	Curfn = nil
	peekitabs()

	// Phase 8: Compile top level functions.
	// Don't use range--walk can add functions to xtop.
	timings.Start("be", "compilefuncs")
	fcount = 0
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if n.Op == ODCLFUNC {
			funccompile(n)
			fcount++
		}
	}
	timings.AddEvent(fcount, "funcs")

	if nsavederrors+nerrors == 0 {
		fninit(xtop)
	}

	compileFunctions()

	if nowritebarrierrecCheck != nil {
		// Write barriers are now known. Check the
		// call graph.
		nowritebarrierrecCheck.check()
		nowritebarrierrecCheck = nil
	}

	// Finalize DWARF inline routine DIEs, then explicitly turn off
	// DWARF inlining gen so as to avoid problems with generated
	// method wrappers.
	if Ctxt.DwFixups != nil {
		Ctxt.DwFixups.Finalize(myimportpath, Debug_gendwarfinl != 0)
		Ctxt.DwFixups = nil
		genDwarfInline = 0
	}

	// Phase 9: Check external declarations.
	timings.Start("be", "externaldcls")
	for i, n := range externdcl {
		if n.Op == ONAME {
			externdcl[i] = typecheck(externdcl[i], ctxExpr)
		}
	}
	// Check the map keys again, since we typechecked the external
	// declarations.
	checkMapKeys()

	if nerrors+nsavederrors != 0 {
		errorexit()
	}

	// Write object data to disk.
	timings.Start("be", "dumpobj")
	dumpdata()
	Ctxt.NumberSyms(false)
	dumpobj()
	if asmhdr != "" {
		dumpasmhdr()
	}

	// Check whether any of the functions we have compiled have gigantic stack frames.
	sort.Slice(largeStackFrames, func(i, j int) bool {
		return largeStackFrames[i].pos.Before(largeStackFrames[j].pos)
	})
	for _, large := range largeStackFrames {
		if large.callee != 0 {
			yyerrorl(large.pos, "stack frame too large (>1GB): %d MB locals + %d MB args + %d MB callee", large.locals>>20, large.args>>20, large.callee>>20)
		} else {
			yyerrorl(large.pos, "stack frame too large (>1GB): %d MB locals + %d MB args", large.locals>>20, large.args>>20)
		}
	}

	if len(compilequeue) != 0 {
		Fatalf("%d uncompiled functions", len(compilequeue))
	}

	logopt.FlushLoggedOpts(Ctxt, myimportpath)

	if nerrors+nsavederrors != 0 {
		errorexit()
	}

	flusherrors()
	timings.Stop()

	if benchfile != "" {
		if err := writebench(benchfile); err != nil {
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
	fmt.Fprintln(&buf, "commit:", objabi.Version)
	fmt.Fprintln(&buf, "goos:", runtime.GOOS)
	fmt.Fprintln(&buf, "goarch:", runtime.GOARCH)
	timings.Write(&buf, "BenchmarkCompile:"+myimportpath+":")

	n, err := f.Write(buf.Bytes())
	if err != nil {
		return err
	}
	if n != buf.Len() {
		panic("bad writer")
	}

	return f.Close()
}

var (
	importMap   = map[string]string{}
	packageFile map[string]string // nil means not in use
)

func addImportMap(s string) {
	if strings.Count(s, "=") != 1 {
		log.Fatal("-importmap argument must be of the form source=actual")
	}
	i := strings.Index(s, "=")
	source, actual := s[:i], s[i+1:]
	if source == "" || actual == "" {
		log.Fatal("-importmap argument must be of the form source=actual; source and actual must be non-empty")
	}
	importMap[source] = actual
}

func readImportCfg(file string) {
	packageFile = map[string]string{}
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("-importcfg: %v", err)
	}

	for lineNum, line := range strings.Split(string(data), "\n") {
		lineNum++ // 1-based
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		var verb, args string
		if i := strings.Index(line, " "); i < 0 {
			verb = line
		} else {
			verb, args = line[:i], strings.TrimSpace(line[i+1:])
		}
		var before, after string
		if i := strings.Index(args, "="); i >= 0 {
			before, after = args[:i], args[i+1:]
		}
		switch verb {
		default:
			log.Fatalf("%s:%d: unknown directive %q", file, lineNum, verb)
		case "importmap":
			if before == "" || after == "" {
				log.Fatalf(`%s:%d: invalid importmap: syntax is "importmap old=new"`, file, lineNum)
			}
			importMap[before] = after
		case "packagefile":
			if before == "" || after == "" {
				log.Fatalf(`%s:%d: invalid packagefile: syntax is "packagefile path=filename"`, file, lineNum)
			}
			packageFile[before] = after
		}
	}
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
			sym, abi := parts[1], parts[2]
			if abi != "ABI0" { // Only supported external ABI right now
				log.Fatalf(`%s:%d: invalid symabi: unknown abi "%s"`, file, lineNum, abi)
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
				symabiDefs[sym] = obj.ABI0
			} else {
				symabiRefs[sym] = obj.ABI0
			}
		default:
			log.Fatalf(`%s:%d: invalid symabi type "%s"`, file, lineNum, parts[0])
		}
	}
}

func saveerrors() {
	nsavederrors += nerrors
	nerrors = 0
}

func arsize(b *bufio.Reader, name string) int {
	var buf [ArhdrSize]byte
	if _, err := io.ReadFull(b, buf[:]); err != nil {
		return -1
	}
	aname := strings.Trim(string(buf[0:16]), " ")
	if !strings.HasPrefix(aname, name) {
		return -1
	}
	asize := strings.Trim(string(buf[48:58]), " ")
	i, _ := strconv.Atoi(asize)
	return i
}

var idirs []string

func addidir(dir string) {
	if dir != "" {
		idirs = append(idirs, dir)
	}
}

func isDriveLetter(b byte) bool {
	return 'a' <= b && b <= 'z' || 'A' <= b && b <= 'Z'
}

// is this path a local name? begins with ./ or ../ or /
func islocalname(name string) bool {
	return strings.HasPrefix(name, "/") ||
		runtime.GOOS == "windows" && len(name) >= 3 && isDriveLetter(name[0]) && name[1] == ':' && name[2] == '/' ||
		strings.HasPrefix(name, "./") || name == "." ||
		strings.HasPrefix(name, "../") || name == ".."
}

func findpkg(name string) (file string, ok bool) {
	if islocalname(name) {
		if nolocalimports {
			return "", false
		}

		if packageFile != nil {
			file, ok = packageFile[name]
			return file, ok
		}

		// try .a before .6.  important for building libraries:
		// if there is an array.6 in the array.a library,
		// want to find all of array.a, not just array.6.
		file = fmt.Sprintf("%s.a", name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s.o", name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		return "", false
	}

	// local imports should be canonicalized already.
	// don't want to see "encoding/../encoding/base64"
	// as different from "encoding/base64".
	if q := path.Clean(name); q != name {
		yyerror("non-canonical import path %q (should be %q)", name, q)
		return "", false
	}

	if packageFile != nil {
		file, ok = packageFile[name]
		return file, ok
	}

	for _, dir := range idirs {
		file = fmt.Sprintf("%s/%s.a", dir, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s/%s.o", dir, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
	}

	if objabi.GOROOT != "" {
		suffix := ""
		suffixsep := ""
		if flag_installsuffix != "" {
			suffixsep = "_"
			suffix = flag_installsuffix
		} else if flag_race {
			suffixsep = "_"
			suffix = "race"
		} else if flag_msan {
			suffixsep = "_"
			suffix = "msan"
		}

		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.a", objabi.GOROOT, objabi.GOOS, objabi.GOARCH, suffixsep, suffix, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.o", objabi.GOROOT, objabi.GOOS, objabi.GOARCH, suffixsep, suffix, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
	}

	return "", false
}

// loadsys loads the definitions for the low-level runtime functions,
// so that the compiler can generate calls to them,
// but does not make them visible to user code.
func loadsys() {
	types.Block = 1

	inimport = true
	typecheckok = true

	typs := runtimeTypes()
	for _, d := range &runtimeDecls {
		sym := Runtimepkg.Lookup(d.name)
		typ := typs[d.typ]
		switch d.tag {
		case funcTag:
			importfunc(Runtimepkg, src.NoXPos, sym, typ)
		case varTag:
			importvar(Runtimepkg, src.NoXPos, sym, typ)
		default:
			Fatalf("unhandled declaration tag %v", d.tag)
		}
	}

	typecheckok = false
	inimport = false
}

// myheight tracks the local package's height based on packages
// imported so far.
var myheight int

func importfile(f *Val) *types.Pkg {
	path_, ok := f.U.(string)
	if !ok {
		yyerror("import path must be a string")
		return nil
	}

	if len(path_) == 0 {
		yyerror("import path is empty")
		return nil
	}

	if isbadimport(path_, false) {
		return nil
	}

	// The package name main is no longer reserved,
	// but we reserve the import path "main" to identify
	// the main package, just as we reserve the import
	// path "math" to identify the standard math package.
	if path_ == "main" {
		yyerror("cannot import \"main\"")
		errorexit()
	}

	if myimportpath != "" && path_ == myimportpath {
		yyerror("import %q while compiling that package (import cycle)", path_)
		errorexit()
	}

	if mapped, ok := importMap[path_]; ok {
		path_ = mapped
	}

	if path_ == "unsafe" {
		imported_unsafe = true
		return unsafepkg
	}

	if islocalname(path_) {
		if path_[0] == '/' {
			yyerror("import path cannot be absolute path")
			return nil
		}

		prefix := Ctxt.Pathname
		if localimport != "" {
			prefix = localimport
		}
		path_ = path.Join(prefix, path_)

		if isbadimport(path_, true) {
			return nil
		}
	}

	file, found := findpkg(path_)
	if !found {
		yyerror("can't find import: %q", path_)
		errorexit()
	}

	importpkg := types.NewPkg(path_, "")
	if importpkg.Imported {
		return importpkg
	}

	importpkg.Imported = true

	imp, err := bio.Open(file)
	if err != nil {
		yyerror("can't open import: %q: %v", path_, err)
		errorexit()
	}
	defer imp.Close()

	// check object header
	p, err := imp.ReadString('\n')
	if err != nil {
		yyerror("import %s: reading input: %v", file, err)
		errorexit()
	}

	if p == "!<arch>\n" { // package archive
		// package export block should be first
		sz := arsize(imp.Reader, "__.PKGDEF")
		if sz <= 0 {
			yyerror("import %s: not a package file", file)
			errorexit()
		}
		p, err = imp.ReadString('\n')
		if err != nil {
			yyerror("import %s: reading input: %v", file, err)
			errorexit()
		}
	}

	if !strings.HasPrefix(p, "go object ") {
		yyerror("import %s: not a go object file: %s", file, p)
		errorexit()
	}
	q := fmt.Sprintf("%s %s %s %s\n", objabi.GOOS, objabi.GOARCH, objabi.Version, objabi.Expstring())
	if p[10:] != q {
		yyerror("import %s: object is [%s] expected [%s]", file, p[10:], q)
		errorexit()
	}

	// process header lines
	for {
		p, err = imp.ReadString('\n')
		if err != nil {
			yyerror("import %s: reading input: %v", file, err)
			errorexit()
		}
		if p == "\n" {
			break // header ends with blank line
		}
	}

	// assume files move (get installed) so don't record the full path
	if packageFile != nil {
		// If using a packageFile map, assume path_ can be recorded directly.
		Ctxt.AddImport(path_)
	} else {
		// For file "/Users/foo/go/pkg/darwin_amd64/math.a" record "math.a".
		Ctxt.AddImport(file[len(file)-len(path_)-len(".a"):])
	}

	// In the importfile, if we find:
	// $$\n  (textual format): not supported anymore
	// $$B\n (binary format) : import directly, then feed the lexer a dummy statement

	// look for $$
	var c byte
	for {
		c, err = imp.ReadByte()
		if err != nil {
			break
		}
		if c == '$' {
			c, err = imp.ReadByte()
			if c == '$' || err != nil {
				break
			}
		}
	}

	// get character after $$
	if err == nil {
		c, _ = imp.ReadByte()
	}

	switch c {
	case '\n':
		yyerror("cannot import %s: old export format no longer supported (recompile library)", path_)
		return nil

	case 'B':
		if Debug_export != 0 {
			fmt.Printf("importing %s (%s)\n", path_, file)
		}
		imp.ReadByte() // skip \n after $$B

		c, err = imp.ReadByte()
		if err != nil {
			yyerror("import %s: reading input: %v", file, err)
			errorexit()
		}

		// Indexed format is distinguished by an 'i' byte,
		// whereas previous export formats started with 'c', 'd', or 'v'.
		if c != 'i' {
			yyerror("import %s: unexpected package format byte: %v", file, c)
			errorexit()
		}
		iimport(importpkg, imp)

	default:
		yyerror("no import in %q", path_)
		errorexit()
	}

	if importpkg.Height >= myheight {
		myheight = importpkg.Height + 1
	}

	return importpkg
}

func pkgnotused(lineno src.XPos, path string, name string) {
	// If the package was imported with a name other than the final
	// import path element, show it explicitly in the error message.
	// Note that this handles both renamed imports and imports of
	// packages containing unconventional package declarations.
	// Note that this uses / always, even on Windows, because Go import
	// paths always use forward slashes.
	elem := path
	if i := strings.LastIndex(elem, "/"); i >= 0 {
		elem = elem[i+1:]
	}
	if name == "" || elem == name {
		yyerrorl(lineno, "imported and not used: %q", path)
	} else {
		yyerrorl(lineno, "imported and not used: %q as %s", path, name)
	}
}

func mkpackage(pkgname string) {
	if localpkg.Name == "" {
		if pkgname == "_" {
			yyerror("invalid package name _")
		}
		localpkg.Name = pkgname
	} else {
		if pkgname != localpkg.Name {
			yyerror("package %s; expected %s", pkgname, localpkg.Name)
		}
	}
}

func clearImports() {
	type importedPkg struct {
		pos  src.XPos
		path string
		name string
	}
	var unused []importedPkg

	for _, s := range localpkg.Syms {
		n := asNode(s.Def)
		if n == nil {
			continue
		}
		if n.Op == OPACK {
			// throw away top-level package name left over
			// from previous file.
			// leave s->block set to cause redeclaration
			// errors if a conflicting top-level name is
			// introduced by a different file.
			if !n.Name.Used() && nsyntaxerrors == 0 {
				unused = append(unused, importedPkg{n.Pos, n.Name.Pkg.Path, s.Name})
			}
			s.Def = nil
			continue
		}
		if IsAlias(s) {
			// throw away top-level name left over
			// from previous import . "x"
			if n.Name != nil && n.Name.Pack != nil && !n.Name.Pack.Name.Used() && nsyntaxerrors == 0 {
				unused = append(unused, importedPkg{n.Name.Pack.Pos, n.Name.Pack.Name.Pkg.Path, ""})
				n.Name.Pack.Name.SetUsed(true)
			}
			s.Def = nil
			continue
		}
	}

	sort.Slice(unused, func(i, j int) bool { return unused[i].pos.Before(unused[j].pos) })
	for _, pkg := range unused {
		pkgnotused(pkg.pos, pkg.path, pkg.name)
	}
}

func IsAlias(sym *types.Sym) bool {
	return sym.Def != nil && asNode(sym.Def).Sym != sym
}

// By default, assume any debug flags are incompatible with concurrent compilation.
// A few are safe and potentially in common use for normal compiles, though; mark them as such here.
var concurrentFlagOK = [256]bool{
	'B': true, // disabled bounds checking
	'C': true, // disable printing of columns in error messages
	'e': true, // no limit on errors; errors all come from non-concurrent code
	'I': true, // add `directory` to import search path
	'N': true, // disable optimizations
	'l': true, // disable inlining
	'w': true, // all printing happens before compilation
	'W': true, // all printing happens before compilation
	'S': true, // printing disassembly happens at the end (but see concurrentBackendAllowed below)
}

func concurrentBackendAllowed() bool {
	for i, x := range &Debug {
		if x != 0 && !concurrentFlagOK[i] {
			return false
		}
	}
	// Debug['S'] by itself is ok, because all printing occurs
	// while writing the object file, and that is non-concurrent.
	// Adding Debug_vlog, however, causes Debug['S'] to also print
	// while flushing the plist, which happens concurrently.
	if Debug_vlog || debugstr != "" || debuglive > 0 {
		return false
	}
	// TODO: Test and delete this condition.
	if objabi.Fieldtrack_enabled != 0 {
		return false
	}
	// TODO: fix races and enable the following flags
	if Ctxt.Flag_shared || Ctxt.Flag_dynlink || flag_race {
		return false
	}
	return true
}

// recordFlags records the specified command-line flags to be placed
// in the DWARF info.
func recordFlags(flags ...string) {
	if myimportpath == "" {
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
	s := Ctxt.Lookup(dwarf.CUInfoPrefix + "producer." + myimportpath)
	s.Type = objabi.SDWARFINFO
	// Sometimes (for example when building tests) we can link
	// together two package main archives. So allow dups.
	s.Set(obj.AttrDuplicateOK, true)
	Ctxt.Data = append(Ctxt.Data, s)
	s.P = cmd.Bytes()[1:]
}

// recordPackageName records the name of the package being
// compiled, so that the linker can save it in the compile unit's DIE.
func recordPackageName() {
	s := Ctxt.Lookup(dwarf.CUInfoPrefix + "packagename." + myimportpath)
	s.Type = objabi.SDWARFINFO
	// Sometimes (for example when building tests) we can link
	// together two package main archives. So allow dups.
	s.Set(obj.AttrDuplicateOK, true)
	Ctxt.Data = append(Ctxt.Data, s)
	s.P = []byte(localpkg.Name)
}

// flag_lang is the language version we are compiling for, set by the -lang flag.
var flag_lang string

// currentLang returns the current language version.
func currentLang() string {
	return fmt.Sprintf("go1.%d", goversion.Version)
}

// goVersionRE is a regular expression that matches the valid
// arguments to the -lang flag.
var goVersionRE = regexp.MustCompile(`^go([1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

// A lang is a language version broken into major and minor numbers.
type lang struct {
	major, minor int
}

// langWant is the desired language version set by the -lang flag.
// If the -lang flag is not set, this is the zero value, meaning that
// any language version is supported.
var langWant lang

// langSupported reports whether language version major.minor is
// supported in a particular package.
func langSupported(major, minor int, pkg *types.Pkg) bool {
	if pkg == nil {
		// TODO(mdempsky): Set Pkg for local types earlier.
		pkg = localpkg
	}
	if pkg != localpkg {
		// Assume imported packages passed type-checking.
		return true
	}

	if langWant.major == 0 && langWant.minor == 0 {
		return true
	}
	return langWant.major > major || (langWant.major == major && langWant.minor >= minor)
}

// checkLang verifies that the -lang flag holds a valid value, and
// exits if not. It initializes data used by langSupported.
func checkLang() {
	if flag_lang == "" {
		return
	}

	var err error
	langWant, err = parseLang(flag_lang)
	if err != nil {
		log.Fatalf("invalid value %q for -lang: %v", flag_lang, err)
	}

	if def := currentLang(); flag_lang != def {
		defVers, err := parseLang(def)
		if err != nil {
			log.Fatalf("internal error parsing default lang %q: %v", def, err)
		}
		if langWant.major > defVers.major || (langWant.major == defVers.major && langWant.minor > defVers.minor) {
			log.Fatalf("invalid value %q for -lang: max known version is %q", flag_lang, def)
		}
	}
}

// parseLang parses a -lang option into a langVer.
func parseLang(s string) (lang, error) {
	matches := goVersionRE.FindStringSubmatch(s)
	if matches == nil {
		return lang{}, fmt.Errorf(`should be something like "go1.12"`)
	}
	major, err := strconv.Atoi(matches[1])
	if err != nil {
		return lang{}, err
	}
	minor, err := strconv.Atoi(matches[2])
	if err != nil {
		return lang{}, err
	}
	return lang{major: major, minor: minor}, nil
}
