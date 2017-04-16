// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run mkbuiltin.go

package gc

import (
	"bufio"
	"bytes"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"cmd/internal/sys"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"runtime"
	"strconv"
	"strings"
)

var imported_unsafe bool

var (
	buildid string
)

var (
	Debug_append   int
	Debug_asm      bool
	Debug_closure  int
	debug_dclstack int
	Debug_panic    int
	Debug_slice    int
	Debug_vlog     bool
	Debug_wb       int
	Debug_pctab    string
)

// Debug arguments.
// These can be specified with the -d flag, as in "-d nil"
// to set the debug_checknil variable.
// Multiple options can be comma-separated.
// Each option accepts an optional argument, as in "gcprog=2"
var debugtab = []struct {
	name string
	val  interface{} // must be *int or *string
}{
	{"append", &Debug_append},         // print information about append compilation
	{"closure", &Debug_closure},       // print information about closure compilation
	{"disablenil", &disable_checknil}, // disable nil checks
	{"dclstack", &debug_dclstack},     // run internal dclstack checks
	{"gcprog", &Debug_gcprog},         // print dump of GC programs
	{"nil", &Debug_checknil},          // print information about nil checks
	{"panic", &Debug_panic},           // do not hide any compiler panic
	{"slice", &Debug_slice},           // print information about slice compilation
	{"typeassert", &Debug_typeassert}, // print information about type assertion inlining
	{"wb", &Debug_wb},                 // print information about write barriers
	{"export", &Debug_export},         // print export data
	{"pctab", &Debug_pctab},           // print named pc-value table
}

func usage() {
	fmt.Printf("usage: compile [options] file.go...\n")
	obj.Flagprint(1)
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

func doversion() {
	p := obj.Expstring()
	if p == "X:none" {
		p = ""
	}
	sep := ""
	if p != "" {
		sep = " "
	}
	fmt.Printf("compile version %s%s%s\n", obj.Version, sep, p)
	os.Exit(0)
}

// supportsDynlink reports whether or not the code generator for the given
// architecture supports the -shared and -dynlink flags.
func supportsDynlink(arch *sys.Arch) bool {
	return arch.InFamily(sys.AMD64, sys.ARM, sys.ARM64, sys.I386, sys.PPC64, sys.S390X)
}

// timing data for compiler phases
var timings Timings
var benchfile string

// Main parses flags and Go source files specified in the command-line
// arguments, type-checks the parsed Go package, compiles functions to machine
// code, and finally writes the compiled package definition to disk.
func Main(archInit func(*Arch)) {
	timings.Start("fe", "init")

	defer hidePanic()

	archInit(&thearch)

	Ctxt = obj.Linknew(thearch.LinkArch)
	Ctxt.DiagFunc = yyerror
	Ctxt.Bso = bufio.NewWriter(os.Stdout)

	localpkg = mkpkg("")
	localpkg.Prefix = "\"\""

	// pseudo-package, for scoping
	builtinpkg = mkpkg("go.builtin")
	builtinpkg.Prefix = "go.builtin" // not go%2ebuiltin

	// pseudo-package, accessed by import "unsafe"
	unsafepkg = mkpkg("unsafe")
	unsafepkg.Name = "unsafe"

	// Pseudo-package that contains the compiler's builtin
	// declarations for package runtime. These are declared in a
	// separate package to avoid conflicts with package runtime's
	// actual declarations, which may differ intentionally but
	// insignificantly.
	Runtimepkg = mkpkg("go.runtime")
	Runtimepkg.Name = "runtime"
	Runtimepkg.Prefix = "runtime"

	// pseudo-packages used in symbol tables
	itabpkg = mkpkg("go.itab")
	itabpkg.Name = "go.itab"
	itabpkg.Prefix = "go.itab" // not go%2eitab

	itablinkpkg = mkpkg("go.itablink")
	itablinkpkg.Name = "go.itablink"
	itablinkpkg.Prefix = "go.itablink" // not go%2eitablink

	trackpkg = mkpkg("go.track")
	trackpkg.Name = "go.track"
	trackpkg.Prefix = "go.track" // not go%2etrack

	typepkg = mkpkg("type")
	typepkg.Name = "type"

	// pseudo-package used for map zero values
	mappkg = mkpkg("go.map")
	mappkg.Name = "go.map"
	mappkg.Prefix = "go.map"

	Nacl = obj.GOOS == "nacl"

	flag.BoolVar(&compiling_runtime, "+", false, "compiling runtime")
	obj.Flagcount("%", "debug non-static initializers", &Debug['%'])
	obj.Flagcount("B", "disable bounds checking", &Debug['B'])
	obj.Flagcount("C", "disable printing of columns in error messages", &Debug['C']) // TODO(gri) remove eventually
	flag.StringVar(&localimport, "D", "", "set relative `path` for local imports")
	obj.Flagcount("E", "debug symbol export", &Debug['E'])
	obj.Flagfn1("I", "add `directory` to import search path", addidir)
	obj.Flagcount("K", "debug missing line numbers", &Debug['K'])
	obj.Flagcount("N", "disable optimizations", &Debug['N'])
	flag.BoolVar(&Debug_asm, "S", false, "print assembly listing")
	obj.Flagfn0("V", "print compiler version", doversion)
	obj.Flagcount("W", "debug parse tree after type checking", &Debug['W'])
	flag.StringVar(&asmhdr, "asmhdr", "", "write assembly header to `file`")
	flag.StringVar(&buildid, "buildid", "", "record `id` as the build id in the export metadata")
	flag.BoolVar(&pure_go, "complete", false, "compiling complete package (no C or assembly)")
	flag.StringVar(&debugstr, "d", "", "print debug information about items in `list`")
	flag.BoolVar(&flagDWARF, "dwarf", true, "generate DWARF symbols")
	obj.Flagcount("e", "no limit on number of errors reported", &Debug['e'])
	obj.Flagcount("f", "debug stack frames", &Debug['f'])
	obj.Flagcount("h", "halt on error", &Debug['h'])
	obj.Flagcount("i", "debug line number stack", &Debug['i'])
	obj.Flagfn1("importmap", "add `definition` of the form source=actual to import map", addImportMap)
	flag.StringVar(&flag_installsuffix, "installsuffix", "", "set pkg directory `suffix`")
	obj.Flagcount("j", "debug runtime-initialized variables", &Debug['j'])
	obj.Flagcount("l", "disable inlining", &Debug['l'])
	flag.StringVar(&linkobj, "linkobj", "", "write linker-specific object to `file`")
	obj.Flagcount("live", "debug liveness analysis", &debuglive)
	obj.Flagcount("m", "print optimization decisions", &Debug['m'])
	flag.BoolVar(&flag_msan, "msan", false, "build code compatible with C/C++ memory sanitizer")
	flag.BoolVar(&dolinkobj, "dolinkobj", true, "generate linker-specific objects; if false, some invalid code may compile")
	flag.BoolVar(&nolocalimports, "nolocalimports", false, "reject local (relative) imports")
	flag.StringVar(&outfile, "o", "", "write output to `file`")
	flag.StringVar(&myimportpath, "p", "", "set expected package import `path`")
	flag.BoolVar(&writearchive, "pack", false, "write package file instead of object file")
	obj.Flagcount("r", "debug generated wrappers", &Debug['r'])
	flag.BoolVar(&flag_race, "race", false, "enable race detector")
	obj.Flagcount("s", "warn about composite literals that can be simplified", &Debug['s'])
	flag.StringVar(&pathPrefix, "trimpath", "", "remove `prefix` from recorded source file paths")
	flag.BoolVar(&safemode, "u", false, "reject unsafe code")
	flag.BoolVar(&Debug_vlog, "v", false, "increase debug verbosity")
	obj.Flagcount("w", "debug type checking", &Debug['w'])
	flag.BoolVar(&use_writebarrier, "wb", true, "enable write barrier")
	var flag_shared bool
	var flag_dynlink bool
	if supportsDynlink(thearch.LinkArch.Arch) {
		flag.BoolVar(&flag_shared, "shared", false, "generate code that can be linked into a shared library")
		flag.BoolVar(&flag_dynlink, "dynlink", false, "support references to Go symbols defined in other shared libraries")
	}
	flag.StringVar(&cpuprofile, "cpuprofile", "", "write cpu profile to `file`")
	flag.StringVar(&memprofile, "memprofile", "", "write memory profile to `file`")
	flag.Int64Var(&memprofilerate, "memprofilerate", 0, "set runtime.MemProfileRate to `rate`")
	flag.StringVar(&traceprofile, "traceprofile", "", "write an execution trace to `file`")
	flag.StringVar(&blockprofile, "blockprofile", "", "write block profile to `file`")
	flag.StringVar(&mutexprofile, "mutexprofile", "", "write mutex profile to `file`")
	flag.StringVar(&benchfile, "bench", "", "append benchmark times to `file`")
	obj.Flagparse(usage)

	Ctxt.Flag_shared = flag_dynlink || flag_shared
	Ctxt.Flag_dynlink = flag_dynlink
	Ctxt.Flag_optimize = Debug['N'] == 0

	Ctxt.Debugasm = Debug_asm
	Ctxt.Debugvlog = Debug_vlog
	if flagDWARF {
		Ctxt.DebugInfo = debuginfo
	}

	if flag.NArg() < 1 {
		usage()
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

	if flag_race {
		racepkg = mkpkg("runtime/race")
		racepkg.Name = "race"
	}
	if flag_msan {
		msanpkg = mkpkg("runtime/msan")
		msanpkg.Name = "msan"
	}
	if flag_race && flag_msan {
		log.Fatal("cannot use both -race and -msan")
	} else if flag_race || flag_msan {
		instrumenting = true
	}
	if compiling_runtime && Debug['N'] != 0 {
		log.Fatal("cannot disable optimizations while compiling runtime")
	}

	// parse -d argument
	if debugstr != "" {
	Split:
		for _, name := range strings.Split(debugstr, ",") {
			if name == "" {
				continue
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

	// set via a -d flag
	Ctxt.Debugpcln = Debug_pctab

	// enable inlining.  for now:
	//	default: inlining on.  (debug['l'] == 1)
	//	-l: inlining off  (debug['l'] == 0)
	//	-ll, -lll: inlining on again, with extra debugging (debug['l'] > 1)
	if Debug['l'] <= 1 {
		Debug['l'] = 1 - Debug['l']
	}

	Widthint = thearch.LinkArch.IntSize
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
	types.Tconv = func(t *types.Type, flag, mode, depth int) string {
		return tconv(t, FmtFlag(flag), fmtMode(mode), depth)
	}
	types.FormatSym = func(sym *types.Sym, s fmt.State, verb rune, mode int) {
		symFormat(sym, s, verb, fmtMode(mode))
	}
	types.FormatType = func(t *types.Type, s fmt.State, verb rune, mode int) {
		typeFormat(t, s, verb, fmtMode(mode))
	}
	types.FieldName = func(f *types.Field) string {
		return f.Sym.Name
	}
	types.TypeLinkSym = func(t *types.Type) *obj.LSym {
		return Linksym(typenamesym(t))
	}
	types.FmtLeft = int(FmtLeft)
	types.FmtUnsigned = int(FmtUnsigned)
	types.FErr = FErr

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

	typecheckok = true
	if Debug['f'] != 0 {
		frame(1)
	}

	// Process top-level declarations in phases.

	// Phase 1: const, type, and names and types of funcs.
	//   This will gather all the information about types
	//   and methods but doesn't depend on any of it.
	//   We also defer type alias declarations until phase 2
	//   to avoid cycles like #18640.
	defercheckwidth()

	// Don't use range--typecheck can add closures to xtop.
	timings.Start("fe", "typecheck", "top1")
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if op := n.Op; op != ODCL && op != OAS && op != OAS2 && (op != ODCLTYPE || !n.Left.Name.Param.Alias) {
			xtop[i] = typecheck(n, Etop)
		}
	}

	// Phase 2: Variable assignments.
	//   To check interface assignments, depends on phase 1.

	// Don't use range--typecheck can add closures to xtop.
	timings.Start("fe", "typecheck", "top2")
	for i := 0; i < len(xtop); i++ {
		n := xtop[i]
		if op := n.Op; op == ODCL || op == OAS || op == OAS2 || op == ODCLTYPE && n.Left.Name.Param.Alias {
			xtop[i] = typecheck(n, Etop)
		}
	}
	resumecheckwidth()

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
			typecheckslice(Curfn.Nbody.Slice(), Etop)
			checkreturn(Curfn)
			if nerrors != 0 {
				Curfn.Nbody.Set(nil) // type errors; do not compile
			}
			fcount++
		}
	}
	timings.AddEvent(fcount, "funcs")

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

	Curfn = nil

	if nsavederrors+nerrors != 0 {
		errorexit()
	}

	// Phase 5: Inlining
	timings.Start("fe", "inlining")
	if Debug['l'] > 1 {
		// Typecheck imported function bodies if debug['l'] > 1,
		// otherwise lazily when used or re-exported.
		for _, n := range importlist {
			if n.Func.Inl.Len() != 0 {
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

	if dolinkobj {
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

		if compiling_runtime {
			checknowritebarrierrec()
		}
		for _, largePos := range largeStackFrames {
			yyerrorl(largePos, "stack frame too large (>2GB)")
		}
	}

	// Phase 9: Check external declarations.
	timings.Start("be", "externaldcls")
	for i, n := range externdcl {
		if n.Op == ONAME {
			externdcl[i] = typecheck(externdcl[i], Erv)
		}
	}

	if nerrors+nsavederrors != 0 {
		errorexit()
	}

	// Write object data to disk.
	timings.Start("be", "dumpobj")
	dumpobj()
	if asmhdr != "" {
		dumpasmhdr()
	}

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
	fmt.Fprintln(&buf, "commit:", obj.Version)
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

var importMap = map[string]string{}

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

func skiptopkgdef(b *bufio.Reader) bool {
	// archive header
	p, err := b.ReadString('\n')
	if err != nil {
		log.Fatalf("reading input: %v", err)
	}
	if p != "!<arch>\n" {
		return false
	}

	// package export block should be first
	sz := arsize(b, "__.PKGDEF")
	return sz > 0
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

// is this path a local name?  begins with ./ or ../ or /
func islocalname(name string) bool {
	return strings.HasPrefix(name, "/") ||
		runtime.GOOS == "windows" && len(name) >= 3 && isDriveLetter(name[0]) && name[1] == ':' && name[2] == '/' ||
		strings.HasPrefix(name, "./") || name == "." ||
		strings.HasPrefix(name, "../") || name == ".."
}

func findpkg(name string) (file string, ok bool) {
	if islocalname(name) {
		if safemode || nolocalimports {
			return "", false
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

	if obj.GOROOT != "" {
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

		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.a", obj.GOROOT, obj.GOOS, obj.GOARCH, suffixsep, suffix, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.o", obj.GOROOT, obj.GOOS, obj.GOARCH, suffixsep, suffix, name)
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
	defercheckwidth()

	typs := runtimeTypes()
	for _, d := range runtimeDecls {
		sym := Runtimepkg.Lookup(d.name)
		typ := typs[d.typ]
		switch d.tag {
		case funcTag:
			importsym(Runtimepkg, sym, ONAME)
			n := newfuncname(sym)
			n.Type = typ
			declare(n, PFUNC)
		case varTag:
			importvar(Runtimepkg, sym, typ)
		default:
			Fatalf("unhandled declaration tag %v", d.tag)
		}
	}

	typecheckok = false
	resumecheckwidth()
	inimport = false
}

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

	if isbadimport(path_) {
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
		if safemode {
			yyerror("cannot import package unsafe")
			errorexit()
		}

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

		if isbadimport(path_) {
			return nil
		}
	}

	file, found := findpkg(path_)
	if !found {
		yyerror("can't find import: %q", path_)
		errorexit()
	}

	importpkg := mkpkg(path_)
	if importpkg.Imported {
		return importpkg
	}

	importpkg.Imported = true

	impf, err := os.Open(file)
	if err != nil {
		yyerror("can't open import: %q: %v", path_, err)
		errorexit()
	}
	defer impf.Close()
	imp := bufio.NewReader(impf)

	const pkgSuffix = ".a"
	if strings.HasSuffix(file, pkgSuffix) {
		if !skiptopkgdef(imp) {
			yyerror("import %s: not a package file", file)
			errorexit()
		}
	}

	// check object header
	p, err := imp.ReadString('\n')
	if err != nil {
		log.Fatalf("reading input: %v", err)
	}
	if len(p) > 0 {
		p = p[:len(p)-1]
	}

	if p != "empty archive" {
		if !strings.HasPrefix(p, "go object ") {
			yyerror("import %s: not a go object file: %s", file, p)
			errorexit()
		}

		q := fmt.Sprintf("%s %s %s %s", obj.GOOS, obj.GOARCH, obj.Version, obj.Expstring())
		if p[10:] != q {
			yyerror("import %s: object is [%s] expected [%s]", file, p[10:], q)
			errorexit()
		}
	}

	// process header lines
	safe := false
	for {
		p, err = imp.ReadString('\n')
		if err != nil {
			log.Fatalf("reading input: %v", err)
		}
		if p == "\n" {
			break // header ends with blank line
		}
		if strings.HasPrefix(p, "safe") {
			safe = true
			break // ok to ignore rest
		}
	}
	if safemode && !safe {
		yyerror("cannot import unsafe package %q", importpkg.Path)
	}

	// assume files move (get installed) so don't record the full path
	// (e.g., for file "/Users/foo/go/pkg/darwin_amd64/math.a" record "math.a")
	Ctxt.AddImport(file[len(file)-len(path_)-len(pkgSuffix):])

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
		Import(importpkg, imp)

	default:
		yyerror("no import in %q", path_)
		errorexit()
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
	for _, s := range localpkg.Syms {
		if asNode(s.Def) == nil {
			continue
		}
		if asNode(s.Def).Op == OPACK {
			// throw away top-level package name leftover
			// from previous file.
			// leave s->block set to cause redeclaration
			// errors if a conflicting top-level name is
			// introduced by a different file.
			if !asNode(s.Def).Used() && nsyntaxerrors == 0 {
				pkgnotused(asNode(s.Def).Pos, asNode(s.Def).Name.Pkg.Path, s.Name)
			}
			s.Def = nil
			continue
		}

		if IsAlias(s) {
			// throw away top-level name left over
			// from previous import . "x"
			if asNode(s.Def).Name != nil && asNode(s.Def).Name.Pack != nil && !asNode(s.Def).Name.Pack.Used() && nsyntaxerrors == 0 {
				pkgnotused(asNode(s.Def).Name.Pack.Pos, asNode(s.Def).Name.Pack.Name.Pkg.Path, "")
				asNode(s.Def).Name.Pack.SetUsed(true)
			}

			s.Def = nil
			continue
		}
	}
}

func IsAlias(sym *types.Sym) bool {
	return sym.Def != nil && asNode(sym.Def).Sym != sym
}
