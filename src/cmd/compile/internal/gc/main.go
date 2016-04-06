// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run mkbuiltin.go

package gc

import (
	"bufio"
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
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
	goos    string
	goarch  string
	goroot  string
	buildid string
)

var (
	Debug_append  int
	Debug_closure int
	Debug_panic   int
	Debug_slice   int
	Debug_wb      int
)

// Debug arguments.
// These can be specified with the -d flag, as in "-d nil"
// to set the debug_checknil variable. In general the list passed
// to -d can be comma-separated.
var debugtab = []struct {
	name string
	val  *int
}{
	{"append", &Debug_append},         // print information about append compilation
	{"closure", &Debug_closure},       // print information about closure compilation
	{"disablenil", &Disable_checknil}, // disable nil checks
	{"gcprog", &Debug_gcprog},         // print dump of GC programs
	{"nil", &Debug_checknil},          // print information about nil checks
	{"panic", &Debug_panic},           // do not hide any compiler panic
	{"slice", &Debug_slice},           // print information about slice compilation
	{"typeassert", &Debug_typeassert}, // print information about type assertion inlining
	{"wb", &Debug_wb},                 // print information about write barriers
	{"export", &Debug_export},         // print export data
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
	fmt.Printf("compile version %s%s%s\n", obj.Getgoversion(), sep, p)
	os.Exit(0)
}

func Main() {
	defer hidePanic()

	// Allow GOARCH=thearch.thestring or GOARCH=thearch.thestringsuffix,
	// but not other values.
	p := obj.Getgoarch()

	if !strings.HasPrefix(p, Thearch.LinkArch.Name) {
		log.Fatalf("cannot use %cg with GOARCH=%s", Thearch.LinkArch.Family, p)
	}
	goarch = p

	Ctxt = obj.Linknew(Thearch.LinkArch)
	Ctxt.DiagFunc = Yyerror
	Ctxt.Bso = &bstdout
	bstdout = *obj.Binitw(os.Stdout)

	localpkg = mkpkg("")
	localpkg.Prefix = "\"\""

	// pseudo-package, for scoping
	builtinpkg = mkpkg("go.builtin")
	builtinpkg.Prefix = "go.builtin" // not go%2ebuiltin

	// pseudo-package, accessed by import "unsafe"
	unsafepkg = mkpkg("unsafe")
	unsafepkg.Name = "unsafe"

	// real package, referred to by generated runtime calls
	Runtimepkg = mkpkg("runtime")
	Runtimepkg.Name = "runtime"

	// pseudo-packages used in symbol tables
	itabpkg = mkpkg("go.itab")
	itabpkg.Name = "go.itab"
	itabpkg.Prefix = "go.itab" // not go%2eitab

	typelinkpkg = mkpkg("go.typelink")
	typelinkpkg.Name = "go.typelink"
	typelinkpkg.Prefix = "go.typelink" // not go%2etypelink

	itablinkpkg = mkpkg("go.itablink")
	itablinkpkg.Name = "go.itablink"
	itablinkpkg.Prefix = "go.itablink" // not go%2eitablink

	trackpkg = mkpkg("go.track")
	trackpkg.Name = "go.track"
	trackpkg.Prefix = "go.track" // not go%2etrack

	typepkg = mkpkg("type")
	typepkg.Name = "type"

	goroot = obj.Getgoroot()
	goos = obj.Getgoos()

	Nacl = goos == "nacl"
	if Nacl {
		flag_largemodel = 1
	}

	outfile = ""
	obj.Flagcount("+", "compiling runtime", &compiling_runtime)
	obj.Flagcount("%", "debug non-static initializers", &Debug['%'])
	obj.Flagcount("A", "for bootstrapping, allow 'any' type", &Debug['A'])
	obj.Flagcount("B", "disable bounds checking", &Debug['B'])
	obj.Flagstr("D", "set relative `path` for local imports", &localimport)
	obj.Flagcount("E", "debug symbol export", &Debug['E'])
	obj.Flagfn1("I", "add `directory` to import search path", addidir)
	obj.Flagcount("K", "debug missing line numbers", &Debug['K'])
	obj.Flagcount("L", "use full (long) path in error messages", &Debug['L'])
	obj.Flagcount("M", "debug move generation", &Debug['M'])
	obj.Flagcount("N", "disable optimizations", &Debug['N'])
	obj.Flagcount("P", "debug peephole optimizer", &Debug['P'])
	obj.Flagcount("R", "debug register optimizer", &Debug['R'])
	obj.Flagcount("S", "print assembly listing", &Debug['S'])
	obj.Flagfn0("V", "print compiler version", doversion)
	obj.Flagcount("W", "debug parse tree after type checking", &Debug['W'])
	obj.Flagstr("asmhdr", "write assembly header to `file`", &asmhdr)
	obj.Flagstr("buildid", "record `id` as the build id in the export metadata", &buildid)
	obj.Flagcount("complete", "compiling complete package (no C or assembly)", &pure_go)
	obj.Flagstr("d", "print debug information about items in `list`", &debugstr)
	obj.Flagcount("e", "no limit on number of errors reported", &Debug['e'])
	obj.Flagcount("f", "debug stack frames", &Debug['f'])
	obj.Flagcount("g", "debug code generation", &Debug['g'])
	obj.Flagcount("h", "halt on error", &Debug['h'])
	obj.Flagcount("i", "debug line number stack", &Debug['i'])
	obj.Flagfn1("importmap", "add `definition` of the form source=actual to import map", addImportMap)
	obj.Flagstr("installsuffix", "set pkg directory `suffix`", &flag_installsuffix)
	obj.Flagcount("j", "debug runtime-initialized variables", &Debug['j'])
	obj.Flagcount("l", "disable inlining", &Debug['l'])
	obj.Flagcount("live", "debug liveness analysis", &debuglive)
	obj.Flagcount("m", "print optimization decisions", &Debug['m'])
	obj.Flagcount("msan", "build code compatible with C/C++ memory sanitizer", &flag_msan)
	obj.Flagcount("newexport", "use new export format", &newexport) // TODO(gri) remove eventually (issue 13241)
	obj.Flagcount("nolocalimports", "reject local (relative) imports", &nolocalimports)
	obj.Flagstr("o", "write output to `file`", &outfile)
	obj.Flagstr("p", "set expected package import `path`", &myimportpath)
	obj.Flagcount("pack", "write package file instead of object file", &writearchive)
	obj.Flagcount("r", "debug generated wrappers", &Debug['r'])
	obj.Flagcount("race", "enable race detector", &flag_race)
	obj.Flagcount("s", "warn about composite literals that can be simplified", &Debug['s'])
	obj.Flagstr("trimpath", "remove `prefix` from recorded source file paths", &Ctxt.LineHist.TrimPathPrefix)
	obj.Flagcount("u", "reject unsafe code", &safemode)
	obj.Flagcount("v", "increase debug verbosity", &Debug['v'])
	obj.Flagcount("w", "debug type checking", &Debug['w'])
	use_writebarrier = 1
	obj.Flagcount("wb", "enable write barrier", &use_writebarrier)
	obj.Flagcount("x", "debug lexer", &Debug['x'])
	obj.Flagcount("y", "debug declarations in canned imports (with -d)", &Debug['y'])
	var flag_shared int
	var flag_dynlink bool
	if Thearch.LinkArch.InFamily(sys.ARM, sys.AMD64, sys.ARM64, sys.I386, sys.PPC64) {
		obj.Flagcount("shared", "generate code that can be linked into a shared library", &flag_shared)
	}
	if Thearch.LinkArch.Family == sys.AMD64 {
		obj.Flagcount("largemodel", "generate code that assumes a large memory model", &flag_largemodel)
	}
	if Thearch.LinkArch.InFamily(sys.ARM, sys.AMD64, sys.ARM64, sys.I386, sys.PPC64) {
		flag.BoolVar(&flag_dynlink, "dynlink", false, "support references to Go symbols defined in other shared libraries")
	}
	obj.Flagstr("cpuprofile", "write cpu profile to `file`", &cpuprofile)
	obj.Flagstr("memprofile", "write memory profile to `file`", &memprofile)
	obj.Flagint64("memprofilerate", "set runtime.MemProfileRate to `rate`", &memprofilerate)
	flag.BoolVar(&ssaEnabled, "ssa", true, "use SSA backend to generate code")
	obj.Flagparse(usage)

	if flag_dynlink {
		flag_shared = 1
	}
	Ctxt.Flag_shared = int32(flag_shared)
	Ctxt.Flag_dynlink = flag_dynlink
	Ctxt.Flag_optimize = Debug['N'] == 0

	Ctxt.Debugasm = int32(Debug['S'])
	Ctxt.Debugvlog = int32(Debug['v'])

	if flag.NArg() < 1 {
		usage()
	}

	startProfile()

	if flag_race != 0 {
		racepkg = mkpkg("runtime/race")
		racepkg.Name = "race"
	}
	if flag_msan != 0 {
		msanpkg = mkpkg("runtime/msan")
		msanpkg.Name = "msan"
	}
	if flag_race != 0 && flag_msan != 0 {
		log.Fatal("cannot use both -race and -msan")
	} else if flag_race != 0 || flag_msan != 0 {
		instrumenting = true
	}

	// parse -d argument
	if debugstr != "" {
	Split:
		for _, name := range strings.Split(debugstr, ",") {
			if name == "" {
				continue
			}
			val := 1
			if i := strings.Index(name, "="); i >= 0 {
				var err error
				val, err = strconv.Atoi(name[i+1:])
				if err != nil {
					log.Fatalf("invalid debug value %v", name)
				}
				name = name[:i]
			}
			for _, t := range debugtab {
				if t.name == name {
					if t.val != nil {
						*t.val = val
						continue Split
					}
				}
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
				err := ssa.PhaseOption(phase, flag, val)
				if err != "" {
					log.Fatalf(err)
				}
				continue Split
			}
			log.Fatalf("unknown debug key -d %s\n", name)
		}
	}

	// enable inlining.  for now:
	//	default: inlining on.  (debug['l'] == 1)
	//	-l: inlining off  (debug['l'] == 0)
	//	-ll, -lll: inlining on again, with extra debugging (debug['l'] > 1)
	if Debug['l'] <= 1 {
		Debug['l'] = 1 - Debug['l']
	}

	Thearch.Betypeinit()
	Widthint = Thearch.LinkArch.IntSize
	Widthptr = Thearch.LinkArch.PtrSize
	Widthreg = Thearch.LinkArch.RegSize

	initUniverse()

	blockgen = 1
	dclcontext = PEXTERN
	nerrors = 0
	lexlineno = 1

	loadsys()

	for _, infile = range flag.Args() {
		if trace && Debug['x'] != 0 {
			fmt.Printf("--- %s ---\n", infile)
		}

		linehistpush(infile)

		f, err := os.Open(infile)
		if err != nil {
			fmt.Printf("open %s: %v\n", infile, err)
			errorexit()
		}
		bin := bufio.NewReader(f)

		// Skip initial BOM if present.
		if r, _, _ := bin.ReadRune(); r != BOM {
			bin.UnreadRune()
		}

		block = 1
		iota_ = -1000000

		imported_unsafe = false

		parse_file(bin)
		if nsyntaxerrors != 0 {
			errorexit()
		}

		// Instead of converting EOF into '\n' in getc and count it as an extra line
		// for the line history to work, and which then has to be corrected elsewhere,
		// just add a line here.
		lexlineno++

		linehistpop()
		f.Close()
	}

	testdclstack()
	mkpackage(localpkg.Name) // final import not used checks
	finishUniverse()

	typecheckok = true
	if Debug['f'] != 0 {
		frame(1)
	}

	// Process top-level declarations in phases.

	// Phase 1: const, type, and names and types of funcs.
	//   This will gather all the information about types
	//   and methods but doesn't depend on any of it.
	defercheckwidth()

	// Don't use range--typecheck can add closures to xtop.
	for i := 0; i < len(xtop); i++ {
		if xtop[i].Op != ODCL && xtop[i].Op != OAS && xtop[i].Op != OAS2 {
			xtop[i] = typecheck(xtop[i], Etop)
		}
	}

	// Phase 2: Variable assignments.
	//   To check interface assignments, depends on phase 1.

	// Don't use range--typecheck can add closures to xtop.
	for i := 0; i < len(xtop); i++ {
		if xtop[i].Op == ODCL || xtop[i].Op == OAS || xtop[i].Op == OAS2 {
			xtop[i] = typecheck(xtop[i], Etop)
		}
	}
	resumecheckwidth()

	// Phase 3: Type check function bodies.
	// Don't use range--typecheck can add closures to xtop.
	for i := 0; i < len(xtop); i++ {
		if xtop[i].Op == ODCLFUNC || xtop[i].Op == OCLOSURE {
			Curfn = xtop[i]
			decldepth = 1
			saveerrors()
			typecheckslice(Curfn.Nbody.Slice(), Etop)
			checkreturn(Curfn)
			if nerrors != 0 {
				Curfn.Nbody.Set(nil) // type errors; do not compile
			}
		}
	}

	// Phase 4: Decide how to capture closed variables.
	// This needs to run before escape analysis,
	// because variables captured by value do not escape.
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
	if Debug['l'] > 1 {
		// Typecheck imported function bodies if debug['l'] > 1,
		// otherwise lazily when used or re-exported.
		for _, n := range importlist {
			if len(n.Func.Inl.Slice()) != 0 {
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
				if n.Op == ODCLFUNC {
					caninl(n)
					inlcalls(n)
				}
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
	escapes(xtop)

	// Phase 7: Transform closure bodies to properly reference captured variables.
	// This needs to happen before walk, because closures must be transformed
	// before walk reaches a call of a closure.
	for _, n := range xtop {
		if n.Op == ODCLFUNC && n.Func.Closure != nil {
			Curfn = n
			transformclosure(n)
		}
	}

	Curfn = nil

	// Phase 8: Compile top level functions.
	// Don't use range--walk can add functions to xtop.
	for i := 0; i < len(xtop); i++ {
		if xtop[i].Op == ODCLFUNC {
			funccompile(xtop[i])
		}
	}

	if nsavederrors+nerrors == 0 {
		fninit(xtop)
	}

	if compiling_runtime != 0 {
		checknowritebarrierrec()
	}

	// Phase 9: Check external declarations.
	for i, n := range externdcl {
		if n.Op == ONAME {
			externdcl[i] = typecheck(externdcl[i], Erv)
		}
	}

	if nerrors+nsavederrors != 0 {
		errorexit()
	}

	dumpobj()

	if asmhdr != "" {
		dumpasmhdr()
	}

	if nerrors+nsavederrors != 0 {
		errorexit()
	}

	Flusherrors()
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
		if safemode != 0 || nolocalimports != 0 {
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
		Yyerror("non-canonical import path %q (should be %q)", name, q)
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

	if goroot != "" {
		suffix := ""
		suffixsep := ""
		if flag_installsuffix != "" {
			suffixsep = "_"
			suffix = flag_installsuffix
		} else if flag_race != 0 {
			suffixsep = "_"
			suffix = "race"
		} else if flag_msan != 0 {
			suffixsep = "_"
			suffix = "msan"
		}

		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.a", goroot, goos, goarch, suffixsep, suffix, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s/pkg/%s_%s%s%s/%s.o", goroot, goos, goarch, suffixsep, suffix, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
	}

	return "", false
}

// loadsys loads the definitions for the low-level runtime and unsafe functions,
// so that the compiler can generate calls to them,
// but does not make the names "runtime" or "unsafe" visible as packages.
func loadsys() {
	if Debug['A'] != 0 {
		return
	}

	block = 1
	iota_ = -1000000
	incannedimport = 1

	importpkg = Runtimepkg
	parse_import(bufio.NewReader(strings.NewReader(runtimeimport)), nil)

	importpkg = unsafepkg
	parse_import(bufio.NewReader(strings.NewReader(unsafeimport)), nil)

	importpkg = nil
	incannedimport = 0
}

func importfile(f *Val, indent []byte) {
	if importpkg != nil {
		Fatalf("importpkg not nil")
	}

	path_, ok := f.U.(string)
	if !ok {
		Yyerror("import statement not a string")
		return
	}

	if len(path_) == 0 {
		Yyerror("import path is empty")
		return
	}

	if isbadimport(path_) {
		return
	}

	// The package name main is no longer reserved,
	// but we reserve the import path "main" to identify
	// the main package, just as we reserve the import
	// path "math" to identify the standard math package.
	if path_ == "main" {
		Yyerror("cannot import \"main\"")
		errorexit()
	}

	if myimportpath != "" && path_ == myimportpath {
		Yyerror("import %q while compiling that package (import cycle)", path_)
		errorexit()
	}

	if mapped, ok := importMap[path_]; ok {
		path_ = mapped
	}

	if path_ == "unsafe" {
		if safemode != 0 {
			Yyerror("cannot import package unsafe")
			errorexit()
		}

		importpkg = unsafepkg
		imported_unsafe = true
		return
	}

	if islocalname(path_) {
		if path_[0] == '/' {
			Yyerror("import path cannot be absolute path")
			return
		}

		prefix := Ctxt.Pathname
		if localimport != "" {
			prefix = localimport
		}
		path_ = path.Join(prefix, path_)

		if isbadimport(path_) {
			return
		}
	}

	file, found := findpkg(path_)
	if !found {
		Yyerror("can't find import: %q", path_)
		errorexit()
	}

	importpkg = mkpkg(path_)

	if importpkg.Imported {
		return
	}

	importpkg.Imported = true

	impf, err := os.Open(file)
	if err != nil {
		Yyerror("can't open import: %q: %v", path_, err)
		errorexit()
	}
	defer impf.Close()
	imp := bufio.NewReader(impf)

	if strings.HasSuffix(file, ".a") {
		if !skiptopkgdef(imp) {
			Yyerror("import %s: not a package file", file)
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
			Yyerror("import %s: not a go object file", file)
			errorexit()
		}

		q := fmt.Sprintf("%s %s %s %s", obj.Getgoos(), obj.Getgoarch(), obj.Getgoversion(), obj.Expstring())
		if p[10:] != q {
			Yyerror("import %s: object is [%s] expected [%s]", file, p[10:], q)
			errorexit()
		}
	}

	// assume files move (get installed)
	// so don't record the full path.
	linehistpragma(file[len(file)-len(path_)-2:]) // acts as #pragma lib

	// In the importfile, if we find:
	// $$\n  (old format): position the input right after $$\n and return
	// $$B\n (new format): import directly, then feed the lexer a dummy statement

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
		// old export format
		parse_import(imp, indent)

	case 'B':
		// new export format
		if Debug_export != 0 {
			fmt.Printf("importing %s (%s)\n", path_, file)
		}
		imp.ReadByte() // skip \n after $$B
		Import(imp)

	default:
		Yyerror("no import in %q", path_)
		errorexit()
	}

	if safemode != 0 && !importpkg.Safe {
		Yyerror("cannot import unsafe package %q", importpkg.Path)
	}
}

func pkgnotused(lineno int32, path string, name string) {
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
			Yyerror("invalid package name _")
		}
		localpkg.Name = pkgname
	} else {
		if pkgname != localpkg.Name {
			Yyerror("package %s; expected %s", pkgname, localpkg.Name)
		}
		for _, s := range localpkg.Syms {
			if s.Def == nil {
				continue
			}
			if s.Def.Op == OPACK {
				// throw away top-level package name leftover
				// from previous file.
				// leave s->block set to cause redeclaration
				// errors if a conflicting top-level name is
				// introduced by a different file.
				if !s.Def.Used && nsyntaxerrors == 0 {
					pkgnotused(s.Def.Lineno, s.Def.Name.Pkg.Path, s.Name)
				}
				s.Def = nil
				continue
			}

			if s.Def.Sym != s {
				// throw away top-level name left over
				// from previous import . "x"
				if s.Def.Name != nil && s.Def.Name.Pack != nil && !s.Def.Name.Pack.Used && nsyntaxerrors == 0 {
					pkgnotused(s.Def.Name.Pack.Lineno, s.Def.Name.Pack.Name.Pkg.Path, "")
					s.Def.Name.Pack.Used = true
				}

				s.Def = nil
				continue
			}
		}
	}

	if outfile == "" {
		p := infile
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
		if writearchive > 0 {
			suffix = ".a"
		}
		outfile = p + suffix
	}
}
