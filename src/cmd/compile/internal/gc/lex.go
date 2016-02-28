// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run mkbuiltin.go

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/internal/obj"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

var imported_unsafe bool

var (
	goos    string
	goarch  string
	goroot  string
	buildid string
)

var (
	Debug_append int
	Debug_panic  int
	Debug_slice  int
	Debug_wb     int
)

const BOM = 0xFEFF

// Debug arguments.
// These can be specified with the -d flag, as in "-d nil"
// to set the debug_checknil variable. In general the list passed
// to -d can be comma-separated.
var debugtab = []struct {
	name string
	val  *int
}{
	{"append", &Debug_append},         // print information about append compilation
	{"disablenil", &Disable_checknil}, // disable nil checks
	{"gcprog", &Debug_gcprog},         // print dump of GC programs
	{"nil", &Debug_checknil},          // print information about nil checks
	{"panic", &Debug_panic},           // do not hide any compiler panic
	{"slice", &Debug_slice},           // print information about slice compilation
	{"typeassert", &Debug_typeassert}, // print information about type assertion inlining
	{"wb", &Debug_wb},                 // print information about write barriers
	{"export", &Debug_export},         // print export data
}

const (
	EOF = -1
)

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

	if !strings.HasPrefix(p, Thearch.Thestring) {
		log.Fatalf("cannot use %cg with GOARCH=%s", Thearch.Thechar, p)
	}
	goarch = p

	Thearch.Linkarchinit()
	Ctxt = obj.Linknew(Thearch.Thelinkarch)
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
	gostringpkg = mkpkg("go.string")

	gostringpkg.Name = "go.string"
	gostringpkg.Prefix = "go.string" // not go%2estring

	itabpkg = mkpkg("go.itab")

	itabpkg.Name = "go.itab"
	itabpkg.Prefix = "go.itab" // not go%2eitab

	weaktypepkg = mkpkg("go.weak.type")

	weaktypepkg.Name = "go.weak.type"
	weaktypepkg.Prefix = "go.weak.type" // not go%2eweak%2etype

	typelinkpkg = mkpkg("go.typelink")
	typelinkpkg.Name = "go.typelink"
	typelinkpkg.Prefix = "go.typelink" // not go%2etypelink

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
	switch Thearch.Thechar {
	case '5', '6', '7', '8', '9':
		obj.Flagcount("shared", "generate code that can be linked into a shared library", &flag_shared)
	}
	if Thearch.Thechar == '6' {
		obj.Flagcount("largemodel", "generate code that assumes a large memory model", &flag_largemodel)
	}
	switch Thearch.Thechar {
	case '5', '6', '7', '8', '9':
		flag.BoolVar(&flag_dynlink, "dynlink", false, "support references to Go symbols defined in other shared libraries")
	}
	obj.Flagstr("cpuprofile", "write cpu profile to `file`", &cpuprofile)
	obj.Flagstr("memprofile", "write memory profile to `file`", &memprofile)
	obj.Flagint64("memprofilerate", "set runtime.MemProfileRate to `rate`", &memprofilerate)
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
	if Widthptr == 0 {
		Fatalf("betypeinit failed")
	}

	lexinit()
	typeinit()
	lexinit1()

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

		bin, err := obj.Bopenr(infile)
		if err != nil {
			fmt.Printf("open %s: %v\n", infile, err)
			errorexit()
		}

		// Skip initial BOM if present.
		if obj.Bgetrune(bin) != BOM {
			obj.Bungetrune(bin)
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
		obj.Bterm(bin)
	}

	testdclstack()
	mkpackage(localpkg.Name) // final import not used checks
	lexfini()

	typecheckok = true
	if Debug['f'] != 0 {
		frame(1)
	}

	// Process top-level declarations in phases.

	// Phase 1: const, type, and names and types of funcs.
	//   This will gather all the information about types
	//   and methods but doesn't depend on any of it.
	defercheckwidth()

	for l := xtop; l != nil; l = l.Next {
		if l.N.Op != ODCL && l.N.Op != OAS && l.N.Op != OAS2 {
			typecheck(&l.N, Etop)
		}
	}

	// Phase 2: Variable assignments.
	//   To check interface assignments, depends on phase 1.
	for l := xtop; l != nil; l = l.Next {
		if l.N.Op == ODCL || l.N.Op == OAS || l.N.Op == OAS2 {
			typecheck(&l.N, Etop)
		}
	}
	resumecheckwidth()

	// Phase 3: Type check function bodies.
	for l := xtop; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC || l.N.Op == OCLOSURE {
			Curfn = l.N
			decldepth = 1
			saveerrors()
			typechecklist(l.N.Nbody, Etop)
			checkreturn(l.N)
			if nerrors != 0 {
				l.N.Nbody = nil // type errors; do not compile
			}
		}
	}

	// Phase 4: Decide how to capture closed variables.
	// This needs to run before escape analysis,
	// because variables captured by value do not escape.
	for l := xtop; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC && l.N.Func.Closure != nil {
			Curfn = l.N
			capturevars(l.N)
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
			// TODO: use a range statement here if the order does not matter
			for i := len(list) - 1; i >= 0; i-- {
				n := list[i]
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
	for l := xtop; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC && l.N.Func.Closure != nil {
			Curfn = l.N
			transformclosure(l.N)
		}
	}

	Curfn = nil

	// Phase 8: Compile top level functions.
	for l := xtop; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC {
			funccompile(l.N)
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
			typecheck(&externdcl[i], Erv)
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

func arsize(b *obj.Biobuf, name string) int {
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

func skiptopkgdef(b *obj.Biobuf) bool {
	// archive header
	p := obj.Brdline(b, '\n')
	if p == "" {
		return false
	}
	if obj.Blinelen(b) != 8 {
		return false
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
		Ctxt.Windows != 0 && len(name) >= 3 && isDriveLetter(name[0]) && name[1] == ':' && name[2] == '/' ||
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
	parse_import(obj.Binitr(strings.NewReader(runtimeimport)), nil)

	importpkg = unsafepkg
	parse_import(obj.Binitr(strings.NewReader(unsafeimport)), nil)

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

	imp, err := obj.Bopenr(file)
	if err != nil {
		Yyerror("can't open import: %q: %v", path_, err)
		errorexit()
	}
	defer obj.Bterm(imp)

	if strings.HasSuffix(file, ".a") {
		if !skiptopkgdef(imp) {
			Yyerror("import %s: not a package file", file)
			errorexit()
		}
	}

	// check object header
	p := obj.Brdstr(imp, '\n', 1)

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
	var c int
	for {
		c = obj.Bgetc(imp)
		if c < 0 {
			break
		}
		if c == '$' {
			c = obj.Bgetc(imp)
			if c == '$' || c < 0 {
				break
			}
		}
	}

	// get character after $$
	if c >= 0 {
		c = obj.Bgetc(imp)
	}

	switch c {
	case '\n':
		// old export format
		parse_import(imp, indent)

	case 'B':
		// new export format
		obj.Bgetc(imp) // skip \n after $$B
		Import(imp)

	default:
		Yyerror("no import in %q", path_)
		errorexit()
	}

	if safemode != 0 && !importpkg.Safe {
		Yyerror("cannot import unsafe package %q", importpkg.Path)
	}
}

func isSpace(c rune) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func isLetter(c rune) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || c == '_'
}

func isDigit(c rune) bool {
	return '0' <= c && c <= '9'
}

func plan9quote(s string) string {
	if s == "" {
		return "''"
	}
	for _, c := range s {
		if c <= ' ' || c == '\'' {
			return "'" + strings.Replace(s, "'", "''", -1) + "'"
		}
	}
	return s
}

type Pragma uint16

const (
	Nointerface       Pragma = 1 << iota
	Noescape                 // func parameters don't escape
	Norace                   // func must not have race detector annotations
	Nosplit                  // func should not execute on separate stack
	Noinline                 // func should not be inlined
	Systemstack              // func must run on system stack
	Nowritebarrier           // emit compiler error instead of write barrier
	Nowritebarrierrec        // error on write barrier in this or recursive callees
	CgoUnsafeArgs            // treat a pointer to one arg as a pointer to them all
)

type lexer struct {
	// source
	bin    *obj.Biobuf
	peekr1 rune
	peekr2 rune // second peekc for ...

	nlsemi bool // if set, '\n' and EOF translate to ';'

	// pragma flags
	// accumulated by lexer; reset by parser
	pragma Pragma

	// current token
	tok  int32
	sym_ *Sym   // valid if tok == LNAME
	val  Val    // valid if tok == LLITERAL
	op   Op     // valid if tok == LASOP or LINCOP, or prec > 0
	prec OpPrec // operator precedence; 0 if not a binary operator
}

type OpPrec int

const (
	// Precedences of binary operators (must be > 0).
	PCOMM OpPrec = 1 + iota
	POROR
	PANDAND
	PCMP
	PADD
	PMUL
)

const (
	// The value of single-char tokens is just their character's Unicode value.
	// They are all below utf8.RuneSelf. Shift other tokens up to avoid conflicts.
	LLITERAL = utf8.RuneSelf + iota
	LASOP
	LCOLAS
	LBREAK
	LCASE
	LCHAN
	LCONST
	LCONTINUE
	LDDD
	LDEFAULT
	LDEFER
	LELSE
	LFALL
	LFOR
	LFUNC
	LGO
	LGOTO
	LIF
	LIMPORT
	LINTERFACE
	LMAP
	LNAME
	LPACKAGE
	LRANGE
	LRETURN
	LSELECT
	LSTRUCT
	LSWITCH
	LTYPE
	LVAR
	LANDAND
	LANDNOT
	LCOMM
	LEQ
	LGE
	LGT
	LIGNORE
	LINCOP
	LLE
	LLSH
	LLT
	LNE
	LOROR
	LRSH
)

func (l *lexer) next() {
	nlsemi := l.nlsemi
	l.nlsemi = false
	l.prec = 0

l0:
	// skip white space
	c := l.getr()
	for isSpace(c) {
		if c == '\n' && nlsemi {
			if Debug['x'] != 0 {
				fmt.Printf("lex: implicit semi\n")
			}
			// Insert implicit semicolon on previous line,
			// before the newline character.
			lineno = lexlineno - 1
			l.tok = ';'
			return
		}
		c = l.getr()
	}

	// start of token
	lineno = lexlineno

	// identifiers and keywords
	// (for better error messages consume all chars >= utf8.RuneSelf for identifiers)
	if isLetter(c) || c >= utf8.RuneSelf {
		l.ident(c)
		if l.tok == LIGNORE {
			goto l0
		}
		return
	}
	// c < utf8.RuneSelf

	var c1 rune
	var op Op
	var prec OpPrec

	switch c {
	case EOF:
		l.ungetr(EOF) // return EOF again in future next call
		// Treat EOF as "end of line" for the purposes
		// of inserting a semicolon.
		if nlsemi {
			if Debug['x'] != 0 {
				fmt.Printf("lex: implicit semi\n")
			}
			l.tok = ';'
			return
		}
		l.tok = -1
		return

	case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		l.number(c)
		return

	case '.':
		c1 = l.getr()
		if isDigit(c1) {
			l.ungetr(c1)
			l.number('.')
			return
		}

		if c1 == '.' {
			c1 = l.getr()
			if c1 == '.' {
				c = LDDD
				goto lx
			}

			l.ungetr(c1)
			c1 = '.'
		}

	case '"':
		l.stdString()
		return

	case '`':
		l.rawString()
		return

	case '\'':
		l.rune()
		return

	case '/':
		c1 = l.getr()
		if c1 == '*' {
			c = l.getr()
			for {
				if c == '*' {
					c = l.getr()
					if c == '/' {
						break
					}
					continue
				}
				if c == EOF {
					Yyerror("eof in comment")
					errorexit()
				}
				c = l.getr()
			}

			// A comment containing newlines acts like a newline.
			if lexlineno > lineno && nlsemi {
				if Debug['x'] != 0 {
					fmt.Printf("lex: implicit semi\n")
				}
				l.tok = ';'
				return
			}
			goto l0
		}

		if c1 == '/' {
			c = l.getlinepragma()
			for {
				if c == '\n' || c == EOF {
					l.ungetr(c)
					goto l0
				}

				c = l.getr()
			}
		}

		op = ODIV
		prec = PMUL
		goto binop1

	case ':':
		c1 = l.getr()
		if c1 == '=' {
			c = LCOLAS
			goto lx
		}

	case '*':
		op = OMUL
		prec = PMUL
		goto binop

	case '%':
		op = OMOD
		prec = PMUL
		goto binop

	case '+':
		op = OADD
		goto incop

	case '-':
		op = OSUB
		goto incop

	case '>':
		c1 = l.getr()
		if c1 == '>' {
			c = LRSH
			op = ORSH
			prec = PMUL
			goto binop
		}

		l.prec = PCMP
		if c1 == '=' {
			c = LGE
			l.op = OGE
			goto lx
		}
		c = LGT
		l.op = OGT

	case '<':
		c1 = l.getr()
		if c1 == '<' {
			c = LLSH
			op = OLSH
			prec = PMUL
			goto binop
		}

		if c1 == '-' {
			c = LCOMM
			// Not a binary operator, but parsed as one
			// so we can give a good error message when used
			// in an expression context.
			l.prec = PCOMM
			l.op = OSEND
			goto lx
		}

		l.prec = PCMP
		if c1 == '=' {
			c = LLE
			l.op = OLE
			goto lx
		}
		c = LLT
		l.op = OLT

	case '=':
		c1 = l.getr()
		if c1 == '=' {
			c = LEQ
			l.prec = PCMP
			l.op = OEQ
			goto lx
		}

	case '!':
		c1 = l.getr()
		if c1 == '=' {
			c = LNE
			l.prec = PCMP
			l.op = ONE
			goto lx
		}

	case '&':
		c1 = l.getr()
		if c1 == '&' {
			c = LANDAND
			l.prec = PANDAND
			l.op = OANDAND
			goto lx
		}

		if c1 == '^' {
			c = LANDNOT
			op = OANDNOT
			prec = PMUL
			goto binop
		}

		op = OAND
		prec = PMUL
		goto binop1

	case '|':
		c1 = l.getr()
		if c1 == '|' {
			c = LOROR
			l.prec = POROR
			l.op = OOROR
			goto lx
		}

		op = OOR
		prec = PADD
		goto binop1

	case '^':
		op = OXOR
		prec = PADD
		goto binop

	case '(', '[', '{', ',', ';':
		goto lx

	case ')', ']', '}':
		l.nlsemi = true
		goto lx

	case '#', '$', '?', '@', '\\':
		if importpkg != nil {
			goto lx
		}
		fallthrough

	default:
		// anything else is illegal
		Yyerror("syntax error: illegal character %#U", c)
		goto l0
	}

	l.ungetr(c1)

lx:
	if Debug['x'] != 0 {
		if c >= utf8.RuneSelf {
			fmt.Printf("%v lex: TOKEN %s\n", Ctxt.Line(int(lineno)), lexname(c))
		} else {
			fmt.Printf("%v lex: TOKEN '%c'\n", Ctxt.Line(int(lineno)), c)
		}
	}

	l.tok = c
	return

incop:
	c1 = l.getr()
	if c1 == c {
		l.nlsemi = true
		l.op = op
		c = LINCOP
		goto lx
	}
	prec = PADD
	goto binop1

binop:
	c1 = l.getr()
binop1:
	if c1 != '=' {
		l.ungetr(c1)
		l.op = op
		l.prec = prec
		goto lx
	}

	l.op = op
	if Debug['x'] != 0 {
		fmt.Printf("lex: TOKEN ASOP %s=\n", goopnames[op])
	}
	l.tok = LASOP
}

func (l *lexer) ident(c rune) {
	cp := &lexbuf
	cp.Reset()

	// accelerate common case (7bit ASCII)
	for isLetter(c) || isDigit(c) {
		cp.WriteByte(byte(c))
		c = l.getr()
	}

	// general case
	for {
		if c >= utf8.RuneSelf {
			if unicode.IsLetter(c) || c == '_' || unicode.IsDigit(c) || importpkg != nil && c == 0xb7 {
				if cp.Len() == 0 && unicode.IsDigit(c) {
					Yyerror("identifier cannot begin with digit %#U", c)
				}
			} else {
				Yyerror("invalid identifier character %#U", c)
			}
			cp.WriteRune(c)
		} else if isLetter(c) || isDigit(c) {
			cp.WriteByte(byte(c))
		} else {
			break
		}
		c = l.getr()
	}

	cp = nil
	l.ungetr(c)

	name := lexbuf.Bytes()

	if len(name) >= 2 {
		if tok, ok := keywords[string(name)]; ok {
			if Debug['x'] != 0 {
				fmt.Printf("lex: %s\n", lexname(tok))
			}
			switch tok {
			case LBREAK, LCONTINUE, LFALL, LRETURN:
				l.nlsemi = true
			}
			l.tok = tok
			return
		}
	}

	s := LookupBytes(name)
	if Debug['x'] != 0 {
		fmt.Printf("lex: ident %s\n", s)
	}
	l.sym_ = s
	l.nlsemi = true
	l.tok = LNAME
}

var keywords = map[string]int32{
	"break":       LBREAK,
	"case":        LCASE,
	"chan":        LCHAN,
	"const":       LCONST,
	"continue":    LCONTINUE,
	"default":     LDEFAULT,
	"defer":       LDEFER,
	"else":        LELSE,
	"fallthrough": LFALL,
	"for":         LFOR,
	"func":        LFUNC,
	"go":          LGO,
	"goto":        LGOTO,
	"if":          LIF,
	"import":      LIMPORT,
	"interface":   LINTERFACE,
	"map":         LMAP,
	"package":     LPACKAGE,
	"range":       LRANGE,
	"return":      LRETURN,
	"select":      LSELECT,
	"struct":      LSTRUCT,
	"switch":      LSWITCH,
	"type":        LTYPE,
	"var":         LVAR,

	// 💩
	"notwithstanding":      LIGNORE,
	"thetruthofthematter":  LIGNORE,
	"despiteallobjections": LIGNORE,
	"whereas":              LIGNORE,
	"insofaras":            LIGNORE,
}

func (l *lexer) number(c rune) {
	// TODO(gri) this can be done nicely with fewer or even without labels

	var str string
	cp := &lexbuf
	cp.Reset()

	if c != '.' {
		if c != '0' {
			for isDigit(c) {
				cp.WriteByte(byte(c))
				c = l.getr()
			}
			if c == '.' {
				goto casedot
			}
			if c == 'e' || c == 'E' || c == 'p' || c == 'P' {
				goto caseep
			}
			if c == 'i' {
				goto casei
			}
			goto ncu
		}

		// c == 0
		cp.WriteByte('0')
		c = l.getr()
		if c == 'x' || c == 'X' {
			cp.WriteByte(byte(c))
			c = l.getr()
			for isDigit(c) || 'a' <= c && c <= 'f' || 'A' <= c && c <= 'F' {
				cp.WriteByte(byte(c))
				c = l.getr()
			}
			if lexbuf.Len() == 2 {
				Yyerror("malformed hex constant")
			}
			if c == 'p' {
				goto caseep
			}
			goto ncu
		}

		if c == 'p' { // 0p begins floating point zero
			goto caseep
		}

		has8or9 := false
		for isDigit(c) {
			if c > '7' {
				has8or9 = true
			}
			cp.WriteByte(byte(c))
			c = l.getr()
		}
		if c == '.' {
			goto casedot
		}
		if c == 'e' || c == 'E' {
			goto caseep
		}
		if c == 'i' {
			goto casei
		}
		if has8or9 {
			Yyerror("malformed octal constant")
		}
		goto ncu
	}

casedot:
	// fraction
	// c == '.'
	cp.WriteByte('.')
	c = l.getr()
	for isDigit(c) {
		cp.WriteByte(byte(c))
		c = l.getr()
	}
	if c == 'i' {
		goto casei
	}
	if c != 'e' && c != 'E' {
		goto caseout
	}
	// base-2-exponents (p or P) don't appear in numbers
	// with fractions - ok to not test for 'p' or 'P'
	// above

caseep:
	// exponent
	if importpkg == nil && (c == 'p' || c == 'P') {
		// <mantissa>p<base-2-exponent> is allowed in .a/.o imports,
		// but not in .go sources.  See #9036.
		Yyerror("malformed floating point constant")
	}
	cp.WriteByte(byte(c))
	c = l.getr()
	if c == '+' || c == '-' {
		cp.WriteByte(byte(c))
		c = l.getr()
	}

	if !isDigit(c) {
		Yyerror("malformed floating point constant exponent")
	}
	for isDigit(c) {
		cp.WriteByte(byte(c))
		c = l.getr()
	}

	if c != 'i' {
		goto caseout
	}

casei:
	// imaginary constant
	cp = nil

	str = lexbuf.String()
	l.val.U = new(Mpcplx)
	Mpmovecflt(&l.val.U.(*Mpcplx).Real, 0.0)
	mpatoflt(&l.val.U.(*Mpcplx).Imag, str)
	if l.val.U.(*Mpcplx).Imag.Val.IsInf() {
		Yyerror("overflow in imaginary constant")
		Mpmovecflt(&l.val.U.(*Mpcplx).Imag, 0.0)
	}

	if Debug['x'] != 0 {
		fmt.Printf("lex: imaginary literal\n")
	}
	goto done

caseout:
	cp = nil
	l.ungetr(c)

	str = lexbuf.String()
	l.val.U = newMpflt()
	mpatoflt(l.val.U.(*Mpflt), str)
	if l.val.U.(*Mpflt).Val.IsInf() {
		Yyerror("overflow in float constant")
		Mpmovecflt(l.val.U.(*Mpflt), 0.0)
	}

	if Debug['x'] != 0 {
		fmt.Printf("lex: floating literal\n")
	}
	goto done

ncu:
	cp = nil
	l.ungetr(c)

	str = lexbuf.String()
	l.val.U = new(Mpint)
	mpatofix(l.val.U.(*Mpint), str)
	if l.val.U.(*Mpint).Ovf {
		Yyerror("overflow in constant")
		Mpmovecfix(l.val.U.(*Mpint), 0)
	}

	if Debug['x'] != 0 {
		fmt.Printf("lex: integer literal\n")
	}

done:
	litbuf = "literal " + str
	l.nlsemi = true
	l.tok = LLITERAL
}

func (l *lexer) stdString() {
	lexbuf.Reset()
	lexbuf.WriteString(`"<string>"`)

	cp := &strbuf
	cp.Reset()

	for {
		r, b, ok := l.onechar('"')
		if !ok {
			break
		}
		if r == 0 {
			cp.WriteByte(b)
		} else {
			cp.WriteRune(r)
		}
	}

	l.val.U = internString(cp.Bytes())
	if Debug['x'] != 0 {
		fmt.Printf("lex: string literal\n")
	}
	litbuf = "string literal"
	l.nlsemi = true
	l.tok = LLITERAL
}

func (l *lexer) rawString() {
	lexbuf.Reset()
	lexbuf.WriteString("`<string>`")

	cp := &strbuf
	cp.Reset()

	for {
		c := l.getr()
		if c == '\r' {
			continue
		}
		if c == EOF {
			Yyerror("eof in string")
			break
		}
		if c == '`' {
			break
		}
		cp.WriteRune(c)
	}

	l.val.U = internString(cp.Bytes())
	if Debug['x'] != 0 {
		fmt.Printf("lex: string literal\n")
	}
	litbuf = "string literal"
	l.nlsemi = true
	l.tok = LLITERAL
}

func (l *lexer) rune() {
	r, b, ok := l.onechar('\'')
	if !ok {
		Yyerror("empty character literal or unescaped ' in character literal")
		r = '\''
	}
	if r == 0 {
		r = rune(b)
	}

	if c := l.getr(); c != '\'' {
		Yyerror("missing '")
		l.ungetr(c)
	}

	x := new(Mpint)
	l.val.U = x
	Mpmovecfix(x, int64(r))
	x.Rune = true
	if Debug['x'] != 0 {
		fmt.Printf("lex: codepoint literal\n")
	}
	litbuf = "rune literal"
	l.nlsemi = true
	l.tok = LLITERAL
}

var internedStrings = map[string]string{}

func internString(b []byte) string {
	s, ok := internedStrings[string(b)] // string(b) here doesn't allocate
	if !ok {
		s = string(b)
		internedStrings[s] = s
	}
	return s
}

func more(pp *string) bool {
	p := *pp
	for p != "" && isSpace(rune(p[0])) {
		p = p[1:]
	}
	*pp = p
	return p != ""
}

// read and interpret syntax that looks like
// //line parse.y:15
// as a discontinuity in sequential line numbers.
// the next line of input comes from parse.y:15
func (l *lexer) getlinepragma() rune {
	c := l.getr()
	if c == 'g' { // check for //go: directive
		cp := &lexbuf
		cp.Reset()
		cp.WriteByte('g') // already read
		for {
			c = l.getr()
			if c == EOF || c >= utf8.RuneSelf {
				return c
			}
			if c == '\n' {
				break
			}
			cp.WriteByte(byte(c))
		}
		cp = nil

		text := strings.TrimSuffix(lexbuf.String(), "\r")

		if strings.HasPrefix(text, "go:cgo_") {
			pragcgo(text)
		}

		verb := text
		if i := strings.Index(text, " "); i >= 0 {
			verb = verb[:i]
		}

		switch verb {
		case "go:linkname":
			if !imported_unsafe {
				Yyerror("//go:linkname only allowed in Go files that import \"unsafe\"")
			}
			f := strings.Fields(text)
			if len(f) != 3 {
				Yyerror("usage: //go:linkname localname linkname")
				break
			}
			Lookup(f[1]).Linkname = f[2]
		case "go:nointerface":
			if obj.Fieldtrack_enabled != 0 {
				l.pragma |= Nointerface
			}
		case "go:noescape":
			l.pragma |= Noescape
		case "go:norace":
			l.pragma |= Norace
		case "go:nosplit":
			l.pragma |= Nosplit
		case "go:noinline":
			l.pragma |= Noinline
		case "go:systemstack":
			if compiling_runtime == 0 {
				Yyerror("//go:systemstack only allowed in runtime")
			}
			l.pragma |= Systemstack
		case "go:nowritebarrier":
			if compiling_runtime == 0 {
				Yyerror("//go:nowritebarrier only allowed in runtime")
			}
			l.pragma |= Nowritebarrier
		case "go:nowritebarrierrec":
			if compiling_runtime == 0 {
				Yyerror("//go:nowritebarrierrec only allowed in runtime")
			}
			l.pragma |= Nowritebarrierrec | Nowritebarrier // implies Nowritebarrier
		case "go:cgo_unsafe_args":
			l.pragma |= CgoUnsafeArgs
		}
		return c
	}

	// check for //line directive
	if c != 'l' {
		return c
	}
	for i := 1; i < 5; i++ {
		c = l.getr()
		if c != rune("line "[i]) {
			return c
		}
	}

	cp := &lexbuf
	cp.Reset()
	linep := 0
	for {
		c = l.getr()
		if c == EOF {
			return c
		}
		if c == '\n' {
			break
		}
		if c == ' ' {
			continue
		}
		if c == ':' {
			linep = cp.Len() + 1
		}
		cp.WriteByte(byte(c))
	}
	cp = nil

	if linep == 0 {
		return c
	}
	text := strings.TrimSuffix(lexbuf.String(), "\r")
	n, err := strconv.Atoi(text[linep:])
	if err != nil {
		return c // todo: make this an error instead? it is almost certainly a bug.
	}
	if n > 1e8 {
		Yyerror("line number out of range")
		errorexit()
	}
	if n <= 0 {
		return c
	}

	linehistupdate(text[:linep-1], n)
	return c
}

func getimpsym(pp *string) string {
	more(pp) // skip spaces
	p := *pp
	if p == "" || p[0] == '"' {
		return ""
	}
	i := 0
	for i < len(p) && !isSpace(rune(p[i])) && p[i] != '"' {
		i++
	}
	sym := p[:i]
	*pp = p[i:]
	return sym
}

func getquoted(pp *string) (string, bool) {
	more(pp) // skip spaces
	p := *pp
	if p == "" || p[0] != '"' {
		return "", false
	}
	p = p[1:]
	i := strings.Index(p, `"`)
	if i < 0 {
		return "", false
	}
	*pp = p[i+1:]
	return p[:i], true
}

// Copied nearly verbatim from the C compiler's #pragma parser.
// TODO: Rewrite more cleanly once the compiler is written in Go.
func pragcgo(text string) {
	var q string

	if i := strings.Index(text, " "); i >= 0 {
		text, q = text[:i], text[i:]
	}

	verb := text[3:] // skip "go:"

	if verb == "cgo_dynamic_linker" || verb == "dynlinker" {
		p, ok := getquoted(&q)
		if !ok {
			Yyerror("usage: //go:cgo_dynamic_linker \"path\"")
			return
		}
		pragcgobuf += fmt.Sprintf("cgo_dynamic_linker %v\n", plan9quote(p))
		return

	}

	if verb == "dynexport" {
		verb = "cgo_export_dynamic"
	}
	if verb == "cgo_export_static" || verb == "cgo_export_dynamic" {
		local := getimpsym(&q)
		var remote string
		if local == "" {
			goto err2
		}
		if !more(&q) {
			pragcgobuf += fmt.Sprintf("%s %v\n", verb, plan9quote(local))
			return
		}

		remote = getimpsym(&q)
		if remote == "" {
			goto err2
		}
		pragcgobuf += fmt.Sprintf("%s %v %v\n", verb, plan9quote(local), plan9quote(remote))
		return

	err2:
		Yyerror("usage: //go:%s local [remote]", verb)
		return
	}

	if verb == "cgo_import_dynamic" || verb == "dynimport" {
		var ok bool
		local := getimpsym(&q)
		var p string
		var remote string
		if local == "" {
			goto err3
		}
		if !more(&q) {
			pragcgobuf += fmt.Sprintf("cgo_import_dynamic %v\n", plan9quote(local))
			return
		}

		remote = getimpsym(&q)
		if remote == "" {
			goto err3
		}
		if !more(&q) {
			pragcgobuf += fmt.Sprintf("cgo_import_dynamic %v %v\n", plan9quote(local), plan9quote(remote))
			return
		}

		p, ok = getquoted(&q)
		if !ok {
			goto err3
		}
		pragcgobuf += fmt.Sprintf("cgo_import_dynamic %v %v %v\n", plan9quote(local), plan9quote(remote), plan9quote(p))
		return

	err3:
		Yyerror("usage: //go:cgo_import_dynamic local [remote [\"library\"]]")
		return
	}

	if verb == "cgo_import_static" {
		local := getimpsym(&q)
		if local == "" || more(&q) {
			Yyerror("usage: //go:cgo_import_static local")
			return
		}
		pragcgobuf += fmt.Sprintf("cgo_import_static %v\n", plan9quote(local))
		return

	}

	if verb == "cgo_ldflag" {
		p, ok := getquoted(&q)
		if !ok {
			Yyerror("usage: //go:cgo_ldflag \"arg\"")
			return
		}
		pragcgobuf += fmt.Sprintf("cgo_ldflag %v\n", plan9quote(p))
		return

	}
}

func (l *lexer) getr() rune {
	// unread rune != 0 available
	if r := l.peekr1; r != 0 {
		l.peekr1 = l.peekr2
		l.peekr2 = 0
		if r == '\n' && importpkg == nil {
			lexlineno++
		}
		return r
	}

redo:
	// common case: 7bit ASCII
	c := obj.Bgetc(l.bin)
	if c < utf8.RuneSelf {
		if c == 0 {
			yyerrorl(int(lexlineno), "illegal NUL byte")
			return 0
		}
		if c == '\n' && importpkg == nil {
			lexlineno++
		}
		return rune(c)
	}
	// c >= utf8.RuneSelf

	// uncommon case: non-ASCII
	var buf [utf8.UTFMax]byte
	buf[0] = byte(c)
	buf[1] = byte(obj.Bgetc(l.bin))
	i := 2
	for ; i < len(buf) && !utf8.FullRune(buf[:i]); i++ {
		buf[i] = byte(obj.Bgetc(l.bin))
	}

	r, w := utf8.DecodeRune(buf[:i])
	if r == utf8.RuneError && w == 1 {
		// The string conversion here makes a copy for passing
		// to fmt.Printf, so that buf itself does not escape and
		// can be allocated on the stack.
		yyerrorl(int(lexlineno), "illegal UTF-8 sequence % x", string(buf[:i]))
	}

	if r == BOM {
		yyerrorl(int(lexlineno), "Unicode (UTF-8) BOM in middle of file")
		goto redo
	}

	return r
}

func (l *lexer) ungetr(r rune) {
	l.peekr2 = l.peekr1
	l.peekr1 = r
	if r == '\n' && importpkg == nil {
		lexlineno--
	}
}

// onechar lexes a single character within a rune or interpreted string literal,
// handling escape sequences as necessary.
func (l *lexer) onechar(quote rune) (r rune, b byte, ok bool) {
	c := l.getr()
	switch c {
	case EOF:
		Yyerror("eof in string")
		l.ungetr(EOF)
		return

	case '\n':
		Yyerror("newline in string")
		l.ungetr('\n')
		return

	case '\\':
		break

	case quote:
		return

	default:
		return c, 0, true
	}

	c = l.getr()
	switch c {
	case 'x':
		return 0, byte(l.hexchar(2)), true

	case 'u':
		return l.unichar(4), 0, true

	case 'U':
		return l.unichar(8), 0, true

	case '0', '1', '2', '3', '4', '5', '6', '7':
		x := c - '0'
		for i := 2; i > 0; i-- {
			c = l.getr()
			if c >= '0' && c <= '7' {
				x = x*8 + c - '0'
				continue
			}

			Yyerror("non-octal character in escape sequence: %c", c)
			l.ungetr(c)
		}

		if x > 255 {
			Yyerror("octal escape value > 255: %d", x)
		}

		return 0, byte(x), true

	case 'a':
		c = '\a'
	case 'b':
		c = '\b'
	case 'f':
		c = '\f'
	case 'n':
		c = '\n'
	case 'r':
		c = '\r'
	case 't':
		c = '\t'
	case 'v':
		c = '\v'
	case '\\':
		c = '\\'

	default:
		if c != quote {
			Yyerror("unknown escape sequence: %c", c)
		}
	}

	return c, 0, true
}

func (l *lexer) unichar(n int) rune {
	x := l.hexchar(n)
	if x > utf8.MaxRune || 0xd800 <= x && x < 0xe000 {
		Yyerror("invalid Unicode code point in escape sequence: %#x", x)
		x = utf8.RuneError
	}
	return rune(x)
}

func (l *lexer) hexchar(n int) uint32 {
	var x uint32

	for ; n > 0; n-- {
		var d uint32
		switch c := l.getr(); {
		case isDigit(c):
			d = uint32(c - '0')
		case 'a' <= c && c <= 'f':
			d = uint32(c - 'a' + 10)
		case 'A' <= c && c <= 'F':
			d = uint32(c - 'A' + 10)
		default:
			Yyerror("non-hex character in escape sequence: %c", c)
			l.ungetr(c)
			return x
		}
		x = x*16 + d
	}

	return x
}

var basicTypes = [...]struct {
	name  string
	etype EType
}{
	{"int8", TINT8},
	{"int16", TINT16},
	{"int32", TINT32},
	{"int64", TINT64},
	{"uint8", TUINT8},
	{"uint16", TUINT16},
	{"uint32", TUINT32},
	{"uint64", TUINT64},
	{"float32", TFLOAT32},
	{"float64", TFLOAT64},
	{"complex64", TCOMPLEX64},
	{"complex128", TCOMPLEX128},
	{"bool", TBOOL},
	{"string", TSTRING},
	{"any", TANY},
}

var builtinFuncs = [...]struct {
	name string
	op   Op
}{
	{"append", OAPPEND},
	{"cap", OCAP},
	{"close", OCLOSE},
	{"complex", OCOMPLEX},
	{"copy", OCOPY},
	{"delete", ODELETE},
	{"imag", OIMAG},
	{"len", OLEN},
	{"make", OMAKE},
	{"new", ONEW},
	{"panic", OPANIC},
	{"print", OPRINT},
	{"println", OPRINTN},
	{"real", OREAL},
	{"recover", ORECOVER},
}

// lexinit initializes known symbols and the basic types.
func lexinit() {
	for _, s := range basicTypes {
		etype := s.etype
		if int(etype) >= len(Types) {
			Fatalf("lexinit: %s bad etype", s.name)
		}
		s2 := Pkglookup(s.name, builtinpkg)
		t := Types[etype]
		if t == nil {
			t = typ(etype)
			t.Sym = s2
			if etype != TANY && etype != TSTRING {
				dowidth(t)
			}
			Types[etype] = t
		}
		s2.Def = typenod(t)
		s2.Def.Name = new(Name)
	}

	for _, s := range builtinFuncs {
		// TODO(marvin): Fix Node.EType type union.
		s2 := Pkglookup(s.name, builtinpkg)
		s2.Def = Nod(ONAME, nil, nil)
		s2.Def.Sym = s2
		s2.Def.Etype = EType(s.op)
	}

	// logically, the type of a string literal.
	// types[TSTRING] is the named type string
	// (the type of x in var x string or var x = "hello").
	// this is the ideal form
	// (the type of x in const x = "hello").
	idealstring = typ(TSTRING)

	idealbool = typ(TBOOL)

	s := Pkglookup("true", builtinpkg)
	s.Def = Nodbool(true)
	s.Def.Sym = Lookup("true")
	s.Def.Name = new(Name)
	s.Def.Type = idealbool

	s = Pkglookup("false", builtinpkg)
	s.Def = Nodbool(false)
	s.Def.Sym = Lookup("false")
	s.Def.Name = new(Name)
	s.Def.Type = idealbool

	s = Lookup("_")
	s.Block = -100
	s.Def = Nod(ONAME, nil, nil)
	s.Def.Sym = s
	Types[TBLANK] = typ(TBLANK)
	s.Def.Type = Types[TBLANK]
	nblank = s.Def

	s = Pkglookup("_", builtinpkg)
	s.Block = -100
	s.Def = Nod(ONAME, nil, nil)
	s.Def.Sym = s
	Types[TBLANK] = typ(TBLANK)
	s.Def.Type = Types[TBLANK]

	Types[TNIL] = typ(TNIL)
	s = Pkglookup("nil", builtinpkg)
	var v Val
	v.U = new(NilVal)
	s.Def = nodlit(v)
	s.Def.Sym = s
	s.Def.Name = new(Name)

	s = Pkglookup("iota", builtinpkg)
	s.Def = Nod(OIOTA, nil, nil)
	s.Def.Sym = s
	s.Def.Name = new(Name)
}

func lexinit1() {
	// t = interface { Error() string }
	rcvr := typ(TSTRUCT)

	rcvr.Type = typ(TFIELD)
	rcvr.Type.Type = Ptrto(typ(TSTRUCT))
	rcvr.Funarg = true
	in := typ(TSTRUCT)
	in.Funarg = true
	out := typ(TSTRUCT)
	out.Type = typ(TFIELD)
	out.Type.Type = Types[TSTRING]
	out.Funarg = true
	f := typ(TFUNC)
	*getthis(f) = rcvr
	*Getoutarg(f) = out
	*getinarg(f) = in
	f.Thistuple = 1
	f.Intuple = 0
	f.Outnamed = false
	f.Outtuple = 1
	t := typ(TINTER)
	t.Type = typ(TFIELD)
	t.Type.Sym = Lookup("Error")
	t.Type.Type = f

	// error type
	s := Pkglookup("error", builtinpkg)
	errortype = t
	errortype.Sym = s
	s.Def = typenod(errortype)

	// byte alias
	s = Pkglookup("byte", builtinpkg)
	bytetype = typ(TUINT8)
	bytetype.Sym = s
	s.Def = typenod(bytetype)
	s.Def.Name = new(Name)

	// rune alias
	s = Pkglookup("rune", builtinpkg)
	runetype = typ(TINT32)
	runetype.Sym = s
	s.Def = typenod(runetype)
	s.Def.Name = new(Name)

	// backend-specific builtin types (e.g. int).
	for i := range Thearch.Typedefs {
		s := Pkglookup(Thearch.Typedefs[i].Name, builtinpkg)
		s.Def = typenod(Types[Thearch.Typedefs[i].Etype])
		s.Def.Name = new(Name)
		s.Origpkg = builtinpkg
	}
}

func lexfini() {
	for _, s := range builtinpkg.Syms {
		if s.Def == nil {
			continue
		}
		s1 := Lookup(s.Name)
		if s1.Def != nil {
			continue
		}

		s1.Def = s.Def
		s1.Block = s.Block
	}

	nodfp = Nod(ONAME, nil, nil)
	nodfp.Type = Types[TINT32]
	nodfp.Xoffset = 0
	nodfp.Class = PPARAM
	nodfp.Sym = Lookup(".fp")
}

var lexn = map[rune]string{
	LANDAND:    "ANDAND",
	LANDNOT:    "ANDNOT",
	LASOP:      "ASOP",
	LBREAK:     "BREAK",
	LCASE:      "CASE",
	LCHAN:      "CHAN",
	LCOLAS:     "COLAS",
	LCOMM:      "<-",
	LCONST:     "CONST",
	LCONTINUE:  "CONTINUE",
	LDDD:       "...",
	LDEFAULT:   "DEFAULT",
	LDEFER:     "DEFER",
	LELSE:      "ELSE",
	LEQ:        "EQ",
	LFALL:      "FALL",
	LFOR:       "FOR",
	LFUNC:      "FUNC",
	LGE:        "GE",
	LGO:        "GO",
	LGOTO:      "GOTO",
	LGT:        "GT",
	LIF:        "IF",
	LIMPORT:    "IMPORT",
	LINCOP:     "INCOP",
	LINTERFACE: "INTERFACE",
	LLE:        "LE",
	LLITERAL:   "LITERAL",
	LLSH:       "LSH",
	LLT:        "LT",
	LMAP:       "MAP",
	LNAME:      "NAME",
	LNE:        "NE",
	LOROR:      "OROR",
	LPACKAGE:   "PACKAGE",
	LRANGE:     "RANGE",
	LRETURN:    "RETURN",
	LRSH:       "RSH",
	LSELECT:    "SELECT",
	LSTRUCT:    "STRUCT",
	LSWITCH:    "SWITCH",
	LTYPE:      "TYPE",
	LVAR:       "VAR",
}

func lexname(lex rune) string {
	if s, ok := lexn[lex]; ok {
		return s
	}
	return fmt.Sprintf("LEX-%d", lex)
}

func pkgnotused(lineno int, path string, name string) {
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
		yyerrorl(int(lineno), "imported and not used: %q", path)
	} else {
		yyerrorl(int(lineno), "imported and not used: %q as %s", path, name)
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
					pkgnotused(int(s.Def.Lineno), s.Def.Name.Pkg.Path, s.Name)
				}
				s.Def = nil
				continue
			}

			if s.Def.Sym != s {
				// throw away top-level name left over
				// from previous import . "x"
				if s.Def.Name != nil && s.Def.Name.Pack != nil && !s.Def.Name.Pack.Used && nsyntaxerrors == 0 {
					pkgnotused(int(s.Def.Name.Pack.Lineno), s.Def.Name.Pack.Name.Pkg.Path, "")
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
		if Ctxt.Windows != 0 {
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
