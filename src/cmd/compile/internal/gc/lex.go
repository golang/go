// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run mkbuiltin.go runtime unsafe

package gc

import (
	"bytes"
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
	Ctxt.Diag = Yyerror
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
		log.Fatal("can not use both -race and -msan")
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
	const BOM = 0xFEFF

	for _, infile = range flag.Args() {
		if trace && Debug['x'] != 0 {
			fmt.Printf("--- %s ---\n", infile)
		}

		linehistpush(infile)

		curio.infile = infile
		var err error
		curio.bin, err = obj.Bopenr(infile)
		if err != nil {
			fmt.Printf("open %s: %v\n", infile, err)
			errorexit()
		}

		curio.peekc = 0
		curio.peekc1 = 0
		curio.nlsemi = false
		curio.eofnl = false
		curio.last = 0

		// Skip initial BOM if present.
		if obj.Bgetrune(curio.bin) != BOM {
			obj.Bungetrune(curio.bin)
		}

		block = 1
		iota_ = -1000000

		imported_unsafe = false

		parse_file()
		if nsyntaxerrors != 0 {
			errorexit()
		}

		linehistpop()
		if curio.bin != nil {
			obj.Bterm(curio.bin)
		}
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

	// symbol table may be first; skip it
	sz := arsize(b, "__.GOSYMDEF")

	if sz >= 0 {
		obj.Bseek(b, int64(sz), 1)
	} else {
		obj.Bseek(b, 8, 0)
	}

	// package export block is next
	sz = arsize(b, "__.PKGDEF")

	if sz <= 0 {
		return false
	}
	return true
}

func addidir(dir string) {
	if dir == "" {
		return
	}

	var pp **Idir
	for pp = &idirs; *pp != nil; pp = &(*pp).link {
	}
	*pp = new(Idir)
	(*pp).link = nil
	(*pp).dir = dir
}

// is this path a local name?  begins with ./ or ../ or /
func islocalname(name string) bool {
	return strings.HasPrefix(name, "/") ||
		Ctxt.Windows != 0 && len(name) >= 3 && isAlpha(int(name[0])) && name[1] == ':' && name[2] == '/' ||
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

	for p := idirs; p != nil; p = p.link {
		file = fmt.Sprintf("%s/%s.a", p.dir, name)
		if _, err := os.Stat(file); err == nil {
			return file, true
		}
		file = fmt.Sprintf("%s/%s.o", p.dir, name)
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

func fakeimport() {
	importpkg = mkpkg("fake")
	cannedimports("fake.o", "$$\n")
}

// TODO(gri) line argument doesn't appear to be used
func importfile(f *Val, line int) {
	if _, ok := f.U.(string); !ok {
		Yyerror("import statement not a string")
		fakeimport()
		return
	}

	if len(f.U.(string)) == 0 {
		Yyerror("import path is empty")
		fakeimport()
		return
	}

	if isbadimport(f.U.(string)) {
		fakeimport()
		return
	}

	// The package name main is no longer reserved,
	// but we reserve the import path "main" to identify
	// the main package, just as we reserve the import
	// path "math" to identify the standard math package.
	if f.U.(string) == "main" {
		Yyerror("cannot import \"main\"")
		errorexit()
	}

	if myimportpath != "" && f.U.(string) == myimportpath {
		Yyerror("import %q while compiling that package (import cycle)", f.U.(string))
		errorexit()
	}

	path_ := f.U.(string)

	if mapped, ok := importMap[path_]; ok {
		path_ = mapped
	}

	if path_ == "unsafe" {
		if safemode != 0 {
			Yyerror("cannot import package unsafe")
			errorexit()
		}

		importpkg = mkpkg(f.U.(string))
		cannedimports("unsafe.o", unsafeimport)
		imported_unsafe = true
		return
	}

	if islocalname(path_) {
		if path_[0] == '/' {
			Yyerror("import path cannot be absolute path")
			fakeimport()
			return
		}

		prefix := Ctxt.Pathname
		if localimport != "" {
			prefix = localimport
		}
		cleanbuf := prefix
		cleanbuf += "/"
		cleanbuf += path_
		cleanbuf = path.Clean(cleanbuf)
		path_ = cleanbuf

		if isbadimport(path_) {
			fakeimport()
			return
		}
	}

	file, found := findpkg(path_)
	if !found {
		Yyerror("can't find import: %q", f.U.(string))
		errorexit()
	}

	importpkg = mkpkg(path_)

	// If we already saw that package, feed a dummy statement
	// to the lexer to avoid parsing export data twice.
	if importpkg.Imported {
		tag := ""
		if importpkg.Safe {
			tag = "safe"
		}

		p := fmt.Sprintf("package %s %s\n$$\n", importpkg.Name, tag)
		cannedimports(file, p)
		return
	}

	importpkg.Imported = true

	var err error
	var imp *obj.Biobuf
	imp, err = obj.Bopenr(file)
	if err != nil {
		Yyerror("can't open import: %q: %v", f.U.(string), err)
		errorexit()
	}

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
		pushedio = curio

		curio.bin = imp
		curio.peekc = 0
		curio.peekc1 = 0
		curio.infile = file
		curio.nlsemi = false
		typecheckok = true

		push_parser()

	case 'B':
		// new export format
		obj.Bgetc(imp) // skip \n after $$B
		Import(imp)

		// continue as if the package was imported before (see above)
		tag := ""
		if importpkg.Safe {
			tag = "safe"
		}
		p := fmt.Sprintf("package %s %s\n$$\n", importpkg.Name, tag)
		cannedimports(file, p)

	default:
		Yyerror("no import in %q", f.U.(string))
	}
}

func unimportfile() {
	pop_parser()

	if curio.bin != nil {
		obj.Bterm(curio.bin)
		curio.bin = nil
	} else {
		lexlineno-- // re correct sys.6 line number
	}

	curio = pushedio

	pushedio.bin = nil
	incannedimport = 0
	typecheckok = false
}

func cannedimports(file string, cp string) {
	lexlineno++ // if sys.6 is included on line 1,

	pushedio = curio

	curio.bin = nil
	curio.peekc = 0
	curio.peekc1 = 0
	curio.infile = file
	curio.cp = cp
	curio.nlsemi = false
	curio.importsafe = false

	typecheckok = true
	incannedimport = 1

	push_parser()
}

func isSpace(c int) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func isAlpha(c int) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z'
}

func isDigit(c int) bool {
	return '0' <= c && c <= '9'
}
func isAlnum(c int) bool {
	return isAlpha(c) || isDigit(c)
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

func isfrog(c int) bool {
	// complain about possibly invisible control characters
	if c < ' ' {
		return !isSpace(c) // exclude good white space
	}

	if 0x7f <= c && c <= 0xa0 { // DEL, unicode block including unbreakable space.
		return true
	}
	return false
}

type yySymType struct {
	yys  int
	node *Node
	list *NodeList
	typ  *Type
	sym  *Sym
	val  Val
	op   Op
}

const (
	LLITERAL = 57346 + iota
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
	LDEC
	LEQ
	LGE
	LGT
	LIGNORE
	LINC
	LLE
	LLSH
	LLT
	LNE
	LOROR
	LRSH
)

func _yylex(yylval *yySymType) int32 {
	var c1 int
	var op Op
	var escflag int
	var v int64
	var cp *bytes.Buffer
	var s *Sym
	var str string

	prevlineno = lineno

l0:
	c := getc()
	if isSpace(c) {
		if c == '\n' && curio.nlsemi {
			ungetc(c)
			if Debug['x'] != 0 {
				fmt.Printf("lex: implicit semi\n")
			}
			return ';'
		}

		goto l0
	}

	lineno = lexlineno // start of token

	if c >= utf8.RuneSelf {
		// all multibyte runes are alpha
		cp = &lexbuf
		cp.Reset()

		goto talph
	}

	if isAlpha(c) {
		cp = &lexbuf
		cp.Reset()
		goto talph
	}

	if isDigit(c) {
		cp = &lexbuf
		cp.Reset()
		if c != '0' {
			for {
				cp.WriteByte(byte(c))
				c = getc()
				if isDigit(c) {
					continue
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
		}

		cp.WriteByte(byte(c))
		c = getc()
		if c == 'x' || c == 'X' {
			for {
				cp.WriteByte(byte(c))
				c = getc()
				if isDigit(c) {
					continue
				}
				if c >= 'a' && c <= 'f' {
					continue
				}
				if c >= 'A' && c <= 'F' {
					continue
				}
				if lexbuf.Len() == 2 {
					Yyerror("malformed hex constant")
				}
				if c == 'p' {
					goto caseep
				}
				goto ncu
			}
		}

		if c == 'p' { // 0p begins floating point zero
			goto caseep
		}

		c1 = 0
		for {
			if !isDigit(c) {
				break
			}
			if c < '0' || c > '7' {
				c1 = 1 // not octal
			}
			cp.WriteByte(byte(c))
			c = getc()
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
		if c1 != 0 {
			Yyerror("malformed octal constant")
		}
		goto ncu
	}

	switch c {
	case EOF:
		lineno = prevlineno
		ungetc(EOF)
		return -1

	case '_':
		cp = &lexbuf
		cp.Reset()
		goto talph

	case '.':
		c1 = getc()
		if isDigit(c1) {
			cp = &lexbuf
			cp.Reset()
			cp.WriteByte(byte(c))
			c = c1
			goto casedot
		}

		if c1 == '.' {
			c1 = getc()
			if c1 == '.' {
				c = LDDD
				goto lx
			}

			ungetc(c1)
			c1 = '.'
		}

		// "..."
	case '"':
		lexbuf.Reset()
		lexbuf.WriteString(`"<string>"`)

		cp = &strbuf
		cp.Reset()

		for {
			if escchar('"', &escflag, &v) {
				break
			}
			if v < utf8.RuneSelf || escflag != 0 {
				cp.WriteByte(byte(v))
			} else {
				cp.WriteRune(rune(v))
			}
		}

		goto strlit

		// `...`
	case '`':
		lexbuf.Reset()
		lexbuf.WriteString("`<string>`")

		cp = &strbuf
		cp.Reset()

		for {
			c = int(getr())
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
			cp.WriteRune(rune(c))
		}

		goto strlit

		// '.'
	case '\'':
		if escchar('\'', &escflag, &v) {
			Yyerror("empty character literal or unescaped ' in character literal")
			v = '\''
		}

		if !escchar('\'', &escflag, &v) {
			Yyerror("missing '")
			ungetc(int(v))
		}

		x := new(Mpint)
		yylval.val.U = x
		Mpmovecfix(x, v)
		x.Rune = true
		if Debug['x'] != 0 {
			fmt.Printf("lex: codepoint literal\n")
		}
		litbuf = "string literal"
		return LLITERAL

	case '/':
		c1 = getc()
		if c1 == '*' {
			nl := false
			for {
				c = int(getr())
				if c == '\n' {
					nl = true
				}
				for c == '*' {
					c = int(getr())
					if c == '/' {
						if nl {
							ungetc('\n')
						}
						goto l0
					}

					if c == '\n' {
						nl = true
					}
				}

				if c == EOF {
					Yyerror("eof in comment")
					errorexit()
				}
			}
		}

		if c1 == '/' {
			c = getlinepragma()
			for {
				if c == '\n' || c == EOF {
					ungetc(c)
					goto l0
				}

				c = int(getr())
			}
		}

		if c1 == '=' {
			op = ODIV
			goto asop
		}

	case ':':
		c1 = getc()
		if c1 == '=' {
			c = int(LCOLAS)
			goto lx
		}

	case '*':
		c1 = getc()
		if c1 == '=' {
			op = OMUL
			goto asop
		}

	case '%':
		c1 = getc()
		if c1 == '=' {
			op = OMOD
			goto asop
		}

	case '+':
		c1 = getc()
		if c1 == '+' {
			c = int(LINC)
			goto lx
		}

		if c1 == '=' {
			op = OADD
			goto asop
		}

	case '-':
		c1 = getc()
		if c1 == '-' {
			c = int(LDEC)
			goto lx
		}

		if c1 == '=' {
			op = OSUB
			goto asop
		}

	case '>':
		c1 = getc()
		if c1 == '>' {
			c = int(LRSH)
			c1 = getc()
			if c1 == '=' {
				op = ORSH
				goto asop
			}

			break
		}

		if c1 == '=' {
			c = int(LGE)
			goto lx
		}

		c = int(LGT)

	case '<':
		c1 = getc()
		if c1 == '<' {
			c = int(LLSH)
			c1 = getc()
			if c1 == '=' {
				op = OLSH
				goto asop
			}

			break
		}

		if c1 == '=' {
			c = int(LLE)
			goto lx
		}

		if c1 == '-' {
			c = int(LCOMM)
			goto lx
		}

		c = int(LLT)

	case '=':
		c1 = getc()
		if c1 == '=' {
			c = int(LEQ)
			goto lx
		}

	case '!':
		c1 = getc()
		if c1 == '=' {
			c = int(LNE)
			goto lx
		}

	case '&':
		c1 = getc()
		if c1 == '&' {
			c = int(LANDAND)
			goto lx
		}

		if c1 == '^' {
			c = int(LANDNOT)
			c1 = getc()
			if c1 == '=' {
				op = OANDNOT
				goto asop
			}

			break
		}

		if c1 == '=' {
			op = OAND
			goto asop
		}

	case '|':
		c1 = getc()
		if c1 == '|' {
			c = int(LOROR)
			goto lx
		}

		if c1 == '=' {
			op = OOR
			goto asop
		}

	case '^':
		c1 = getc()
		if c1 == '=' {
			op = OXOR
			goto asop
		}

	default:
		goto lx
	}

	ungetc(c1)

lx:
	if Debug['x'] != 0 {
		if c > 0xff {
			fmt.Printf("%v lex: TOKEN %s\n", Ctxt.Line(int(lexlineno)), lexname(c))
		} else {
			fmt.Printf("%v lex: TOKEN '%c'\n", Ctxt.Line(int(lexlineno)), c)
		}
	}
	if isfrog(c) {
		Yyerror("illegal character 0x%x", uint(c))
		goto l0
	}

	if importpkg == nil && (c == '#' || c == '$' || c == '?' || c == '@' || c == '\\') {
		Yyerror("%s: unexpected %c", "syntax error", c)
		goto l0
	}

	return int32(c)

asop:
	yylval.op = op
	if Debug['x'] != 0 {
		fmt.Printf("lex: TOKEN ASOP %s=\n", goopnames[op])
	}
	return LASOP

	// cp is set to lexbuf and some
	// prefix has been stored
talph:
	for {
		if c >= utf8.RuneSelf {
			ungetc(c)
			r := rune(getr())

			// 0xb7 · is used for internal names
			if !unicode.IsLetter(r) && !unicode.IsDigit(r) && (importpkg == nil || r != 0xb7) {
				Yyerror("invalid identifier character U+%04x", r)
			}
			if cp.Len() == 0 && unicode.IsDigit(r) {
				Yyerror("identifier cannot begin with digit U+%04x", r)
			}
			cp.WriteRune(r)
		} else if !isAlnum(c) && c != '_' {
			break
		} else {
			cp.WriteByte(byte(c))
		}
		c = getc()
	}

	cp = nil
	ungetc(c)

	s = LookupBytes(lexbuf.Bytes())
	if s.Lexical == LIGNORE {
		goto l0
	}

	if Debug['x'] != 0 {
		fmt.Printf("lex: %s %s\n", s, lexname(int(s.Lexical)))
	}
	yylval.sym = s
	return int32(s.Lexical)

ncu:
	cp = nil
	ungetc(c)

	str = lexbuf.String()
	yylval.val.U = new(Mpint)
	mpatofix(yylval.val.U.(*Mpint), str)
	if yylval.val.U.(*Mpint).Ovf {
		Yyerror("overflow in constant")
		Mpmovecfix(yylval.val.U.(*Mpint), 0)
	}

	if Debug['x'] != 0 {
		fmt.Printf("lex: integer literal\n")
	}
	litbuf = "literal " + str
	return LLITERAL

casedot:
	for {
		cp.WriteByte(byte(c))
		c = getc()
		if !isDigit(c) {
			break
		}
	}

	if c == 'i' {
		goto casei
	}
	if c != 'e' && c != 'E' {
		goto caseout
	}

caseep:
	if importpkg == nil && (c == 'p' || c == 'P') {
		// <mantissa>p<base-2-exponent> is allowed in .a/.o imports,
		// but not in .go sources.  See #9036.
		Yyerror("malformed floating point constant")
	}
	cp.WriteByte(byte(c))
	c = getc()
	if c == '+' || c == '-' {
		cp.WriteByte(byte(c))
		c = getc()
	}

	if !isDigit(c) {
		Yyerror("malformed floating point constant exponent")
	}
	for isDigit(c) {
		cp.WriteByte(byte(c))
		c = getc()
	}

	if c == 'i' {
		goto casei
	}
	goto caseout

	// imaginary constant
casei:
	cp = nil

	str = lexbuf.String()
	yylval.val.U = new(Mpcplx)
	Mpmovecflt(&yylval.val.U.(*Mpcplx).Real, 0.0)
	mpatoflt(&yylval.val.U.(*Mpcplx).Imag, str)
	if yylval.val.U.(*Mpcplx).Imag.Val.IsInf() {
		Yyerror("overflow in imaginary constant")
		Mpmovecflt(&yylval.val.U.(*Mpcplx).Imag, 0.0)
	}

	if Debug['x'] != 0 {
		fmt.Printf("lex: imaginary literal\n")
	}
	litbuf = "literal " + str
	return LLITERAL

caseout:
	cp = nil
	ungetc(c)

	str = lexbuf.String()
	yylval.val.U = newMpflt()
	mpatoflt(yylval.val.U.(*Mpflt), str)
	if yylval.val.U.(*Mpflt).Val.IsInf() {
		Yyerror("overflow in float constant")
		Mpmovecflt(yylval.val.U.(*Mpflt), 0.0)
	}

	if Debug['x'] != 0 {
		fmt.Printf("lex: floating literal\n")
	}
	litbuf = "literal " + str
	return LLITERAL

strlit:
	yylval.val.U = internString(cp.Bytes())
	if Debug['x'] != 0 {
		fmt.Printf("lex: string literal\n")
	}
	litbuf = "string literal"
	return LLITERAL
}

var internedStrings = map[string]string{}

func internString(b []byte) string {
	s, ok := internedStrings[string(b)] // string(b) here doesn't allocate
	if ok {
		return s
	}
	s = string(b)
	internedStrings[s] = s
	return s
}

func more(pp *string) bool {
	p := *pp
	for p != "" && isSpace(int(p[0])) {
		p = p[1:]
	}
	*pp = p
	return p != ""
}

// read and interpret syntax that looks like
// //line parse.y:15
// as a discontinuity in sequential line numbers.
// the next line of input comes from parse.y:15
func getlinepragma() int {
	var cmd, verb, name string

	c := int(getr())
	if c == 'g' {
		cp := &lexbuf
		cp.Reset()
		cp.WriteByte('g') // already read
		for {
			c = int(getr())
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

		cmd = text
		verb = cmd
		if i := strings.Index(verb, " "); i >= 0 {
			verb = verb[:i]
		}

		if verb == "go:linkname" {
			if !imported_unsafe {
				Yyerror("//go:linkname only allowed in Go files that import \"unsafe\"")
			}
			f := strings.Fields(cmd)
			if len(f) != 3 {
				Yyerror("usage: //go:linkname localname linkname")
				return c
			}

			Lookup(f[1]).Linkname = f[2]
			return c
		}

		if verb == "go:nointerface" && obj.Fieldtrack_enabled != 0 {
			nointerface = true
			return c
		}

		if verb == "go:noescape" {
			noescape = true
			return c
		}

		if verb == "go:norace" {
			norace = true
			return c
		}

		if verb == "go:nosplit" {
			nosplit = true
			return c
		}

		if verb == "go:noinline" {
			noinline = true
			return c
		}

		if verb == "go:systemstack" {
			if compiling_runtime == 0 {
				Yyerror("//go:systemstack only allowed in runtime")
			}
			systemstack = true
			return c
		}

		if verb == "go:nowritebarrier" {
			if compiling_runtime == 0 {
				Yyerror("//go:nowritebarrier only allowed in runtime")
			}
			nowritebarrier = true
			return c
		}

		if verb == "go:nowritebarrierrec" {
			if compiling_runtime == 0 {
				Yyerror("//go:nowritebarrierrec only allowed in runtime")
			}
			nowritebarrierrec = true
			nowritebarrier = true // Implies nowritebarrier
			return c
		}
		return c
	}
	if c != 'l' {
		return c
	}
	for i := 1; i < 5; i++ {
		c = int(getr())
		if c != int("line "[i]) {
			return c
		}
	}

	cp := &lexbuf
	cp.Reset()
	linep := 0
	for {
		c = int(getr())
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
	n := 0
	for _, c := range text[linep:] {
		if c < '0' || c > '9' {
			goto out
		}
		n = n*10 + int(c) - '0'
		if n > 1e8 {
			Yyerror("line number out of range")
			errorexit()
		}
	}

	if n <= 0 {
		return c
	}

	name = text[:linep-1]
	linehistupdate(name, n)
	return c

out:
	return c
}

func getimpsym(pp *string) string {
	more(pp) // skip spaces
	p := *pp
	if p == "" || p[0] == '"' {
		return ""
	}
	i := 0
	for i < len(p) && !isSpace(int(p[i])) && p[i] != '"' {
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

func yylex(yylval *yySymType) int32 {
	lx := _yylex(yylval)

	if curio.nlsemi && lx == EOF {
		// Treat EOF as "end of line" for the purposes
		// of inserting a semicolon.
		lx = ';'
	}

	switch lx {
	case LNAME,
		LLITERAL,
		LBREAK,
		LCONTINUE,
		LFALL,
		LRETURN,
		LINC,
		LDEC,
		')',
		'}',
		']':
		curio.nlsemi = true

	default:
		curio.nlsemi = false
	}

	return lx
}

func getc() int {
	c := curio.peekc
	if c != 0 {
		curio.peekc = curio.peekc1
		curio.peekc1 = 0
		goto check
	}

	if curio.bin == nil {
		if len(curio.cp) == 0 {
			c = 0
		} else {
			c = int(curio.cp[0])
			curio.cp = curio.cp[1:]
		}
	} else {
	loop:
		c = obj.Bgetc(curio.bin)
		// recognize BOM (U+FEFF): UTF-8 encoding is 0xef 0xbb 0xbf
		if c == 0xef {
			buf, err := curio.bin.Peek(2)
			if err != nil {
				yyerrorl(int(lexlineno), "illegal UTF-8 sequence ef % x followed by read error (%v)", string(buf), err)
				errorexit()
			}
			if buf[0] == 0xbb && buf[1] == 0xbf {
				yyerrorl(int(lexlineno), "Unicode (UTF-8) BOM in middle of file")

				// consume BOM bytes
				obj.Bgetc(curio.bin)
				obj.Bgetc(curio.bin)
				goto loop
			}
		}
	}

check:
	switch c {
	case 0:
		if curio.bin != nil {
			Yyerror("illegal NUL byte")
			break
		}
		fallthrough

		// insert \n at EOF
	case EOF:
		if curio.eofnl || curio.last == '\n' {
			return EOF
		}
		curio.eofnl = true
		c = '\n'
		fallthrough

	case '\n':
		if pushedio.bin == nil {
			lexlineno++
		}
	}

	curio.last = c
	return c
}

func ungetc(c int) {
	curio.peekc1 = curio.peekc
	curio.peekc = c
	if c == '\n' && pushedio.bin == nil {
		lexlineno--
	}
}

func getr() int32 {
	var buf [utf8.UTFMax]byte

	for i := 0; ; i++ {
		c := getc()
		if i == 0 && c < utf8.RuneSelf {
			return int32(c)
		}
		buf[i] = byte(c)
		if i+1 == len(buf) || utf8.FullRune(buf[:i+1]) {
			r, w := utf8.DecodeRune(buf[:i+1])
			if r == utf8.RuneError && w == 1 {
				lineno = lexlineno
				// The string conversion here makes a copy for passing
				// to fmt.Printf, so that buf itself does not escape and can
				// be allocated on the stack.
				Yyerror("illegal UTF-8 sequence % x", string(buf[:i+1]))
			}
			return int32(r)
		}
	}
}

func escchar(e int, escflg *int, val *int64) bool {
	*escflg = 0

	c := int(getr())
	switch c {
	case EOF:
		Yyerror("eof in string")
		return true

	case '\n':
		Yyerror("newline in string")
		return true

	case '\\':
		break

	default:
		if c == e {
			return true
		}
		*val = int64(c)
		return false
	}

	u := 0
	c = int(getr())
	var i int
	switch c {
	case 'x':
		*escflg = 1 // it's a byte
		i = 2
		goto hex

	case 'u':
		i = 4
		u = 1
		goto hex

	case 'U':
		i = 8
		u = 1
		goto hex

	case '0',
		'1',
		'2',
		'3',
		'4',
		'5',
		'6',
		'7':
		*escflg = 1 // it's a byte
		l := int64(c) - '0'
		for i := 2; i > 0; i-- {
			c = getc()
			if c >= '0' && c <= '7' {
				l = l*8 + int64(c) - '0'
				continue
			}

			Yyerror("non-octal character in escape sequence: %c", c)
			ungetc(c)
		}

		if l > 255 {
			Yyerror("octal escape value > 255: %d", l)
		}

		*val = l
		return false

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
		if c != e {
			Yyerror("unknown escape sequence: %c", c)
		}
	}

	*val = int64(c)
	return false

hex:
	l := int64(0)
	for ; i > 0; i-- {
		c = getc()
		if c >= '0' && c <= '9' {
			l = l*16 + int64(c) - '0'
			continue
		}

		if c >= 'a' && c <= 'f' {
			l = l*16 + int64(c) - 'a' + 10
			continue
		}

		if c >= 'A' && c <= 'F' {
			l = l*16 + int64(c) - 'A' + 10
			continue
		}

		Yyerror("non-hex character in escape sequence: %c", c)
		ungetc(c)
		break
	}

	if u != 0 && (l > utf8.MaxRune || (0xd800 <= l && l < 0xe000)) {
		Yyerror("invalid Unicode code point in escape sequence: %#x", l)
		l = utf8.RuneError
	}

	*val = l
	return false
}

var syms = []struct {
	name    string
	lexical int
	etype   EType
	op      Op
}{
	// basic types
	{"int8", LNAME, TINT8, OXXX},
	{"int16", LNAME, TINT16, OXXX},
	{"int32", LNAME, TINT32, OXXX},
	{"int64", LNAME, TINT64, OXXX},
	{"uint8", LNAME, TUINT8, OXXX},
	{"uint16", LNAME, TUINT16, OXXX},
	{"uint32", LNAME, TUINT32, OXXX},
	{"uint64", LNAME, TUINT64, OXXX},
	{"float32", LNAME, TFLOAT32, OXXX},
	{"float64", LNAME, TFLOAT64, OXXX},
	{"complex64", LNAME, TCOMPLEX64, OXXX},
	{"complex128", LNAME, TCOMPLEX128, OXXX},
	{"bool", LNAME, TBOOL, OXXX},
	{"string", LNAME, TSTRING, OXXX},
	{"any", LNAME, TANY, OXXX},
	{"break", LBREAK, Txxx, OXXX},
	{"case", LCASE, Txxx, OXXX},
	{"chan", LCHAN, Txxx, OXXX},
	{"const", LCONST, Txxx, OXXX},
	{"continue", LCONTINUE, Txxx, OXXX},
	{"default", LDEFAULT, Txxx, OXXX},
	{"else", LELSE, Txxx, OXXX},
	{"defer", LDEFER, Txxx, OXXX},
	{"fallthrough", LFALL, Txxx, OXXX},
	{"for", LFOR, Txxx, OXXX},
	{"func", LFUNC, Txxx, OXXX},
	{"go", LGO, Txxx, OXXX},
	{"goto", LGOTO, Txxx, OXXX},
	{"if", LIF, Txxx, OXXX},
	{"import", LIMPORT, Txxx, OXXX},
	{"interface", LINTERFACE, Txxx, OXXX},
	{"map", LMAP, Txxx, OXXX},
	{"package", LPACKAGE, Txxx, OXXX},
	{"range", LRANGE, Txxx, OXXX},
	{"return", LRETURN, Txxx, OXXX},
	{"select", LSELECT, Txxx, OXXX},
	{"struct", LSTRUCT, Txxx, OXXX},
	{"switch", LSWITCH, Txxx, OXXX},
	{"type", LTYPE, Txxx, OXXX},
	{"var", LVAR, Txxx, OXXX},
	{"append", LNAME, Txxx, OAPPEND},
	{"cap", LNAME, Txxx, OCAP},
	{"close", LNAME, Txxx, OCLOSE},
	{"complex", LNAME, Txxx, OCOMPLEX},
	{"copy", LNAME, Txxx, OCOPY},
	{"delete", LNAME, Txxx, ODELETE},
	{"imag", LNAME, Txxx, OIMAG},
	{"len", LNAME, Txxx, OLEN},
	{"make", LNAME, Txxx, OMAKE},
	{"new", LNAME, Txxx, ONEW},
	{"panic", LNAME, Txxx, OPANIC},
	{"print", LNAME, Txxx, OPRINT},
	{"println", LNAME, Txxx, OPRINTN},
	{"real", LNAME, Txxx, OREAL},
	{"recover", LNAME, Txxx, ORECOVER},
	{"notwithstanding", LIGNORE, Txxx, OXXX},
	{"thetruthofthematter", LIGNORE, Txxx, OXXX},
	{"despiteallobjections", LIGNORE, Txxx, OXXX},
	{"whereas", LIGNORE, Txxx, OXXX},
	{"insofaras", LIGNORE, Txxx, OXXX},
}

// lexinit initializes known symbols and the basic types.
func lexinit() {
	for _, s := range syms {
		lex := s.lexical
		s1 := Lookup(s.name)
		s1.Lexical = uint16(lex)

		if etype := s.etype; etype != Txxx {
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

			s2.Lexical = LNAME
			s2.Def = typenod(t)
			s2.Def.Name = new(Name)
			continue
		}

		// TODO(marvin): Fix Node.EType type union.
		if etype := s.op; etype != OXXX {
			s2 := Pkglookup(s.name, builtinpkg)
			s2.Lexical = LNAME
			s2.Def = Nod(ONAME, nil, nil)
			s2.Def.Sym = s2
			s2.Def.Etype = EType(etype)
		}
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
	s := Lookup("error")

	s.Lexical = LNAME
	s1 := Pkglookup("error", builtinpkg)
	errortype = t
	errortype.Sym = s1
	s1.Lexical = LNAME
	s1.Def = typenod(errortype)

	// byte alias
	s = Lookup("byte")

	s.Lexical = LNAME
	s1 = Pkglookup("byte", builtinpkg)
	bytetype = typ(TUINT8)
	bytetype.Sym = s1
	s1.Lexical = LNAME
	s1.Def = typenod(bytetype)
	s1.Def.Name = new(Name)

	// rune alias
	s = Lookup("rune")

	s.Lexical = LNAME
	s1 = Pkglookup("rune", builtinpkg)
	runetype = typ(TINT32)
	runetype.Sym = s1
	s1.Lexical = LNAME
	s1.Def = typenod(runetype)
	s1.Def.Name = new(Name)
}

func lexfini() {
	for i := range syms {
		lex := syms[i].lexical
		if lex != LNAME {
			continue
		}
		s := Lookup(syms[i].name)
		s.Lexical = uint16(lex)

		etype := syms[i].etype
		if etype != Txxx && (etype != TANY || Debug['A'] != 0) && s.Def == nil {
			s.Def = typenod(Types[etype])
			s.Def.Name = new(Name)
			s.Origpkg = builtinpkg
		}

		// TODO(marvin): Fix Node.EType type union.
		etype = EType(syms[i].op)
		if etype != EType(OXXX) && s.Def == nil {
			s.Def = Nod(ONAME, nil, nil)
			s.Def.Sym = s
			s.Def.Etype = etype
			s.Origpkg = builtinpkg
		}
	}

	// backend-specific builtin types (e.g. int).
	for i := range Thearch.Typedefs {
		s := Lookup(Thearch.Typedefs[i].Name)
		if s.Def == nil {
			s.Def = typenod(Types[Thearch.Typedefs[i].Etype])
			s.Def.Name = new(Name)
			s.Origpkg = builtinpkg
		}
	}

	// there's only so much table-driven we can handle.
	// these are special cases.
	if s := Lookup("byte"); s.Def == nil {
		s.Def = typenod(bytetype)
		s.Def.Name = new(Name)
		s.Origpkg = builtinpkg
	}

	if s := Lookup("error"); s.Def == nil {
		s.Def = typenod(errortype)
		s.Def.Name = new(Name)
		s.Origpkg = builtinpkg
	}

	if s := Lookup("rune"); s.Def == nil {
		s.Def = typenod(runetype)
		s.Def.Name = new(Name)
		s.Origpkg = builtinpkg
	}

	if s := Lookup("nil"); s.Def == nil {
		var v Val
		v.U = new(NilVal)
		s.Def = nodlit(v)
		s.Def.Sym = s
		s.Def.Name = new(Name)
		s.Origpkg = builtinpkg
	}

	if s := Lookup("iota"); s.Def == nil {
		s.Def = Nod(OIOTA, nil, nil)
		s.Def.Sym = s
		s.Origpkg = builtinpkg
	}

	if s := Lookup("true"); s.Def == nil {
		s.Def = Nodbool(true)
		s.Def.Sym = s
		s.Def.Name = new(Name)
		s.Origpkg = builtinpkg
	}

	if s := Lookup("false"); s.Def == nil {
		s.Def = Nodbool(false)
		s.Def.Sym = s
		s.Def.Name = new(Name)
		s.Origpkg = builtinpkg
	}

	nodfp = Nod(ONAME, nil, nil)
	nodfp.Type = Types[TINT32]
	nodfp.Xoffset = 0
	nodfp.Class = PPARAM
	nodfp.Sym = Lookup(".fp")
}

var lexn = map[int]string{
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
	LDEC:       "DEC",
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
	LINC:       "INC",
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

func lexname(lex int) string {
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
