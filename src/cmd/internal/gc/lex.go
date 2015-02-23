// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

var yychar_lex int

var yyprev int

var yylast int

var imported_unsafe int

var goos string

var goarch string

var goroot string

// Debug arguments.
// These can be specified with the -d flag, as in "-d nil"
// to set the debug_checknil variable. In general the list passed
// to -d can be comma-separated.
var debugtab = []struct {
	name string
	val  *int
}{struct {
	name string
	val  *int
}{"nil", &Debug_checknil}}

// Our own isdigit, isspace, isalpha, isalnum that take care
// of EOF and other out of range arguments.
func yy_isdigit(c int) bool {
	return c >= 0 && c <= 0xFF && isdigit(c)
}

func yy_isspace(c int) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func yy_isalpha(c int) bool {
	return 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z'
}

func yy_isalnum(c int) bool {
	return c >= 0 && c <= 0xFF && isalnum(c)
}

// Disallow use of isdigit etc.

const (
	EOF = -1
)

func usage() {
	fmt.Printf("usage: %cg [options] file.go...\n", Thearch.Thechar)
	obj.Flagprint(1)
	Exit(2)
}

func hidePanic() {
	if nsavederrors+nerrors > 0 {
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
	fmt.Printf("%cg version %s%s%s\n", Thearch.Thechar, obj.Getgoversion(), sep, p)
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

	localpkg = mkpkg(newstrlit(""))
	localpkg.Prefix = "\"\""

	// pseudo-package, for scoping
	builtinpkg = mkpkg(newstrlit("go.builtin"))

	builtinpkg.Prefix = "go.builtin" // not go%2ebuiltin

	// pseudo-package, accessed by import "unsafe"
	unsafepkg = mkpkg(newstrlit("unsafe"))

	unsafepkg.Name = "unsafe"

	// real package, referred to by generated runtime calls
	Runtimepkg = mkpkg(newstrlit("runtime"))

	Runtimepkg.Name = "runtime"

	// pseudo-packages used in symbol tables
	gostringpkg = mkpkg(newstrlit("go.string"))

	gostringpkg.Name = "go.string"
	gostringpkg.Prefix = "go.string" // not go%2estring

	itabpkg = mkpkg(newstrlit("go.itab"))

	itabpkg.Name = "go.itab"
	itabpkg.Prefix = "go.itab" // not go%2eitab

	weaktypepkg = mkpkg(newstrlit("go.weak.type"))

	weaktypepkg.Name = "go.weak.type"
	weaktypepkg.Prefix = "go.weak.type" // not go%2eweak%2etype

	typelinkpkg = mkpkg(newstrlit("go.typelink"))
	typelinkpkg.Name = "go.typelink"
	typelinkpkg.Prefix = "go.typelink" // not go%2etypelink

	trackpkg = mkpkg(newstrlit("go.track"))

	trackpkg.Name = "go.track"
	trackpkg.Prefix = "go.track" // not go%2etrack

	typepkg = mkpkg(newstrlit("type"))

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
	obj.Flagstr("D", "path: set relative path for local imports", &localimport)
	obj.Flagcount("E", "debug symbol export", &Debug['E'])
	obj.Flagfn1("I", "dir: add dir to import search path", addidir)
	obj.Flagcount("K", "debug missing line numbers", &Debug['K'])
	obj.Flagcount("L", "use full (long) path in error messages", &Debug['L'])
	obj.Flagcount("M", "debug move generation", &Debug['M'])
	obj.Flagcount("N", "disable optimizations", &Debug['N'])
	obj.Flagcount("P", "debug peephole optimizer", &Debug['P'])
	obj.Flagcount("R", "debug register optimizer", &Debug['R'])
	obj.Flagcount("S", "print assembly listing", &Debug['S'])
	obj.Flagfn0("V", "print compiler version", doversion)
	obj.Flagcount("W", "debug parse tree after type checking", &Debug['W'])
	obj.Flagstr("asmhdr", "file: write assembly header to named file", &asmhdr)
	obj.Flagcount("complete", "compiling complete package (no C or assembly)", &pure_go)
	obj.Flagstr("d", "list: print debug information about items in list", &debugstr)
	obj.Flagcount("e", "no limit on number of errors reported", &Debug['e'])
	obj.Flagcount("f", "debug stack frames", &Debug['f'])
	obj.Flagcount("g", "debug code generation", &Debug['g'])
	obj.Flagcount("h", "halt on error", &Debug['h'])
	obj.Flagcount("i", "debug line number stack", &Debug['i'])
	obj.Flagstr("installsuffix", "pkg directory suffix", &flag_installsuffix)
	obj.Flagcount("j", "debug runtime-initialized variables", &Debug['j'])
	obj.Flagcount("l", "disable inlining", &Debug['l'])
	obj.Flagcount("live", "debug liveness analysis", &debuglive)
	obj.Flagcount("m", "print optimization decisions", &Debug['m'])
	obj.Flagcount("nolocalimports", "reject local (relative) imports", &nolocalimports)
	obj.Flagstr("o", "obj: set output file", &outfile)
	obj.Flagstr("p", "path: set expected package import path", &myimportpath)
	obj.Flagcount("pack", "write package file instead of object file", &writearchive)
	obj.Flagcount("r", "debug generated wrappers", &Debug['r'])
	obj.Flagcount("race", "enable race detector", &flag_race)
	obj.Flagcount("s", "warn about composite literals that can be simplified", &Debug['s'])
	obj.Flagstr("trimpath", "prefix: remove prefix from recorded source file paths", &Ctxt.Trimpath)
	obj.Flagcount("u", "reject unsafe code", &safemode)
	obj.Flagcount("v", "increase debug verbosity", &Debug['v'])
	obj.Flagcount("w", "debug type checking", &Debug['w'])
	use_writebarrier = 1
	obj.Flagcount("wb", "enable write barrier", &use_writebarrier)
	obj.Flagcount("x", "debug lexer", &Debug['x'])
	obj.Flagcount("y", "debug declarations in canned imports (with -d)", &Debug['y'])
	if Thearch.Thechar == '6' {
		obj.Flagcount("largemodel", "generate code that assumes a large memory model", &flag_largemodel)
	}

	obj.Flagstr("cpuprofile", "file: write cpu profile to file", &cpuprofile)
	obj.Flagstr("memprofile", "file: write memory profile to file", &memprofile)
	obj.Flagparse(usage)
	Ctxt.Debugasm = int32(Debug['S'])
	Ctxt.Debugvlog = int32(Debug['v'])

	if flag.NArg() < 1 {
		usage()
	}

	startProfile()

	if flag_race != 0 {
		racepkg = mkpkg(newstrlit("runtime/race"))
		racepkg.Name = "race"
	}

	// parse -d argument
	if debugstr != "" {
		var j int
		f := strings.Split(debugstr, ",")
		for i := range f {
			if f[i] == "" {
				continue
			}
			for j = 0; j < len(debugtab); j++ {
				if debugtab[j].name == f[i] {
					if debugtab[j].val != nil {
						*debugtab[j].val = 1
					}
					break
				}
			}

			if j >= len(debugtab) {
				log.Fatalf("unknown debug information -d '%s'\n", f[i])
			}
		}
	}

	// enable inlining.  for now:
	//	default: inlining on.  (debug['l'] == 1)
	//	-l: inlining off  (debug['l'] == 0)
	//	-ll, -lll: inlining on again, with extra debugging (debug['l'] > 1)
	if Debug['l'] <= 1 {
		Debug['l'] = 1 - Debug['l']
	}

	if Thearch.Thechar == '8' {
		p := obj.Getgo386()
		if p == "387" {
			Use_sse = 0
		} else if p == "sse2" {
			Use_sse = 1
		} else {
			log.Fatalf("unsupported setting GO386=%s", p)
		}
	}

	Thearch.Betypeinit()
	if Widthptr == 0 {
		Fatal("betypeinit failed")
	}

	lexinit()
	typeinit()
	lexinit1()
	// TODO(rsc): Restore yytinit?

	blockgen = 1
	dclcontext = PEXTERN
	nerrors = 0
	lexlineno = 1

	for _, infile = range flag.Args() {
		linehist(infile, 0, 0)

		curio.infile = infile
		var err error
		curio.bin, err = obj.Bopenr(infile)
		if err != nil {
			fmt.Printf("open %s: %v\n", infile, err)
			errorexit()
		}

		curio.peekc = 0
		curio.peekc1 = 0
		curio.nlsemi = 0
		curio.eofnl = 0
		curio.last = 0

		// Skip initial BOM if present.
		if obj.Bgetrune(curio.bin) != obj.BOM {
			obj.Bungetrune(curio.bin)
		}

		block = 1
		iota_ = -1000000

		imported_unsafe = 0

		yyparse()
		if nsyntaxerrors != 0 {
			errorexit()
		}

		linehist("<pop>", 0, 0)
		if curio.bin != nil {
			obj.Bterm(curio.bin)
		}
	}

	testdclstack()
	mkpackage(localpkg.Name) // final import not used checks
	lexfini()

	typecheckok = 1
	if Debug['f'] != 0 {
		frame(1)
	}

	// Process top-level declarations in phases.

	// Phase 1: const, type, and names and types of funcs.
	//   This will gather all the information about types
	//   and methods but doesn't depend on any of it.
	defercheckwidth()

	for l := xtop; l != nil; l = l.Next {
		if l.N.Op != ODCL && l.N.Op != OAS {
			typecheck(&l.N, Etop)
		}
	}

	// Phase 2: Variable assignments.
	//   To check interface assignments, depends on phase 1.
	for l := xtop; l != nil; l = l.Next {
		if l.N.Op == ODCL || l.N.Op == OAS {
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
		if l.N.Op == ODCLFUNC && l.N.Closure != nil {
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
		for l := importlist; l != nil; l = l.Next {
			if l.N.Inl != nil {
				saveerrors()
				typecheckinl(l.N)
			}
		}

		if nsavederrors+nerrors != 0 {
			errorexit()
		}
	}

	if Debug['l'] != 0 {
		// Find functions that can be inlined and clone them before walk expands them.
		for l := xtop; l != nil; l = l.Next {
			if l.N.Op == ODCLFUNC {
				caninl(l.N)
			}
		}

		// Expand inlineable calls in all functions
		for l := xtop; l != nil; l = l.Next {
			if l.N.Op == ODCLFUNC {
				inlcalls(l.N)
			}
		}
	}

	// Phase 6: Escape analysis.
	// Required for moving heap allocations onto stack,
	// which in turn is required by the closure implementation,
	// which stores the addresses of stack variables into the closure.
	// If the closure does not escape, it needs to be on the stack
	// or else the stack copier will not update it.
	escapes(xtop)

	// Escape analysis moved escaped values off stack.
	// Move large values off stack too.
	movelarge(xtop)

	// Phase 7: Transform closure bodies to properly reference captured variables.
	// This needs to happen before walk, because closures must be transformed
	// before walk reaches a call of a closure.
	for l := xtop; l != nil; l = l.Next {
		if l.N.Op == ODCLFUNC && l.N.Closure != nil {
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

	// Phase 9: Check external declarations.
	for l := externdcl; l != nil; l = l.Next {
		if l.N.Op == ONAME {
			typecheck(&l.N, Erv)
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
	/* archive header */
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

	/* symbol table may be first; skip it */
	sz := arsize(b, "__.GOSYMDEF")

	if sz >= 0 {
		obj.Bseek(b, int64(sz), 1)
	} else {
		obj.Bseek(b, 8, 0)
	}

	/* package export block is next */
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
func islocalname(name *Strlit) bool {
	return strings.HasPrefix(name.S, "/") ||
		Ctxt.Windows != 0 && len(name.S) >= 3 && yy_isalpha(int(name.S[0])) && name.S[1] == ':' && name.S[2] == '/' ||
		strings.HasPrefix(name.S, "./") || name.S == "." ||
		strings.HasPrefix(name.S, "../") || name.S == ".."
}

func findpkg(name *Strlit) bool {
	if islocalname(name) {
		if safemode != 0 || nolocalimports != 0 {
			return false
		}

		// try .a before .6.  important for building libraries:
		// if there is an array.6 in the array.a library,
		// want to find all of array.a, not just array.6.
		namebuf = fmt.Sprintf("%v.a", Zconv(name, 0))

		if obj.Access(namebuf, 0) >= 0 {
			return true
		}
		namebuf = fmt.Sprintf("%v.%c", Zconv(name, 0), Thearch.Thechar)
		if obj.Access(namebuf, 0) >= 0 {
			return true
		}
		return false
	}

	// local imports should be canonicalized already.
	// don't want to see "encoding/../encoding/base64"
	// as different from "encoding/base64".
	var q string
	_ = q
	if path.Clean(name.S) != name.S {
		Yyerror("non-canonical import path %v (should be %s)", Zconv(name, 0), q)
		return false
	}

	for p := idirs; p != nil; p = p.link {
		namebuf = fmt.Sprintf("%s/%v.a", p.dir, Zconv(name, 0))
		if obj.Access(namebuf, 0) >= 0 {
			return true
		}
		namebuf = fmt.Sprintf("%s/%v.%c", p.dir, Zconv(name, 0), Thearch.Thechar)
		if obj.Access(namebuf, 0) >= 0 {
			return true
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
		}

		namebuf = fmt.Sprintf("%s/pkg/%s_%s%s%s/%v.a", goroot, goos, goarch, suffixsep, suffix, Zconv(name, 0))
		if obj.Access(namebuf, 0) >= 0 {
			return true
		}
		namebuf = fmt.Sprintf("%s/pkg/%s_%s%s%s/%v.%c", goroot, goos, goarch, suffixsep, suffix, Zconv(name, 0), Thearch.Thechar)
		if obj.Access(namebuf, 0) >= 0 {
			return true
		}
	}

	return false
}

func fakeimport() {
	importpkg = mkpkg(newstrlit("fake"))
	cannedimports("fake.6", "$$\n")
}

func importfile(f *Val, line int) {
	if f.Ctype != CTSTR {
		Yyerror("import statement not a string")
		fakeimport()
		return
	}

	if len(f.U.Sval.S) == 0 {
		Yyerror("import path is empty")
		fakeimport()
		return
	}

	if isbadimport(f.U.Sval) {
		fakeimport()
		return
	}

	// The package name main is no longer reserved,
	// but we reserve the import path "main" to identify
	// the main package, just as we reserve the import
	// path "math" to identify the standard math package.
	if f.U.Sval.S == "main" {
		Yyerror("cannot import \"main\"")
		errorexit()
	}

	if myimportpath != "" && f.U.Sval.S == myimportpath {
		Yyerror("import \"%v\" while compiling that package (import cycle)", Zconv(f.U.Sval, 0))
		errorexit()
	}

	if f.U.Sval.S == "unsafe" {
		if safemode != 0 {
			Yyerror("cannot import package unsafe")
			errorexit()
		}

		importpkg = mkpkg(f.U.Sval)
		cannedimports("unsafe.6", unsafeimport)
		imported_unsafe = 1
		return
	}

	path_ := f.U.Sval
	if islocalname(path_) {
		if path_.S[0] == '/' {
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
		cleanbuf += path_.S
		cleanbuf = path.Clean(cleanbuf)
		path_ = newstrlit(cleanbuf)

		if isbadimport(path_) {
			fakeimport()
			return
		}
	}

	if !findpkg(path_) {
		Yyerror("can't find import: \"%v\"", Zconv(f.U.Sval, 0))
		errorexit()
	}

	importpkg = mkpkg(path_)

	// If we already saw that package, feed a dummy statement
	// to the lexer to avoid parsing export data twice.
	if importpkg.Imported != 0 {
		file := namebuf
		tag := ""
		if importpkg.Safe {
			tag = "safe"
		}

		p := fmt.Sprintf("package %s %s\n$$\n", importpkg.Name, tag)
		cannedimports(file, p)
		return
	}

	importpkg.Imported = 1

	var err error
	var imp *obj.Biobuf
	imp, err = obj.Bopenr(namebuf)
	if err != nil {
		Yyerror("can't open import: \"%v\": %v", Zconv(f.U.Sval, 0), err)
		errorexit()
	}

	file := namebuf

	n := len(namebuf)
	if n > 2 && namebuf[n-2] == '.' && namebuf[n-1] == 'a' {
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
	linehist(file[n-len(path_.S)-2:], -1, 1) // acts as #pragma lib

	/*
	 * position the input right
	 * after $$ and return
	 */
	pushedio = curio

	curio.bin = imp
	curio.peekc = 0
	curio.peekc1 = 0
	curio.infile = file
	curio.nlsemi = 0
	typecheckok = 1

	var c int32
	for {
		c = int32(getc())
		if c == EOF {
			break
		}
		if c != '$' {
			continue
		}
		c = int32(getc())
		if c == EOF {
			break
		}
		if c != '$' {
			continue
		}
		return
	}

	Yyerror("no import in \"%v\"", Zconv(f.U.Sval, 0))
	unimportfile()
}

func unimportfile() {
	if curio.bin != nil {
		obj.Bterm(curio.bin)
		curio.bin = nil
	} else {
		lexlineno-- // re correct sys.6 line number
	}

	curio = pushedio

	pushedio.bin = nil
	incannedimport = 0
	typecheckok = 0
}

func cannedimports(file string, cp string) {
	lexlineno++ // if sys.6 is included on line 1,

	pushedio = curio

	curio.bin = nil
	curio.peekc = 0
	curio.peekc1 = 0
	curio.infile = file
	curio.cp = cp
	curio.nlsemi = 0
	curio.importsafe = false

	typecheckok = 1
	incannedimport = 1
}

func isfrog(c int) bool {
	// complain about possibly invisible control characters
	if c < ' ' {
		return !yy_isspace(c) // exclude good white space
	}

	if 0x7f <= c && c <= 0xa0 { // DEL, unicode block including unbreakable space.
		return true
	}
	return false
}

type Loophack struct {
	v    int
	next *Loophack
}

var _yylex_lstk *Loophack

func _yylex(yylval *yySymType) int32 {
	var c int
	var c1 int
	var escflag int
	var v int64
	var cp *bytes.Buffer
	var rune_ uint
	var s *Sym
	var h *Loophack

	prevlineno = lineno

l0:
	c = getc()
	if yy_isspace(c) {
		if c == '\n' && curio.nlsemi != 0 {
			ungetc(c)
			DBG("lex: implicit semi\n")
			return ';'
		}

		goto l0
	}

	lineno = lexlineno /* start of token */

	if c >= utf8.RuneSelf {
		/* all multibyte runes are alpha */
		cp = &lexbuf
		cp.Reset()

		goto talph
	}

	if yy_isalpha(c) {
		cp = &lexbuf
		cp.Reset()
		goto talph
	}

	if yy_isdigit(c) {
		goto tnum
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
		if yy_isdigit(c1) {
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

		/* "..." */
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
				rune_ = uint(v)
				cp.WriteRune(rune(rune_))
			}
		}

		goto strlit

		/* `...` */
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

		/* '.' */
	case '\'':
		if escchar('\'', &escflag, &v) {
			Yyerror("empty character literal or unescaped ' in character literal")
			v = '\''
		}

		if !escchar('\'', &escflag, &v) {
			Yyerror("missing '")
			ungetc(int(v))
		}

		yylval.val.U.Xval = new(Mpint)
		Mpmovecfix(yylval.val.U.Xval, v)
		yylval.val.Ctype = CTRUNE
		DBG("lex: codepoint literal\n")
		litbuf = "string literal"
		return LLITERAL

	case '/':
		c1 = getc()
		if c1 == '*' {
			nl := 0
			for {
				c = int(getr())
				if c == '\n' {
					nl = 1
				}
				for c == '*' {
					c = int(getr())
					if c == '/' {
						if nl != 0 {
							ungetc('\n')
						}
						goto l0
					}

					if c == '\n' {
						nl = 1
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
			c = ODIV
			goto asop
		}

	case ':':
		c1 = getc()
		if c1 == '=' {
			c = LCOLAS
			yylval.i = int(lexlineno)
			goto lx
		}

	case '*':
		c1 = getc()
		if c1 == '=' {
			c = OMUL
			goto asop
		}

	case '%':
		c1 = getc()
		if c1 == '=' {
			c = OMOD
			goto asop
		}

	case '+':
		c1 = getc()
		if c1 == '+' {
			c = LINC
			goto lx
		}

		if c1 == '=' {
			c = OADD
			goto asop
		}

	case '-':
		c1 = getc()
		if c1 == '-' {
			c = LDEC
			goto lx
		}

		if c1 == '=' {
			c = OSUB
			goto asop
		}

	case '>':
		c1 = getc()
		if c1 == '>' {
			c = LRSH
			c1 = getc()
			if c1 == '=' {
				c = ORSH
				goto asop
			}

			break
		}

		if c1 == '=' {
			c = LGE
			goto lx
		}

		c = LGT

	case '<':
		c1 = getc()
		if c1 == '<' {
			c = LLSH
			c1 = getc()
			if c1 == '=' {
				c = OLSH
				goto asop
			}

			break
		}

		if c1 == '=' {
			c = LLE
			goto lx
		}

		if c1 == '-' {
			c = LCOMM
			goto lx
		}

		c = LLT

	case '=':
		c1 = getc()
		if c1 == '=' {
			c = LEQ
			goto lx
		}

	case '!':
		c1 = getc()
		if c1 == '=' {
			c = LNE
			goto lx
		}

	case '&':
		c1 = getc()
		if c1 == '&' {
			c = LANDAND
			goto lx
		}

		if c1 == '^' {
			c = LANDNOT
			c1 = getc()
			if c1 == '=' {
				c = OANDNOT
				goto asop
			}

			break
		}

		if c1 == '=' {
			c = OAND
			goto asop
		}

	case '|':
		c1 = getc()
		if c1 == '|' {
			c = LOROR
			goto lx
		}

		if c1 == '=' {
			c = OOR
			goto asop
		}

	case '^':
		c1 = getc()
		if c1 == '=' {
			c = OXOR
			goto asop
		}

		/*
		 * clumsy dance:
		 * to implement rule that disallows
		 *	if T{1}[0] { ... }
		 * but allows
		 * 	if (T{1}[0]) { ... }
		 * the block bodies for if/for/switch/select
		 * begin with an LBODY token, not '{'.
		 *
		 * when we see the keyword, the next
		 * non-parenthesized '{' becomes an LBODY.
		 * loophack is normally 0.
		 * a keyword makes it go up to 1.
		 * parens push loophack onto a stack and go back to 0.
		 * a '{' with loophack == 1 becomes LBODY and disables loophack.
		 *
		 * i said it was clumsy.
		 */
	case '(',
		'[':
		if loophack != 0 || _yylex_lstk != nil {
			h = new(Loophack)
			if h == nil {
				Flusherrors()
				Yyerror("out of memory")
				errorexit()
			}

			h.v = loophack
			h.next = _yylex_lstk
			_yylex_lstk = h
			loophack = 0
		}

		goto lx

	case ')',
		']':
		if _yylex_lstk != nil {
			h = _yylex_lstk
			loophack = h.v
			_yylex_lstk = h.next
		}

		goto lx

	case '{':
		if loophack == 1 {
			DBG("%L lex: LBODY\n", lexlineno)
			loophack = 0
			return LBODY
		}

		goto lx

	default:
		goto lx
	}

	ungetc(c1)

lx:
	if c > 0xff {
		DBG("%L lex: TOKEN %s\n", lexlineno, lexname(c))
	} else {
		DBG("%L lex: TOKEN '%c'\n", lexlineno, c)
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
	yylval.i = c // rathole to hold which asop
	DBG("lex: TOKEN ASOP %c\n", c)
	return LASOP

	/*
	 * cp is set to lexbuf and some
	 * prefix has been stored
	 */
talph:
	for {
		if c >= utf8.RuneSelf {
			ungetc(c)
			rune_ = uint(getr())

			// 0xb7 · is used for internal names
			if !unicode.IsLetter(rune(rune_)) && !unicode.IsDigit(rune(rune_)) && (importpkg == nil || rune_ != 0xb7) {
				Yyerror("invalid identifier character U+%04x", rune_)
			}
			cp.WriteRune(rune(rune_))
		} else if !yy_isalnum(c) && c != '_' {
			break
		} else {
			cp.WriteByte(byte(c))
		}
		c = getc()
	}

	cp = nil
	ungetc(c)

	s = Lookup(lexbuf.String())
	switch s.Lexical {
	case LIGNORE:
		goto l0

	case LFOR,
		LIF,
		LSWITCH,
		LSELECT:
		loophack = 1 // see comment about loophack above
	}

	DBG("lex: %S %s\n", s, lexname(int(s.Lexical)))
	yylval.sym = s
	return int32(s.Lexical)

tnum:
	cp = &lexbuf
	cp.Reset()
	if c != '0' {
		for {
			cp.WriteByte(byte(c))
			c = getc()
			if yy_isdigit(c) {
				continue
			}
			goto dc
		}
	}

	cp.WriteByte(byte(c))
	c = getc()
	if c == 'x' || c == 'X' {
		for {
			cp.WriteByte(byte(c))
			c = getc()
			if yy_isdigit(c) {
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
		if !yy_isdigit(c) {
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

dc:
	if c == '.' {
		goto casedot
	}
	if c == 'e' || c == 'E' || c == 'p' || c == 'P' {
		goto caseep
	}
	if c == 'i' {
		goto casei
	}

ncu:
	cp = nil
	ungetc(c)

	yylval.val.U.Xval = new(Mpint)
	mpatofix(yylval.val.U.Xval, lexbuf.String())
	if yylval.val.U.Xval.Ovf != 0 {
		Yyerror("overflow in constant")
		Mpmovecfix(yylval.val.U.Xval, 0)
	}

	yylval.val.Ctype = CTINT
	DBG("lex: integer literal\n")
	litbuf = "literal "
	litbuf += lexbuf.String()
	return LLITERAL

casedot:
	for {
		cp.WriteByte(byte(c))
		c = getc()
		if !yy_isdigit(c) {
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
	cp.WriteByte(byte(c))
	c = getc()
	if c == '+' || c == '-' {
		cp.WriteByte(byte(c))
		c = getc()
	}

	if !yy_isdigit(c) {
		Yyerror("malformed fp constant exponent")
	}
	for yy_isdigit(c) {
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

	yylval.val.U.Cval = new(Mpcplx)
	Mpmovecflt(&yylval.val.U.Cval.Real, 0.0)
	mpatoflt(&yylval.val.U.Cval.Imag, lexbuf.String())
	if yylval.val.U.Cval.Imag.Val.Ovf != 0 {
		Yyerror("overflow in imaginary constant")
		Mpmovecflt(&yylval.val.U.Cval.Real, 0.0)
	}

	yylval.val.Ctype = CTCPLX
	DBG("lex: imaginary literal\n")
	litbuf = "literal "
	litbuf += lexbuf.String()
	return LLITERAL

caseout:
	cp = nil
	ungetc(c)

	yylval.val.U.Fval = new(Mpflt)
	mpatoflt(yylval.val.U.Fval, lexbuf.String())
	if yylval.val.U.Fval.Val.Ovf != 0 {
		Yyerror("overflow in float constant")
		Mpmovecflt(yylval.val.U.Fval, 0.0)
	}

	yylval.val.Ctype = CTFLT
	DBG("lex: floating literal\n")
	litbuf = "literal "
	litbuf += lexbuf.String()
	return LLITERAL

strlit:
	yylval.val.U.Sval = &Strlit{S: cp.String()}
	yylval.val.Ctype = CTSTR
	DBG("lex: string literal\n")
	litbuf = "string literal"
	return LLITERAL
}

func more(pp *string) bool {
	p := *pp
	for p != "" && yy_isspace(int(p[0])) {
		p = p[1:]
	}
	*pp = p
	return p != ""
}

/*
 * read and interpret syntax that looks like
 * //line parse.y:15
 * as a discontinuity in sequential line numbers.
 * the next line of input comes from parse.y:15
 */
func getlinepragma() int {
	var cmd, verb, name string
	var n int
	var cp *bytes.Buffer
	var linep int

	c := int(getr())
	if c == 'g' {
		goto go_
	}
	if c != 'l' {
		goto out
	}
	for i := 1; i < 5; i++ {
		c = int(getr())
		if c != int("line "[i]) {
			goto out
		}
	}

	cp = &lexbuf
	cp.Reset()
	linep = 0
	for {
		c = int(getr())
		if c == EOF {
			goto out
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
		goto out
	}
	n = 0
	for _, c := range lexbuf.String()[linep:] {
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
		goto out
	}

	// try to avoid allocating file name over and over
	name = lexbuf.String()[:linep-1]
	for h := Ctxt.Hist; h != nil; h = h.Link {
		if h.Name != "" && h.Name == name {
			linehist(h.Name, int32(n), 0)
			goto out
		}
	}

	linehist(name, int32(n), 0)
	goto out

go_:
	cp = &lexbuf
	cp.Reset()
	cp.WriteByte('g') // already read
	for {
		c = int(getr())
		if c == EOF || c >= utf8.RuneSelf {
			goto out
		}
		if c == '\n' {
			break
		}
		cp.WriteByte(byte(c))
	}

	cp = nil

	if strings.HasPrefix(lexbuf.String(), "go:cgo_") {
		pragcgo(lexbuf.String())
	}

	cmd = lexbuf.String()
	verb = cmd
	if i := strings.Index(verb, " "); i >= 0 {
		verb = verb[:i]
	}

	if verb == "go:linkname" {
		if imported_unsafe == 0 {
			Yyerror("//go:linkname only allowed in Go files that import \"unsafe\"")
		}
		f := strings.Fields(cmd)
		if len(f) != 3 {
			Yyerror("usage: //go:linkname localname linkname")
			goto out
		}

		Lookup(f[1]).Linkname = f[2]
		goto out
	}

	if verb == "go:nointerface" && obj.Fieldtrack_enabled != 0 {
		nointerface = true
		goto out
	}

	if verb == "go:noescape" {
		noescape = true
		goto out
	}

	if verb == "go:nosplit" {
		nosplit = true
		goto out
	}

	if verb == "go:nowritebarrier" {
		if compiling_runtime == 0 {
			Yyerror("//go:nowritebarrier only allowed in runtime")
		}
		nowritebarrier = true
		goto out
	}

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
	for i < len(p) && !yy_isspace(int(p[i])) && p[i] != '"' {
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
		var ok bool
		var p string
		p, ok = getquoted(&q)
		if !ok {
			goto err1
		}
		pragcgobuf += fmt.Sprintf("cgo_dynamic_linker %v\n", plan9quote(p))
		goto out

	err1:
		Yyerror("usage: //go:cgo_dynamic_linker \"path\"")
		goto out
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
			goto out
		}

		remote = getimpsym(&q)
		if remote == "" {
			goto err2
		}
		pragcgobuf += fmt.Sprintf("%s %v %v\n", verb, plan9quote(local), plan9quote(remote))
		goto out

	err2:
		Yyerror("usage: //go:%s local [remote]", verb)
		goto out
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
			goto out
		}

		remote = getimpsym(&q)
		if remote == "" {
			goto err3
		}
		if !more(&q) {
			pragcgobuf += fmt.Sprintf("cgo_import_dynamic %v %v\n", plan9quote(local), plan9quote(remote))
			goto out
		}

		p, ok = getquoted(&q)
		if !ok {
			goto err3
		}
		pragcgobuf += fmt.Sprintf("cgo_import_dynamic %v %v %v\n", plan9quote(local), plan9quote(remote), plan9quote(p))
		goto out

	err3:
		Yyerror("usage: //go:cgo_import_dynamic local [remote [\"library\"]]")
		goto out
	}

	if verb == "cgo_import_static" {
		local := getimpsym(&q)
		if local == "" || more(&q) {
			goto err4
		}
		pragcgobuf += fmt.Sprintf("cgo_import_static %v\n", plan9quote(local))
		goto out

	err4:
		Yyerror("usage: //go:cgo_import_static local")
		goto out
	}

	if verb == "cgo_ldflag" {
		var ok bool
		var p string
		p, ok = getquoted(&q)
		if !ok {
			goto err5
		}
		pragcgobuf += fmt.Sprintf("cgo_ldflag %v\n", plan9quote(p))
		goto out

	err5:
		Yyerror("usage: //go:cgo_ldflag \"arg\"")
		goto out
	}

out:
}

type yy struct{}

var yymsg []struct {
	yystate, yychar int
	msg             string
}

func (yy) Lex(v *yySymType) int {
	return int(yylex(v))
}

func (yy) Error(msg string) {
	Yyerror("%s", msg)
}

var theparser yyParser
var parsing bool

func yyparse() {
	theparser = yyNewParser()
	parsing = true
	theparser.Parse(yy{})
	parsing = false
}

func yylex(yylval *yySymType) int32 {
	lx := int(_yylex(yylval))

	if curio.nlsemi != 0 && lx == EOF {
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
		curio.nlsemi = 1

	default:
		curio.nlsemi = 0
	}

	// Track last two tokens returned by yylex.
	yyprev = yylast

	yylast = lx
	return int32(lx)
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
		var c1 int
		var c2 int
	loop:
		c = obj.Bgetc(curio.bin)
		if c == 0xef {
			c1 = obj.Bgetc(curio.bin)
			c2 = obj.Bgetc(curio.bin)
			if c1 == 0xbb && c2 == 0xbf {
				yyerrorl(int(lexlineno), "Unicode (UTF-8) BOM in middle of file")
				goto loop
			}

			obj.Bungetc(curio.bin)
			obj.Bungetc(curio.bin)
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
		if curio.eofnl != 0 || curio.last == '\n' {
			return EOF
		}
		curio.eofnl = 1
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
				Yyerror("illegal UTF-8 sequence % x", buf[:i+1])
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
	var l int64
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
		goto oct

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
	l = 0
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

oct:
	l = int64(c) - '0'
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
}

var syms = []struct {
	name    string
	lexical int
	etype   int
	op      int
}{
	/*	name		lexical		etype		op
	 */
	/* basic types */
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"int8", LNAME, TINT8, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"int16", LNAME, TINT16, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"int32", LNAME, TINT32, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"int64", LNAME, TINT64, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"uint8", LNAME, TUINT8, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"uint16", LNAME, TUINT16, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"uint32", LNAME, TUINT32, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"uint64", LNAME, TUINT64, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"float32", LNAME, TFLOAT32, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"float64", LNAME, TFLOAT64, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"complex64", LNAME, TCOMPLEX64, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"complex128", LNAME, TCOMPLEX128, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"bool", LNAME, TBOOL, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"string", LNAME, TSTRING, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"any", LNAME, TANY, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"break", LBREAK, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"case", LCASE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"chan", LCHAN, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"const", LCONST, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"continue", LCONTINUE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"default", LDEFAULT, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"else", LELSE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"defer", LDEFER, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"fallthrough", LFALL, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"for", LFOR, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"func", LFUNC, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"go", LGO, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"goto", LGOTO, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"if", LIF, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"import", LIMPORT, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"interface", LINTERFACE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"map", LMAP, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"package", LPACKAGE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"range", LRANGE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"return", LRETURN, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"select", LSELECT, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"struct", LSTRUCT, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"switch", LSWITCH, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"type", LTYPE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"var", LVAR, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"append", LNAME, Txxx, OAPPEND},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"cap", LNAME, Txxx, OCAP},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"close", LNAME, Txxx, OCLOSE},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"complex", LNAME, Txxx, OCOMPLEX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"copy", LNAME, Txxx, OCOPY},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"delete", LNAME, Txxx, ODELETE},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"imag", LNAME, Txxx, OIMAG},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"len", LNAME, Txxx, OLEN},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"make", LNAME, Txxx, OMAKE},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"new", LNAME, Txxx, ONEW},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"panic", LNAME, Txxx, OPANIC},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"print", LNAME, Txxx, OPRINT},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"println", LNAME, Txxx, OPRINTN},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"real", LNAME, Txxx, OREAL},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"recover", LNAME, Txxx, ORECOVER},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"notwithstanding", LIGNORE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"thetruthofthematter", LIGNORE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"despiteallobjections", LIGNORE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"whereas", LIGNORE, Txxx, OXXX},
	struct {
		name    string
		lexical int
		etype   int
		op      int
	}{"insofaras", LIGNORE, Txxx, OXXX},
}

func lexinit() {
	var lex int
	var s *Sym
	var s1 *Sym
	var t *Type
	var etype int

	/*
	 * initialize basic types array
	 * initialize known symbols
	 */
	for i := 0; i < len(syms); i++ {
		lex = syms[i].lexical
		s = Lookup(syms[i].name)
		s.Lexical = uint16(lex)

		etype = syms[i].etype
		if etype != Txxx {
			if etype < 0 || etype >= len(Types) {
				Fatal("lexinit: %s bad etype", s.Name)
			}
			s1 = Pkglookup(syms[i].name, builtinpkg)
			t = Types[etype]
			if t == nil {
				t = typ(etype)
				t.Sym = s1

				if etype != TANY && etype != TSTRING {
					dowidth(t)
				}
				Types[etype] = t
			}

			s1.Lexical = LNAME
			s1.Def = typenod(t)
			continue
		}

		etype = syms[i].op
		if etype != OXXX {
			s1 = Pkglookup(syms[i].name, builtinpkg)
			s1.Lexical = LNAME
			s1.Def = Nod(ONAME, nil, nil)
			s1.Def.Sym = s1
			s1.Def.Etype = uint8(etype)
			s1.Def.Builtin = 1
		}
	}

	// logically, the type of a string literal.
	// types[TSTRING] is the named type string
	// (the type of x in var x string or var x = "hello").
	// this is the ideal form
	// (the type of x in const x = "hello").
	idealstring = typ(TSTRING)

	idealbool = typ(TBOOL)

	s = Pkglookup("true", builtinpkg)
	s.Def = Nodbool(true)
	s.Def.Sym = Lookup("true")
	s.Def.Type = idealbool

	s = Pkglookup("false", builtinpkg)
	s.Def = Nodbool(false)
	s.Def.Sym = Lookup("false")
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
	v.Ctype = CTNIL
	s.Def = nodlit(v)
	s.Def.Sym = s
}

func lexinit1() {
	// t = interface { Error() string }
	rcvr := typ(TSTRUCT)

	rcvr.Type = typ(TFIELD)
	rcvr.Type.Type = Ptrto(typ(TSTRUCT))
	rcvr.Funarg = 1
	in := typ(TSTRUCT)
	in.Funarg = 1
	out := typ(TSTRUCT)
	out.Type = typ(TFIELD)
	out.Type.Type = Types[TSTRING]
	out.Funarg = 1
	f := typ(TFUNC)
	*getthis(f) = rcvr
	*Getoutarg(f) = out
	*getinarg(f) = in
	f.Thistuple = 1
	f.Intuple = 0
	f.Outnamed = 0
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

	// rune alias
	s = Lookup("rune")

	s.Lexical = LNAME
	s1 = Pkglookup("rune", builtinpkg)
	runetype = typ(TINT32)
	runetype.Sym = s1
	s1.Lexical = LNAME
	s1.Def = typenod(runetype)
}

func lexfini() {
	var s *Sym
	var lex int
	var etype int
	var i int

	for i = 0; i < len(syms); i++ {
		lex = syms[i].lexical
		if lex != LNAME {
			continue
		}
		s = Lookup(syms[i].name)
		s.Lexical = uint16(lex)

		etype = syms[i].etype
		if etype != Txxx && (etype != TANY || Debug['A'] != 0) && s.Def == nil {
			s.Def = typenod(Types[etype])
			s.Origpkg = builtinpkg
		}

		etype = syms[i].op
		if etype != OXXX && s.Def == nil {
			s.Def = Nod(ONAME, nil, nil)
			s.Def.Sym = s
			s.Def.Etype = uint8(etype)
			s.Def.Builtin = 1
			s.Origpkg = builtinpkg
		}
	}

	// backend-specific builtin types (e.g. int).
	for i = range Thearch.Typedefs {
		s = Lookup(Thearch.Typedefs[i].Name)
		if s.Def == nil {
			s.Def = typenod(Types[Thearch.Typedefs[i].Etype])
			s.Origpkg = builtinpkg
		}
	}

	// there's only so much table-driven we can handle.
	// these are special cases.
	s = Lookup("byte")

	if s.Def == nil {
		s.Def = typenod(bytetype)
		s.Origpkg = builtinpkg
	}

	s = Lookup("error")
	if s.Def == nil {
		s.Def = typenod(errortype)
		s.Origpkg = builtinpkg
	}

	s = Lookup("rune")
	if s.Def == nil {
		s.Def = typenod(runetype)
		s.Origpkg = builtinpkg
	}

	s = Lookup("nil")
	if s.Def == nil {
		var v Val
		v.Ctype = CTNIL
		s.Def = nodlit(v)
		s.Def.Sym = s
		s.Origpkg = builtinpkg
	}

	s = Lookup("iota")
	if s.Def == nil {
		s.Def = Nod(OIOTA, nil, nil)
		s.Def.Sym = s
		s.Origpkg = builtinpkg
	}

	s = Lookup("true")
	if s.Def == nil {
		s.Def = Nodbool(true)
		s.Def.Sym = s
		s.Origpkg = builtinpkg
	}

	s = Lookup("false")
	if s.Def == nil {
		s.Def = Nodbool(false)
		s.Def.Sym = s
		s.Origpkg = builtinpkg
	}

	nodfp = Nod(ONAME, nil, nil)
	nodfp.Type = Types[TINT32]
	nodfp.Xoffset = 0
	nodfp.Class = PPARAM
	nodfp.Sym = Lookup(".fp")
}

var lexn = []struct {
	lex  int
	name string
}{
	struct {
		lex  int
		name string
	}{LANDAND, "ANDAND"},
	struct {
		lex  int
		name string
	}{LANDNOT, "ANDNOT"},
	struct {
		lex  int
		name string
	}{LASOP, "ASOP"},
	struct {
		lex  int
		name string
	}{LBREAK, "BREAK"},
	struct {
		lex  int
		name string
	}{LCASE, "CASE"},
	struct {
		lex  int
		name string
	}{LCHAN, "CHAN"},
	struct {
		lex  int
		name string
	}{LCOLAS, "COLAS"},
	struct {
		lex  int
		name string
	}{LCOMM, "<-"},
	struct {
		lex  int
		name string
	}{LCONST, "CONST"},
	struct {
		lex  int
		name string
	}{LCONTINUE, "CONTINUE"},
	struct {
		lex  int
		name string
	}{LDDD, "..."},
	struct {
		lex  int
		name string
	}{LDEC, "DEC"},
	struct {
		lex  int
		name string
	}{LDEFAULT, "DEFAULT"},
	struct {
		lex  int
		name string
	}{LDEFER, "DEFER"},
	struct {
		lex  int
		name string
	}{LELSE, "ELSE"},
	struct {
		lex  int
		name string
	}{LEQ, "EQ"},
	struct {
		lex  int
		name string
	}{LFALL, "FALL"},
	struct {
		lex  int
		name string
	}{LFOR, "FOR"},
	struct {
		lex  int
		name string
	}{LFUNC, "FUNC"},
	struct {
		lex  int
		name string
	}{LGE, "GE"},
	struct {
		lex  int
		name string
	}{LGO, "GO"},
	struct {
		lex  int
		name string
	}{LGOTO, "GOTO"},
	struct {
		lex  int
		name string
	}{LGT, "GT"},
	struct {
		lex  int
		name string
	}{LIF, "IF"},
	struct {
		lex  int
		name string
	}{LIMPORT, "IMPORT"},
	struct {
		lex  int
		name string
	}{LINC, "INC"},
	struct {
		lex  int
		name string
	}{LINTERFACE, "INTERFACE"},
	struct {
		lex  int
		name string
	}{LLE, "LE"},
	struct {
		lex  int
		name string
	}{LLITERAL, "LITERAL"},
	struct {
		lex  int
		name string
	}{LLSH, "LSH"},
	struct {
		lex  int
		name string
	}{LLT, "LT"},
	struct {
		lex  int
		name string
	}{LMAP, "MAP"},
	struct {
		lex  int
		name string
	}{LNAME, "NAME"},
	struct {
		lex  int
		name string
	}{LNE, "NE"},
	struct {
		lex  int
		name string
	}{LOROR, "OROR"},
	struct {
		lex  int
		name string
	}{LPACKAGE, "PACKAGE"},
	struct {
		lex  int
		name string
	}{LRANGE, "RANGE"},
	struct {
		lex  int
		name string
	}{LRETURN, "RETURN"},
	struct {
		lex  int
		name string
	}{LRSH, "RSH"},
	struct {
		lex  int
		name string
	}{LSELECT, "SELECT"},
	struct {
		lex  int
		name string
	}{LSTRUCT, "STRUCT"},
	struct {
		lex  int
		name string
	}{LSWITCH, "SWITCH"},
	struct {
		lex  int
		name string
	}{LTYPE, "TYPE"},
	struct {
		lex  int
		name string
	}{LVAR, "VAR"},
}

var lexname_buf string

func lexname(lex int) string {
	for i := 0; i < len(lexn); i++ {
		if lexn[i].lex == lex {
			return lexn[i].name
		}
	}
	lexname_buf = fmt.Sprintf("LEX-%d", lex)
	return lexname_buf
}

var yytfix = []struct {
	have string
	want string
}{
	struct {
		have string
		want string
	}{"$end", "EOF"},
	struct {
		have string
		want string
	}{"LLITERAL", "literal"},
	struct {
		have string
		want string
	}{"LASOP", "op="},
	struct {
		have string
		want string
	}{"LBREAK", "break"},
	struct {
		have string
		want string
	}{"LCASE", "case"},
	struct {
		have string
		want string
	}{"LCHAN", "chan"},
	struct {
		have string
		want string
	}{"LCOLAS", ":="},
	struct {
		have string
		want string
	}{"LCONST", "const"},
	struct {
		have string
		want string
	}{"LCONTINUE", "continue"},
	struct {
		have string
		want string
	}{"LDDD", "..."},
	struct {
		have string
		want string
	}{"LDEFAULT", "default"},
	struct {
		have string
		want string
	}{"LDEFER", "defer"},
	struct {
		have string
		want string
	}{"LELSE", "else"},
	struct {
		have string
		want string
	}{"LFALL", "fallthrough"},
	struct {
		have string
		want string
	}{"LFOR", "for"},
	struct {
		have string
		want string
	}{"LFUNC", "func"},
	struct {
		have string
		want string
	}{"LGO", "go"},
	struct {
		have string
		want string
	}{"LGOTO", "goto"},
	struct {
		have string
		want string
	}{"LIF", "if"},
	struct {
		have string
		want string
	}{"LIMPORT", "import"},
	struct {
		have string
		want string
	}{"LINTERFACE", "interface"},
	struct {
		have string
		want string
	}{"LMAP", "map"},
	struct {
		have string
		want string
	}{"LNAME", "name"},
	struct {
		have string
		want string
	}{"LPACKAGE", "package"},
	struct {
		have string
		want string
	}{"LRANGE", "range"},
	struct {
		have string
		want string
	}{"LRETURN", "return"},
	struct {
		have string
		want string
	}{"LSELECT", "select"},
	struct {
		have string
		want string
	}{"LSTRUCT", "struct"},
	struct {
		have string
		want string
	}{"LSWITCH", "switch"},
	struct {
		have string
		want string
	}{"LTYPE", "type"},
	struct {
		have string
		want string
	}{"LVAR", "var"},
	struct {
		have string
		want string
	}{"LANDAND", "&&"},
	struct {
		have string
		want string
	}{"LANDNOT", "&^"},
	struct {
		have string
		want string
	}{"LBODY", "{"},
	struct {
		have string
		want string
	}{"LCOMM", "<-"},
	struct {
		have string
		want string
	}{"LDEC", "--"},
	struct {
		have string
		want string
	}{"LINC", "++"},
	struct {
		have string
		want string
	}{"LEQ", "=="},
	struct {
		have string
		want string
	}{"LGE", ">="},
	struct {
		have string
		want string
	}{"LGT", ">"},
	struct {
		have string
		want string
	}{"LLE", "<="},
	struct {
		have string
		want string
	}{"LLT", "<"},
	struct {
		have string
		want string
	}{"LLSH", "<<"},
	struct {
		have string
		want string
	}{"LRSH", ">>"},
	struct {
		have string
		want string
	}{"LOROR", "||"},
	struct {
		have string
		want string
	}{"LNE", "!="},
	// spell out to avoid confusion with punctuation in error messages
	struct {
		have string
		want string
	}{"';'", "semicolon or newline"},
	struct {
		have string
		want string
	}{"','", "comma"},
}

func pkgnotused(lineno int, path_ *Strlit, name string) {
	// If the package was imported with a name other than the final
	// import path element, show it explicitly in the error message.
	// Note that this handles both renamed imports and imports of
	// packages containing unconventional package declarations.
	// Note that this uses / always, even on Windows, because Go import
	// paths always use forward slashes.
	elem := path_.S
	if i := strings.LastIndex(elem, "/"); i >= 0 {
		elem = elem[i+1:]
	}
	if name == "" || elem == name {
		yyerrorl(int(lineno), "imported and not used: \"%v\"", Zconv(path_, 0))
	} else {
		yyerrorl(int(lineno), "imported and not used: \"%v\" as %s", Zconv(path_, 0), name)
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
		var s *Sym
		for h := int32(0); h < NHASH; h++ {
			for s = hash[h]; s != nil; s = s.Link {
				if s.Def == nil || s.Pkg != localpkg {
					continue
				}
				if s.Def.Op == OPACK {
					// throw away top-level package name leftover
					// from previous file.
					// leave s->block set to cause redeclaration
					// errors if a conflicting top-level name is
					// introduced by a different file.
					if s.Def.Used == 0 && nsyntaxerrors == 0 {
						pkgnotused(int(s.Def.Lineno), s.Def.Pkg.Path, s.Name)
					}
					s.Def = nil
					continue
				}

				if s.Def.Sym != s {
					// throw away top-level name left over
					// from previous import . "x"
					if s.Def.Pack != nil && s.Def.Pack.Used == 0 && nsyntaxerrors == 0 {
						pkgnotused(int(s.Def.Pack.Lineno), s.Def.Pack.Pkg.Path, "")
						s.Def.Pack.Used = 1
					}

					s.Def = nil
					continue
				}
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
		namebuf = p
		if i := strings.LastIndex(namebuf, "."); i >= 0 {
			namebuf = namebuf[:i]
		}
		outfile = fmt.Sprintf("%s.%c", namebuf, Thearch.Thechar)
	}
}
