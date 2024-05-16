// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cgo; see doc.go for an overview.

// TODO(rsc):
//	Emit correct line number annotations.
//	Make gc understand the annotations.

package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"internal/buildcfg"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"

	"cmd/internal/edit"
	"cmd/internal/notsha256"
	"cmd/internal/objabi"
	"cmd/internal/telemetry"
)

// A Package collects information about the package we're going to write.
type Package struct {
	PackageName string // name of package
	PackagePath string
	PtrSize     int64
	IntSize     int64
	GccOptions  []string
	GccIsClang  bool
	LdFlags     []string // #cgo LDFLAGS
	Written     map[string]bool
	Name        map[string]*Name // accumulated Name from Files
	ExpFunc     []*ExpFunc       // accumulated ExpFunc from Files
	Decl        []ast.Decl
	GoFiles     []string        // list of Go files
	GccFiles    []string        // list of gcc output files
	Preamble    string          // collected preamble for _cgo_export.h
	typedefs    map[string]bool // type names that appear in the types of the objects we're interested in
	typedefList []typedefInfo
	noCallbacks map[string]bool // C function names with #cgo nocallback directive
	noEscapes   map[string]bool // C function names with #cgo noescape directive
}

// A typedefInfo is an element on Package.typedefList: a typedef name
// and the position where it was required.
type typedefInfo struct {
	typedef string
	pos     token.Pos
}

// A File collects information about a single Go input file.
type File struct {
	AST         *ast.File           // parsed AST
	Comments    []*ast.CommentGroup // comments from file
	Package     string              // Package name
	Preamble    string              // C preamble (doc comment on import "C")
	Ref         []*Ref              // all references to C.xxx in AST
	Calls       []*Call             // all calls to C.xxx in AST
	ExpFunc     []*ExpFunc          // exported functions for this file
	Name        map[string]*Name    // map from Go name to Name
	NamePos     map[*Name]token.Pos // map from Name to position of the first reference
	NoCallbacks map[string]bool     // C function names that with #cgo nocallback directive
	NoEscapes   map[string]bool     // C function names that with #cgo noescape directive
	Edit        *edit.Buffer
}

func (f *File) offset(p token.Pos) int {
	return fset.Position(p).Offset
}

func nameKeys(m map[string]*Name) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}

// A Call refers to a call of a C.xxx function in the AST.
type Call struct {
	Call     *ast.CallExpr
	Deferred bool
	Done     bool
}

// A Ref refers to an expression of the form C.xxx in the AST.
type Ref struct {
	Name    *Name
	Expr    *ast.Expr
	Context astContext
	Done    bool
}

func (r *Ref) Pos() token.Pos {
	return (*r.Expr).Pos()
}

var nameKinds = []string{"iconst", "fconst", "sconst", "type", "var", "fpvar", "func", "macro", "not-type"}

// A Name collects information about C.xxx.
type Name struct {
	Go       string // name used in Go referring to package C
	Mangle   string // name used in generated Go
	C        string // name used in C
	Define   string // #define expansion
	Kind     string // one of the nameKinds
	Type     *Type  // the type of xxx
	FuncType *FuncType
	AddError bool
	Const    string // constant definition
}

// IsVar reports whether Kind is either "var" or "fpvar"
func (n *Name) IsVar() bool {
	return n.Kind == "var" || n.Kind == "fpvar"
}

// IsConst reports whether Kind is either "iconst", "fconst" or "sconst"
func (n *Name) IsConst() bool {
	return strings.HasSuffix(n.Kind, "const")
}

// An ExpFunc is an exported function, callable from C.
// Such functions are identified in the Go input file
// by doc comments containing the line //export ExpName
type ExpFunc struct {
	Func    *ast.FuncDecl
	ExpName string // name to use from C
	Doc     string
}

// A TypeRepr contains the string representation of a type.
type TypeRepr struct {
	Repr       string
	FormatArgs []interface{}
}

// A Type collects information about a type in both the C and Go worlds.
type Type struct {
	Size       int64
	Align      int64
	C          *TypeRepr
	Go         ast.Expr
	EnumValues map[string]int64
	Typedef    string
	BadPointer bool // this pointer type should be represented as a uintptr (deprecated)
}

// A FuncType collects information about a function type in both the C and Go worlds.
type FuncType struct {
	Params []*Type
	Result *Type
	Go     *ast.FuncType
}

func usage() {
	fmt.Fprint(os.Stderr, "usage: cgo -- [compiler options] file.go ...\n")
	flag.PrintDefaults()
	os.Exit(2)
}

var ptrSizeMap = map[string]int64{
	"386":      4,
	"alpha":    8,
	"amd64":    8,
	"arm":      4,
	"arm64":    8,
	"loong64":  8,
	"m68k":     4,
	"mips":     4,
	"mipsle":   4,
	"mips64":   8,
	"mips64le": 8,
	"nios2":    4,
	"ppc":      4,
	"ppc64":    8,
	"ppc64le":  8,
	"riscv":    4,
	"riscv64":  8,
	"s390":     4,
	"s390x":    8,
	"sh":       4,
	"shbe":     4,
	"sparc":    4,
	"sparc64":  8,
}

var intSizeMap = map[string]int64{
	"386":      4,
	"alpha":    8,
	"amd64":    8,
	"arm":      4,
	"arm64":    8,
	"loong64":  8,
	"m68k":     4,
	"mips":     4,
	"mipsle":   4,
	"mips64":   8,
	"mips64le": 8,
	"nios2":    4,
	"ppc":      4,
	"ppc64":    8,
	"ppc64le":  8,
	"riscv":    4,
	"riscv64":  8,
	"s390":     4,
	"s390x":    8,
	"sh":       4,
	"shbe":     4,
	"sparc":    4,
	"sparc64":  8,
}

var cPrefix string

var fset = token.NewFileSet()

var dynobj = flag.String("dynimport", "", "if non-empty, print dynamic import data for that file")
var dynout = flag.String("dynout", "", "write -dynimport output to this file")
var dynpackage = flag.String("dynpackage", "main", "set Go package for -dynimport output")
var dynlinker = flag.Bool("dynlinker", false, "record dynamic linker information in -dynimport mode")

// This flag is for bootstrapping a new Go implementation,
// to generate Go types that match the data layout and
// constant values used in the host's C libraries and system calls.
var godefs = flag.Bool("godefs", false, "for bootstrap: write Go definitions for C file to standard output")

var srcDir = flag.String("srcdir", "", "source directory")
var objDir = flag.String("objdir", "", "object directory")
var importPath = flag.String("importpath", "", "import path of package being built (for comments in generated files)")
var exportHeader = flag.String("exportheader", "", "where to write export header if any exported functions")

var ldflags = flag.String("ldflags", "", "flags to pass to C linker")

var gccgo = flag.Bool("gccgo", false, "generate files for use with gccgo")
var gccgoprefix = flag.String("gccgoprefix", "", "-fgo-prefix option used with gccgo")
var gccgopkgpath = flag.String("gccgopkgpath", "", "-fgo-pkgpath option used with gccgo")
var gccgoMangler func(string) string
var gccgoDefineCgoIncomplete = flag.Bool("gccgo_define_cgoincomplete", false, "define cgo.Incomplete for older gccgo/GoLLVM")
var importRuntimeCgo = flag.Bool("import_runtime_cgo", true, "import runtime/cgo in generated code")
var importSyscall = flag.Bool("import_syscall", true, "import syscall in generated code")
var trimpath = flag.String("trimpath", "", "applies supplied rewrites or trims prefixes to recorded source file paths")

var goarch, goos, gomips, gomips64 string
var gccBaseCmd []string

func main() {
	telemetry.Start()
	objabi.AddVersionFlag() // -V
	objabi.Flagparse(usage)
	telemetry.Inc("cgo/invocations")
	telemetry.CountFlags("cgo/flag:", *flag.CommandLine)

	if *gccgoDefineCgoIncomplete {
		if !*gccgo {
			fmt.Fprintf(os.Stderr, "cgo: -gccgo_define_cgoincomplete without -gccgo\n")
			os.Exit(2)
		}
		incomplete = "_cgopackage_Incomplete"
	}

	if *dynobj != "" {
		// cgo -dynimport is essentially a separate helper command
		// built into the cgo binary. It scans a gcc-produced executable
		// and dumps information about the imported symbols and the
		// imported libraries. The 'go build' rules for cgo prepare an
		// appropriate executable and then use its import information
		// instead of needing to make the linkers duplicate all the
		// specialized knowledge gcc has about where to look for imported
		// symbols and which ones to use.
		dynimport(*dynobj)
		return
	}

	if *godefs {
		// Generating definitions pulled from header files,
		// to be checked into Go repositories.
		// Line numbers are just noise.
		conf.Mode &^= printer.SourcePos
	}

	args := flag.Args()
	if len(args) < 1 {
		usage()
	}

	// Find first arg that looks like a go file and assume everything before
	// that are options to pass to gcc.
	var i int
	for i = len(args); i > 0; i-- {
		if !strings.HasSuffix(args[i-1], ".go") {
			break
		}
	}
	if i == len(args) {
		usage()
	}

	// Save original command line arguments for the godefs generated comment. Relative file
	// paths in os.Args will be rewritten to absolute file paths in the loop below.
	osArgs := make([]string, len(os.Args))
	copy(osArgs, os.Args[:])
	goFiles := args[i:]

	for _, arg := range args[:i] {
		if arg == "-fsanitize=thread" {
			tsanProlog = yesTsanProlog
		}
		if arg == "-fsanitize=memory" {
			msanProlog = yesMsanProlog
		}
	}

	p := newPackage(args[:i])

	// We need a C compiler to be available. Check this.
	var err error
	gccBaseCmd, err = checkGCCBaseCmd()
	if err != nil {
		fatalf("%v", err)
		os.Exit(2)
	}

	// Record linker flags for external linking.
	if *ldflags != "" {
		args, err := splitQuoted(*ldflags)
		if err != nil {
			fatalf("bad -ldflags option: %q (%s)", *ldflags, err)
		}
		p.addToFlag("LDFLAGS", args)
	}

	// Need a unique prefix for the global C symbols that
	// we use to coordinate between gcc and ourselves.
	// We already put _cgo_ at the beginning, so the main
	// concern is other cgo wrappers for the same functions.
	// Use the beginning of the notsha256 of the input to disambiguate.
	h := notsha256.New()
	io.WriteString(h, *importPath)
	fs := make([]*File, len(goFiles))
	for i, input := range goFiles {
		if *srcDir != "" {
			input = filepath.Join(*srcDir, input)
		}

		// Create absolute path for file, so that it will be used in error
		// messages and recorded in debug line number information.
		// This matches the rest of the toolchain. See golang.org/issue/5122.
		if aname, err := filepath.Abs(input); err == nil {
			input = aname
		}

		b, err := os.ReadFile(input)
		if err != nil {
			fatalf("%s", err)
		}
		if _, err = h.Write(b); err != nil {
			fatalf("%s", err)
		}

		// Apply trimpath to the file path. The path won't be read from after this point.
		input, _ = objabi.ApplyRewrites(input, *trimpath)
		if strings.ContainsAny(input, "\r\n") {
			// ParseGo, (*Package).writeOutput, and printer.Fprint in SourcePos mode
			// all emit line directives, which don't permit newlines in the file path.
			// Bail early if we see anything newline-like in the trimmed path.
			fatalf("input path contains newline character: %q", input)
		}
		goFiles[i] = input

		f := new(File)
		f.Edit = edit.NewBuffer(b)
		f.ParseGo(input, b)
		f.ProcessCgoDirectives()
		fs[i] = f
	}

	cPrefix = fmt.Sprintf("_%x", h.Sum(nil)[0:6])

	if *objDir == "" {
		*objDir = "_obj"
	}
	// make sure that `objDir` directory exists, so that we can write
	// all the output files there.
	os.MkdirAll(*objDir, 0o700)
	*objDir += string(filepath.Separator)

	for i, input := range goFiles {
		f := fs[i]
		p.Translate(f)
		for _, cref := range f.Ref {
			switch cref.Context {
			case ctxCall, ctxCall2:
				if cref.Name.Kind != "type" {
					break
				}
				old := *cref.Expr
				*cref.Expr = cref.Name.Type.Go
				f.Edit.Replace(f.offset(old.Pos()), f.offset(old.End()), gofmt(cref.Name.Type.Go))
			}
		}
		if nerrors > 0 {
			os.Exit(2)
		}
		p.PackagePath = f.Package
		p.Record(f)
		if *godefs {
			os.Stdout.WriteString(p.godefs(f, osArgs))
		} else {
			p.writeOutput(f, input)
		}
	}
	cFunctions := make(map[string]bool)
	for _, key := range nameKeys(p.Name) {
		n := p.Name[key]
		if n.FuncType != nil {
			cFunctions[n.C] = true
		}
	}

	for funcName := range p.noEscapes {
		if _, found := cFunctions[funcName]; !found {
			error_(token.NoPos, "#cgo noescape %s: no matched C function", funcName)
		}
	}

	for funcName := range p.noCallbacks {
		if _, found := cFunctions[funcName]; !found {
			error_(token.NoPos, "#cgo nocallback %s: no matched C function", funcName)
		}
	}

	if !*godefs {
		p.writeDefs()
	}
	if nerrors > 0 {
		os.Exit(2)
	}
}

// newPackage returns a new Package that will invoke
// gcc with the additional arguments specified in args.
func newPackage(args []string) *Package {
	goarch = runtime.GOARCH
	if s := os.Getenv("GOARCH"); s != "" {
		goarch = s
	}
	goos = runtime.GOOS
	if s := os.Getenv("GOOS"); s != "" {
		goos = s
	}
	buildcfg.Check()
	gomips = buildcfg.GOMIPS
	gomips64 = buildcfg.GOMIPS64
	ptrSize := ptrSizeMap[goarch]
	if ptrSize == 0 {
		fatalf("unknown ptrSize for $GOARCH %q", goarch)
	}
	intSize := intSizeMap[goarch]
	if intSize == 0 {
		fatalf("unknown intSize for $GOARCH %q", goarch)
	}

	// Reset locale variables so gcc emits English errors [sic].
	os.Setenv("LANG", "en_US.UTF-8")
	os.Setenv("LC_ALL", "C")

	p := &Package{
		PtrSize:     ptrSize,
		IntSize:     intSize,
		Written:     make(map[string]bool),
		noCallbacks: make(map[string]bool),
		noEscapes:   make(map[string]bool),
	}
	p.addToFlag("CFLAGS", args)
	return p
}

// Record what needs to be recorded about f.
func (p *Package) Record(f *File) {
	if p.PackageName == "" {
		p.PackageName = f.Package
	} else if p.PackageName != f.Package {
		error_(token.NoPos, "inconsistent package names: %s, %s", p.PackageName, f.Package)
	}

	if p.Name == nil {
		p.Name = f.Name
	} else {
		for k, v := range f.Name {
			if p.Name[k] == nil {
				p.Name[k] = v
			} else if p.incompleteTypedef(p.Name[k].Type) {
				p.Name[k] = v
			} else if p.incompleteTypedef(v.Type) {
				// Nothing to do.
			} else if _, ok := nameToC[k]; ok {
				// Names we predefine may appear inconsistent
				// if some files typedef them and some don't.
				// Issue 26743.
			} else if !reflect.DeepEqual(p.Name[k], v) {
				error_(token.NoPos, "inconsistent definitions for C.%s", fixGo(k))
			}
		}
	}

	// merge nocallback & noescape
	for k, v := range f.NoCallbacks {
		p.noCallbacks[k] = v
	}
	for k, v := range f.NoEscapes {
		p.noEscapes[k] = v
	}

	if f.ExpFunc != nil {
		p.ExpFunc = append(p.ExpFunc, f.ExpFunc...)
		p.Preamble += "\n" + f.Preamble
	}
	p.Decl = append(p.Decl, f.AST.Decls...)
}

// incompleteTypedef reports whether t appears to be an incomplete
// typedef definition.
func (p *Package) incompleteTypedef(t *Type) bool {
	return t == nil || (t.Size == 0 && t.Align == -1)
}
