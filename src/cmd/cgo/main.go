// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Cgo; see gmp.go for an overview.

// TODO(rsc):
//	Emit correct line number annotations.
//	Make gc understand the annotations.

package main

import (
	"crypto/md5"
	"flag"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"

	"cmd/internal/edit"
	"cmd/internal/objabi"
)

// A Package collects information about the package we're going to write.
type Package struct {
	PackageName string // name of package
	PackagePath string
	PtrSize     int64
	IntSize     int64
	GccOptions  []string
	GccIsClang  bool
	CgoFlags    map[string][]string // #cgo flags (CFLAGS, LDFLAGS)
	Written     map[string]bool
	Name        map[string]*Name // accumulated Name from Files
	ExpFunc     []*ExpFunc       // accumulated ExpFunc from Files
	Decl        []ast.Decl
	GoFiles     []string        // list of Go files
	GccFiles    []string        // list of gcc output files
	Preamble    string          // collected preamble for _cgo_export.h
	typedefs    map[string]bool // type names that appear in the types of the objects we're interested in
	typedefList []typedefInfo
}

// A typedefInfo is an element on Package.typedefList: a typedef name
// and the position where it was required.
type typedefInfo struct {
	typedef string
	pos     token.Pos
}

// A File collects information about a single Go input file.
type File struct {
	AST      *ast.File           // parsed AST
	Comments []*ast.CommentGroup // comments from file
	Package  string              // Package name
	Preamble string              // C preamble (doc comment on import "C")
	Ref      []*Ref              // all references to C.xxx in AST
	Calls    []*Call             // all calls to C.xxx in AST
	ExpFunc  []*ExpFunc          // exported functions for this file
	Name     map[string]*Name    // map from Go name to Name
	NamePos  map[*Name]token.Pos // map from Name to position of the first reference
	Edit     *edit.Buffer
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
	BadPointer bool
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
	"amd64":    8,
	"arm":      4,
	"arm64":    8,
	"mips":     4,
	"mipsle":   4,
	"mips64":   8,
	"mips64le": 8,
	"ppc64":    8,
	"ppc64le":  8,
	"riscv64":  8,
	"s390":     4,
	"s390x":    8,
	"sparc64":  8,
}

var intSizeMap = map[string]int64{
	"386":      4,
	"amd64":    8,
	"arm":      4,
	"arm64":    8,
	"mips":     4,
	"mipsle":   4,
	"mips64":   8,
	"mips64le": 8,
	"ppc64":    8,
	"ppc64le":  8,
	"riscv64":  8,
	"s390":     4,
	"s390x":    8,
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

var gccgo = flag.Bool("gccgo", false, "generate files for use with gccgo")
var gccgoprefix = flag.String("gccgoprefix", "", "-fgo-prefix option used with gccgo")
var gccgopkgpath = flag.String("gccgopkgpath", "", "-fgo-pkgpath option used with gccgo")
var gccgoMangleCheckDone bool
var gccgoNewmanglingInEffect bool
var importRuntimeCgo = flag.Bool("import_runtime_cgo", true, "import runtime/cgo in generated code")
var importSyscall = flag.Bool("import_syscall", true, "import syscall in generated code")
var goarch, goos string

func main() {
	objabi.AddVersionFlag() // -V
	flag.Usage = usage
	flag.Parse()

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

	// Record CGO_LDFLAGS from the environment for external linking.
	if ldflags := os.Getenv("CGO_LDFLAGS"); ldflags != "" {
		args, err := splitQuoted(ldflags)
		if err != nil {
			fatalf("bad CGO_LDFLAGS: %q (%s)", ldflags, err)
		}
		p.addToFlag("LDFLAGS", args)
	}

	// Need a unique prefix for the global C symbols that
	// we use to coordinate between gcc and ourselves.
	// We already put _cgo_ at the beginning, so the main
	// concern is other cgo wrappers for the same functions.
	// Use the beginning of the md5 of the input to disambiguate.
	h := md5.New()
	io.WriteString(h, *importPath)
	fs := make([]*File, len(goFiles))
	for i, input := range goFiles {
		if *srcDir != "" {
			input = filepath.Join(*srcDir, input)
		}

		b, err := ioutil.ReadFile(input)
		if err != nil {
			fatalf("%s", err)
		}
		if _, err = h.Write(b); err != nil {
			fatalf("%s", err)
		}

		f := new(File)
		f.Edit = edit.NewBuffer(b)
		f.ParseGo(input, b)
		f.DiscardCgoDirectives()
		fs[i] = f
	}

	cPrefix = fmt.Sprintf("_%x", h.Sum(nil)[0:6])

	if *objDir == "" {
		// make sure that _obj directory exists, so that we can write
		// all the output files there.
		os.Mkdir("_obj", 0777)
		*objDir = "_obj"
	}
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
			os.Stdout.WriteString(p.godefs(f, input))
		} else {
			p.writeOutput(f, input)
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
		PtrSize:  ptrSize,
		IntSize:  intSize,
		CgoFlags: make(map[string][]string),
		Written:  make(map[string]bool),
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
