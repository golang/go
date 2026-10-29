// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"cmp"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"go/types"
	"internal/syslist"
	"io"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	"cmd/internal/edit"
	"cmd/internal/hash"
	"cmd/internal/objabi"
	"cmd/internal/par"
)

// BindingVar describes a C global variable bound with //cgo:var.
type BindingVar struct {
	Name    string // Internal identifier (used in _Cvar_<Name>)
	GoName  string // Go variable named by the pragma
	CName   string // C symbol name
	BindC   bool   // The pragma used the C.<CName> form, so C.<CName> is bound as well
	Type    string // Go type expression
	Library string // Shared library or DLL name; empty for libc and static
	LibC    bool   // The symbol comes from the platform C runtime (libc)
	Static  bool   // The symbol is resolved at link time from a .syso or .a (static)

	// Methods of //cgo:dynload types:
	RecvType string // The //cgo:dynload type, for a //cgo:var LibT.GoName pragma
	SymIndex int    // Index of the symbol in RecvType's per-instance symbol table
}

// Bindings holds the parsed declarations and linker flags collected
// from the //cgo: pragmas in a package.
type Bindings struct {
	mu              sync.Mutex
	Package         string
	LdFlags         []string
	Vars            []BindingVar
	Funcs           []BindingFunc
	UsedBuiltins    map[string]bool   // cgo builtins (such as CString) referenced as C.name
	TypeGoToC       map[string]string // Go type name -> C name, for //cgo:type GoName C.cName
	TypeGoToAnyC    map[string]string // Go type name -> C name, for all //cgo:type pragmas
	ConstCToGo      map[string]string // C name -> Go const name, for //cgo:const GoName C.cName
	GoOnlyCNames    map[string]bool   // C names bound by pragmas without the C. prefix
	DynloadTypes    []string          // //cgo:dynload types, in pragma order
	DynloadCounts   map[string]int    // number of symbols bound for each //cgo:dynload type
	DynloadVars     []BindingVar      // //cgo:var LibT.GoName methods
	TypeMap         map[string]BindingType
	UsedFuncPtrs    map[string]bool // bound C function names referenced in non-call context
	UsedExportCalls map[string]bool // //export function names called via C.<name>(...)

	typeSpecs             map[string]*ast.TypeSpec
	resolveUsedTypeInFile func(expr ast.Expr, inPtr bool, imp fileImports)
	tpkg                  *types.Package
	typeInfo              *types.Info
	constPkg              *types.Package
	constFset             *token.FileSet
	constFilePos          map[string]token.Pos
	constSources          []bindingSource
}

// bindingSource holds the raw bytes and parsed AST of a source file inspected
// for //cgo: pragmas.
type bindingSource struct {
	path     string // trimmed path for diagnostics
	origPath string // untrimmed path for ${SRCDIR} expansion; defaults to path
	data     []byte
	file     *ast.File
}

func (bf *Bindings) needCMalloc() bool {
	return bf.UsedBuiltins["CString"] || bf.UsedBuiltins["CBytes"] || bf.UsedBuiltins["_CMalloc"]
}

func targetPtrSize() int64 {
	sz := ptrSizeMap[goarch]
	if sz == 0 {
		fatalf("unknown ptrSize for $GOARCH %q", goarch)
	}
	return sz
}

func targetCLongSize() int64 {
	if goos == "windows" {
		return 4
	}
	return targetPtrSize()
}

// forEachImportC calls fn for every `import "C"` spec in file, passing its
// enclosing GenDecl, the ImportSpec, and its associated doc comment group.
func forEachImportC(file *ast.File, fn func(gd *ast.GenDecl, spec *ast.ImportSpec, doc *ast.CommentGroup)) {
	if file == nil {
		return
	}
	for _, decl := range file.Decls {
		gd, ok := decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.IMPORT {
			continue
		}
		for _, spec := range gd.Specs {
			imp, ok := spec.(*ast.ImportSpec)
			if !ok || imp.Path.Value != `"C"` {
				continue
			}
			doc := imp.Doc
			if doc == nil && len(gd.Specs) == 1 {
				doc = gd.Doc
			}
			fn(gd, imp, doc)
		}
	}
}

func primitiveCtypeToGo(name string) (string, bool) {
	switch name {
	case "_Ctype_char":
		if targetCCharIsUnsigned(goos, goarch) {
			return "uint8", true
		}
		return "int8", true
	case "_Ctype_schar", "_Ctype_int8_t":
		return "int8", true
	case "_Ctype_uchar", "_Ctype_uint8_t":
		return "uint8", true
	case "_Ctype_short", "_Ctype_int16_t":
		return "int16", true
	case "_Ctype_ushort", "_Ctype_uint16_t":
		return "uint16", true
	case "_Ctype_int", "_Ctype_int32_t":
		return "int32", true
	case "_Ctype_uint", "_Ctype_uint32_t":
		return "uint32", true
	case "_Ctype_long":
		if targetCLongSize() == 4 {
			return "int32", true
		}
		return "int64", true
	case "_Ctype_ulong":
		if targetCLongSize() == 4 {
			return "uint32", true
		}
		return "uint64", true
	case "_Ctype_longlong", "_Ctype_int64_t":
		return "int64", true
	case "_Ctype_ulonglong", "_Ctype_uint64_t":
		return "uint64", true
	case "_Ctype_size_t", "_Ctype_uintptr_t":
		if targetPtrSize() == 4 {
			return "uint32", true
		}
		return "uint64", true
	case "_Ctype_ssize_t", "_Ctype_ptrdiff_t", "_Ctype_intptr_t", "_Ctype_intgo":
		if targetPtrSize() == 4 {
			return "int32", true
		}
		return "int64", true
	case "_Ctype_float":
		return "float32", true
	case "_Ctype_double":
		return "float64", true
	case "_Ctype__Bool", "_Ctype_bool":
		return "bool", true
	case "_Ctype_void":
		return "[0]byte", true
	case "_Ctype__GoString_":
		return "string", true
	case "_Ctype__GoBytes_":
		return "[]byte", true
	}
	return "", false
}

// isStandardC reports whether lib names a standard C runtime or system library
// that is implicitly linked by the platform runtime and should not be emitted
// as an explicit shared-library dependency.
func isStandardC(lib string) bool {
	base := filepath.Base(lib)
	lower := strings.ToLower(base)
	switch lower {
	case "c", "libc", "pthread", "libpthread", "dl", "libdl",
		"libc.dylib", "libpthread.dylib",
		"msvcrt", "msvcrt.dll", "ucrtbase", "ucrtbase.dll", "kernel32", "kernel32.dll":
		return true
	}
	if goos == "darwin" || goos == "ios" {
		if lower == "system" || lower == "libsystem" ||
			strings.HasPrefix(lower, "libsystem.") || strings.HasPrefix(lower, "libsystem_") {
			return true
		}
	}
	return strings.HasPrefix(base, "libc.so") ||
		strings.HasPrefix(base, "libpthread.so") ||
		strings.HasPrefix(base, "libdl.so") ||
		strings.HasPrefix(base, "ld-linux") ||
		strings.HasPrefix(base, "ld-musl")
}

// dynLibName returns the name under which a dynamically linked library lib
// is recorded as a dependency of the binary (for example, in an ELF DT_NEEDED
// entry or a Mach-O LC_LOAD_DYLIB command). Elsewhere only the base name is
// recorded and the library is found through the dynamic linker's search path,
// but on Darwin an install name is a load path that dyld uses as written, so
// absolute and @rpath, @loader_path, and @executable_path names are kept intact.
func dynLibName(lib string) string {
	lib = strings.TrimPrefix(lib, "./")
	if (goos == "darwin" || goos == "ios") && (strings.HasPrefix(lib, "/") || strings.HasPrefix(lib, "@")) {
		return lib
	}
	return filepath.Base(lib)
}

func windowsDLLName(lib string) string {
	lib = strings.TrimSpace(strings.TrimPrefix(lib, "./"))
	if lib == "" {
		return "msvcrt.dll"
	}
	base := filepath.Base(lib)
	switch strings.ToLower(base) {
	case "c", "libc", "msvcrt", "msvcrt.dll":
		return "msvcrt.dll"
	case "ucrtbase", "ucrtbase.dll":
		return "ucrtbase.dll"
	case "kernel32", "kernel32.dll":
		return "kernel32.dll"
	}
	if !strings.HasSuffix(strings.ToLower(base), ".dll") {
		return base + ".dll"
	}
	return base
}

// cgoPragma is a parsed cgo binding pragma:
//
//	//cgo:func GoName [C.]cName {"lib" | static | libc}
//	//cgo:var GoName [C.]cName {"lib" | static | libc}
//	//cgo:type GoName [C.]cName
//	//cgo:const GoName [C.]cName
//	//cgo:dynload LibT
//	//cgo:func LibT.GoName cName
//	//cgo:var LibT.GoName cName
//
// Except for //cgo:dynload, a pragma names both a Go declaration, GoName, and
// the C name, cName, that it binds it to. Writing the C name as C.cName also
// lets Go code in the package refer to it as C.cName, as with traditional cgo;
// without the C. prefix, it is reachable only through GoName.
//
// Pragmas are self-contained and may appear anywhere in a file that
// imports "C".
type cgoPragma struct {
	Pos      token.Position
	TokPos   token.Pos // position in the file set the pragma was parsed with
	Text     string    // the pragma comment text
	Kind     string    // pragma name: "func", "var", "type", "const", or "dynload"
	RecvType string    // for func and var, the //cgo:dynload type LibT in the LibT.GoName form
	GoName   string    // Go identifier being bound (the method name in the LibT.GoName form)
	CName    string    // C symbol name
	BindC    bool      // cName was written as C.cName, so Go code may also use C.cName

	// Library specifier, for func and var.
	Library string // shared library for the "lib" form
	LibC    bool   // libc: the platform C runtime
	Static  bool   // static: resolved at link time from a .syso or .a
}

// cgoPragmaSyntax maps each supported //cgo: pragma name to its syntax, for
// error messages.
var cgoPragmaSyntax = map[string]string{
	"func":    `//cgo:func GoName [C.]cName {"lib" | static | libc}`,
	"var":     `//cgo:var GoName [C.]cName {"lib" | static | libc}`,
	"type":    `//cgo:type GoName [C.]cName`,
	"const":   `//cgo:const GoName [C.]cName`,
	"dynload": `//cgo:dynload LibT`,
}

// parseCgoDirective parses a "#cgo [cond] VERB: flags" line (where line starts
// with "#cgo"). It returns the verb, raw flags string, whether any platform
// condition matched the target goos/goarch, and whether the syntax is valid.
func parseCgoDirective(line string) (verb, flags string, matched, ok bool) {
	rest := strings.TrimSpace(line[4:])
	before, after, hasColon := strings.Cut(rest, ":")
	if !hasColon {
		return "", "", false, false
	}
	head := strings.Fields(before)
	if len(head) == 0 {
		return "", "", false, false
	}
	verb = head[len(head)-1]
	switch verb {
	case "CFLAGS", "CPPFLAGS", "CXXFLAGS", "FFLAGS", "LDFLAGS", "pkg-config":
	default:
		return "", "", false, false
	}
	cond := strings.Join(head[:len(head)-1], " ")
	if cond != "" && !matchCgoCond(cond) {
		return verb, "", false, true
	}
	return verb, strings.TrimSpace(after), true, true
}

func matchCgoCond(cond string) bool {
	expr, err := constraint.Parse("// +build " + cond)
	if err != nil {
		return false
	}
	return expr.Eval(func(tag string) bool {
		switch tag {
		case goos, goarch, "cgo", "gc":
			return true
		case "unix":
			return syslist.UnixOS[goos]
		case "linux":
			return goos == "linux" || goos == "android"
		case "darwin":
			return goos == "darwin" || goos == "ios"
		case "solaris":
			return goos == "solaris" || goos == "illumos"
		}
		return false
	})
}

// isCgoPragmaComment reports whether the comment text is a //cgo: binding
// pragma (such as //cgo:func).
func isCgoPragmaComment(text string) bool {
	d, ok := ast.ParseDirective(0, text)
	return ok && d.Tool == "cgo" && cgoPragmaSyntax[d.Name] != ""
}

// parseCgoPragma parses c as a //cgo: binding pragma. It reports false if c
// is not such a pragma and exits with an error if c is malformed.
func parseCgoPragma(fset *token.FileSet, c *ast.Comment) (cgoPragma, bool) {
	d, ok := ast.ParseDirective(c.Pos(), c.Text)
	if !ok || d.Tool != "cgo" || cgoPragmaSyntax[d.Name] == "" {
		return cgoPragma{}, false
	}
	syntax := cgoPragmaSyntax[d.Name]
	bad := func(why string) {
		fatalf("%s: invalid //cgo:%s pragma: %s (want %s)", fset.Position(c.Pos()), d.Name, why, syntax)
	}
	args, err := d.ParseArgs()
	if err != nil {
		bad(err.Error())
	}
	// quoted reports whether arg was written as a quoted string.
	quoted := func(arg ast.DirectiveArg) bool {
		off := int(arg.Pos - d.ArgsPos)
		return off >= 0 && off < len(d.Args) && (d.Args[off] == '"' || d.Args[off] == '`')
	}
	pr := cgoPragma{Pos: fset.Position(c.Pos()), TokPos: c.Pos(), Text: strings.TrimSpace(c.Text), Kind: d.Name}
	if d.Name == "dynload" {
		if len(args) != 1 {
			bad("wrong number of arguments")
		}
		if quoted(args[0]) || !token.IsIdentifier(args[0].Arg) || args[0].Arg == "_" {
			bad(fmt.Sprintf("invalid Go type name %q", args[0].Arg))
		}
		pr.GoName = args[0].Arg
		return pr, true
	}
	if len(args) > 0 && !quoted(args[0]) && (d.Name == "func" || d.Name == "var") {
		if recv, name, ok := strings.Cut(args[0].Arg, "."); ok {
			syntax = "//cgo:" + d.Name + " LibT.GoName cName"
			if len(args) != 2 {
				bad("wrong number of arguments; a //cgo:dynload method takes no library specifier")
			}
			if !token.IsIdentifier(recv) || !token.IsIdentifier(name) || name == "_" {
				bad(fmt.Sprintf("invalid Go name %q", args[0].Arg))
			}
			if quoted(args[1]) {
				bad(fmt.Sprintf("invalid C name %q", args[1].Arg))
			}
			if strings.HasPrefix(args[1].Arg, "C.") {
				bad("a //cgo:dynload method cannot bind a C. name")
			}
			if !token.IsIdentifier(args[1].Arg) {
				bad(fmt.Sprintf("invalid C name %q", args[1].Arg))
			}
			pr.RecvType, pr.GoName, pr.CName = recv, name, args[1].Arg
			return pr, true
		}
	}
	hasLib := d.Name == "func" || d.Name == "var"
	want := 2
	if hasLib {
		want = 3
	}
	if len(args) != want {
		bad("wrong number of arguments")
	}
	if quoted(args[0]) || !token.IsIdentifier(args[0].Arg) || args[0].Arg == "_" {
		bad(fmt.Sprintf("invalid Go name %q", args[0].Arg))
	}
	pr.GoName = args[0].Arg
	pr.CName, pr.BindC = strings.CutPrefix(args[1].Arg, "C.")
	if quoted(args[1]) || !token.IsIdentifier(pr.CName) {
		bad(fmt.Sprintf("invalid C name %q", args[1].Arg))
	}
	if !hasLib {
		return pr, true
	}
	lib := args[2]
	switch {
	case quoted(lib):
		pr.Library = strings.TrimPrefix(lib.Arg, "./")
		if pr.Library == "" {
			bad("empty library name")
		}
	case lib.Arg == "static":
		pr.Static = true
	case lib.Arg == "libc":
		pr.LibC = true
	default:
		bad(fmt.Sprintf("invalid library specifier %s", lib.Arg))
	}
	return pr, true
}

// parseCgoPragmas returns the //cgo: binding pragmas in file, in source order.
// Pragmas are only allowed in files that import "C".
func parseCgoPragmas(fset *token.FileSet, file *ast.File) []cgoPragma {
	var pragmas []cgoPragma
	importsC := hasImportCFile(file)
	for _, cg := range file.Comments {
		for _, c := range cg.List {
			pr, ok := parseCgoPragma(fset, c)
			if !ok {
				continue
			}
			if !importsC {
				fatalf("%s: //cgo:%s pragma is only allowed in a file that imports \"C\"", pr.Pos, pr.Kind)
			}
			pragmas = append(pragmas, pr)
		}
	}
	return pragmas
}

// fileImports records the local import aliases used in a single Go source file
// for the standard packages recognized by cgo bindings.
type fileImports struct {
	hasUnsafe  bool
	unsafe     map[string]bool // local alias -> true ("." for dot import)
	structs    map[string]bool
	runtimeCgo map[string]bool
}

func collectFileImports(file *ast.File) fileImports {
	imp := fileImports{
		unsafe:     make(map[string]bool),
		structs:    make(map[string]bool),
		runtimeCgo: make(map[string]bool),
	}
	if file == nil {
		return imp
	}
	for _, s := range file.Imports {
		name := ""
		if s.Name != nil {
			name = s.Name.Name
		}
		switch s.Path.Value {
		case `"unsafe"`:
			imp.hasUnsafe = true
			if name == "" {
				imp.unsafe["unsafe"] = true
			} else if name != "_" {
				imp.unsafe[name] = true
			}
		case `"structs"`:
			if name == "" {
				imp.structs["structs"] = true
			} else if name != "_" {
				imp.structs[name] = true
			}
		case `"runtime/cgo"`:
			if name == "" {
				imp.runtimeCgo["cgo"] = true
			} else if name != "_" {
				imp.runtimeCgo[name] = true
			}
		}
	}
	return imp
}

func structHasHostLayoutOrIncomplete(st *ast.StructType, imp fileImports) bool {
	if st.Fields == nil || len(st.Fields.List) == 0 {
		return true
	}
	for _, field := range st.Fields.List {
		switch ft := field.Type.(type) {
		case *ast.SelectorExpr:
			if id, ok := ft.X.(*ast.Ident); ok {
				if (imp.structs[id.Name] && ft.Sel.Name == "HostLayout") ||
					(imp.runtimeCgo[id.Name] && ft.Sel.Name == "Incomplete") {
					return true
				}
			}
		case *ast.Ident:
			if (imp.structs["."] && ft.Name == "HostLayout") ||
				(imp.runtimeCgo["."] && ft.Name == "Incomplete") {
				return true
			}
		}
	}
	return false
}

// cVarExpr returns the Go expression for the bound C variable v.
func cVarExpr(v BindingVar) string {
	return "(*_Cvar_" + v.Name + ")"
}

// varImportSym returns the local symbol name for the dynamically imported
// C variable v. On Windows, the symbol is the variable's import address
// table slot rather than the variable itself; see writeGoTypes.
func varImportSym(v BindingVar) string {
	if goos == "windows" {
		return fmt.Sprintf("_cgo%s_var_addr_%s", cPrefix, v.Name)
	}
	return fmt.Sprintf("_cgo%s_var_%s", cPrefix, v.Name)
}

func hasImportCFile(file *ast.File) bool {
	return file != nil && slices.ContainsFunc(file.Imports, func(imp *ast.ImportSpec) bool {
		return imp.Path.Value == `"C"`
	})
}

func replaceCConstNodes(eb *edit.Buffer, tf *token.File, node ast.Node, constCToGo map[string]string) {
	ast.Inspect(node, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if id, ok := sel.X.(*ast.Ident); ok && id.Name == "C" {
			if goName, ok := constCToGo[sel.Sel.Name]; ok {
				eb.Replace(tf.Offset(sel.Pos()), tf.Offset(sel.End()), goName)
				return false
			}
		}
		return true
	})
}

func (p *Package) buildConstPkg(entry *Bindings) {
	constFset := token.NewFileSet()
	entry.constFset = constFset
	var files []*ast.File
	filePos := make(map[string]token.Pos, len(entry.constSources))
	for _, src := range entry.constSources {
		if src.file == nil || src.data == nil {
			continue
		}
		data := src.data
		if hasImportCFile(src.file) {
			if tf := p.fset.File(src.file.Package); tf != nil {
				eb := edit.NewBuffer(data)
				for _, decl := range src.file.Decls {
					if gd, ok := decl.(*ast.GenDecl); ok && (gd.Tok == token.CONST || gd.Tok == token.TYPE) {
						replaceCConstNodes(eb, tf, gd, entry.ConstCToGo)
					}
				}
				data = eb.Bytes()
			}
		}
		f, err := parser.ParseFile(constFset, src.path, data, 0)
		if err != nil {
			continue
		}
		files = append(files, f)
		filePos[src.path] = f.Name.Pos()
	}
	if len(files) == 0 {
		return
	}
	conf := types.Config{
		Importer:                 bindingStubImporter{},
		Sizes:                    types.SizesFor("gc", goarch),
		Error:                    func(error) {},
		DisableUnusedImportCheck: true,
	}
	entry.constPkg, _ = conf.Check(entry.Package, constFset, files, nil)
	entry.constFilePos = filePos
}

// evalArrayLenIn evaluates exprStr as a Go array length in pkg at pos.
// Evaluating "[" + exprStr + "]struct{}" lets go/types apply Go's array-length
// rules and return the length via (*types.Array).Len without importing
// go/constant (which cmd/dist rewrites to bootstrap/go/constant during
// bootstrap while keeping go/types from the bootstrap toolchain).
func evalArrayLenIn(fset *token.FileSet, pkg *types.Package, pos token.Pos, exprStr string) (int64, bool) {
	tv, err := types.Eval(fset, pkg, pos, "["+exprStr+"]struct{}")
	if err != nil {
		return 0, false
	}
	if arr, ok := tv.Type.(*types.Array); ok && arr.Len() >= 0 {
		return arr.Len(), true
	}
	return 0, false
}

func (p *Package) evalBindingArrayLen(expr ast.Expr, entry *Bindings) (int64, bool) {
	var buf bytes.Buffer
	noSourceConf.Fprint(&buf, p.fset, expr)
	src := buf.Bytes()
	if entry != nil && len(entry.ConstCToGo) > 0 {
		tmpFset := token.NewFileSet()
		if parsed, err := parser.ParseExprFrom(tmpFset, "", src, 0); err == nil {
			if tf := tmpFset.File(parsed.Pos()); tf != nil {
				eb := edit.NewBuffer(src)
				replaceCConstNodes(eb, tf, parsed, entry.ConstCToGo)
				src = eb.Bytes()
			}
		}
	}
	exprStr := string(src)
	if entry != nil && entry.tpkg != nil {
		if n, ok := evalArrayLenIn(p.fset, entry.tpkg, expr.Pos(), exprStr); ok {
			return n, true
		}
		if len(entry.ConstCToGo) > 0 {
			if entry.constFset == nil {
				p.buildConstPkg(entry)
			}
			if entry.constPkg != nil {
				pos := token.NoPos
				if tf := p.fset.File(expr.Pos()); tf != nil {
					pos = entry.constFilePos[tf.Name()]
				}
				if n, ok := evalArrayLenIn(entry.constFset, entry.constPkg, pos, exprStr); ok {
					return n, true
				}
			}
		}
	} else {
		if n, ok := evalArrayLenIn(token.NewFileSet(), nil, token.NoPos, exprStr); ok {
			return n, true
		}
	}
	return 0, false
}

func (p *Package) evalArrayLen(arr *ast.ArrayType, entry *Bindings) int64 {
	if entry != nil && entry.typeInfo != nil {
		if tv, ok := entry.typeInfo.Types[arr]; ok {
			if t, ok := tv.Type.(*types.Array); ok && t.Len() >= 0 {
				return t.Len()
			}
		}
	}
	if n, ok := p.evalBindingArrayLen(arr.Len, entry); ok {
		return n
	}
	var lb bytes.Buffer
	noSourceConf.Fprint(&lb, p.fset, arr.Len)
	fatalf("%s: cannot evaluate array length %q as a non-negative integer constant", p.fset.Position(arr.Len.Pos()), lb.String())
	return 0
}

func (p *Package) parseBinding(bindingPath string, data []byte) *Bindings {
	return p.parseBindingFiles([]bindingSource{{path: bindingPath, data: data}})
}

// recvTypeName returns the name of the receiver base type of method d, or ""
// if it is not a (possibly pointer to a) named type.
func recvTypeName(d *ast.FuncDecl) string {
	if d.Recv == nil || len(d.Recv.List) != 1 {
		return ""
	}
	t := d.Recv.List[0].Type
	if star, ok := t.(*ast.StarExpr); ok {
		t = star.X
	}
	if id, ok := t.(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

// structEmbedsLibrary reports whether ts is a non-generic, non-alias struct
// type that embeds runtime/cgo.Library.
func structEmbedsLibrary(ts *ast.TypeSpec, imp fileImports) bool {
	st, ok := ts.Type.(*ast.StructType)
	if !ok || ts.TypeParams != nil || ts.Assign.IsValid() {
		return false
	}
	for _, fld := range st.Fields.List {
		if len(fld.Names) != 0 {
			continue
		}
		switch ft := fld.Type.(type) {
		case *ast.SelectorExpr:
			if id, ok := ft.X.(*ast.Ident); ok && imp.runtimeCgo[id.Name] && ft.Sel.Name == "Library" {
				return true
			}
		case *ast.Ident:
			if imp.runtimeCgo["."] && ft.Name == "Library" {
				return true
			}
		}
	}
	return false
}

func (p *Package) checkDynloadType(pr cgoPragma, ts *ast.TypeSpec, imp fileImports) {
	if ts == nil {
		fatalf("%s: //cgo:dynload %s: missing declaration of type %s", pr.Pos, pr.GoName, pr.GoName)
	}
	if !structEmbedsLibrary(ts, imp) {
		fatalf("%s: //cgo:dynload type %s must be a struct type that embeds cgo.Library", p.fset.Position(ts.Pos()), pr.GoName)
	}
}

func (p *Package) parseBindingFiles(sources []bindingSource) *Bindings {
	if p.noCallbacks == nil {
		p.noCallbacks = make(map[string]bool)
	}
	if p.noEscapes == nil {
		p.noEscapes = make(map[string]bool)
	}
	entry := &Bindings{
		LdFlags:         append([]string(nil), p.LdFlags...),
		UsedBuiltins:    make(map[string]bool),
		TypeGoToC:       make(map[string]string),
		TypeGoToAnyC:    make(map[string]string),
		ConstCToGo:      make(map[string]string),
		GoOnlyCNames:    make(map[string]bool),
		DynloadCounts:   make(map[string]int),
		UsedFuncPtrs:    make(map[string]bool),
		UsedExportCalls: make(map[string]bool),
		constSources:    sources,
	}

	var pragmas []cgoPragma
	for i := range sources {
		src := &sources[i]
		if src.file == nil {
			fileAST, err := parser.ParseFile(p.fset, src.path, src.data, parser.ParseComments)
			if err != nil {
				fatalf("parsing %s: %s", src.path, err)
			}
			src.file = fileAST
		}
		if entry.Package == "" {
			entry.Package = src.file.Name.Name
		}
		pragmas = append(pragmas, parseCgoPragmas(p.fset, src.file)...)

		// Collect #cgo noescape and nocallback directives, and #cgo LDFLAGS
		// directives (honoring their build constraints), from the import "C"
		// preamble. When cmd/go passes -ldflags, it has already collected
		// (and checked against CGO_LDFLAGS_ALLOW) the package's flags, so the
		// preamble is not consulted for them.
		if !hasImportCFile(src.file) {
			continue
		}
		origPath := src.origPath
		if origPath == "" {
			origPath = src.path
		}
		srcDirVal := filepath.ToSlash(filepath.Dir(origPath))
		forEachImportC(src.file, func(_ *ast.GenDecl, _ *ast.ImportSpec, doc *ast.CommentGroup) {
			if doc == nil {
				return
			}
			for line := range strings.SplitSeq(commentText(doc), "\n") {
				cgoLine := strings.TrimSpace(line)
				if !strings.HasPrefix(cgoLine, "#cgo") || len(cgoLine) < 5 || (cgoLine[4] != ' ' && cgoLine[4] != '\t') {
					continue
				}
				if fields := strings.Fields(cgoLine); len(fields) == 3 {
					switch fields[1] {
					case "nocallback":
						p.noCallbacks[fields[2]] = true
						continue
					case "noescape":
						p.noEscapes[fields[2]] = true
						continue
					}
				}
				if ldflagsSet {
					continue
				}
				verb, raw, matched, ok := parseCgoDirective(cgoLine)
				if !ok || !matched || verb != "LDFLAGS" || raw == "" {
					continue
				}
				fields, err := splitQuoted(raw)
				if err != nil {
					fatalf("%s: bad #cgo LDFLAGS: %q (%s)", src.path, raw, err)
				}
				for i, f := range fields {
					fields[i] = strings.ReplaceAll(f, "${SRCDIR}", srcDirVal)
				}
				entry.LdFlags = append(entry.LdFlags, fields...)
			}
		})
	}

	// Index the package-level declarations that pragmas may name.
	funcDecls := make(map[string]*ast.FuncDecl)
	funcImports := make(map[string]fileImports)
	methodDecls := make(map[string]*ast.FuncDecl) // keyed by "T.M"
	methodImports := make(map[string]fileImports)
	typeSpecs := make(map[string]*ast.TypeSpec)
	typeImports := make(map[string]fileImports)
	constSpecs := make(map[string]*ast.ValueSpec)
	varSpecs := make(map[string]*ast.ValueSpec)
	varImports := make(map[string]fileImports)
	var typeOrder []string
	for _, src := range sources {
		imp := collectFileImports(src.file)
		for _, decl := range src.file.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if d.Recv == nil {
					funcDecls[d.Name.Name] = d
					funcImports[d.Name.Name] = imp
				} else if recv := recvTypeName(d); recv != "" {
					methodDecls[recv+"."+d.Name.Name] = d
					methodImports[recv+"."+d.Name.Name] = imp
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if _, ok := typeSpecs[s.Name.Name]; !ok {
							typeOrder = append(typeOrder, s.Name.Name)
							typeSpecs[s.Name.Name] = s
							typeImports[s.Name.Name] = imp
						}
					case *ast.ValueSpec:
						m := varSpecs
						if d.Tok == token.CONST {
							m = constSpecs
						}
						for _, id := range s.Names {
							m[id.Name] = s
							if d.Tok == token.VAR {
								varImports[id.Name] = imp
							}
						}
					}
				}
			}
		}
	}
	entry.typeSpecs = typeSpecs

	// Check that each pragma names a declaration of the right kind, and
	// that no Go or C. name is bound twice.
	boundGo := make(map[string]bool)
	boundC := make(map[string]bool)
	for _, pr := range pragmas {
		if pr.Kind == "dynload" {
			p.checkDynloadType(pr, typeSpecs[pr.GoName], typeImports[pr.GoName])
			if _, ok := entry.DynloadCounts[pr.GoName]; ok {
				fatalf("%s: %s is declared by more than one //cgo:dynload pragma", pr.Pos, pr.GoName)
			}
			boundGo[pr.GoName] = true
			entry.DynloadCounts[pr.GoName] = 0
			entry.DynloadTypes = append(entry.DynloadTypes, pr.GoName)
		}
	}
	for _, pr := range pragmas {
		if pr.Kind == "dynload" {
			continue
		}
		if pr.RecvType != "" {
			key := pr.RecvType + "." + pr.GoName
			if _, ok := entry.DynloadCounts[pr.RecvType]; !ok {
				fatalf("%s: //cgo:%s %s: %s is not declared with //cgo:dynload", pr.Pos, pr.Kind, key, pr.RecvType)
			}
			if boundGo[key] {
				fatalf("%s: %s is bound by more than one //cgo: pragma", pr.Pos, key)
			}
			boundGo[key] = true
			d := methodDecls[key]
			if d == nil {
				fatalf("%s: //cgo:%s %s: missing declaration of method (*%s).%s", pr.Pos, pr.Kind, key, pr.RecvType, pr.GoName)
			}
			if _, ok := d.Recv.List[0].Type.(*ast.StarExpr); !ok {
				fatalf("%s: //cgo:%s method %s must have a pointer receiver (*%s)", p.fset.Position(d.Pos()), pr.Kind, pr.GoName, pr.RecvType)
			}
			continue
		}
		if boundGo[pr.GoName] {
			fatalf("%s: %s is bound by more than one //cgo: pragma", pr.Pos, pr.GoName)
		}
		boundGo[pr.GoName] = true
		if pr.BindC {
			if pr.CName == "malloc" || (pr.CName != "_CMalloc" && builtinDefs[pr.CName] != "") {
				fatalf("%s: C.%s is a cgo builtin and cannot be bound by a //cgo: pragma", pr.Pos, pr.CName)
			}
			if boundC[pr.CName] {
				fatalf("%s: C.%s is bound by more than one //cgo: pragma", pr.Pos, pr.CName)
			}
			boundC[pr.CName] = true
		} else {
			entry.GoOnlyCNames[pr.CName] = true
		}
		var found bool
		switch pr.Kind {
		case "func":
			_, found = funcDecls[pr.GoName]
		case "type":
			_, found = typeSpecs[pr.GoName]
		case "const":
			_, found = constSpecs[pr.GoName]
			if found && pr.BindC {
				entry.ConstCToGo[pr.CName] = pr.GoName
			}
		case "var":
			_, found = varSpecs[pr.GoName]
		}
		if !found {
			fatalf("%s: //cgo:%s %s: missing declaration of %s %s", pr.Pos, pr.Kind, pr.GoName, pr.Kind, pr.GoName)
		}
	}

	if len(sources) > 0 {
		astFiles := make([]*ast.File, len(sources))
		for i, src := range sources {
			astFiles[i] = src.file
		}
		entry.typeInfo = &types.Info{
			Types:  make(map[ast.Expr]types.TypeAndValue),
			Uses:   make(map[*ast.Ident]types.Object),
			Scopes: make(map[ast.Node]*types.Scope),
		}
		typeConf := types.Config{
			Importer:                 bindingStubImporter{},
			Sizes:                    types.SizesFor("gc", goarch),
			Error:                    func(error) {},
			DisableUnusedImportCheck: true,
		}
		entry.tpkg, _ = typeConf.Check(entry.Package, p.fset, astFiles, entry.typeInfo)
	}

	type rawTypeInfo struct {
		ts    *ast.TypeSpec
		imp   fileImports
		cName string
	}
	rawTypes := make(map[string]rawTypeInfo)
	cToGoType := make(map[string]string)
	var pragmaTypes []string
	for _, pr := range pragmas {
		if pr.Kind != "type" {
			continue
		}
		ts := typeSpecs[pr.GoName]
		if ts.TypeParams != nil {
			fatalf("%s: //cgo:type %s cannot have type parameters", p.fset.Position(ts.Pos()), pr.GoName)
		}
		if st, ok := ts.Type.(*ast.StructType); ok && !structHasHostLayoutOrIncomplete(st, typeImports[pr.GoName]) {
			fatalf("%s: //cgo:type struct %s must embed structs.HostLayout", p.fset.Position(ts.Pos()), pr.GoName)
		}
		entry.TypeGoToAnyC[pr.GoName] = pr.CName
		if pr.BindC {
			entry.TypeGoToC[pr.GoName] = pr.CName
			cToGoType[pr.CName] = pr.GoName
		}
		pragmaTypes = append(pragmaTypes, pr.GoName)
	}
	for _, goName := range typeOrder {
		rawTypes[goName] = rawTypeInfo{ts: typeSpecs[goName], imp: typeImports[goName], cName: entry.TypeGoToC[goName]}
	}

	typeMap := make(map[string]BindingType)
	entry.TypeMap = typeMap
	resolving := make(map[string]bool)
	resolved := make(map[string]bool)

	var resolveType func(goName string)
	var validateUsedTypeExpr func(expr ast.Expr, inPtr bool, imp fileImports)

	validateUsedTypeExpr = func(expr ast.Expr, inPtr bool, imp fileImports) {
		if expr == nil {
			return
		}
		switch e := expr.(type) {
		case *ast.ParenExpr:
			validateUsedTypeExpr(e.X, inPtr, imp)
		case *ast.StarExpr:
			validateUsedTypeExpr(e.X, true, imp)
		case *ast.ArrayType:
			if e.Len != nil {
				p.evalArrayLen(e, entry)
			}
			validateUsedTypeExpr(e.Elt, false, imp)
		case *ast.StructType:
			if e.Fields != nil {
				for _, fld := range e.Fields.List {
					validateUsedTypeExpr(fld.Type, false, imp)
				}
			}
		case *ast.SelectorExpr:
			id, ok := e.X.(*ast.Ident)
			if !ok {
				return
			}
			if (imp.unsafe[id.Name] && e.Sel.Name == "Pointer") ||
				(imp.structs[id.Name] && e.Sel.Name == "HostLayout") ||
				(imp.runtimeCgo[id.Name] && e.Sel.Name == "Incomplete") {
				return
			}
			if id.Name == "C" {
				if _, isPrim := primitiveCtypeToGo("_Ctype_" + e.Sel.Name); isPrim {
					return
				}
				if gName, ok := cToGoType[e.Sel.Name]; ok {
					resolveType(gName)
					return
				}
				if inPtr {
					return
				}
				fatalf("%s: cannot resolve C ABI layout of undefined type C.%s", p.fset.Position(e.Pos()), e.Sel.Name)
			}
			fatalf("%s: cannot resolve C ABI layout of external type %s.%s", p.fset.Position(e.Pos()), id.Name, e.Sel.Name)
		case *ast.Ident:
			switch e.Name {
			case "bool", "byte", "rune",
				"int", "int8", "int16", "int32", "int64",
				"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
				"float32", "float64", "complex64", "complex128",
				"string", "any", "error":
				return
			}
			if (imp.unsafe["."] && e.Name == "Pointer") ||
				(imp.structs["."] && e.Name == "HostLayout") ||
				(imp.runtimeCgo["."] && e.Name == "Incomplete") {
				return
			}
			if _, isPrim := primitiveCtypeToGo(e.Name); isPrim {
				return
			}
			if _, exists := rawTypes[e.Name]; exists {
				resolveType(e.Name)
				return
			}
			fatalf("%s: cannot resolve C ABI layout of undefined type %s", p.fset.Position(e.Pos()), e.Name)
		}
	}

	abiInspector := &ABIConfig{OS: goos, Arch: goarch, PtrSize: targetPtrSize(), CLongSize: targetCLongSize()}
	resolveType = func(goName string) {
		if resolved[goName] || resolving[goName] {
			return
		}
		info, ok := rawTypes[goName]
		if !ok {
			return
		}
		resolving[goName] = true
		validateUsedTypeExpr(info.ts.Type, false, info.imp)
		resolving[goName] = false
		resolved[goName] = true

		defStr := p.cgoASTToGoDefInFile(info.ts.Type, entry, info.imp)
		if targetBT, ok := typeMap[defStr]; ok && targetBT.Def != "" {
			defStr = targetBT.Def
		}
		bt := BindingType{Name: goName, Def: defStr}
		if info.cName != "" {
			bt.Name = "_Ctype_" + info.cName
		}
		// A type that cannot be shared with C has no C ABI layout. The
		// uses of the type report it (see cTypeError), so skip it here.
		if cDeclTypeError(defStr, typeMap) == "" {
			ti := abiInspector.inspectType(defStr, typeMap)
			bt.Size, bt.Align, bt.HasPointers = ti.size, ti.align, ti.hasPointers
		}
		if info.cName != "" {
			typeMap[bt.Name] = bt
		}
		typeMap[goName] = bt
	}

	for _, goName := range pragmaTypes {
		resolveType(goName)
		if why := cDeclTypeError(goName, typeMap); why != "" {
			fatalf("%s: //cgo:type %s cannot be shared with C: %s", p.fset.Position(typeSpecs[goName].Pos()), goName, why)
		}
	}
	entry.resolveUsedTypeInFile = validateUsedTypeExpr

	allocDynSym := func(recvType, goName string) (string, int) {
		dynName := fmt.Sprintf("_dyn%d_%s_%s", len(recvType), recvType, goName)
		for boundGo[dynName] {
			dynName = "_" + dynName
		}
		idx := entry.DynloadCounts[recvType]
		entry.DynloadCounts[recvType] = idx + 1
		return dynName, idx
	}

	for _, pr := range pragmas {
		switch pr.Kind {
		case "var":
			if pr.RecvType != "" {
				key := pr.RecvType + "." + pr.GoName
				d := methodDecls[key]
				pos := p.fset.Position(d.Pos())
				if d.Body != nil {
					fatalf("%s: //cgo:var method %s cannot have a body", pos, pr.GoName)
				}
				var star *ast.StarExpr
				if d.Type.Params.NumFields() == 0 && d.Type.Results.NumFields() == 1 {
					star, _ = d.Type.Results.List[0].Type.(*ast.StarExpr)
				}
				if star == nil || d.Type.TypeParams != nil {
					fatalf("%s: //cgo:var method %s must have the form func (*%s) %s() *T", pos, pr.GoName, pr.RecvType, pr.GoName)
				}
				mImp := methodImports[key]
				validateUsedTypeExpr(star.X, true, mImp)
				vType := p.cgoASTToGoDefInFile(star.X, entry, mImp)
				if why := cDeclTypeError(vType, entry.TypeMap); why != "" {
					fatalf("%s: //cgo:var method %s: variable type %s cannot be shared with C: %s", p.fset.Position(star.X.Pos()), pr.GoName, vType, why)
				}
				dynName, symIdx := allocDynSym(pr.RecvType, pr.GoName)
				entry.DynloadVars = append(entry.DynloadVars, BindingVar{
					Name:     dynName,
					GoName:   pr.GoName,
					CName:    pr.CName,
					Type:     vType,
					RecvType: pr.RecvType,
					SymIndex: symIdx,
				})
				continue
			}
			if token.IsExported(pr.GoName) {
				fatalf("%s: //cgo:var %s: Go variable name must be unexported", pr.Pos, pr.GoName)
			}
			vs := varSpecs[pr.GoName]
			pos := p.fset.Position(vs.Pos())
			if len(vs.Names) != 1 {
				fatalf("%s: //cgo:var %s: the variable declaration must declare only %s", pos, pr.GoName, pr.GoName)
			}
			if vs.Type == nil {
				fatalf("%s: //cgo:var variable %s must specify an explicit type", pos, pr.GoName)
			}
			if len(vs.Values) > 0 {
				fatalf("%s: //cgo:var variable %s cannot have an initializer", pos, pr.GoName)
			}
			vImp := varImports[pr.GoName]
			validateUsedTypeExpr(vs.Type, false, vImp)
			vType := p.cgoASTToGoDefInFile(vs.Type, entry, vImp)
			if why := cDeclTypeError(vType, entry.TypeMap); why != "" {
				fatalf("%s: //cgo:var variable %s of type %s cannot be shared with C: %s", p.fset.Position(vs.Type.Pos()), pr.GoName, vType, why)
			}
			entry.Vars = append(entry.Vars, BindingVar{
				Name:    pr.GoName,
				GoName:  pr.GoName,
				CName:   pr.CName,
				BindC:   pr.BindC,
				Type:    vType,
				Library: pr.Library,
				LibC:    pr.LibC,
				Static:  pr.Static,
			})
		case "func":
			d := funcDecls[pr.GoName]
			if pr.RecvType != "" {
				d = methodDecls[pr.RecvType+"."+pr.GoName]
			}
			pos := p.fset.Position(d.Pos())
			if d.Body != nil {
				fatalf("%s: //cgo:func function %s cannot have a body", pos, pr.GoName)
			}
			fnName := pr.GoName
			if builtinDefs[fnName] != "" || fnName == "_cgo_malloc" {
				fnName = "_go_" + fnName
				for boundGo[fnName] {
					fnName = "_" + fnName
				}
			}
			fn := BindingFunc{
				Name:    fnName,
				GoName:  pr.GoName,
				CName:   pr.CName,
				BindC:   pr.BindC,
				Library: pr.Library,
				LibC:    pr.LibC,
				Static:  pr.Static,
			}
			fImp := funcImports[pr.GoName]
			if pr.RecvType != "" {
				fImp = methodImports[pr.RecvType+"."+pr.GoName]
				fn.Name, fn.SymIndex = allocDynSym(pr.RecvType, pr.GoName)
				fn.Indirect = true
				fn.RecvType = pr.RecvType
			}
			if d.Type.Params != nil {
				for _, field := range d.Type.Params.List {
					validateUsedTypeExpr(field.Type, false, fImp)
					pType := p.cgoASTToGoDefInFile(field.Type, entry, fImp)
					if why := cTypeError(pType, entry.TypeMap); why != "" {
						fatalf("%s: //cgo:func %s: parameter type %s cannot be passed to C: %s", p.fset.Position(field.Type.Pos()), pr.GoName, pType, why)
					}
					for range max(1, len(field.Names)) {
						fn.Params = append(fn.Params, pType)
					}
				}
			}
			if d.Type.Results != nil && len(d.Type.Results.List) > 0 {
				res := d.Type.Results.List[0].Type
				validateUsedTypeExpr(res, false, fImp)
				fn.Ret = p.cgoASTToGoDefInFile(res, entry, fImp)
				if why := cTypeError(fn.Ret, entry.TypeMap); why != "" {
					fatalf("%s: //cgo:func %s: result type %s cannot be returned from C: %s", p.fset.Position(res.Pos()), pr.GoName, fn.Ret, why)
				}
			}
			entry.Funcs = append(entry.Funcs, fn)
		}
	}
	return entry
}

// processBinding consumes the //cgo: pragmas in all of the package's Go files,
// rewrites the Go source files, and generates Go wrappers and assembly trampolines
// without requiring a C compiler.
func (p *Package) processBinding(goFiles []string) {
	type parsedFile struct {
		srcfile    string
		data       []byte
		parsed     *ast.File
		hasImportC bool
	}

	parsedFiles := make([]parsedFile, len(goFiles))
	sources := make([]bindingSource, len(goFiles))
	q := par.NewQueue(runtime.GOMAXPROCS(0))
	for i, srcfile := range goFiles {
		q.Add(func() {
			origPath := srcfile
			if *srcDir != "" && !filepath.IsAbs(srcfile) {
				origPath = filepath.Join(*srcDir, srcfile)
			}
			if aname, err := filepath.Abs(origPath); err == nil {
				origPath = aname
			}
			b, err := os.ReadFile(origPath)
			if err != nil {
				fatalf("%s", err)
			}
			trimmedPath, _ := objabi.ApplyRewrites(origPath, *trimpath)
			if strings.ContainsAny(trimmedPath, "\r\n") {
				fatalf("input path contains newline character: %q", trimmedPath)
			}
			parsed, err := parser.ParseFile(p.fset, trimmedPath, b, parser.ParseComments)
			if err != nil {
				fatalf("%s", err)
			}
			hasImportC := hasImportCFile(parsed)
			parsedFiles[i] = parsedFile{
				srcfile:    trimmedPath,
				data:       b,
				parsed:     parsed,
				hasImportC: hasImportC,
			}
			sources[i] = bindingSource{
				path:     trimmedPath,
				origPath: origPath,
				data:     b,
				file:     parsed,
			}
		})
	}
	<-q.Idle()

	h := hash.New32()
	io.WriteString(h, *importPath)
	var rawExports []rawBindingExport
	expMap := make(map[string]bool)
	for _, pf := range parsedFiles {
		h.Write(pf.data)
		if pf.hasImportC {
			rawExports = append(rawExports, p.collectBindingExports(pf.parsed, expMap)...)
		}
	}
	if nerrors > 0 {
		os.Exit(2)
	}

	entry := p.parseBindingFiles(sources)
	p.PackageName = entry.Package
	p.LdFlags = entry.LdFlags
	expFuncs := p.resolveBindingExports(entry, rawExports)
	if nerrors > 0 {
		os.Exit(2)
	}

	// knownBoundNames are names that must not be referred to as C.name:
	// Go names bound by pragmas, and C names bound without the C. prefix.
	// References to them are rewritten to undefined identifiers.
	knownBoundNames := maps.Clone(entry.GoOnlyCNames)
	for _, goConst := range entry.ConstCToGo {
		knownBoundNames[goConst] = true
	}
	for goType := range entry.TypeGoToC {
		knownBoundNames[goType] = true
	}

	// varMap and cFuncMap map C names bound with the C.cName form.
	varMap := make(map[string]BindingVar)
	boundGoVars := make(map[string]BindingVar)
	for _, v := range entry.Vars {
		if v.BindC {
			varMap[v.CName] = v
		}
		boundGoVars[v.GoName] = v
		knownBoundNames[v.GoName] = true
	}

	cFuncMap := make(map[string]BindingFunc)
	goFuncMap := make(map[string]BindingFunc)
	boundGoFuncs := make(map[string]bool)
	boundMethods := make(map[string]bool) // "LibT.GoName" for //cgo:dynload methods
	boundCFuncs := make(map[string]bool)
	for _, v := range entry.DynloadVars {
		boundMethods[v.RecvType+"."+v.GoName] = true
	}
	for _, fn := range entry.Funcs {
		if fn.RecvType != "" {
			boundMethods[fn.RecvType+"."+fn.GoName] = true
			boundCFuncs[fn.CName] = true
			continue
		}
		if fn.BindC {
			cFuncMap[fn.CName] = fn
		}
		boundGoFuncs[fn.GoName] = true
		goFuncMap[fn.GoName] = fn
		boundCFuncs[fn.CName] = true
		knownBoundNames[fn.Name] = true
		knownBoundNames[fn.GoName] = true
	}

	for _, cName := range slices.Sorted(maps.Keys(p.noEscapes)) {
		if !boundCFuncs[cName] {
			error_(token.NoPos, "#cgo noescape %s: no matched C function", cName)
		}
	}
	for _, cName := range slices.Sorted(maps.Keys(p.noCallbacks)) {
		if !boundCFuncs[cName] {
			error_(token.NoPos, "#cgo nocallback %s: no matched C function", cName)
		}
	}
	if nerrors > 0 {
		os.Exit(2)
	}
	for i := range entry.Funcs {
		fn := &entry.Funcs[i]
		fn.MayCallback = !p.noCallbacks[fn.CName]
	}

	expFuncMap := make(map[string]BindingFunc, len(expFuncs))
	for _, ef := range expFuncs {
		expFuncMap[ef.Name] = exportBindingFunc(ef)
	}

	rewriter := &bindingRewriter{
		p:               p,
		tpkg:            entry.tpkg,
		typeInfo:        entry.typeInfo,
		entry:           entry,
		varMap:          varMap,
		boundGoVars:     boundGoVars,
		cFuncMap:        cFuncMap,
		goFuncMap:       goFuncMap,
		knownBoundNames: knownBoundNames,
		expFuncMap:      expFuncMap,
	}

	outGoFiles := make([]string, len(parsedFiles))
	rewriteQ := par.NewQueue(runtime.GOMAXPROCS(0))
	for i, pf := range parsedFiles {
		rewriteQ.Add(func() {
			trimmedSrc := pf.srcfile
			b := pf.data
			parsed := pf.parsed
			tf := p.fset.File(parsed.Package)
			editBuf := edit.NewBuffer(b)

			// Strip //cgo: pragmas and bound func/var declarations;
			// _cgo_gotypes.go supplies their wrappers and C symbol imports.
			var stripped [][2]token.Pos
			stripNode := func(pos, end token.Pos) {
				editBuf.Replace(tf.Offset(pos), tf.Offset(end), "")
				stripped = append(stripped, [2]token.Pos{pos, end})
			}
			isBoundVar := func(spec ast.Spec) bool {
				vs, ok := spec.(*ast.ValueSpec)
				if !ok || len(vs.Names) != 1 {
					return false
				}
				_, ok = boundGoVars[vs.Names[0].Name]
				return ok
			}
			parsed.Decls = slices.DeleteFunc(parsed.Decls, func(decl ast.Decl) bool {
				switch d := decl.(type) {
				case *ast.FuncDecl:
					if d.Body == nil && (d.Recv == nil && boundGoFuncs[d.Name.Name] || d.Recv != nil && boundMethods[recvTypeName(d)+"."+d.Name.Name]) {
						stripNode(d.Pos(), d.End())
						return true
					}
				case *ast.GenDecl:
					if d.Tok == token.VAR {
						if len(d.Specs) > 0 && !slices.ContainsFunc(d.Specs, func(spec ast.Spec) bool { return !isBoundVar(spec) }) {
							stripNode(d.Pos(), d.End())
							return true
						}
						d.Specs = slices.DeleteFunc(d.Specs, func(spec ast.Spec) bool {
							if isBoundVar(spec) {
								stripNode(spec.Pos(), spec.End())
								return true
							}
							return false
						})
					}
				}
				return false
			})

			for _, cg := range parsed.Comments {
				for _, c := range cg.List {
					if isCgoPragmaComment(c.Text) && !slices.ContainsFunc(stripped, func(r [2]token.Pos) bool {
						return c.Pos() < r[1] && c.End() > r[0]
					}) {
						editBuf.Replace(tf.Offset(c.Pos()), tf.Offset(c.End()), "")
					}
				}
			}

			imp := collectFileImports(parsed)
			hasUnsafeImport := imp.hasUnsafe

			if len(entry.DynloadTypes) > 0 {
				// Make registering the //cgo:dynload types the first
				// declaration in the file (see writeGoTypes). Put it right
				// after the imports (or the package clause), on the same
				// line, so that line numbers do not change and the first
				// declaration keeps its doc comment and directives.
				// Skip an import "C" declaration that is removed below.
				insertAt := tf.Offset(parsed.Name.End())
				for _, d := range parsed.Decls {
					gd, ok := d.(*ast.GenDecl)
					if !ok || gd.Tok != token.IMPORT {
						break
					}
					if hasUnsafeImport && len(gd.Specs) == 1 && gd.Specs[0].(*ast.ImportSpec).Path.Value == `"C"` {
						continue
					}
					insertAt = tf.Offset(gd.End())
				}
				editBuf.Insert(insertAt, "; "+dynloadRegisterDecl)
			}

			if pf.hasImportC {
				forEachImportC(parsed, func(d *ast.GenDecl, s *ast.ImportSpec, _ *ast.CommentGroup) {
					if hasUnsafeImport {
						if len(d.Specs) == 1 {
							editBuf.Replace(tf.Offset(d.Pos()), tf.Offset(d.End()), "")
						} else {
							editBuf.Replace(tf.Offset(s.Pos()), tf.Offset(s.End()), "")
						}
					} else {
						editBuf.Replace(tf.Offset(s.Path.Pos()), tf.Offset(s.Path.End()), `_ "unsafe"`)
						hasUnsafeImport = true
					}
				})
			}
			if len(stripped) > 0 || pf.hasImportC {
				var keepImports strings.Builder
				for _, s := range parsed.Imports {
					alias := ""
					if s.Name != nil {
						alias = s.Name.Name
					}
					if alias == "_" || alias == "." {
						continue
					}
					switch s.Path.Value {
					case `"unsafe"`:
						fmt.Fprintf(&keepImports, "\nvar _ %s.Pointer\n", cmp.Or(alias, "unsafe"))
					case `"structs"`:
						fmt.Fprintf(&keepImports, "\nvar _ %s.HostLayout\n", cmp.Or(alias, "structs"))
					case `"runtime/cgo"`:
						fmt.Fprintf(&keepImports, "\ntype _ %s.Incomplete\n", cmp.Or(alias, "cgo"))
					}
				}
				if keepImports.Len() > 0 {
					editBuf.Insert(len(b), keepImports.String())
				}
			}

			rewriter.rewriteFile(editBuf, tf, parsed, b, pf.hasImportC)

			base := strings.TrimSuffix(filepath.Base(trimmedSrc), ".go")
			outName := base + ".cgobinding.go"
			fgo1 := creat(outName)
			fmt.Fprintf(fgo1, "// Code generated by cmd/cgo; DO NOT EDIT.\n\n")
			fmt.Fprintf(fgo1, "//line %s:1:1\n", filepath.ToSlash(trimmedSrc))
			fgo1.Write(editBuf.Bytes())
			fgo1.Close()
			outGoFiles[i] = outName
		})
	}
	<-rewriteQ.Idle()
	p.GoFiles = append(p.GoFiles, outGoFiles...)

	origCPrefix := cPrefix
	cPrefix = fmt.Sprintf("_%x", h.Sum(nil)[0:6])
	p.writeGoTypes(entry, expFuncs)
	p.writeTrampolines(entry, expFuncs)
	cPrefix = origCPrefix

	if *exportHeader != "" && len(expFuncs) > 0 {
		p.writeBindingExportHeader(*exportHeader, entry, expFuncs, sources)
	}

	if len(p.LdFlags) > 0 {
		fflg := creat("_cgo_flags")
		for _, flag := range p.LdFlags {
			fmt.Fprintf(fflg, "_CGO_LDFLAGS_=%s\n", flag)
		}
		fflg.Close()
	}
}

type bindingStubImporter struct{}

func (bindingStubImporter) Import(path string) (*types.Package, error) {
	if path == "unsafe" {
		return types.Unsafe, nil
	}
	name := path
	if i := strings.LastIndex(path, "/"); i >= 0 {
		name = path[i+1:]
	}
	pkg := types.NewPackage(path, name)
	pkg.MarkComplete()
	return pkg, nil
}

type bindingRewriter struct {
	p               *Package
	tpkg            *types.Package
	typeInfo        *types.Info
	entry           *Bindings
	varMap          map[string]BindingVar
	boundGoVars     map[string]BindingVar
	cFuncMap        map[string]BindingFunc
	goFuncMap       map[string]BindingFunc
	knownBoundNames map[string]bool
	expFuncMap      map[string]BindingFunc
}

func (r *bindingRewriter) rewriteFile(editBuf *edit.Buffer, tf *token.File, file *ast.File, src []byte, hasImportC bool) {
	isImportC := func(id *ast.Ident) bool {
		obj := r.typeInfo.Uses[id]
		if obj == nil && r.tpkg != nil {
			if scope := r.tpkg.Scope().Innermost(id.Pos()); scope != nil {
				_, obj = scope.LookupParent(id.Name, id.Pos())
			}
		}
		pkgName, ok := obj.(*types.PkgName)
		return ok && pkgName.Imported().Path() == "C"
	}
	// isBoundGoFunc reports whether id refers to the package-level
	// declaration of a bound function, rather than a local name.
	isBoundGoFunc := func(id *ast.Ident) bool {
		obj := r.typeInfo.Uses[id]
		if obj == nil && r.tpkg != nil {
			if scope := r.tpkg.Scope().Innermost(id.Pos()); scope != nil {
				_, obj = scope.LookupParent(id.Name, id.Pos())
			}
		}
		fn, ok := obj.(*types.Func)
		return ok && r.tpkg != nil && fn.Parent() == r.tpkg.Scope()
	}

	// Rewrite references to C names, and calls of bound functions that
	// pass pointers, which get the same cgocheck pointer checks as calls
	// in traditional cgo.
	edits := &bindingEdits{src: src}
	boundVarObjs := make(map[types.Object]BindingVar, len(r.boundGoVars))
	if r.tpkg != nil {
		for name, v := range r.boundGoVars {
			if obj := r.tpkg.Scope().Lookup(name); obj != nil {
				boundVarObjs[obj] = v
			}
		}
	}
	// Record selector expressions in call position (CallExpr.Fun, ignoring
	// enclosing parentheses) so C.<func> and C.<export> are rewritten as calls
	// there and as non-assignable C function pointers elsewhere.
	callFuns := make(map[*ast.SelectorExpr]bool)
	if hasImportC {
		ast.Inspect(file, func(node ast.Node) bool {
			if call, ok := node.(*ast.CallExpr); ok {
				if sel, ok := unparen(call.Fun).(*ast.SelectorExpr); ok {
					callFuns[sel] = true
				}
			}
			return true
		})
	}
	ast.Inspect(file, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.SelectorExpr:
			if !hasImportC {
				return true
			}
			id, ok := n.X.(*ast.Ident)
			if !ok || id.Name != "C" || !isImportC(id) {
				return true
			}
			name := n.Sel.Name
			if name == "malloc" {
				name = "_CMalloc"
			}
			var repl string
			if goConst, isConst := r.entry.ConstCToGo[name]; isConst {
				repl = goConst
			} else if v, isVar := r.varMap[name]; isVar {
				repl = cVarExpr(v)
			} else if ef, isExp := r.expFuncMap[name]; isExp {
				if callFuns[n] {
					r.entry.mu.Lock()
					r.entry.UsedExportCalls[name] = true
					r.entry.mu.Unlock()
					repl = bindingFuncEntry(ef, r.entry.TypeMap)
				} else {
					repl = "_Cgo_ptr(_Cfpvar_fp_" + name + ")"
				}
			} else if builtinDefs[name] != "" {
				r.entry.mu.Lock()
				r.entry.UsedBuiltins[name] = true
				r.entry.mu.Unlock()
				repl = "_Cfunc_" + name
			} else if fn, isFunc := r.cFuncMap[name]; isFunc {
				if callFuns[n] {
					repl = bindingFuncEntry(fn, r.entry.TypeMap)
				} else {
					r.entry.mu.Lock()
					r.entry.UsedFuncPtrs[name] = true
					r.entry.mu.Unlock()
					repl = "_Cgo_ptr(_Cfpvar_fp_" + name + ")"
				}
			} else if _, isType := r.entry.TypeMap["_Ctype_"+name]; isType {
				repl = "_Ctype_" + name
			} else if goType, isPrim := primitiveCtypeToGo("_Ctype_" + name); isPrim {
				repl = goType
			} else if r.knownBoundNames[name] {
				repl = "_Cundef_" + name
			} else {
				repl = "_Cfunc_" + name
			}
			edits.replace(tf.Offset(n.Pos()), tf.Offset(n.End()), repl)
			return false
		case *ast.Ident:
			if v, ok := boundVarObjs[r.typeInfo.Uses[n]]; ok {
				edits.replace(tf.Offset(n.Pos()), tf.Offset(n.End()), cVarExpr(v))
			}
			return false
		}
		return true
	})
	type callSite struct {
		call *ast.CallExpr
		fn   BindingFunc
	}
	var calls []callSite
	deferred := make(map[*ast.CallExpr]bool)
	unused := make(map[*ast.CallExpr]bool)
	ast.Inspect(file, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.DeferStmt:
			deferred[n.Call] = true
		case *ast.GoStmt:
			deferred[n.Call] = true
		case *ast.ExprStmt:
			if call, ok := n.X.(*ast.CallExpr); ok {
				unused[call] = true
			}
		case *ast.CallExpr:
			switch fun := unparen(n.Fun).(type) {
			case *ast.SelectorExpr:
				if id, ok := fun.X.(*ast.Ident); ok && hasImportC && id.Name == "C" && isImportC(id) {
					if fn, ok := r.cFuncMap[fun.Sel.Name]; ok {
						calls = append(calls, callSite{n, fn})
					} else if ef, ok := r.expFuncMap[fun.Sel.Name]; ok {
						calls = append(calls, callSite{n, ef})
					}
				}
			case *ast.Ident:
				if fn, ok := r.goFuncMap[fun.Name]; ok && isBoundGoFunc(fun) {
					calls = append(calls, callSite{n, fn})
				}
			}
		}
		return true
	})
	// Rewrite inner calls first so that outer calls include them.
	for _, c := range slices.Backward(calls) {
		if repl := r.p.rewriteBindingCall(edits, tf.Offset, c.call, c.fn, r.entry.TypeMap, deferred[c.call], unused[c.call]); repl != "" {
			edits.replace(tf.Offset(c.call.Pos()), tf.Offset(c.call.End()), repl)
		}
	}
	edits.apply(editBuf)
}

func (p *Package) cgoASTToGoDefInFile(expr ast.Expr, entry *Bindings, imp fileImports) string {
	switch e := expr.(type) {
	case *ast.ParenExpr:
		return p.cgoASTToGoDefInFile(e.X, entry, imp)
	case *ast.Ident:
		if imp.structs["."] && e.Name == "HostLayout" {
			return "structs.HostLayout"
		}
		if imp.runtimeCgo["."] && e.Name == "Incomplete" {
			return "_cgopackage.Incomplete"
		}
		if imp.unsafe["."] && e.Name == "Pointer" {
			return "unsafe.Pointer"
		}
		return e.Name
	case *ast.SelectorExpr:
		if id, ok := e.X.(*ast.Ident); ok {
			if imp.structs[id.Name] && e.Sel.Name == "HostLayout" {
				return "structs.HostLayout"
			}
			if imp.unsafe[id.Name] && e.Sel.Name == "Pointer" {
				return "unsafe.Pointer"
			}
			if imp.runtimeCgo[id.Name] && e.Sel.Name == "Incomplete" {
				return "_cgopackage.Incomplete"
			}
			if id.Name == "C" {
				if goType, ok := primitiveCtypeToGo("_Ctype_" + e.Sel.Name); ok {
					return goType
				}
				return "_Ctype_" + e.Sel.Name
			}
		}
		return p.cgoASTToGoDefInFile(e.X, entry, imp) + "." + e.Sel.Name
	case *ast.StarExpr:
		return "*" + p.cgoASTToGoDefInFile(e.X, entry, imp)
	case *ast.ArrayType:
		lenStr := ""
		if e.Len != nil {
			lenStr = strconv.FormatInt(p.evalArrayLen(e, entry), 10)
		}
		return "[" + lenStr + "]" + p.cgoASTToGoDefInFile(e.Elt, entry, imp)
	case *ast.StructType:
		var buf bytes.Buffer
		buf.WriteString("struct {\n")
		for _, fld := range e.Fields.List {
			typeStr := p.cgoASTToGoDefInFile(fld.Type, entry, imp)
			var names []string
			for _, id := range fld.Names {
				names = append(names, id.Name)
			}
			nameStr := strings.Join(names, ", ")
			if nameStr == "" {
				fmt.Fprintf(&buf, "\t%s\n", typeStr)
			} else {
				fmt.Fprintf(&buf, "\t%s %s\n", nameStr, typeStr)
			}
		}
		buf.WriteString("}")
		return buf.String()
	default:
		var buf bytes.Buffer
		noSourceConf.Fprint(&buf, p.fset, expr)
		return buf.String()
	}
}

// dynloadRegisterDecl is the declaration that registers the symbols of
// the //cgo:dynload types of a package during package initialization.
const dynloadRegisterDecl = "var _ = _cgo_register_libraries()"

const bindingRuntimeProlog = `func _Cgo_ptr(ptr unsafe.Pointer) unsafe.Pointer { return ptr }

//go:linkname _cgo_runtime_cgocall runtime.cgocall
func _cgo_runtime_cgocall(fn unsafe.Pointer, arg uintptr) int32

//go:linkname _Cgo_always_false runtime.cgoAlwaysFalse
var _Cgo_always_false bool

//go:linkname _Cgo_use runtime.cgoUse
func _Cgo_use(interface{})

//go:linkname _Cgo_keepalive runtime.cgoKeepAlive
//go:noescape
func _Cgo_keepalive(interface{})

//go:linkname _cgoCheckPointer runtime.cgoCheckPointer
//go:noescape
func _cgoCheckPointer(interface{}, interface{})

//go:linkname _Cgo_no_callback runtime.cgoNoCallback
func _Cgo_no_callback(bool)

`

const bindingCMallocProlog = `//go:linkname _cgo_runtime_throw runtime.throw
func _cgo_runtime_throw(string)

//go:linkname _cgo_trampoline__cgo_malloc _cgo%s_trampoline__cgo_malloc
var _cgo_trampoline__cgo_malloc byte

type _cgo_arg__cgo_malloc struct {
	p0  _Ctype_size_t
	ret unsafe.Pointer
}

func _cgo_cmalloc(n uint64) unsafe.Pointer {
	if n == 0 {
		n = 1
	}
	args := _cgo_arg__cgo_malloc{p0: _Ctype_size_t(n)}
	_cgo_runtime_cgocall(unsafe.Pointer(&_cgo_trampoline__cgo_malloc), uintptr(unsafe.Pointer(&args)))
	if _Cgo_always_false {
		_Cgo_keepalive(&args)
	}
	if args.ret == nil {
		_cgo_runtime_throw("runtime: C malloc failed")
	}
	return args.ret
}

`

// cTypeError reports why values of the Go type expression t cannot be passed
// to or returned from C, or "" if they can. Go strings, slices, maps,
// channels, functions, interfaces, and instantiated generic types have no C
// equivalent, so they (and pointers to, arrays of, and structs containing
// them) are rejected rather than passed in their Go runtime representation.
// Every struct type reachable from t, including through pointers, must
// embed structs.HostLayout so that its Go layout matches C's, unless it is
// an empty struct or an incomplete (opaque) C type embedding cgo.Incomplete.
// Incomplete types may only be used behind pointers. Named types are
// resolved through types, which maps a type name to its definition.
func cTypeError(t string, types map[string]BindingType) string {
	return cTypeErrorIn(t, types, make(map[string]bool), true)
}

// cDeclTypeError is like cTypeError, but for the type of a declaration
// shared with C, such as a //cgo:type, rather than of a value passed by
// value, so the type itself may be incomplete.
func cDeclTypeError(t string, types map[string]BindingType) string {
	return cTypeErrorIn(t, types, make(map[string]bool), false)
}

// cTypeErrorIn is like cTypeError. If byValue is false, t itself is not
// passed by value, so it may be an incomplete type.
func cTypeErrorIn(t string, types map[string]BindingType, seen map[string]bool, byValue bool) string {
	t = strings.TrimSpace(t)
	if strings.HasPrefix(t, "...") {
		return "Go variadic parameters are not supported"
	}
	expr, err := parser.ParseExpr(t)
	if err != nil {
		// Leave reporting malformed types to the ABI layout code.
		return ""
	}
	return cTypeExprError(expr, types, seen, byValue)
}

func cTypeExprError(expr ast.Expr, types map[string]BindingType, seen map[string]bool, byValue bool) string {
	switch e := expr.(type) {
	case *ast.ParenExpr:
		return cTypeExprError(e.X, types, seen, byValue)
	case *ast.StarExpr:
		return cTypeExprError(e.X, types, seen, false)
	case *ast.ArrayType:
		if e.Len == nil {
			return "it contains a Go slice"
		}
		return cTypeExprError(e.Elt, types, seen, byValue)
	case *ast.Ellipsis:
		return "Go variadic parameters are not supported"
	case *ast.MapType:
		return "it contains a Go map"
	case *ast.ChanType:
		return "it contains a Go channel"
	case *ast.FuncType:
		return "it contains a Go func value"
	case *ast.InterfaceType:
		return "it contains a Go interface"
	case *ast.IndexExpr, *ast.IndexListExpr:
		return "it contains an instantiated generic type"
	case *ast.StructType:
		if structIsIncomplete(e) {
			if byValue {
				return "it contains an incomplete C type by value"
			}
			return ""
		}
		for _, field := range e.Fields.List {
			if why := cTypeExprError(field.Type, types, seen, true); why != "" {
				return why
			}
		}
		if len(e.Fields.List) > 0 && !slices.ContainsFunc(e.Fields.List, func(f *ast.Field) bool { return isHostLayoutExpr(f.Type) }) {
			return "it contains a struct type that does not embed structs.HostLayout"
		}
		return ""
	case *ast.Ident:
		switch e.Name {
		case "string":
			return "it contains a Go string"
		case "any", "error":
			return "it contains a Go interface"
		}
		return cTypeNameError(e.Name, types, seen, byValue)
	case *ast.SelectorExpr:
		if x, ok := e.X.(*ast.Ident); ok {
			return cTypeNameError(x.Name+"."+e.Sel.Name, types, seen, byValue)
		}
	}
	return ""
}

func cTypeNameError(name string, types map[string]BindingType, seen map[string]bool, byValue bool) string {
	key := name
	if byValue {
		key += " (by value)"
	}
	if seen[key] {
		return ""
	}
	seen[key] = true
	if bt, ok := types[name]; ok && bt.Def != "" {
		return cTypeErrorIn(bt.Def, types, seen, byValue)
	}
	return ""
}

// structIsIncomplete reports whether st is an incomplete C type, which
// embeds cgo.Incomplete (spelled _cgopackage.Incomplete in Go type
// definitions generated for bindings).
func structIsIncomplete(st *ast.StructType) bool {
	return slices.ContainsFunc(st.Fields.List, func(f *ast.Field) bool {
		sel, ok := f.Type.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		id, ok := sel.X.(*ast.Ident)
		return ok && (id.Name == "_cgopackage" || id.Name == "cgo") && sel.Sel.Name == "Incomplete"
	})
}

// bindingEdits is a set of non-overlapping replacements in a source file.
// A replacement that covers others absorbs them, so a rewritten call may
// include the rewritten text of its arguments.
type bindingEdits struct {
	src   []byte
	edits []bindingEdit
}

type bindingEdit struct {
	start, end int
	text       string
}

// replace replaces src[start:end] with text, which must already include
// the effect of any earlier replacements within src[start:end].
func (e *bindingEdits) replace(start, end int, text string) {
	e.edits = slices.DeleteFunc(e.edits, func(x bindingEdit) bool {
		return start <= x.start && x.end <= end
	})
	e.edits = append(e.edits, bindingEdit{start, end, text})
}

// text returns src[start:end] with the replacements within it applied.
func (e *bindingEdits) text(start, end int) string {
	var inside []bindingEdit
	for _, x := range e.edits {
		if start <= x.start && x.end <= end {
			inside = append(inside, x)
		}
	}
	slices.SortFunc(inside, func(a, b bindingEdit) int { return cmp.Compare(a.start, b.start) })
	var sb strings.Builder
	pos := start
	for _, x := range inside {
		sb.Write(e.src[pos:x.start])
		sb.WriteString(x.text)
		pos = x.end
	}
	sb.Write(e.src[pos:end])
	return sb.String()
}

func (e *bindingEdits) apply(buf *edit.Buffer) {
	for _, x := range e.edits {
		buf.Replace(x.start, x.end, x.text)
	}
}

// bindingPointerChecks reports, for each parameter of fn, whether an
// argument passed for it needs a cgocheck pointer check.
func bindingPointerChecks(fn BindingFunc, types map[string]BindingType) (checks []bool, any bool) {
	checks = make([]bool, len(fn.Params))
	for i, pt := range fn.Params {
		checks[i] = bindingHasPointer(pt, types, true)
		any = any || checks[i]
	}
	return checks, any
}

// bindingFuncEntry returns the name of the Go function to call for fn
// when its arguments have not been checked by the caller: the checking
// wrapper _Cgocheck_<name> if any argument might need a pointer check, or
// the unchecked _Cfunc_<name> otherwise.
func bindingFuncEntry(fn BindingFunc, types map[string]BindingType) string {
	if _, any := bindingPointerChecks(fn, types); any {
		return "_Cgocheck_" + fn.Name
	}
	return "_Cfunc_" + fn.Name
}

// bindingHasPointer is like hasPointer, for a Go type expression t in the
// signature of a bound function. If top is true, it reports whether t is or
// contains a pointer that might point to a Go pointer, so that values of
// type t passed to C need a cgocheck pointer check; if top is false, it
// reports whether t is or contains a pointer. Named types are resolved
// through types; types that cannot be resolved are assumed to contain
// pointers.
func bindingHasPointer(t string, types map[string]BindingType, top bool) bool {
	expr, err := parser.ParseExpr(t)
	if err != nil {
		return true
	}
	return bindingHasPointerExpr(expr, types, top, make(map[string]bool))
}

func bindingHasPointerExpr(expr ast.Expr, types map[string]BindingType, top bool, seen map[string]bool) bool {
	switch t := expr.(type) {
	case *ast.ParenExpr:
		return bindingHasPointerExpr(t.X, types, top, seen)
	case *ast.ArrayType:
		if t.Len == nil {
			return true
		}
		return bindingHasPointerExpr(t.Elt, types, top, seen)
	case *ast.StructType:
		return slices.ContainsFunc(t.Fields.List, func(f *ast.Field) bool {
			return bindingHasPointerExpr(f.Type, types, top, seen)
		})
	case *ast.StarExpr:
		if !top {
			return true
		}
		return bindingHasPointerExpr(t.X, types, false, seen)
	case *ast.Ident:
		switch t.Name {
		case "bool", "byte", "rune", "uintptr",
			"int", "int8", "int16", "int32", "int64",
			"uint", "uint8", "uint16", "uint32", "uint64",
			"float32", "float64", "complex64", "complex128":
			return false
		}
		return bindingHasPointerName(t.Name, types, top, seen)
	case *ast.SelectorExpr:
		if x, ok := t.X.(*ast.Ident); ok {
			switch {
			case x.Name == "unsafe" && t.Sel.Name == "Pointer":
				return true
			case x.Name == "structs" && t.Sel.Name == "HostLayout":
				return false
			}
			return bindingHasPointerName(x.Name+"."+t.Sel.Name, types, top, seen)
		}
	}
	return true
}

func bindingHasPointerName(name string, types map[string]BindingType, top bool, seen map[string]bool) bool {
	bt, ok := types[name]
	if !ok || bt.Def == "" {
		return true
	}
	key := fmt.Sprintf("%s/%v", name, top)
	if seen[key] {
		return false
	}
	seen[key] = true
	expr, err := parser.ParseExpr(bt.Def)
	if err != nil {
		return true
	}
	return bindingHasPointerExpr(expr, types, top, seen)
}

// rewriteBindingCall returns the replacement text for call, a call of the
// bound function fn, that performs the cgocheck pointer checks of
// traditional cgo (see rewriteCall, checkIndex, checkAddr, and checkSlice)
// and then calls the unchecked _Cfunc_ entry for fn. It returns "" if no
// argument needs a check, in which case the call is left to fn's checking
// Go wrapper. The arguments are evaluated once, in order, into a
// _cgo_arg_<name> value, which also converts untyped constant arguments
// to the parameter types. For example, with p and q pointers, the call
// F(&x.f, p) becomes
//
//	func() (_cgoa _cgo_arg_F) {
//		_cgoBase0 := &x.f; _cgoa.p0 = _cgoBase0; _cgoa.p1 = p
//		_cgoCheckPointer(_cgoBase0, 0 == 0); _cgoCheckPointer(_cgoa.p1, nil)
//		_cgoa.ret = _Cfunc_F(_cgoa.p0, _cgoa.p1); return
//	}().ret
//
// on a single line. A call whose result is unused discards it. A deferred
// call or go statement evaluates the arguments immediately and checks and
// calls when the returned function literal runs.
func (p *Package) rewriteBindingCall(e *bindingEdits, off func(token.Pos) int, call *ast.CallExpr, fn BindingFunc, types map[string]BindingType, deferred, unused bool) string {
	if call.Ellipsis.IsValid() || len(call.Args) != len(fn.Params) {
		return ""
	}
	isNil := func(x ast.Expr) bool {
		id, ok := x.(*ast.Ident)
		return ok && id.Name == "nil"
	}
	needs, _ := bindingPointerChecks(fn, types)
	anyCheck := false
	for i, arg := range call.Args {
		anyCheck = anyCheck || needs[i] && !isNil(arg)
	}
	if !anyCheck {
		return ""
	}
	// isType reports whether x is a type in a conversion.
	var isType func(x ast.Expr) bool
	isType = func(x ast.Expr) bool {
		switch t := x.(type) {
		case *ast.ParenExpr:
			return isType(t.X)
		case *ast.StarExpr:
			return isType(t.X)
		case *ast.SelectorExpr:
			if id, ok := t.X.(*ast.Ident); ok && id.Name == "C" {
				if _, ok := types["_Ctype_"+t.Sel.Name]; ok {
					return true
				}
				_, isPrim := primitiveCtypeToGo("_Ctype_" + t.Sel.Name)
				return isPrim
			}
		}
		return p.isType(x)
	}
	text := func(start, end token.Pos) string { return e.text(off(start), off(end)) }

	var stmts, checks strings.Builder
	for i, arg := range call.Args {
		field := fmt.Sprintf("_cgoa.p%d", i)
		if !needs[i] || isNil(arg) || p.checkUnsafeStringData(arg) {
			fmt.Fprintf(&stmts, "%s = %s; ", field, text(arg.Pos(), arg.End()))
			continue
		}
		// Strip type conversions.
		core := arg
		for {
			c, ok := core.(*ast.CallExpr)
			if !ok || len(c.Args) != 1 || c.Ellipsis.IsValid() || !(isType(c.Fun) || p.isUnsafeData(c.Fun, false)) {
				break
			}
			core = c.Args[0]
		}
		// withCore returns arg with core replaced by repl.
		withCore := func(repl string) string {
			return text(arg.Pos(), core.Pos()) + repl + text(core.End(), arg.End())
		}
		if u, ok := core.(*ast.UnaryExpr); ok && u.Op == token.AND {
			if index, ok := u.X.(*ast.IndexExpr); ok {
				// &a[i]: check the complete contents of the slice or array a.
				idx := fmt.Sprintf("_cgoIndex%d", i)
				addr, deref, ref := "", "", idx
				if p.isVariable(index.X) {
					addr, deref, ref = "&", "*", "(*"+idx+")"
				}
				fmt.Fprintf(&stmts, "%s := %s%s; ", idx, addr, text(index.X.Pos(), index.X.End()))
				fmt.Fprintf(&stmts, "%s = %s; ", field, withCore("&"+ref+text(index.Lbrack, index.End())))
				fmt.Fprintf(&checks, "_cgoCheckPointer(%s, %s%s); ", field, deref, idx)
				continue
			}
			// &x: check just the memory that x occupies.
			base := fmt.Sprintf("_cgoBase%d", i)
			fmt.Fprintf(&stmts, "%s := %s; ", base, text(core.Pos(), core.End()))
			fmt.Fprintf(&stmts, "%s = %s; ", field, withCore(base))
			fmt.Fprintf(&checks, "_cgoCheckPointer(%s, 0 == 0); ", base)
			continue
		}
		if _, ok := core.(*ast.SliceExpr); ok {
			// a[i:j]: check just the contents of the slice.
			sl := fmt.Sprintf("_cgoSlice%d", i)
			fmt.Fprintf(&stmts, "%s := %s; ", sl, text(core.Pos(), core.End()))
			fmt.Fprintf(&stmts, "%s = %s; ", field, withCore(sl))
			fmt.Fprintf(&checks, "_cgoCheckPointer(%s, 0 == 0); ", sl)
			continue
		}
		fmt.Fprintf(&stmts, "%s = %s; ", field, text(arg.Pos(), arg.End()))
		fmt.Fprintf(&checks, "_cgoCheckPointer(%s, nil); ", field)
	}

	var args []string
	for i := range call.Args {
		args = append(args, fmt.Sprintf("_cgoa.p%d", i))
	}
	callText := fmt.Sprintf("_Cfunc_%s(%s)", fn.Name, strings.Join(args, ", "))

	var repl string
	switch {
	case deferred:
		repl = fmt.Sprintf("func() func() { var _cgoa _cgo_arg_%s; %sreturn func() { %s%s } }()()",
			fn.Name, &stmts, &checks, callText)
	case fn.Ret != "" && !unused:
		repl = fmt.Sprintf("func() (_cgoa _cgo_arg_%s) { %s%s_cgoa.ret = %s; return }().ret",
			fn.Name, &stmts, &checks, callText)
	default:
		repl = fmt.Sprintf("func() { var _cgoa _cgo_arg_%s; %s%s%s }()",
			fn.Name, &stmts, &checks, callText)
	}
	// Keep the line count, so that positions after the call do not move:
	// add the newlines of any dropped comments before the closing brace.
	if n := strings.Count(text(call.Pos(), call.End()), "\n") - strings.Count(repl, "\n"); n > 0 {
		i := strings.LastIndex(repl, "}")
		repl = repl[:i] + strings.Repeat("\n", n) + repl[i:]
	}
	return repl
}

func (p *Package) writeGoTypes(bf *Bindings, expFuncs []ExportFunc) {
	f := creat("_cgo_gotypes.go")
	defer f.Close()

	needsStructs := false
	for _, v := range bf.Vars {
		if strings.Contains(v.Type, "structs.HostLayout") {
			needsStructs = true
			break
		}
	}
	if !needsStructs {
		for _, fn := range bf.Funcs {
			if strings.Contains(fn.Ret, "structs.HostLayout") ||
				slices.ContainsFunc(fn.Params, func(pt string) bool { return strings.Contains(pt, "structs.HostLayout") }) {
				needsStructs = true
				break
			}
		}
	}

	if !needsStructs {
		for _, v := range bf.DynloadVars {
			if strings.Contains(v.Type, "structs.HostLayout") {
				needsStructs = true
				break
			}
		}
	}
	if !needsStructs {
		for _, ef := range expFuncs {
			if strings.Contains(ef.Ret, "structs.HostLayout") ||
				slices.ContainsFunc(ef.Params, func(pt string) bool { return strings.Contains(pt, "structs.HostLayout") }) {
				needsStructs = true
				break
			}
		}
	}

	fmt.Fprintf(f, "// Code generated by cmd/cgo; DO NOT EDIT.\n\n")
	fmt.Fprintf(f, "package %s\n\n", bf.Package)
	fmt.Fprintf(f, "import \"unsafe\"\n\n")
	if needsStructs {
		fmt.Fprintf(f, "import \"structs\"\n\n")
	}
	if *importRuntimeCgo || len(bf.DynloadTypes) > 0 {
		fmt.Fprintf(f, "import _cgopackage \"runtime/cgo\"\n\n")
		fmt.Fprintf(f, "type _ _cgopackage.Incomplete\n\n")
	}
	fmt.Fprintf(f, "var _ unsafe.Pointer\n\n")

	if len(bf.DynloadTypes) > 0 {
		fmt.Fprintf(f, "//go:linkname _cgo_register_library runtime/cgo.registerLibrary\n")
		fmt.Fprintf(f, "func _cgo_register_library(typ any, syms []string)\n\n")
		fmt.Fprintf(f, "//go:linkname _cgo_library_symbol runtime/cgo.librarySymbol\n")
		fmt.Fprintf(f, "func _cgo_library_symbol(lib *_cgopackage.Library, index int) unsafe.Pointer\n\n")

		// Register the symbols bound to each //cgo:dynload type, in symbol
		// index order, so that cgo.Open can resolve all of them when it
		// loads the library. Package initialization may call cgo.Open, so
		// the registration must happen first: every file of the package
		// starts with a variable declaration that calls
		// _cgo_register_libraries (see dynloadRegisterDecl), and since
		// that call depends on no package-level variables, the variable in
		// the first file is initialized before any other variable, and
		// before any init function runs. Registering again is harmless.
		fmt.Fprintf(f, "func _cgo_register_libraries() bool {\n")
		for _, t := range bf.DynloadTypes {
			syms := make([]string, bf.DynloadCounts[t])
			for _, fn := range bf.Funcs {
				if fn.Indirect && fn.RecvType == t {
					syms[fn.SymIndex] = strconv.Quote(fn.CName)
				}
			}
			for _, v := range bf.DynloadVars {
				if v.RecvType == t {
					syms[v.SymIndex] = strconv.Quote(v.CName)
				}
			}
			fmt.Fprintf(f, "\t_cgo_register_library((*%s)(nil), []string{%s})\n", t, strings.Join(syms, ", "))
		}
		fmt.Fprintf(f, "\treturn true\n}\n\n")
		fmt.Fprintf(f, "%s\n\n", dynloadRegisterDecl)
	}
	resolveDynSym := func(index int) string {
		return fmt.Sprintf("_cgo_library_symbol(&_cgo_recv.Library, %d)", index)
	}
	for _, v := range bf.DynloadVars {
		fmt.Fprintf(f, "func (_cgo_recv *%s) %s() *%s {\n\treturn (*%s)(%s)\n}\n\n", v.RecvType, v.GoName, v.Type, v.Type, resolveDynSym(v.SymIndex))
	}

	impLocal := func(sym string) string {
		if goos == "aix" {
			return "__cgo_imp_" + sym
		}
		return sym
	}
	if len(bf.Funcs) > 0 || len(bf.Vars) > 0 || len(bf.LdFlags) > 0 || bf.needCMalloc() {
		importedLibs := make(map[string]bool)
		for _, fn := range bf.Funcs {
			if fn.Library != "" && !isStandardC(fn.Library) {
				importedLibs[dynLibName(fn.Library)] = true
			}
		}
		for _, v := range bf.Vars {
			if v.Library != "" && !isStandardC(v.Library) {
				importedLibs[dynLibName(v.Library)] = true
			}
		}
		for _, flag := range bf.LdFlags {
			fields, _ := splitQuoted(flag)
			for _, field := range fields {
				if !strings.HasPrefix(field, "-") && (strings.HasSuffix(field, ".so") || strings.Contains(field, ".so.") || strings.HasSuffix(field, ".dylib") || strings.HasSuffix(field, ".dll")) {
					if !isStandardC(field) {
						importedLibs[dynLibName(field)] = true
					}
				}
			}
		}
		sortedLibs := slices.Sorted(maps.Keys(importedLibs))
		if goos != "windows" && goos != "aix" {
			for _, lib := range sortedLibs {
				fmt.Fprintf(f, "//go:cgo_import_dynamic _ _ %q\n", lib)
			}
			if len(sortedLibs) > 0 {
				fmt.Fprintf(f, "\n")
			}
		}

		dynImpLib := func(rawLib string) string {
			switch goos {
			case "windows":
				return windowsDLLName(rawLib)
			case "aix":
				if rawLib != "" && !isStandardC(rawLib) {
					lib := strings.TrimPrefix(rawLib, "./")
					if n := strings.Split(lib, "/"); len(n) == 2 && strings.HasSuffix(n[0], ".a") && (strings.HasSuffix(n[1], ".o") || strings.Contains(n[1], ".so.")) {
						return lib
					}
					return ""
				}
				return "libc.a/shr_64.o"
			default:
				return ""
			}
		}
		emittedImp := make(map[string]bool)
		for _, fn := range bf.Funcs {
			cSym := fn.CName
			if fn.Static || fn.Indirect {
				// Resolved at link time (.syso/.a) or at run time (cgo.Open).
				continue
			}
			if !emittedImp[cSym] {
				emittedImp[cSym] = true
				if lib := dynImpLib(fn.Library); lib != "" {
					fmt.Fprintf(f, "//go:cgo_import_dynamic %s %s %q\n", impLocal(cSym), cSym, lib)
				} else {
					fmt.Fprintf(f, "//go:cgo_import_dynamic %s %s\n", impLocal(cSym), cSym)
				}
			}
		}
		if bf.needCMalloc() {
			// C.CString, C.CBytes, and C.malloc allocate with libc malloc.
			// Use a package-scoped import symbol if the package binds a
			// static or non-libc malloc.
			cMallocSym := cMallocImportSym(bf)
			if !emittedImp[cMallocSym] {
				emittedImp[cMallocSym] = true
				if lib := dynImpLib(""); lib != "" {
					fmt.Fprintf(f, "//go:cgo_import_dynamic %s malloc %q\n", impLocal(cMallocSym), lib)
				} else {
					fmt.Fprintf(f, "//go:cgo_import_dynamic %s malloc\n", impLocal(cMallocSym))
				}
			}
		}
		if len(emittedImp) > 0 {
			fmt.Fprintf(f, "\n")
		}

		emittedStatic := make(map[string]bool)
		emittedVarImp := false
		for _, fn := range bf.Funcs {
			if fn.Static && fn.BindC && bf.UsedFuncPtrs[fn.CName] && !emittedStatic[fn.CName] {
				emittedStatic[fn.CName] = true
				emittedVarImp = true
				fmt.Fprintf(f, "//go:cgo_import_static %s\n", fn.CName)
			}
		}
		for _, v := range bf.Vars {
			if v.Static {
				if !emittedStatic[v.CName] {
					emittedStatic[v.CName] = true
					emittedVarImp = true
					// Resolved at link time from a .syso or .a file.
					fmt.Fprintf(f, "//go:cgo_import_static %s\n", v.CName)
				}
			} else {
				emittedVarImp = true
				localSym := varImportSym(v)
				if lib := dynImpLib(v.Library); lib != "" {
					fmt.Fprintf(f, "//go:cgo_import_dynamic %s %s %q\n", localSym, v.CName, lib)
				} else {
					fmt.Fprintf(f, "//go:cgo_import_dynamic %s %s\n", localSym, v.CName)
				}
			}
		}
		if emittedVarImp {
			fmt.Fprintf(f, "\n")
		}
	}

	for _, flag := range bf.LdFlags {
		fmt.Fprintf(f, "//go:cgo_ldflag %q\n", flag)
	}
	if len(bf.LdFlags) > 0 {
		fmt.Fprintf(f, "\n")
	}

	// _Cvar_<Name> points to the C variable. Usually it is initialized
	// with the address of __cgo_var_<Name>, which is the variable itself.
	// A dynamically imported symbol on Windows is instead a slot in the PE
	// import address table, which the Windows loader fills in with the
	// address of the variable in the DLL: PE has no copy relocations, so
	// code outside a DLL can only reach the DLL's data through such a slot.
	// There, _Cvar_<Name> is the slot itself.
	for _, v := range bf.Vars {
		if goos == "windows" && !v.Static {
			fmt.Fprintf(f, "//go:linkname _Cvar_%s %s\n", v.Name, varImportSym(v))
			fmt.Fprintf(f, "var _Cvar_%s *%s\n\n", v.Name, v.Type)
			continue
		}
		goSym := "__cgo_var_" + v.Name
		target := v.CName
		if !v.Static {
			target = varImportSym(v)
		}
		fmt.Fprintf(f, "//go:linkname %s %s\n", goSym, target)
		fmt.Fprintf(f, "var %s byte\n", goSym)
		fmt.Fprintf(f, "var _Cvar_%s *%s = (*%s)(unsafe.Pointer(&%s))\n\n", v.Name, v.Type, v.Type, goSym)
	}

	// _Cfpvar_fp_<CName> holds the C function pointer for a bound C function
	// referenced in a non-call expression. On Windows, a dynamically imported
	// symbol is a PE IAT slot holding the function's address; on AIX, it is an
	// XCOFF TOC slot holding the pointer to the function descriptor.
	for _, fn := range bf.Funcs {
		if !fn.BindC || fn.Indirect || !bf.UsedFuncPtrs[fn.CName] {
			continue
		}
		if !fn.Static && (goos == "windows" || goos == "aix") {
			fmt.Fprintf(f, "//go:linkname _Cfpvar_fp_%s %s\n", fn.CName, impLocal(fn.CName))
			fmt.Fprintf(f, "var _Cfpvar_fp_%s unsafe.Pointer\n\n", fn.CName)
			continue
		}
		target := fn.CName
		if !fn.Static {
			target = impLocal(fn.CName)
		}
		fmt.Fprintf(f, "//go:linkname __cgo_fp_%s %s\n", fn.CName, target)
		fmt.Fprintf(f, "var __cgo_fp_%s byte\n", fn.CName)
		fmt.Fprintf(f, "var _Cfpvar_fp_%s unsafe.Pointer = unsafe.Pointer(&__cgo_fp_%s)\n\n", fn.CName, fn.CName)
	}

	abiCfg := &ABIConfig{OS: goos, Arch: goarch, PtrSize: targetPtrSize(), CLongSize: targetCLongSize()}

	fmt.Fprint(f, bindingRuntimeProlog)
	if len(expFuncs) > 0 {
		fmt.Fprintf(f, "//go:linkname _cgoCheckResult runtime.cgoCheckResult\n//go:noescape\nfunc _cgoCheckResult(interface{})\n\n")
		p.writeBindingExports(f, expFuncs, bf.TypeMap, abiCfg)
	}

	emittedAlias := make(map[string]bool)
	if len(bf.TypeGoToC) > 0 {
		for _, goName := range slices.Sorted(maps.Keys(bf.TypeGoToC)) {
			cName := bf.TypeGoToC[goName]
			aliasName := "_Ctype_" + cName
			if !emittedAlias[aliasName] && aliasName != goName {
				emittedAlias[aliasName] = true
				fmt.Fprintf(f, "type %s = %s\n", aliasName, goName)
			}
		}
		fmt.Fprintf(f, "\n")
	}

	definedCfuncs := make(map[string]bool)
	if len(bf.UsedBuiltins) > 0 {
		charGoType, _ := primitiveCtypeToGo("_Ctype_char")
		sizeGoType, _ := primitiveCtypeToGo("_Ctype_size_t")
		for _, bType := range []struct{ c, goType string }{
			{"_Ctype_char", charGoType},
			{"_Ctype_int", "int32"},
			{"_Ctype_size_t", sizeGoType},
		} {
			if !emittedAlias[bType.c] {
				emittedAlias[bType.c] = true
				fmt.Fprintf(f, "type %s = %s\n", bType.c, bType.goType)
			}
		}
		fmt.Fprintf(f, "\n")
		if bf.needCMalloc() {
			fmt.Fprintf(f, bindingCMallocProlog, cPrefix)
		}
		for _, name := range slices.Sorted(maps.Keys(bf.UsedBuiltins)) {
			definedCfuncs[name] = true
			fmt.Fprintf(f, "%s\n\n", strings.TrimSpace(builtinDefs[name]))
		}
	}

	for _, fn := range bf.Funcs {
		definedCfuncs[fn.Name] = true
	}
	for _, ef := range expFuncs {
		if bf.UsedExportCalls[ef.Name] {
			definedCfuncs[ef.Name] = true
		}
	}
	emitFuncWrapper := func(fn BindingFunc, emitArgStruct bool) {
		fmt.Fprintf(f, "//go:linkname _cgo_trampoline_%s _cgo%s_trampoline_%s\n", fn.Name, cPrefix, fn.Name)
		fmt.Fprintf(f, "var _cgo_trampoline_%s byte\n\n", fn.Name)

		if emitArgStruct && (fn.Indirect || len(fn.Params) > 0 || fn.Ret != "") {
			fmt.Fprintf(f, "type _cgo_arg_%s struct {\n", fn.Name)
			if fn.Indirect {
				fmt.Fprintf(f, "\tfn uintptr\n")
			}
			for i, pt := range fn.Params {
				fmt.Fprintf(f, "\tp%d %s\n", i, pt)
			}
			if fn.Ret != "" {
				fmt.Fprintf(f, "\tret %s\n", fn.Ret)
			}
			fmt.Fprintf(f, "}\n\n")
		}

		var paramDecls []string
		var paramPass []string
		var paramNames []string
		if fn.Indirect {
			paramDecls = append(paramDecls, "fn uintptr")
			paramPass = append(paramPass, "fn: fn")
		}
		for i, pt := range fn.Params {
			paramDecls = append(paramDecls, fmt.Sprintf("p%d %s", i, pt))
			paramPass = append(paramPass, fmt.Sprintf("p%d: p%d", i, i))
			paramNames = append(paramNames, fmt.Sprintf("p%d", i))
		}
		retSig := ""
		if fn.Ret != "" {
			retSig = " " + fn.Ret
		}
		noCallback := p.noCallbacks[fn.CName]
		noEscape := p.noEscapes[fn.CName]
		hasArgs := fn.Indirect || len(fn.Params) > 0 || fn.Ret != ""
		// When a C call with a return value may invoke Go callbacks, the
		// trampoline relocates &args by the _cgo_topofstack delta if the
		// goroutine stack grows. Pass args as a value parameter to a
		// //go:cgo_unsafe_args helper so the compiler always places it on
		// the goroutine stack (never heap-allocating even if > MaxStackVarSize).
		pinArgsOnStack := hasArgs && !noCallback && fn.Ret != ""

		if pinArgsOnStack {
			fmt.Fprintf(f, "//go:cgo_unsafe_args\n")
			fmt.Fprintf(f, "func _cgo_call_%s(args _cgo_arg_%s)%s {\n", fn.Name, fn.Name, retSig)
			fmt.Fprintf(f, "\t_cgo_runtime_cgocall(unsafe.Pointer(&_cgo_trampoline_%s), uintptr(unsafe.Pointer(&args)))\n", fn.Name)
			fmt.Fprintf(f, "\tif _Cgo_always_false {\n")
			fmt.Fprintf(f, "\t\t_Cgo_keepalive(&args)\n")
			fmt.Fprintf(f, "\t}\n")
			fmt.Fprintf(f, "\treturn args.ret\n")
			fmt.Fprintf(f, "}\n\n")
		}
		fmt.Fprintf(f, "func _Cfunc_%s(%s)%s {\n", fn.Name, strings.Join(paramDecls, ", "), retSig)
		if pinArgsOnStack {
			fmt.Fprintf(f, "\tr := _cgo_call_%s(_cgo_arg_%s{%s})\n", fn.Name, fn.Name, strings.Join(paramPass, ", "))
		} else {
			if noCallback {
				fmt.Fprintf(f, "\t_Cgo_no_callback(true)\n")
			}
			if !hasArgs {
				fmt.Fprintf(f, "\t_cgo_runtime_cgocall(unsafe.Pointer(&_cgo_trampoline_%s), 0)\n", fn.Name)
			} else {
				fmt.Fprintf(f, "\targs := _cgo_arg_%s{%s}\n", fn.Name, strings.Join(paramPass, ", "))
				fmt.Fprintf(f, "\t_cgo_runtime_cgocall(unsafe.Pointer(&_cgo_trampoline_%s), uintptr(unsafe.Pointer(&args)))\n", fn.Name)
			}
			if noCallback {
				fmt.Fprintf(f, "\t_Cgo_no_callback(false)\n")
			}
		}
		touchFunc := "_Cgo_use"
		if noEscape && noCallback {
			touchFunc = "_Cgo_keepalive"
		}
		var touchParams []string
		for i, name := range paramNames {
			if abiCfg.inspectType(fn.Params[i], bf.TypeMap).hasPointers {
				touchParams = append(touchParams, name)
			}
		}
		keepArgsHere := hasArgs && !pinArgsOnStack
		if len(touchParams) > 0 || keepArgsHere {
			fmt.Fprintf(f, "\tif _Cgo_always_false {\n")
			for _, name := range touchParams {
				fmt.Fprintf(f, "\t\t%s(%s)\n", touchFunc, name)
			}
			if keepArgsHere {
				fmt.Fprintf(f, "\t\t_Cgo_keepalive(&args)\n")
			}
			fmt.Fprintf(f, "\t}\n")
		}
		if fn.Ret != "" {
			if pinArgsOnStack {
				fmt.Fprintf(f, "\treturn r\n")
			} else {
				fmt.Fprintf(f, "\treturn args.ret\n")
			}
		}
		fmt.Fprintf(f, "}\n\n")

		var fnArg string
		if fn.Indirect {
			fnArg = "uintptr(" + resolveDynSym(fn.SymIndex) + ")"
		}
		emitGoFuncWrapper(f, fn, fnArg, definedCfuncs, bf.TypeMap)
	}
	for _, fn := range bf.Funcs {
		emitFuncWrapper(fn, true)
	}
	for _, ef := range expFuncs {
		if bf.UsedExportCalls[ef.Name] {
			emitFuncWrapper(exportBindingFunc(ef), false)
		}
	}
}

// emitGoFuncWrapper writes the Go function or //cgo:dynload method wrapper calling _Cfunc_<fn.Name>.
func emitGoFuncWrapper(f *os.File, fn BindingFunc, fnArg string, definedCfuncs map[string]bool, types map[string]BindingType) {
	if fn.GoName == "" && !fn.BindC {
		return
	}
	var paramDecls []string
	var passArgs []string
	recv := ""
	if fn.Indirect {
		recv = "(_cgo_recv *" + fn.RecvType + ") "
		passArgs = append(passArgs, fnArg)
	}
	for i, pt := range fn.Params {
		pName := fmt.Sprintf("p%d", i)
		paramDecls = append(paramDecls, fmt.Sprintf("%s %s", pName, pt))
		passArgs = append(passArgs, pName)
	}
	retSig := ""
	retPrefix := ""
	if fn.Ret != "" {
		retSig = " " + fn.Ret
		retPrefix = "return "
	}
	cSym := fn.CName
	targetFunc := fn.Name
	callExpr := fmt.Sprintf("_Cfunc_%s(%s)", targetFunc, strings.Join(passArgs, ", "))

	// Calls that are not rewritten to check their own arguments (calls
	// from other packages and from files without import "C", method and
	// indirect calls, and calls whose arguments cannot be analyzed) go
	// through the wrapper, which checks every argument that might contain
	// a Go pointer. Without knowing how an argument was formed, it checks
	// the complete memory allocation that the argument points to.
	var checks strings.Builder
	needs, _ := bindingPointerChecks(fn, types)
	for i, need := range needs {
		if need {
			fmt.Fprintf(&checks, "\t_cgoCheckPointer(p%d, nil)\n", i)
		}
	}
	if checks.Len() > 0 && !fn.Indirect {
		// Calls to C.cName that rewriteBindingCall leaves as direct calls
		// (such as multi-value argument forwarding) use _Cgocheck_<name>;
		// see bindingFuncEntry.
		entry := bindingFuncEntry(fn, types)
		fmt.Fprintf(f, "func %s(%s)%s {\n%s\t%s%s\n}\n\n",
			entry, strings.Join(paramDecls, ", "), retSig, &checks, retPrefix, callExpr)
		checks.Reset()
		callExpr = fmt.Sprintf("%s(%s)", entry, strings.Join(passArgs, ", "))
	}
	if fn.GoName == "" {
		return
	}
	fmt.Fprintf(f, "func %s%s(%s)%s {\n%s\t%s%s\n}\n\n",
		recv, fn.GoName, strings.Join(paramDecls, ", "), retSig, &checks, retPrefix, callExpr)
	if fn.BindC && cSym != targetFunc && !definedCfuncs[cSym] {
		definedCfuncs[cSym] = true
		fmt.Fprintf(f, "var _Cfunc_%s = _Cfunc_%s\n\n", cSym, targetFunc)
	}
}

func cMallocImportSym(bf *Bindings) string {
	for _, fn := range bf.Funcs {
		if fn.CName == "malloc" && (fn.Static || !fn.LibC) {
			return fmt.Sprintf("_cgo%s_libc_malloc", cPrefix)
		}
	}
	return "malloc"
}

func (p *Package) writeTrampolines(bf *Bindings, expFuncs []ExportFunc) {
	f := creat("_cgo_trampolines.s")
	defer f.Close()

	fmt.Fprintf(f, "// Code generated by cmd/cgo; DO NOT EDIT.\n\n")
	fmt.Fprintf(f, "#include \"textflag.h\"\n\n")

	if len(bf.Funcs) == 0 && !bf.needCMalloc() && len(expFuncs) == 0 {
		return
	}

	abi := getABI(goos, goarch)
	for _, fn := range bf.Funcs {
		abi.writeTrampoline(f, fn, bf.TypeMap)
	}
	if bf.needCMalloc() {
		sizeGoType, _ := primitiveCtypeToGo("_Ctype_size_t")
		abi.writeTrampoline(f, BindingFunc{
			Name:   "_cgo_malloc",
			CName:  cMallocImportSym(bf),
			Params: []string{sizeGoType},
			Ret:    "unsafe.Pointer",
			LibC:   true,
		}, bf.TypeMap)
	}
	for _, ef := range expFuncs {
		abi.writeCallbackTrampoline(f, BindingFunc{
			Name:   ef.Name,
			Params: ef.Params,
			Ret:    ef.Ret,
		}, bf.TypeMap)
		if bf.UsedExportCalls[ef.Name] {
			abi.writeTrampoline(f, exportBindingFunc(ef), bf.TypeMap)
		}
	}
}
