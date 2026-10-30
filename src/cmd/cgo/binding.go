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
	"internal/syslist"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"

	"cmd/internal/edit"
	"cmd/internal/hash"
	"cmd/internal/par"
)

// BindingEntry holds the parsed declarations and linker flags collected
// from the //cgo: pragmas in a package.
type BindingEntry struct {
	Package string
	LdFlags []string
	Funcs   []BindingFunc
}

// bindingSource holds the raw bytes and parsed AST of a source file inspected
// for //cgo: pragmas.
type bindingSource struct {
	path string
	data []byte
	file *ast.File
	fset *token.FileSet
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
	case "_Ctype_char", "_Ctype_schar", "_Ctype_int8_t":
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

// cgoPragma is a parsed //cgo:func pragma:
//
//	//cgo:func GoName [C.]cName {"lib" | static | libc}
//
// A pragma always names both the Go function GoName, which must be declared
// without a body elsewhere in the package, and the C symbol cName that it
// calls. Writing the C name as C.cName additionally lets Go code in the
// package call the function as C.cName, as with traditional cgo; without the
// C. prefix, the function is only reachable through GoName.
//
// Pragmas are self-contained and may appear anywhere in a file that
// imports "C".
type cgoPragma struct {
	Pos    token.Position
	Kind   string // pragma name: "func"
	GoName string // Go identifier being bound
	CName  string // C symbol name
	BindC  bool   // cName was written as C.cName, so Go code may also use C.cName

	// Library specifier.
	Library string // shared library for the "lib" form
	LibC    bool   // libc: the platform C runtime
	Static  bool   // static: resolved at link time from a .syso or .a
}

// cgoPragmaSyntax maps each supported //cgo: pragma name to its syntax, for
// error messages.
var cgoPragmaSyntax = map[string]string{
	"func": `//cgo:func GoName [C.]cName {"lib" | static | libc}`,
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
	bad := func(why string) {
		fatalf("%s: invalid //cgo:%s pragma: %s (want %s)", fset.Position(c.Pos()), d.Name, why, cgoPragmaSyntax[d.Name])
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
	if len(args) != 3 {
		bad("wrong number of arguments")
	}
	pr := cgoPragma{Pos: fset.Position(c.Pos()), Kind: d.Name}
	if quoted(args[0]) || !token.IsIdentifier(args[0].Arg) || args[0].Arg == "_" {
		bad(fmt.Sprintf("invalid Go name %q", args[0].Arg))
	}
	pr.GoName = args[0].Arg
	pr.CName, pr.BindC = strings.CutPrefix(args[1].Arg, "C.")
	if quoted(args[1]) || !token.IsIdentifier(pr.CName) {
		bad(fmt.Sprintf("invalid C name %q", args[1].Arg))
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

func hasImportCFile(file *ast.File) bool {
	return file != nil && slices.ContainsFunc(file.Imports, func(imp *ast.ImportSpec) bool {
		return imp.Path.Value == `"C"`
	})
}

func (p *Package) parseBinding(bindingPath string, data []byte) *BindingEntry {
	return p.parseBindingFiles([]bindingSource{{path: bindingPath, data: data}})
}

func (p *Package) parseBindingFiles(sources []bindingSource) *BindingEntry {
	entry := &BindingEntry{
		LdFlags: append([]string(nil), p.LdFlags...),
	}

	var pragmas []cgoPragma
	for i := range sources {
		src := &sources[i]
		if src.file == nil {
			src.fset = token.NewFileSet()
			fileAST, err := parser.ParseFile(src.fset, src.path, src.data, parser.ParseComments)
			if err != nil {
				fatalf("parsing %s: %s", src.path, err)
			}
			src.file = fileAST
		}
		if entry.Package == "" {
			entry.Package = src.file.Name.Name
		}
		pragmas = append(pragmas, parseCgoPragmas(src.fset, src.file)...)

		// Collect #cgo LDFLAGS directives from the import "C" preamble,
		// honoring their build constraints. When cmd/go passes -ldflags,
		// it has already collected (and checked against CGO_LDFLAGS_ALLOW)
		// the package's flags, so the preamble is not consulted.
		if ldflagsSet {
			continue
		}
		srcDirVal := filepath.Dir(src.path)
		forEachImportC(src.file, func(_ *ast.GenDecl, _ *ast.ImportSpec, doc *ast.CommentGroup) {
			if doc == nil {
				return
			}
			for line := range strings.SplitSeq(commentText(doc), "\n") {
				cgoLine := strings.TrimSpace(line)
				if !strings.HasPrefix(cgoLine, "#cgo") || len(cgoLine) < 5 || (cgoLine[4] != ' ' && cgoLine[4] != '\t') {
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

	// Index the bodyless function declarations that pragmas may name.
	type funcDecl struct {
		d    *ast.FuncDecl
		fset *token.FileSet
	}
	funcDecls := make(map[string]funcDecl)
	for _, src := range sources {
		for _, decl := range src.file.Decls {
			if d, ok := decl.(*ast.FuncDecl); ok && d.Recv == nil {
				funcDecls[d.Name.Name] = funcDecl{d, src.fset}
			}
		}
	}

	boundGo := make(map[string]bool)
	boundC := make(map[string]bool)
	for _, pr := range pragmas {
		pos := pr.Pos
		if boundGo[pr.GoName] {
			fatalf("%s: %s is bound by more than one //cgo: pragma", pos, pr.GoName)
		}
		boundGo[pr.GoName] = true
		if pr.BindC {
			if boundC[pr.CName] {
				fatalf("%s: C.%s is bound by more than one //cgo: pragma", pos, pr.CName)
			}
			boundC[pr.CName] = true
		}

		fd, ok := funcDecls[pr.GoName]
		if !ok {
			fatalf("%s: //cgo:func %s: missing declaration of func %s", pos, pr.GoName, pr.GoName)
		}
		d := fd.d
		if d.Body != nil {
			fatalf("%s: //cgo:func function %s cannot have a body", fd.fset.Position(d.Pos()), pr.GoName)
		}
		fn := BindingFunc{
			Name:    pr.GoName,
			GoName:  pr.GoName,
			CName:   pr.CName,
			BindC:   pr.BindC,
			Library: pr.Library,
			LibC:    pr.LibC,
			Static:  pr.Static,
		}
		if d.Type.Params != nil {
			for _, field := range d.Type.Params.List {
				pType := p.cgoASTToGoDef(field.Type)
				if why := cTypeError(pType, nil); why != "" {
					fatalf("%s: //cgo:func %s: parameter type %s cannot be passed to C: %s", fd.fset.Position(field.Type.Pos()), pr.GoName, pType, why)
				}
				for range max(1, len(field.Names)) {
					fn.Params = append(fn.Params, pType)
				}
			}
		}
		if d.Type.Results != nil && len(d.Type.Results.List) > 0 {
			res := d.Type.Results.List[0].Type
			fn.Ret = p.cgoASTToGoDef(res)
			if why := cTypeError(fn.Ret, nil); why != "" {
				fatalf("%s: //cgo:func %s: result type %s cannot be returned from C: %s", fd.fset.Position(res.Pos()), pr.GoName, fn.Ret, why)
			}
		}
		entry.Funcs = append(entry.Funcs, fn)
	}
	return entry
}

// processBinding consumes the //cgo: pragmas in all of the package's Go files,
// rewrites the Go source files, and generates Go wrappers and assembly trampolines
// without requiring a C compiler.
func (p *Package) processBinding(goFiles []string) {
	type parsedFile struct {
		srcfile    string
		origPath   string
		data       []byte
		fset       *token.FileSet
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
			b, err := os.ReadFile(origPath)
			if err != nil {
				fatalf("%s", err)
			}
			fset := token.NewFileSet()
			parsed, err := parser.ParseFile(fset, origPath, b, parser.ParseComments)
			if err != nil {
				fatalf("%s", err)
			}
			hasImportC := slices.ContainsFunc(parsed.Imports, func(s *ast.ImportSpec) bool {
				return s.Path.Value == `"C"`
			})
			parsedFiles[i] = parsedFile{
				srcfile:    srcfile,
				origPath:   origPath,
				data:       b,
				fset:       fset,
				parsed:     parsed,
				hasImportC: hasImportC,
			}
			sources[i] = bindingSource{
				path: origPath,
				data: b,
				file: parsed,
				fset: fset,
			}
		})
	}
	<-q.Idle()

	h := hash.New32()
	io.WriteString(h, *importPath)
	for _, pf := range parsedFiles {
		h.Write(pf.data)
	}

	entry := p.parseBindingFiles(sources)
	p.PackageName = entry.Package
	p.LdFlags = entry.LdFlags

	// cFuncMap maps C names bound with the C.cName form to their functions.
	// Other bound names (Go names, and C names bound without the C. prefix)
	// are rewritten to undefined identifiers if referenced as C.name.
	cFuncMap := make(map[string]BindingFunc)
	goFuncMap := make(map[string]BindingFunc)
	knownBoundFuncs := make(map[string]bool)
	boundGoFuncs := make(map[string]bool)
	for _, fn := range entry.Funcs {
		if fn.BindC {
			cFuncMap[fn.CName] = fn
		}
		goFuncMap[fn.GoName] = fn
		knownBoundFuncs[fn.CName] = true
		knownBoundFuncs[fn.Name] = true
		knownBoundFuncs[fn.GoName] = true
		boundGoFuncs[fn.GoName] = true
	}

	outGoFiles := make([]string, len(parsedFiles))
	rewriteQ := par.NewQueue(runtime.GOMAXPROCS(0))
	for i, pf := range parsedFiles {
		rewriteQ.Add(func() {
			srcfile := pf.srcfile
			b := pf.data
			parsed := pf.parsed
			editBuf := edit.NewBuffer(b)

			// Strip //cgo: pragmas and the bodyless declarations of bound
			// functions from the input file because _cgo_gotypes.go defines the
			// concrete Go wrapper functions with bodies in the same package; leaving
			// the bodyless declarations in place would cause duplicate declaration
			// errors during compilation.
			for _, cg := range parsed.Comments {
				for _, c := range cg.List {
					if isCgoPragmaComment(c.Text) {
						editBuf.Replace(int(c.Pos()-1), int(c.End()-1), "")
					}
				}
			}
			isBoundDecl := func(n ast.Node) bool {
				d, ok := n.(*ast.FuncDecl)
				return ok && d.Body == nil && d.Recv == nil && boundGoFuncs[d.Name.Name]
			}
			for _, decl := range parsed.Decls {
				if isBoundDecl(decl) {
					editBuf.Replace(int(decl.Pos()-1), int(decl.End()-1), "")
				}
			}

			if pf.hasImportC {
				hasUnsafeImport := false
				hasNamedUnsafeImport := false
				for _, s := range parsed.Imports {
					if s.Path.Value == `"unsafe"` {
						hasUnsafeImport = true
						if s.Name == nil {
							hasNamedUnsafeImport = true
						}
					}
				}

				forEachImportC(parsed, func(d *ast.GenDecl, s *ast.ImportSpec, _ *ast.CommentGroup) {
					if hasUnsafeImport {
						if len(d.Specs) == 1 {
							editBuf.Replace(int(d.Pos()-1), int(d.End()-1), "")
						} else {
							editBuf.Replace(int(s.Pos()-1), int(s.End()-1), "")
						}
					} else {
						editBuf.Replace(int(s.Path.Pos()-1), int(s.Path.End()-1), `_ "unsafe"`)
						hasUnsafeImport = true
					}
				})
				if hasNamedUnsafeImport {
					editBuf.Insert(len(b), "\nvar _ unsafe.Pointer\n")
				}
			}

			// Rewrite references to C names, and calls of bound functions
			// that pass pointers, which get the same cgocheck pointer checks
			// as calls in traditional cgo.
			edits := &bindingEdits{src: b}
			off := func(pos token.Pos) int { return int(pos - 1) }
			if pf.hasImportC {
				ast.Inspect(parsed, func(n ast.Node) bool {
					if isBoundDecl(n) {
						return false
					}
					sel, ok := n.(*ast.SelectorExpr)
					if !ok {
						return true
					}
					id, ok := sel.X.(*ast.Ident)
					if !ok || id.Name != "C" {
						return true
					}
					name := sel.Sel.Name
					start := off(sel.Pos())
					end := off(sel.End())
					if fn, isFunc := cFuncMap[name]; isFunc {
						edits.replace(start, end, bindingFuncEntry(fn, nil))
					} else if goType, isPrim := primitiveCtypeToGo("_Ctype_" + name); isPrim {
						edits.replace(start, end, goType)
					} else if knownBoundFuncs[name] {
						edits.replace(start, end, "_Cundef_"+name)
					} else {
						edits.replace(start, end, "_Cfunc_"+name)
					}
					return true
				})
			}
			shadowed := bindingLocalNames(parsed)
			type callSite struct {
				call *ast.CallExpr
				fn   BindingFunc
			}
			var calls []callSite
			deferred := make(map[*ast.CallExpr]bool)
			unused := make(map[*ast.CallExpr]bool)
			ast.Inspect(parsed, func(n ast.Node) bool {
				switch n := n.(type) {
				case *ast.FuncDecl:
					if isBoundDecl(n) {
						return false
					}
				case *ast.DeferStmt:
					deferred[n.Call] = true
				case *ast.GoStmt:
					deferred[n.Call] = true
				case *ast.ExprStmt:
					if call, ok := n.X.(*ast.CallExpr); ok {
						unused[call] = true
					}
				case *ast.CallExpr:
					switch fun := n.Fun.(type) {
					case *ast.SelectorExpr:
						if id, ok := fun.X.(*ast.Ident); ok && id.Name == "C" && pf.hasImportC {
							if fn, ok := cFuncMap[fun.Sel.Name]; ok {
								calls = append(calls, callSite{n, fn})
							}
						}
					case *ast.Ident:
						if fn, ok := goFuncMap[fun.Name]; ok && !shadowed[fun.Name] {
							calls = append(calls, callSite{n, fn})
						}
					}
				}
				return true
			})
			// Rewrite inner calls first so that outer calls include them.
			for _, c := range slices.Backward(calls) {
				if repl := p.rewriteBindingCall(edits, off, c.call, c.fn, nil, deferred[c.call], unused[c.call], knownBoundFuncs); repl != "" {
					edits.replace(off(c.call.Pos()), off(c.call.End()), repl)
				}
			}
			edits.apply(editBuf)

			base := strings.TrimSuffix(filepath.Base(srcfile), ".go")
			outName := base + ".cgobinding.go"
			fgo1 := creat(outName)
			fmt.Fprintf(fgo1, "// Code generated by cmd/cgo; DO NOT EDIT.\n\n")
			fmt.Fprintf(fgo1, "//line %s:1:1\n", srcfile)
			fgo1.Write(editBuf.Bytes())
			fgo1.Close()
			outGoFiles[i] = outName
		})
	}
	<-rewriteQ.Idle()
	p.GoFiles = append(p.GoFiles, outGoFiles...)

	origCPrefix := cPrefix
	cPrefix = fmt.Sprintf("_%x", h.Sum(nil)[0:6])
	p.writeGoTypes(entry)
	p.writeTrampolines(entry)
	cPrefix = origCPrefix

	if len(p.LdFlags) > 0 {
		fflg := creat("_cgo_flags")
		for _, flag := range p.LdFlags {
			fmt.Fprintf(fflg, "_CGO_LDFLAGS_=%s\n", flag)
		}
		fflg.Close()
	}
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

// bindingLocalNames returns the names that file declares other than by
// package-level declarations: import names and names declared in function
// scopes. A call through such a name may not refer to the bound function of
// that name, so such calls are left to the checking Go wrapper.
func bindingLocalNames(file *ast.File) map[string]bool {
	names := make(map[string]bool)
	for _, imp := range file.Imports {
		if imp.Name != nil {
			names[imp.Name.Name] = true
		}
	}
	addFields := func(fl *ast.FieldList) {
		if fl == nil {
			return
		}
		for _, f := range fl.List {
			for _, n := range f.Names {
				names[n.Name] = true
			}
		}
	}
	addIdents := func(exprs ...ast.Expr) {
		for _, x := range exprs {
			if id, ok := x.(*ast.Ident); ok {
				names[id.Name] = true
			}
		}
	}
	local := func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.FuncLit:
			addFields(n.Type.TypeParams)
			addFields(n.Type.Params)
			addFields(n.Type.Results)
		case *ast.AssignStmt:
			if n.Tok == token.DEFINE {
				addIdents(n.Lhs...)
			}
		case *ast.RangeStmt:
			if n.Tok == token.DEFINE {
				addIdents(n.Key, n.Value)
			}
		case *ast.ValueSpec:
			for _, id := range n.Names {
				names[id.Name] = true
			}
		case *ast.TypeSpec:
			names[n.Name.Name] = true
			addFields(n.TypeParams)
		}
		return true
	}
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			addFields(d.Recv)
			addFields(d.Type.TypeParams)
			addFields(d.Type.Params)
			addFields(d.Type.Results)
			if d.Body != nil {
				ast.Inspect(d.Body, local)
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				if ts, ok := spec.(*ast.TypeSpec); ok {
					addFields(ts.TypeParams)
				}
			}
			ast.Inspect(d, func(n ast.Node) bool {
				if fl, ok := n.(*ast.FuncLit); ok {
					ast.Inspect(fl, local)
					return false
				}
				return true
			})
		}
	}
	return names
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
func (p *Package) rewriteBindingCall(e *bindingEdits, off func(token.Pos) int, call *ast.CallExpr, fn BindingFunc, types map[string]BindingType, deferred, unused bool, knownBoundFuncs map[string]bool) string {
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
	// isType reports whether x is a type in a conversion. A C name that
	// is not a bound function is a C type.
	var isType func(x ast.Expr) bool
	isType = func(x ast.Expr) bool {
		switch t := x.(type) {
		case *ast.ParenExpr:
			return isType(t.X)
		case *ast.StarExpr:
			return isType(t.X)
		case *ast.SelectorExpr:
			if id, ok := t.X.(*ast.Ident); ok && id.Name == "C" {
				return !knownBoundFuncs[t.Sel.Name]
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

func (p *Package) cgoASTToGoDef(expr ast.Expr) string {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		if id, ok := e.X.(*ast.Ident); ok && id.Name == "C" {
			if goType, ok := primitiveCtypeToGo("_Ctype_" + e.Sel.Name); ok {
				return goType
			}
			return "_Ctype_" + e.Sel.Name
		}
		return p.cgoASTToGoDef(e.X) + "." + e.Sel.Name
	case *ast.StarExpr:
		return "*" + p.cgoASTToGoDef(e.X)
	default:
		var buf bytes.Buffer
		noSourceConf.Fprint(&buf, fset, expr)
		return buf.String()
	}
}

// cTypeError reports why values of the Go type expression t cannot be passed
// to or returned from C, or "" if they can. Go strings, slices, maps,
// channels, functions, and interfaces have no C equivalent, so they (and
// pointers to, arrays of, and structs containing them) are rejected rather
// than passed in their Go runtime representation. Named types are resolved
// through types, which maps a type name to its definition.
func cTypeError(t string, types map[string]BindingType) string {
	return cTypeErrorIn(t, types, make(map[string]bool))
}

func cTypeErrorIn(t string, types map[string]BindingType, seen map[string]bool) string {
	t = strings.TrimSpace(t)
	if strings.HasPrefix(t, "...") {
		return "Go variadic parameters are not supported"
	}
	expr, err := parser.ParseExpr(t)
	if err != nil {
		// Leave reporting malformed types to the ABI layout code.
		return ""
	}
	return cTypeExprError(expr, types, seen)
}

func cTypeExprError(expr ast.Expr, types map[string]BindingType, seen map[string]bool) string {
	switch e := expr.(type) {
	case *ast.ParenExpr:
		return cTypeExprError(e.X, types, seen)
	case *ast.StarExpr:
		return cTypeExprError(e.X, types, seen)
	case *ast.ArrayType:
		if e.Len == nil {
			return "it contains a Go slice"
		}
		return cTypeExprError(e.Elt, types, seen)
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
	case *ast.StructType:
		for _, field := range e.Fields.List {
			if why := cTypeExprError(field.Type, types, seen); why != "" {
				return why
			}
		}
		return ""
	case *ast.Ident:
		switch e.Name {
		case "string":
			return "it contains a Go string"
		case "any", "error":
			return "it contains a Go interface"
		}
		return cTypeNameError(e.Name, types, seen)
	case *ast.SelectorExpr:
		if x, ok := e.X.(*ast.Ident); ok {
			return cTypeNameError(x.Name+"."+e.Sel.Name, types, seen)
		}
	}
	return ""
}

func cTypeNameError(name string, types map[string]BindingType, seen map[string]bool) string {
	if seen[name] {
		return ""
	}
	seen[name] = true
	if bt, ok := types[name]; ok && bt.Def != "" {
		return cTypeErrorIn(bt.Def, types, seen)
	}
	return ""
}

func typeHasPointers(typeName string) bool {
	typeName = strings.TrimSpace(typeName)
	return strings.HasPrefix(typeName, "*") || strings.HasPrefix(typeName, "unsafe.Pointer") || strings.HasPrefix(typeName, "[]") || strings.HasPrefix(typeName, "string")
}

func (p *Package) writeGoTypes(bf *BindingEntry) {
	f := creat("_cgo_gotypes.go")
	defer f.Close()

	fmt.Fprintf(f, "// Code generated by cmd/cgo; DO NOT EDIT.\n\n")
	fmt.Fprintf(f, "package %s\n\n", bf.Package)
	fmt.Fprintf(f, "import \"unsafe\"\n\n")
	if *importRuntimeCgo {
		fmt.Fprintf(f, "import _cgopackage \"runtime/cgo\"\n\n")
		fmt.Fprintf(f, "type _ _cgopackage.Incomplete\n\n")
	}
	fmt.Fprintf(f, "var _ unsafe.Pointer\n\n")

	if len(bf.Funcs) > 0 {
		if goos != "windows" && goos != "aix" {
			importedLibs := make(map[string]bool)
			for _, fn := range bf.Funcs {
				if fn.Library != "" && !isStandardC(fn.Library) {
					importedLibs[dynLibName(fn.Library)] = true
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
			var sortedLibs []string
			for lib := range importedLibs {
				sortedLibs = append(sortedLibs, lib)
			}
			sort.Strings(sortedLibs)
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
		impLocal := func(sym string) string {
			if goos == "aix" {
				return "__cgo_imp_" + sym
			}
			return sym
		}

		emittedImp := make(map[string]bool)
		for _, fn := range bf.Funcs {
			cSym := fn.CName
			if fn.Static {
				// Resolved at link time from a .syso or .a file.
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
		if len(emittedImp) > 0 {
			fmt.Fprintf(f, "\n")
		}
	}

	for _, flag := range bf.LdFlags {
		fmt.Fprintf(f, "//go:cgo_ldflag %q\n", flag)
	}
	if len(bf.LdFlags) > 0 {
		fmt.Fprintf(f, "\n")
	}

	fmt.Fprintf(f, "//go:linkname _cgo_runtime_cgocall runtime.cgocall\n")
	fmt.Fprintf(f, "func _cgo_runtime_cgocall(fn unsafe.Pointer, arg uintptr) int32\n\n")
	fmt.Fprintf(f, "//go:linkname _Cgo_always_false runtime.cgoAlwaysFalse\n")
	fmt.Fprintf(f, "var _Cgo_always_false bool\n\n")
	fmt.Fprintf(f, "//go:linkname _Cgo_use runtime.cgoUse\n")
	fmt.Fprintf(f, "func _Cgo_use(interface{})\n\n")
	fmt.Fprintf(f, "//go:linkname _Cgo_keepalive runtime.cgoKeepAlive\n")
	fmt.Fprintf(f, "//go:noescape\n")
	fmt.Fprintf(f, "func _Cgo_keepalive(interface{})\n\n")
	fmt.Fprintf(f, "//go:linkname _cgoCheckPointer runtime.cgoCheckPointer\n")
	fmt.Fprintf(f, "//go:noescape\n")
	fmt.Fprintf(f, "func _cgoCheckPointer(interface{}, interface{})\n\n")

	definedCfuncs := make(map[string]bool)
	for _, fn := range bf.Funcs {
		definedCfuncs[fn.Name] = true
	}
	for _, fn := range bf.Funcs {
		fmt.Fprintf(f, "//go:linkname _cgo_trampoline_%s _cgo%s_trampoline_%s\n", fn.Name, cPrefix, fn.Name)
		fmt.Fprintf(f, "var _cgo_trampoline_%s byte\n\n", fn.Name)

		if len(fn.Params) > 0 || fn.Ret != "" {
			fmt.Fprintf(f, "type _cgo_arg_%s struct {\n", fn.Name)
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
		for i, pt := range fn.Params {
			paramDecls = append(paramDecls, fmt.Sprintf("p%d %s", i, pt))
			paramPass = append(paramPass, fmt.Sprintf("p%d: p%d", i, i))
			paramNames = append(paramNames, fmt.Sprintf("p%d", i))
		}
		retSig := ""
		retPrefix := ""
		if fn.Ret != "" {
			retSig = " " + fn.Ret
			retPrefix = "return "
		}
		fmt.Fprintf(f, "func _Cfunc_%s(%s)%s {\n", fn.Name, strings.Join(paramDecls, ", "), retSig)
		hasArgs := len(fn.Params) > 0 || fn.Ret != ""
		if !hasArgs {
			fmt.Fprintf(f, "\t_cgo_runtime_cgocall(unsafe.Pointer(&_cgo_trampoline_%s), 0)\n", fn.Name)
		} else {
			fmt.Fprintf(f, "\targs := _cgo_arg_%s{%s}\n", fn.Name, strings.Join(paramPass, ", "))
			fmt.Fprintf(f, "\t_cgo_runtime_cgocall(unsafe.Pointer(&_cgo_trampoline_%s), uintptr(unsafe.Pointer(&args)))\n", fn.Name)
		}
		var touchParams []string
		for i, name := range paramNames {
			if typeHasPointers(fn.Params[i]) {
				touchParams = append(touchParams, name)
			}
		}
		// The trampoline reads and writes args through the uintptr passed
		// to _cgo_runtime_cgocall, so keep args alive (and on the stack)
		// until the call returns, even if the function returns no result.
		if len(touchParams) > 0 || hasArgs {
			fmt.Fprintf(f, "\tif _Cgo_always_false {\n")
			for _, name := range touchParams {
				fmt.Fprintf(f, "\t\t_Cgo_use(%s)\n", name)
			}
			if hasArgs {
				fmt.Fprintf(f, "\t\t_Cgo_keepalive(&args)\n")
			}
			fmt.Fprintf(f, "\t}\n")
		}
		if fn.Ret != "" {
			fmt.Fprintf(f, "\treturn args.ret\n")
		}
		fmt.Fprintf(f, "}\n\n")

		// Calls that are not rewritten to check their own arguments (calls
		// from other packages and from files without import "C", indirect
		// calls, and calls whose arguments cannot be analyzed) go through
		// a wrapper that checks every argument that might contain a Go
		// pointer. Without knowing how an argument was formed, it checks
		// the complete memory allocation that the argument points to.
		entryFunc := bindingFuncEntry(fn, nil)
		if needs, any := bindingPointerChecks(fn, nil); any {
			fmt.Fprintf(f, "func %s(%s)%s {\n", entryFunc, strings.Join(paramDecls, ", "), retSig)
			for i, need := range needs {
				if need {
					fmt.Fprintf(f, "\t_cgoCheckPointer(p%d, nil)\n", i)
				}
			}
			fmt.Fprintf(f, "\t%s_Cfunc_%s(%s)\n}\n\n", retPrefix, fn.Name, strings.Join(paramNames, ", "))
		}
		if fn.GoName != "" {
			fmt.Fprintf(f, "func %s(%s)%s {\n\t%s%s(%s)\n}\n\n",
				fn.GoName, strings.Join(paramDecls, ", "), retSig, retPrefix, entryFunc, strings.Join(paramNames, ", "))
		}
		if cSym := fn.CName; fn.BindC && cSym != fn.Name && !definedCfuncs[cSym] {
			definedCfuncs[cSym] = true
			fmt.Fprintf(f, "var _Cfunc_%s = _Cfunc_%s\n\n", cSym, fn.Name)
		}
	}
}

func (p *Package) writeTrampolines(bf *BindingEntry) {
	f := creat("_cgo_trampolines.s")
	defer f.Close()

	fmt.Fprintf(f, "// Code generated by cmd/cgo; DO NOT EDIT.\n\n")
	fmt.Fprintf(f, "#include \"textflag.h\"\n\n")

	if len(bf.Funcs) > 0 {
		abi := getABI(goos, goarch)
		for _, fn := range bf.Funcs {
			abi.writeTrampoline(f, fn, nil)
		}
	}
}
