// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"io"
	"os"
	"sort"
	"strings"
)

var conf = printer.Config{Mode: printer.SourcePos, Tabwidth: 8}

// writeDefs creates output files to be compiled by gc and gcc.
func (p *Package) writeDefs() {
	var fgo2, fc io.Writer
	f := creat(*objDir + "_cgo_gotypes.go")
	defer f.Close()
	fgo2 = f
	if *gccgo {
		f := creat(*objDir + "_cgo_defun.c")
		defer f.Close()
		fc = f
	}
	fm := creat(*objDir + "_cgo_main.c")

	var gccgoInit bytes.Buffer

	fflg := creat(*objDir + "_cgo_flags")
	for k, v := range p.CgoFlags {
		fmt.Fprintf(fflg, "_CGO_%s=%s\n", k, strings.Join(v, " "))
		if k == "LDFLAGS" && !*gccgo {
			for _, arg := range v {
				fmt.Fprintf(fgo2, "//go:cgo_ldflag %q\n", arg)
			}
		}
	}
	fflg.Close()

	// Write C main file for using gcc to resolve imports.
	fmt.Fprintf(fm, "int main() { return 0; }\n")
	if *importRuntimeCgo {
		fmt.Fprintf(fm, "void crosscall2(void(*fn)(void*, int), void *a, int c) { }\n")
		fmt.Fprintf(fm, "void _cgo_wait_runtime_init_done() { }\n")
		fmt.Fprintf(fm, "char* _cgo_topofstack(void) { return (char*)0; }\n")
	} else {
		// If we're not importing runtime/cgo, we *are* runtime/cgo,
		// which provides these functions.  We just need a prototype.
		fmt.Fprintf(fm, "void crosscall2(void(*fn)(void*, int), void *a, int c);\n")
		fmt.Fprintf(fm, "void _cgo_wait_runtime_init_done();\n")
	}
	fmt.Fprintf(fm, "void _cgo_allocate(void *a, int c) { }\n")
	fmt.Fprintf(fm, "void _cgo_panic(void *a, int c) { }\n")
	fmt.Fprintf(fm, "void _cgo_reginit(void) { }\n")

	// Write second Go output: definitions of _C_xxx.
	// In a separate file so that the import of "unsafe" does not
	// pollute the original file.
	fmt.Fprintf(fgo2, "// Created by cgo - DO NOT EDIT\n\n")
	fmt.Fprintf(fgo2, "package %s\n\n", p.PackageName)
	fmt.Fprintf(fgo2, "import \"unsafe\"\n\n")
	if !*gccgo && *importRuntimeCgo {
		fmt.Fprintf(fgo2, "import _ \"runtime/cgo\"\n\n")
	}
	if *importSyscall {
		fmt.Fprintf(fgo2, "import \"syscall\"\n\n")
		fmt.Fprintf(fgo2, "var _ syscall.Errno\n")
	}
	fmt.Fprintf(fgo2, "func _Cgo_ptr(ptr unsafe.Pointer) unsafe.Pointer { return ptr }\n\n")

	if !*gccgo {
		fmt.Fprintf(fgo2, "//go:linkname _Cgo_always_false runtime.cgoAlwaysFalse\n")
		fmt.Fprintf(fgo2, "var _Cgo_always_false bool\n")
		fmt.Fprintf(fgo2, "//go:linkname _Cgo_use runtime.cgoUse\n")
		fmt.Fprintf(fgo2, "func _Cgo_use(interface{})\n")
	}

	typedefNames := make([]string, 0, len(typedef))
	for name := range typedef {
		typedefNames = append(typedefNames, name)
	}
	sort.Strings(typedefNames)
	for _, name := range typedefNames {
		def := typedef[name]
		fmt.Fprintf(fgo2, "type %s ", name)
		conf.Fprint(fgo2, fset, def.Go)
		fmt.Fprintf(fgo2, "\n\n")
	}
	if *gccgo {
		fmt.Fprintf(fgo2, "type _Ctype_void byte\n")
	} else {
		fmt.Fprintf(fgo2, "type _Ctype_void [0]byte\n")
	}

	if *gccgo {
		fmt.Fprint(fc, p.cPrologGccgo())
	} else {
		fmt.Fprint(fgo2, goProlog)
	}

	gccgoSymbolPrefix := p.gccgoSymbolPrefix()

	cVars := make(map[string]bool)
	for _, key := range nameKeys(p.Name) {
		n := p.Name[key]
		if !n.IsVar() {
			continue
		}

		if !cVars[n.C] {
			if *gccgo {
				fmt.Fprintf(fc, "extern byte *%s;\n", n.C)
			} else {
				fmt.Fprintf(fm, "extern char %s[];\n", n.C)
				fmt.Fprintf(fm, "void *_cgohack_%s = %s;\n\n", n.C, n.C)
				fmt.Fprintf(fgo2, "//go:linkname __cgo_%s %s\n", n.C, n.C)
				fmt.Fprintf(fgo2, "//go:cgo_import_static %s\n", n.C)
				fmt.Fprintf(fgo2, "var __cgo_%s byte\n", n.C)
			}
			cVars[n.C] = true
		}

		var node ast.Node
		if n.Kind == "var" {
			node = &ast.StarExpr{X: n.Type.Go}
		} else if n.Kind == "fpvar" {
			node = n.Type.Go
		} else {
			panic(fmt.Errorf("invalid var kind %q", n.Kind))
		}
		if *gccgo {
			fmt.Fprintf(fc, `extern void *%s __asm__("%s.%s");`, n.Mangle, gccgoSymbolPrefix, n.Mangle)
			fmt.Fprintf(&gccgoInit, "\t%s = &%s;\n", n.Mangle, n.C)
			fmt.Fprintf(fc, "\n")
		}

		fmt.Fprintf(fgo2, "var %s ", n.Mangle)
		conf.Fprint(fgo2, fset, node)
		if !*gccgo {
			fmt.Fprintf(fgo2, " = (")
			conf.Fprint(fgo2, fset, node)
			fmt.Fprintf(fgo2, ")(unsafe.Pointer(&__cgo_%s))", n.C)
		}
		fmt.Fprintf(fgo2, "\n")
	}
	if *gccgo {
		fmt.Fprintf(fc, "\n")
	}

	for _, key := range nameKeys(p.Name) {
		n := p.Name[key]
		if n.Const != "" {
			fmt.Fprintf(fgo2, "const _Cconst_%s = %s\n", n.Go, n.Const)
		}
	}
	fmt.Fprintf(fgo2, "\n")

	for _, key := range nameKeys(p.Name) {
		n := p.Name[key]
		if n.FuncType != nil {
			p.writeDefsFunc(fgo2, n)
		}
	}

	fgcc := creat(*objDir + "_cgo_export.c")
	fgcch := creat(*objDir + "_cgo_export.h")
	if *gccgo {
		p.writeGccgoExports(fgo2, fm, fgcc, fgcch)
	} else {
		p.writeExports(fgo2, fm, fgcc, fgcch)
	}
	if err := fgcc.Close(); err != nil {
		fatalf("%s", err)
	}
	if err := fgcch.Close(); err != nil {
		fatalf("%s", err)
	}

	if *exportHeader != "" && len(p.ExpFunc) > 0 {
		fexp := creat(*exportHeader)
		fgcch, err := os.Open(*objDir + "_cgo_export.h")
		if err != nil {
			fatalf("%s", err)
		}
		_, err = io.Copy(fexp, fgcch)
		if err != nil {
			fatalf("%s", err)
		}
		if err = fexp.Close(); err != nil {
			fatalf("%s", err)
		}
	}

	init := gccgoInit.String()
	if init != "" {
		fmt.Fprintln(fc, "static void init(void) __attribute__ ((constructor));")
		fmt.Fprintln(fc, "static void init(void) {")
		fmt.Fprint(fc, init)
		fmt.Fprintln(fc, "}")
	}
}

func dynimport(obj string) {
	stdout := os.Stdout
	if *dynout != "" {
		f, err := os.Create(*dynout)
		if err != nil {
			fatalf("%s", err)
		}
		stdout = f
	}

	fmt.Fprintf(stdout, "package %s\n", *dynpackage)

	if f, err := elf.Open(obj); err == nil {
		if *dynlinker {
			// Emit the cgo_dynamic_linker line.
			if sec := f.Section(".interp"); sec != nil {
				if data, err := sec.Data(); err == nil && len(data) > 1 {
					// skip trailing \0 in data
					fmt.Fprintf(stdout, "//go:cgo_dynamic_linker %q\n", string(data[:len(data)-1]))
				}
			}
		}
		sym, err := f.ImportedSymbols()
		if err != nil {
			fatalf("cannot load imported symbols from ELF file %s: %v", obj, err)
		}
		for _, s := range sym {
			targ := s.Name
			if s.Version != "" {
				targ += "#" + s.Version
			}
			fmt.Fprintf(stdout, "//go:cgo_import_dynamic %s %s %q\n", s.Name, targ, s.Library)
		}
		lib, err := f.ImportedLibraries()
		if err != nil {
			fatalf("cannot load imported libraries from ELF file %s: %v", obj, err)
		}
		for _, l := range lib {
			fmt.Fprintf(stdout, "//go:cgo_import_dynamic _ _ %q\n", l)
		}
		return
	}

	if f, err := macho.Open(obj); err == nil {
		sym, err := f.ImportedSymbols()
		if err != nil {
			fatalf("cannot load imported symbols from Mach-O file %s: %v", obj, err)
		}
		for _, s := range sym {
			if len(s) > 0 && s[0] == '_' {
				s = s[1:]
			}
			fmt.Fprintf(stdout, "//go:cgo_import_dynamic %s %s %q\n", s, s, "")
		}
		lib, err := f.ImportedLibraries()
		if err != nil {
			fatalf("cannot load imported libraries from Mach-O file %s: %v", obj, err)
		}
		for _, l := range lib {
			fmt.Fprintf(stdout, "//go:cgo_import_dynamic _ _ %q\n", l)
		}
		return
	}

	if f, err := pe.Open(obj); err == nil {
		sym, err := f.ImportedSymbols()
		if err != nil {
			fatalf("cannot load imported symbols from PE file %s: %v", obj, err)
		}
		for _, s := range sym {
			ss := strings.Split(s, ":")
			name := strings.Split(ss[0], "@")[0]
			fmt.Fprintf(stdout, "//go:cgo_import_dynamic %s %s %q\n", name, ss[0], strings.ToLower(ss[1]))
		}
		return
	}

	fatalf("cannot parse %s as ELF, Mach-O or PE", obj)
}

// Construct a gcc struct matching the gc argument frame.
// Assumes that in gcc, char is 1 byte, short 2 bytes, int 4 bytes, long long 8 bytes.
// These assumptions are checked by the gccProlog.
// Also assumes that gc convention is to word-align the
// input and output parameters.
func (p *Package) structType(n *Name) (string, int64) {
	var buf bytes.Buffer
	fmt.Fprint(&buf, "struct {\n")
	off := int64(0)
	for i, t := range n.FuncType.Params {
		if off%t.Align != 0 {
			pad := t.Align - off%t.Align
			fmt.Fprintf(&buf, "\t\tchar __pad%d[%d];\n", off, pad)
			off += pad
		}
		c := t.Typedef
		if c == "" {
			c = t.C.String()
		}
		fmt.Fprintf(&buf, "\t\t%s p%d;\n", c, i)
		off += t.Size
	}
	if off%p.PtrSize != 0 {
		pad := p.PtrSize - off%p.PtrSize
		fmt.Fprintf(&buf, "\t\tchar __pad%d[%d];\n", off, pad)
		off += pad
	}
	if t := n.FuncType.Result; t != nil {
		if off%t.Align != 0 {
			pad := t.Align - off%t.Align
			fmt.Fprintf(&buf, "\t\tchar __pad%d[%d];\n", off, pad)
			off += pad
		}
		qual := ""
		if c := t.C.String(); c[len(c)-1] == '*' {
			qual = "const "
		}
		fmt.Fprintf(&buf, "\t\t%s%s r;\n", qual, t.C)
		off += t.Size
	}
	if off%p.PtrSize != 0 {
		pad := p.PtrSize - off%p.PtrSize
		fmt.Fprintf(&buf, "\t\tchar __pad%d[%d];\n", off, pad)
		off += pad
	}
	if off == 0 {
		fmt.Fprintf(&buf, "\t\tchar unused;\n") // avoid empty struct
	}
	fmt.Fprintf(&buf, "\t}")
	return buf.String(), off
}

func (p *Package) writeDefsFunc(fgo2 io.Writer, n *Name) {
	name := n.Go
	gtype := n.FuncType.Go
	void := gtype.Results == nil || len(gtype.Results.List) == 0
	if n.AddError {
		// Add "error" to return type list.
		// Type list is known to be 0 or 1 element - it's a C function.
		err := &ast.Field{Type: ast.NewIdent("error")}
		l := gtype.Results.List
		if len(l) == 0 {
			l = []*ast.Field{err}
		} else {
			l = []*ast.Field{l[0], err}
		}
		t := new(ast.FuncType)
		*t = *gtype
		t.Results = &ast.FieldList{List: l}
		gtype = t
	}

	// Go func declaration.
	d := &ast.FuncDecl{
		Name: ast.NewIdent(n.Mangle),
		Type: gtype,
	}

	// Builtins defined in the C prolog.
	inProlog := builtinDefs[name] != ""
	cname := fmt.Sprintf("_cgo%s%s", cPrefix, n.Mangle)
	paramnames := []string(nil)
	for i, param := range d.Type.Params.List {
		paramName := fmt.Sprintf("p%d", i)
		param.Names = []*ast.Ident{ast.NewIdent(paramName)}
		paramnames = append(paramnames, paramName)
	}

	if *gccgo {
		// Gccgo style hooks.
		fmt.Fprint(fgo2, "\n")
		conf.Fprint(fgo2, fset, d)
		fmt.Fprint(fgo2, " {\n")
		if !inProlog {
			fmt.Fprint(fgo2, "\tdefer syscall.CgocallDone()\n")
			fmt.Fprint(fgo2, "\tsyscall.Cgocall()\n")
		}
		if n.AddError {
			fmt.Fprint(fgo2, "\tsyscall.SetErrno(0)\n")
		}
		fmt.Fprint(fgo2, "\t")
		if !void {
			fmt.Fprint(fgo2, "r := ")
		}
		fmt.Fprintf(fgo2, "%s(%s)\n", cname, strings.Join(paramnames, ", "))

		if n.AddError {
			fmt.Fprint(fgo2, "\te := syscall.GetErrno()\n")
			fmt.Fprint(fgo2, "\tif e != 0 {\n")
			fmt.Fprint(fgo2, "\t\treturn ")
			if !void {
				fmt.Fprint(fgo2, "r, ")
			}
			fmt.Fprint(fgo2, "e\n")
			fmt.Fprint(fgo2, "\t}\n")
			fmt.Fprint(fgo2, "\treturn ")
			if !void {
				fmt.Fprint(fgo2, "r, ")
			}
			fmt.Fprint(fgo2, "nil\n")
		} else if !void {
			fmt.Fprint(fgo2, "\treturn r\n")
		}

		fmt.Fprint(fgo2, "}\n")

		// declare the C function.
		fmt.Fprintf(fgo2, "//extern %s\n", cname)
		d.Name = ast.NewIdent(cname)
		if n.AddError {
			l := d.Type.Results.List
			d.Type.Results.List = l[:len(l)-1]
		}
		conf.Fprint(fgo2, fset, d)
		fmt.Fprint(fgo2, "\n")

		return
	}

	if inProlog {
		fmt.Fprint(fgo2, builtinDefs[name])
		return
	}

	// Wrapper calls into gcc, passing a pointer to the argument frame.
	fmt.Fprintf(fgo2, "//go:cgo_import_static %s\n", cname)
	fmt.Fprintf(fgo2, "//go:linkname __cgofn_%s %s\n", cname, cname)
	fmt.Fprintf(fgo2, "var __cgofn_%s byte\n", cname)
	fmt.Fprintf(fgo2, "var %s = unsafe.Pointer(&__cgofn_%s)\n", cname, cname)

	nret := 0
	if !void {
		d.Type.Results.List[0].Names = []*ast.Ident{ast.NewIdent("r1")}
		nret = 1
	}
	if n.AddError {
		d.Type.Results.List[nret].Names = []*ast.Ident{ast.NewIdent("r2")}
	}

	fmt.Fprint(fgo2, "\n")
	conf.Fprint(fgo2, fset, d)
	fmt.Fprint(fgo2, " {\n")

	// NOTE: Using uintptr to hide from escape analysis.
	arg := "0"
	if len(paramnames) > 0 {
		arg = "uintptr(unsafe.Pointer(&p0))"
	} else if !void {
		arg = "uintptr(unsafe.Pointer(&r1))"
	}

	prefix := ""
	if n.AddError {
		prefix = "errno := "
	}
	fmt.Fprintf(fgo2, "\t%s_cgo_runtime_cgocall(%s, %s)\n", prefix, cname, arg)
	if n.AddError {
		fmt.Fprintf(fgo2, "\tif errno != 0 { r2 = syscall.Errno(errno) }\n")
	}
	fmt.Fprintf(fgo2, "\tif _Cgo_always_false {\n")
	for i := range d.Type.Params.List {
		fmt.Fprintf(fgo2, "\t\t_Cgo_use(p%d)\n", i)
	}
	fmt.Fprintf(fgo2, "\t}\n")
	fmt.Fprintf(fgo2, "\treturn\n")
	fmt.Fprintf(fgo2, "}\n")
}

// writeOutput creates stubs for a specific source file to be compiled by gc
func (p *Package) writeOutput(f *File, srcfile string) {
	base := srcfile
	if strings.HasSuffix(base, ".go") {
		base = base[0 : len(base)-3]
	}
	base = strings.Map(slashToUnderscore, base)
	fgo1 := creat(*objDir + base + ".cgo1.go")
	fgcc := creat(*objDir + base + ".cgo2.c")

	p.GoFiles = append(p.GoFiles, base+".cgo1.go")
	p.GccFiles = append(p.GccFiles, base+".cgo2.c")

	// Write Go output: Go input with rewrites of C.xxx to _C_xxx.
	fmt.Fprintf(fgo1, "// Created by cgo - DO NOT EDIT\n\n")
	conf.Fprint(fgo1, fset, f.AST)

	// While we process the vars and funcs, also write gcc output.
	// Gcc output starts with the preamble.
	fmt.Fprintf(fgcc, "%s\n", f.Preamble)
	fmt.Fprintf(fgcc, "%s\n", gccProlog)

	for _, key := range nameKeys(f.Name) {
		n := f.Name[key]
		if n.FuncType != nil {
			p.writeOutputFunc(fgcc, n)
		}
	}

	fgo1.Close()
	fgcc.Close()
}

// fixGo converts the internal Name.Go field into the name we should show
// to users in error messages. There's only one for now: on input we rewrite
// C.malloc into C._CMalloc, so change it back here.
func fixGo(name string) string {
	if name == "_CMalloc" {
		return "malloc"
	}
	return name
}

var isBuiltin = map[string]bool{
	"_Cfunc_CString":   true,
	"_Cfunc_GoString":  true,
	"_Cfunc_GoStringN": true,
	"_Cfunc_GoBytes":   true,
	"_Cfunc__CMalloc":  true,
}

func (p *Package) writeOutputFunc(fgcc *os.File, n *Name) {
	name := n.Mangle
	if isBuiltin[name] || p.Written[name] {
		// The builtins are already defined in the C prolog, and we don't
		// want to duplicate function definitions we've already done.
		return
	}
	p.Written[name] = true

	if *gccgo {
		p.writeGccgoOutputFunc(fgcc, n)
		return
	}

	ctype, _ := p.structType(n)

	// Gcc wrapper unpacks the C argument struct
	// and calls the actual C function.
	if n.AddError {
		fmt.Fprintf(fgcc, "int\n")
	} else {
		fmt.Fprintf(fgcc, "void\n")
	}
	fmt.Fprintf(fgcc, "_cgo%s%s(void *v)\n", cPrefix, n.Mangle)
	fmt.Fprintf(fgcc, "{\n")
	if n.AddError {
		fmt.Fprintf(fgcc, "\terrno = 0;\n")
	}
	// We're trying to write a gcc struct that matches gc's layout.
	// Use packed attribute to force no padding in this struct in case
	// gcc has different packing requirements.
	fmt.Fprintf(fgcc, "\t%s %v *a = v;\n", ctype, p.packedAttribute())
	if n.FuncType.Result != nil {
		// Save the stack top for use below.
		fmt.Fprintf(fgcc, "\tchar *stktop = _cgo_topofstack();\n")
	}
	fmt.Fprintf(fgcc, "\t")
	if t := n.FuncType.Result; t != nil {
		fmt.Fprintf(fgcc, "__typeof__(a->r) r = ")
		if c := t.C.String(); c[len(c)-1] == '*' {
			fmt.Fprint(fgcc, "(__typeof__(a->r)) ")
		}
	}
	fmt.Fprintf(fgcc, "%s(", n.C)
	for i, t := range n.FuncType.Params {
		if i > 0 {
			fmt.Fprintf(fgcc, ", ")
		}
		// We know the type params are correct, because
		// the Go equivalents had good type params.
		// However, our version of the type omits the magic
		// words const and volatile, which can provoke
		// C compiler warnings.  Silence them by casting
		// all pointers to void*.  (Eventually that will produce
		// other warnings.)
		if c := t.C.String(); c[len(c)-1] == '*' {
			fmt.Fprintf(fgcc, "(void*)")
		}
		fmt.Fprintf(fgcc, "a->p%d", i)
	}
	fmt.Fprintf(fgcc, ");\n")
	if n.FuncType.Result != nil {
		// The cgo call may have caused a stack copy (via a callback).
		// Adjust the return value pointer appropriately.
		fmt.Fprintf(fgcc, "\ta = (void*)((char*)a + (_cgo_topofstack() - stktop));\n")
		// Save the return value.
		fmt.Fprintf(fgcc, "\ta->r = r;\n")
	}
	if n.AddError {
		fmt.Fprintf(fgcc, "\treturn errno;\n")
	}
	fmt.Fprintf(fgcc, "}\n")
	fmt.Fprintf(fgcc, "\n")
}

// Write out a wrapper for a function when using gccgo.  This is a
// simple wrapper that just calls the real function.  We only need a
// wrapper to support static functions in the prologue--without a
// wrapper, we can't refer to the function, since the reference is in
// a different file.
func (p *Package) writeGccgoOutputFunc(fgcc *os.File, n *Name) {
	if t := n.FuncType.Result; t != nil {
		fmt.Fprintf(fgcc, "%s\n", t.C.String())
	} else {
		fmt.Fprintf(fgcc, "void\n")
	}
	fmt.Fprintf(fgcc, "_cgo%s%s(", cPrefix, n.Mangle)
	for i, t := range n.FuncType.Params {
		if i > 0 {
			fmt.Fprintf(fgcc, ", ")
		}
		c := t.Typedef
		if c == "" {
			c = t.C.String()
		}
		fmt.Fprintf(fgcc, "%s p%d", c, i)
	}
	fmt.Fprintf(fgcc, ")\n")
	fmt.Fprintf(fgcc, "{\n")
	fmt.Fprintf(fgcc, "\t")
	if t := n.FuncType.Result; t != nil {
		fmt.Fprintf(fgcc, "return ")
		// Cast to void* to avoid warnings due to omitted qualifiers.
		if c := t.C.String(); c[len(c)-1] == '*' {
			fmt.Fprintf(fgcc, "(void*)")
		}
	}
	fmt.Fprintf(fgcc, "%s(", n.C)
	for i, t := range n.FuncType.Params {
		if i > 0 {
			fmt.Fprintf(fgcc, ", ")
		}
		// Cast to void* to avoid warnings due to omitted qualifiers.
		if c := t.C.String(); c[len(c)-1] == '*' {
			fmt.Fprintf(fgcc, "(void*)")
		}
		fmt.Fprintf(fgcc, "p%d", i)
	}
	fmt.Fprintf(fgcc, ");\n")
	fmt.Fprintf(fgcc, "}\n")
	fmt.Fprintf(fgcc, "\n")
}

// packedAttribute returns host compiler struct attribute that will be
// used to match gc's struct layout. For example, on 386 Windows,
// gcc wants to 8-align int64s, but gc does not.
// Use __gcc_struct__ to work around http://gcc.gnu.org/PR52991 on x86,
// and https://golang.org/issue/5603.
func (p *Package) packedAttribute() string {
	s := "__attribute__((__packed__"
	if !p.GccIsClang && (goarch == "amd64" || goarch == "386") {
		s += ", __gcc_struct__"
	}
	return s + "))"
}

// Write out the various stubs we need to support functions exported
// from Go so that they are callable from C.
func (p *Package) writeExports(fgo2, fm, fgcc, fgcch io.Writer) {
	p.writeExportHeader(fgcch)

	fmt.Fprintf(fgcc, "/* Created by cgo - DO NOT EDIT. */\n")
	fmt.Fprintf(fgcc, "#include \"_cgo_export.h\"\n\n")

	fmt.Fprintf(fgcc, "extern void crosscall2(void (*fn)(void *, int), void *, int);\n")
	fmt.Fprintf(fgcc, "extern void _cgo_wait_runtime_init_done();\n\n")

	for _, exp := range p.ExpFunc {
		fn := exp.Func

		// Construct a gcc struct matching the gc argument and
		// result frame.  The gcc struct will be compiled with
		// __attribute__((packed)) so all padding must be accounted
		// for explicitly.
		ctype := "struct {\n"
		off := int64(0)
		npad := 0
		if fn.Recv != nil {
			t := p.cgoType(fn.Recv.List[0].Type)
			ctype += fmt.Sprintf("\t\t%s recv;\n", t.C)
			off += t.Size
		}
		fntype := fn.Type
		forFieldList(fntype.Params,
			func(i int, atype ast.Expr) {
				t := p.cgoType(atype)
				if off%t.Align != 0 {
					pad := t.Align - off%t.Align
					ctype += fmt.Sprintf("\t\tchar __pad%d[%d];\n", npad, pad)
					off += pad
					npad++
				}
				ctype += fmt.Sprintf("\t\t%s p%d;\n", t.C, i)
				off += t.Size
			})
		if off%p.PtrSize != 0 {
			pad := p.PtrSize - off%p.PtrSize
			ctype += fmt.Sprintf("\t\tchar __pad%d[%d];\n", npad, pad)
			off += pad
			npad++
		}
		forFieldList(fntype.Results,
			func(i int, atype ast.Expr) {
				t := p.cgoType(atype)
				if off%t.Align != 0 {
					pad := t.Align - off%t.Align
					ctype += fmt.Sprintf("\t\tchar __pad%d[%d];\n", npad, pad)
					off += pad
					npad++
				}
				ctype += fmt.Sprintf("\t\t%s r%d;\n", t.C, i)
				off += t.Size
			})
		if off%p.PtrSize != 0 {
			pad := p.PtrSize - off%p.PtrSize
			ctype += fmt.Sprintf("\t\tchar __pad%d[%d];\n", npad, pad)
			off += pad
			npad++
		}
		if ctype == "struct {\n" {
			ctype += "\t\tchar unused;\n" // avoid empty struct
		}
		ctype += "\t}"

		// Get the return type of the wrapper function
		// compiled by gcc.
		gccResult := ""
		if fntype.Results == nil || len(fntype.Results.List) == 0 {
			gccResult = "void"
		} else if len(fntype.Results.List) == 1 && len(fntype.Results.List[0].Names) <= 1 {
			gccResult = p.cgoType(fntype.Results.List[0].Type).C.String()
		} else {
			fmt.Fprintf(fgcch, "\n/* Return type for %s */\n", exp.ExpName)
			fmt.Fprintf(fgcch, "struct %s_return {\n", exp.ExpName)
			forFieldList(fntype.Results,
				func(i int, atype ast.Expr) {
					fmt.Fprintf(fgcch, "\t%s r%d;\n", p.cgoType(atype).C, i)
				})
			fmt.Fprintf(fgcch, "};\n")
			gccResult = "struct " + exp.ExpName + "_return"
		}

		// Build the wrapper function compiled by gcc.
		s := fmt.Sprintf("%s %s(", gccResult, exp.ExpName)
		if fn.Recv != nil {
			s += p.cgoType(fn.Recv.List[0].Type).C.String()
			s += " recv"
		}
		forFieldList(fntype.Params,
			func(i int, atype ast.Expr) {
				if i > 0 || fn.Recv != nil {
					s += ", "
				}
				s += fmt.Sprintf("%s p%d", p.cgoType(atype).C, i)
			})
		s += ")"

		if len(exp.Doc) > 0 {
			fmt.Fprintf(fgcch, "\n%s", exp.Doc)
		}
		fmt.Fprintf(fgcch, "\nextern %s;\n", s)

		fmt.Fprintf(fgcc, "extern void _cgoexp%s_%s(void *, int);\n", cPrefix, exp.ExpName)
		fmt.Fprintf(fgcc, "\n%s\n", s)
		fmt.Fprintf(fgcc, "{\n")
		fmt.Fprintf(fgcc, "\t_cgo_wait_runtime_init_done();\n")
		fmt.Fprintf(fgcc, "\t%s %v a;\n", ctype, p.packedAttribute())
		if gccResult != "void" && (len(fntype.Results.List) > 1 || len(fntype.Results.List[0].Names) > 1) {
			fmt.Fprintf(fgcc, "\t%s r;\n", gccResult)
		}
		if fn.Recv != nil {
			fmt.Fprintf(fgcc, "\ta.recv = recv;\n")
		}
		forFieldList(fntype.Params,
			func(i int, atype ast.Expr) {
				fmt.Fprintf(fgcc, "\ta.p%d = p%d;\n", i, i)
			})
		fmt.Fprintf(fgcc, "\tcrosscall2(_cgoexp%s_%s, &a, %d);\n", cPrefix, exp.ExpName, off)
		if gccResult != "void" {
			if len(fntype.Results.List) == 1 && len(fntype.Results.List[0].Names) <= 1 {
				fmt.Fprintf(fgcc, "\treturn a.r0;\n")
			} else {
				forFieldList(fntype.Results,
					func(i int, atype ast.Expr) {
						fmt.Fprintf(fgcc, "\tr.r%d = a.r%d;\n", i, i)
					})
				fmt.Fprintf(fgcc, "\treturn r;\n")
			}
		}
		fmt.Fprintf(fgcc, "}\n")

		// Build the wrapper function compiled by gc.
		goname := exp.Func.Name.Name
		if fn.Recv != nil {
			goname = "_cgoexpwrap" + cPrefix + "_" + fn.Recv.List[0].Names[0].Name + "_" + goname
		}
		fmt.Fprintf(fgo2, "//go:cgo_export_dynamic %s\n", goname)
		fmt.Fprintf(fgo2, "//go:linkname _cgoexp%s_%s _cgoexp%s_%s\n", cPrefix, exp.ExpName, cPrefix, exp.ExpName)
		fmt.Fprintf(fgo2, "//go:cgo_export_static _cgoexp%s_%s\n", cPrefix, exp.ExpName)
		fmt.Fprintf(fgo2, "//go:nosplit\n") // no split stack, so no use of m or g
		fmt.Fprintf(fgo2, "//go:norace\n")  // must not have race detector calls inserted
		fmt.Fprintf(fgo2, "func _cgoexp%s_%s(a unsafe.Pointer, n int32) {", cPrefix, exp.ExpName)
		fmt.Fprintf(fgo2, "\tfn := %s\n", goname)
		// The indirect here is converting from a Go function pointer to a C function pointer.
		fmt.Fprintf(fgo2, "\t_cgo_runtime_cgocallback(**(**unsafe.Pointer)(unsafe.Pointer(&fn)), a, uintptr(n));\n")
		fmt.Fprintf(fgo2, "}\n")

		fmt.Fprintf(fm, "int _cgoexp%s_%s;\n", cPrefix, exp.ExpName)

		// Calling a function with a receiver from C requires
		// a Go wrapper function.
		if fn.Recv != nil {
			fmt.Fprintf(fgo2, "func %s(recv ", goname)
			conf.Fprint(fgo2, fset, fn.Recv.List[0].Type)
			forFieldList(fntype.Params,
				func(i int, atype ast.Expr) {
					fmt.Fprintf(fgo2, ", p%d ", i)
					conf.Fprint(fgo2, fset, atype)
				})
			fmt.Fprintf(fgo2, ")")
			if gccResult != "void" {
				fmt.Fprint(fgo2, " (")
				forFieldList(fntype.Results,
					func(i int, atype ast.Expr) {
						if i > 0 {
							fmt.Fprint(fgo2, ", ")
						}
						conf.Fprint(fgo2, fset, atype)
					})
				fmt.Fprint(fgo2, ")")
			}
			fmt.Fprint(fgo2, " {\n")
			fmt.Fprint(fgo2, "\t")
			if gccResult != "void" {
				fmt.Fprint(fgo2, "return ")
			}
			fmt.Fprintf(fgo2, "recv.%s(", exp.Func.Name)
			forFieldList(fntype.Params,
				func(i int, atype ast.Expr) {
					if i > 0 {
						fmt.Fprint(fgo2, ", ")
					}
					fmt.Fprintf(fgo2, "p%d", i)
				})
			fmt.Fprint(fgo2, ")\n")
			fmt.Fprint(fgo2, "}\n")
		}
	}

	fmt.Fprintf(fgcch, "%s", gccExportHeaderEpilog)
}

// Write out the C header allowing C code to call exported gccgo functions.
func (p *Package) writeGccgoExports(fgo2, fm, fgcc, fgcch io.Writer) {
	gccgoSymbolPrefix := p.gccgoSymbolPrefix()

	p.writeExportHeader(fgcch)

	fmt.Fprintf(fgcc, "/* Created by cgo - DO NOT EDIT. */\n")
	fmt.Fprintf(fgcc, "#include \"_cgo_export.h\"\n")

	fmt.Fprintf(fgcc, "%s\n", gccgoExportFileProlog)

	for _, exp := range p.ExpFunc {
		fn := exp.Func
		fntype := fn.Type

		cdeclBuf := new(bytes.Buffer)
		resultCount := 0
		forFieldList(fntype.Results,
			func(i int, atype ast.Expr) { resultCount++ })
		switch resultCount {
		case 0:
			fmt.Fprintf(cdeclBuf, "void")
		case 1:
			forFieldList(fntype.Results,
				func(i int, atype ast.Expr) {
					t := p.cgoType(atype)
					fmt.Fprintf(cdeclBuf, "%s", t.C)
				})
		default:
			// Declare a result struct.
			fmt.Fprintf(fgcch, "\n/* Return type for %s */\n", exp.ExpName)
			fmt.Fprintf(fgcch, "struct %s_result {\n", exp.ExpName)
			forFieldList(fntype.Results,
				func(i int, atype ast.Expr) {
					t := p.cgoType(atype)
					fmt.Fprintf(fgcch, "\t%s r%d;\n", t.C, i)
				})
			fmt.Fprintf(fgcch, "};\n")
			fmt.Fprintf(cdeclBuf, "struct %s_result", exp.ExpName)
		}

		cRet := cdeclBuf.String()

		cdeclBuf = new(bytes.Buffer)
		fmt.Fprintf(cdeclBuf, "(")
		if fn.Recv != nil {
			fmt.Fprintf(cdeclBuf, "%s recv", p.cgoType(fn.Recv.List[0].Type).C.String())
		}
		// Function parameters.
		forFieldList(fntype.Params,
			func(i int, atype ast.Expr) {
				if i > 0 || fn.Recv != nil {
					fmt.Fprintf(cdeclBuf, ", ")
				}
				t := p.cgoType(atype)
				fmt.Fprintf(cdeclBuf, "%s p%d", t.C, i)
			})
		fmt.Fprintf(cdeclBuf, ")")
		cParams := cdeclBuf.String()

		if len(exp.Doc) > 0 {
			fmt.Fprintf(fgcch, "\n%s", exp.Doc)
		}

		// We need to use a name that will be exported by the
		// Go code; otherwise gccgo will make it static and we
		// will not be able to link against it from the C
		// code.
		goName := "Cgoexp_" + exp.ExpName
		fmt.Fprintf(fgcch, `extern %s %s %s __asm__("%s.%s");`, cRet, goName, cParams, gccgoSymbolPrefix, goName)
		fmt.Fprint(fgcch, "\n")

		// Use a #define so that the C code that includes
		// cgo_export.h will be able to refer to the Go
		// function using the expected name.
		fmt.Fprintf(fgcch, "#define %s %s\n", exp.ExpName, goName)

		// Use a #undef in _cgo_export.c so that we ignore the
		// #define from cgo_export.h, since here we are
		// defining the real function.
		fmt.Fprintf(fgcc, "#undef %s\n", exp.ExpName)

		fmt.Fprint(fgcc, "\n")
		fmt.Fprintf(fgcc, "%s %s %s {\n", cRet, exp.ExpName, cParams)
		fmt.Fprintf(fgcc, "\tif(_cgo_wait_runtime_init_done)\n")
		fmt.Fprintf(fgcc, "\t\t_cgo_wait_runtime_init_done();\n")
		fmt.Fprint(fgcc, "\t")
		if resultCount > 0 {
			fmt.Fprint(fgcc, "return ")
		}
		fmt.Fprintf(fgcc, "%s(", goName)
		if fn.Recv != nil {
			fmt.Fprint(fgcc, "recv")
		}
		forFieldList(fntype.Params,
			func(i int, atype ast.Expr) {
				if i > 0 || fn.Recv != nil {
					fmt.Fprintf(fgcc, ", ")
				}
				fmt.Fprintf(fgcc, "p%d", i)
			})
		fmt.Fprint(fgcc, ");\n")
		fmt.Fprint(fgcc, "}\n")

		// Dummy declaration for _cgo_main.c
		fmt.Fprintf(fm, `char %s[1] __asm__("%s.%s");`, goName, gccgoSymbolPrefix, goName)
		fmt.Fprint(fm, "\n")

		// For gccgo we use a wrapper function in Go, in order
		// to call CgocallBack and CgocallBackDone.

		// This code uses printer.Fprint, not conf.Fprint,
		// because we don't want //line comments in the middle
		// of the function types.
		fmt.Fprint(fgo2, "\n")
		fmt.Fprintf(fgo2, "func %s(", goName)
		if fn.Recv != nil {
			fmt.Fprint(fgo2, "recv ")
			printer.Fprint(fgo2, fset, fn.Recv.List[0].Type)
		}
		forFieldList(fntype.Params,
			func(i int, atype ast.Expr) {
				if i > 0 || fn.Recv != nil {
					fmt.Fprintf(fgo2, ", ")
				}
				fmt.Fprintf(fgo2, "p%d ", i)
				printer.Fprint(fgo2, fset, atype)
			})
		fmt.Fprintf(fgo2, ")")
		if resultCount > 0 {
			fmt.Fprintf(fgo2, " (")
			forFieldList(fntype.Results,
				func(i int, atype ast.Expr) {
					if i > 0 {
						fmt.Fprint(fgo2, ", ")
					}
					printer.Fprint(fgo2, fset, atype)
				})
			fmt.Fprint(fgo2, ")")
		}
		fmt.Fprint(fgo2, " {\n")
		fmt.Fprint(fgo2, "\tsyscall.CgocallBack()\n")
		fmt.Fprint(fgo2, "\tdefer syscall.CgocallBackDone()\n")
		fmt.Fprint(fgo2, "\t")
		if resultCount > 0 {
			fmt.Fprint(fgo2, "return ")
		}
		if fn.Recv != nil {
			fmt.Fprint(fgo2, "recv.")
		}
		fmt.Fprintf(fgo2, "%s(", exp.Func.Name)
		forFieldList(fntype.Params,
			func(i int, atype ast.Expr) {
				if i > 0 {
					fmt.Fprint(fgo2, ", ")
				}
				fmt.Fprintf(fgo2, "p%d", i)
			})
		fmt.Fprint(fgo2, ")\n")
		fmt.Fprint(fgo2, "}\n")
	}

	fmt.Fprintf(fgcch, "%s", gccExportHeaderEpilog)
}

// writeExportHeader writes out the start of the _cgo_export.h file.
func (p *Package) writeExportHeader(fgcch io.Writer) {
	fmt.Fprintf(fgcch, "/* Created by \"go tool cgo\" - DO NOT EDIT. */\n\n")
	pkg := *importPath
	if pkg == "" {
		pkg = p.PackagePath
	}
	fmt.Fprintf(fgcch, "/* package %s */\n\n", pkg)

	fmt.Fprintf(fgcch, "/* Start of preamble from import \"C\" comments.  */\n\n")
	fmt.Fprintf(fgcch, "%s\n", p.Preamble)
	fmt.Fprintf(fgcch, "\n/* End of preamble from import \"C\" comments.  */\n\n")

	fmt.Fprintf(fgcch, "%s\n", p.gccExportHeaderProlog())
}

// Return the package prefix when using gccgo.
func (p *Package) gccgoSymbolPrefix() string {
	if !*gccgo {
		return ""
	}

	clean := func(r rune) rune {
		switch {
		case 'A' <= r && r <= 'Z', 'a' <= r && r <= 'z',
			'0' <= r && r <= '9':
			return r
		}
		return '_'
	}

	if *gccgopkgpath != "" {
		return strings.Map(clean, *gccgopkgpath)
	}
	if *gccgoprefix == "" && p.PackageName == "main" {
		return "main"
	}
	prefix := strings.Map(clean, *gccgoprefix)
	if prefix == "" {
		prefix = "go"
	}
	return prefix + "." + p.PackageName
}

// Call a function for each entry in an ast.FieldList, passing the
// index into the list and the type.
func forFieldList(fl *ast.FieldList, fn func(int, ast.Expr)) {
	if fl == nil {
		return
	}
	i := 0
	for _, r := range fl.List {
		if r.Names == nil {
			fn(i, r.Type)
			i++
		} else {
			for range r.Names {
				fn(i, r.Type)
				i++
			}
		}
	}
}

func c(repr string, args ...interface{}) *TypeRepr {
	return &TypeRepr{repr, args}
}

// Map predeclared Go types to Type.
var goTypes = map[string]*Type{
	"bool":       {Size: 1, Align: 1, C: c("GoUint8")},
	"byte":       {Size: 1, Align: 1, C: c("GoUint8")},
	"int":        {Size: 0, Align: 0, C: c("GoInt")},
	"uint":       {Size: 0, Align: 0, C: c("GoUint")},
	"rune":       {Size: 4, Align: 4, C: c("GoInt32")},
	"int8":       {Size: 1, Align: 1, C: c("GoInt8")},
	"uint8":      {Size: 1, Align: 1, C: c("GoUint8")},
	"int16":      {Size: 2, Align: 2, C: c("GoInt16")},
	"uint16":     {Size: 2, Align: 2, C: c("GoUint16")},
	"int32":      {Size: 4, Align: 4, C: c("GoInt32")},
	"uint32":     {Size: 4, Align: 4, C: c("GoUint32")},
	"int64":      {Size: 8, Align: 8, C: c("GoInt64")},
	"uint64":     {Size: 8, Align: 8, C: c("GoUint64")},
	"float32":    {Size: 4, Align: 4, C: c("GoFloat32")},
	"float64":    {Size: 8, Align: 8, C: c("GoFloat64")},
	"complex64":  {Size: 8, Align: 8, C: c("GoComplex64")},
	"complex128": {Size: 16, Align: 16, C: c("GoComplex128")},
}

// Map an ast type to a Type.
func (p *Package) cgoType(e ast.Expr) *Type {
	switch t := e.(type) {
	case *ast.StarExpr:
		x := p.cgoType(t.X)
		return &Type{Size: p.PtrSize, Align: p.PtrSize, C: c("%s*", x.C)}
	case *ast.ArrayType:
		if t.Len == nil {
			// Slice: pointer, len, cap.
			return &Type{Size: p.PtrSize * 3, Align: p.PtrSize, C: c("GoSlice")}
		}
	case *ast.StructType:
		// TODO
	case *ast.FuncType:
		return &Type{Size: p.PtrSize, Align: p.PtrSize, C: c("void*")}
	case *ast.InterfaceType:
		return &Type{Size: 2 * p.PtrSize, Align: p.PtrSize, C: c("GoInterface")}
	case *ast.MapType:
		return &Type{Size: p.PtrSize, Align: p.PtrSize, C: c("GoMap")}
	case *ast.ChanType:
		return &Type{Size: p.PtrSize, Align: p.PtrSize, C: c("GoChan")}
	case *ast.Ident:
		// Look up the type in the top level declarations.
		// TODO: Handle types defined within a function.
		for _, d := range p.Decl {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for _, spec := range gd.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				if ts.Name.Name == t.Name {
					return p.cgoType(ts.Type)
				}
			}
		}
		if def := typedef[t.Name]; def != nil {
			return def
		}
		if t.Name == "uintptr" {
			return &Type{Size: p.PtrSize, Align: p.PtrSize, C: c("GoUintptr")}
		}
		if t.Name == "string" {
			// The string data is 1 pointer + 1 (pointer-sized) int.
			return &Type{Size: 2 * p.PtrSize, Align: p.PtrSize, C: c("GoString")}
		}
		if t.Name == "error" {
			return &Type{Size: 2 * p.PtrSize, Align: p.PtrSize, C: c("GoInterface")}
		}
		if r, ok := goTypes[t.Name]; ok {
			if r.Size == 0 { // int or uint
				rr := new(Type)
				*rr = *r
				rr.Size = p.IntSize
				rr.Align = p.IntSize
				r = rr
			}
			if r.Align > p.PtrSize {
				r.Align = p.PtrSize
			}
			return r
		}
		error_(e.Pos(), "unrecognized Go type %s", t.Name)
		return &Type{Size: 4, Align: 4, C: c("int")}
	case *ast.SelectorExpr:
		id, ok := t.X.(*ast.Ident)
		if ok && id.Name == "unsafe" && t.Sel.Name == "Pointer" {
			return &Type{Size: p.PtrSize, Align: p.PtrSize, C: c("void*")}
		}
	}
	error_(e.Pos(), "Go type not supported in export: %s", gofmt(e))
	return &Type{Size: 4, Align: 4, C: c("int")}
}

const gccProlog = `
// Usual nonsense: if x and y are not equal, the type will be invalid
// (have a negative array count) and an inscrutable error will come
// out of the compiler and hopefully mention "name".
#define __cgo_compile_assert_eq(x, y, name) typedef char name[(x-y)*(x-y)*-2+1];

// Check at compile time that the sizes we use match our expectations.
#define __cgo_size_assert(t, n) __cgo_compile_assert_eq(sizeof(t), n, _cgo_sizeof_##t##_is_not_##n)

__cgo_size_assert(char, 1)
__cgo_size_assert(short, 2)
__cgo_size_assert(int, 4)
typedef long long __cgo_long_long;
__cgo_size_assert(__cgo_long_long, 8)
__cgo_size_assert(float, 4)
__cgo_size_assert(double, 8)

extern char* _cgo_topofstack(void);

#include <errno.h>
#include <string.h>
`

const builtinProlog = `
#include <stddef.h> /* for ptrdiff_t and size_t below */

/* Define intgo when compiling with GCC.  */
typedef ptrdiff_t intgo;

typedef struct { char *p; intgo n; } _GoString_;
typedef struct { char *p; intgo n; intgo c; } _GoBytes_;
_GoString_ GoString(char *p);
_GoString_ GoStringN(char *p, int l);
_GoBytes_ GoBytes(void *p, int n);
char *CString(_GoString_);
void *_CMalloc(size_t);
`

const goProlog = `
//go:linkname _cgo_runtime_cgocall runtime.cgocall
func _cgo_runtime_cgocall(unsafe.Pointer, uintptr) int32

//go:linkname _cgo_runtime_cmalloc runtime.cmalloc
func _cgo_runtime_cmalloc(uintptr) unsafe.Pointer

//go:linkname _cgo_runtime_cgocallback runtime.cgocallback
func _cgo_runtime_cgocallback(unsafe.Pointer, unsafe.Pointer, uintptr)
`

const goStringDef = `
//go:linkname _cgo_runtime_gostring runtime.gostring
func _cgo_runtime_gostring(*_Ctype_char) string

func _Cfunc_GoString(p *_Ctype_char) string {
	return _cgo_runtime_gostring(p)
}
`

const goStringNDef = `
//go:linkname _cgo_runtime_gostringn runtime.gostringn
func _cgo_runtime_gostringn(*_Ctype_char, int) string

func _Cfunc_GoStringN(p *_Ctype_char, l _Ctype_int) string {
	return _cgo_runtime_gostringn(p, int(l))
}
`

const goBytesDef = `
//go:linkname _cgo_runtime_gobytes runtime.gobytes
func _cgo_runtime_gobytes(unsafe.Pointer, int) []byte

func _Cfunc_GoBytes(p unsafe.Pointer, l _Ctype_int) []byte {
	return _cgo_runtime_gobytes(p, int(l))
}
`

const cStringDef = `
func _Cfunc_CString(s string) *_Ctype_char {
	p := _cgo_runtime_cmalloc(uintptr(len(s)+1))
	pp := (*[1<<30]byte)(p)
	copy(pp[:], s)
	pp[len(s)] = 0
	return (*_Ctype_char)(p)
}
`

const cMallocDef = `
func _Cfunc__CMalloc(n _Ctype_size_t) unsafe.Pointer {
	return _cgo_runtime_cmalloc(uintptr(n))
}
`

var builtinDefs = map[string]string{
	"GoString":  goStringDef,
	"GoStringN": goStringNDef,
	"GoBytes":   goBytesDef,
	"CString":   cStringDef,
	"_CMalloc":  cMallocDef,
}

func (p *Package) cPrologGccgo() string {
	return strings.Replace(cPrologGccgo, "PREFIX", cPrefix, -1)
}

const cPrologGccgo = `
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char byte;
typedef intptr_t intgo;

struct __go_string {
	const unsigned char *__data;
	intgo __length;
};

typedef struct __go_open_array {
	void* __values;
	intgo __count;
	intgo __capacity;
} Slice;

struct __go_string __go_byte_array_to_string(const void* p, intgo len);
struct __go_open_array __go_string_to_byte_array (struct __go_string str);

const char *_cgoPREFIX_Cfunc_CString(struct __go_string s) {
	char *p = malloc(s.__length+1);
	memmove(p, s.__data, s.__length);
	p[s.__length] = 0;
	return p;
}

struct __go_string _cgoPREFIX_Cfunc_GoString(char *p) {
	intgo len = (p != NULL) ? strlen(p) : 0;
	return __go_byte_array_to_string(p, len);
}

struct __go_string _cgoPREFIX_Cfunc_GoStringN(char *p, int32_t n) {
	return __go_byte_array_to_string(p, n);
}

Slice _cgoPREFIX_Cfunc_GoBytes(char *p, int32_t n) {
	struct __go_string s = { (const unsigned char *)p, n };
	return __go_string_to_byte_array(s);
}

extern void runtime_throw(const char *);
void *_cgoPREFIX_Cfunc__CMalloc(size_t n) {
        void *p = malloc(n);
        if(p == NULL && n == 0)
                p = malloc(1);
        if(p == NULL)
                runtime_throw("runtime: C malloc failed");
        return p;
}
`

func (p *Package) gccExportHeaderProlog() string {
	return strings.Replace(gccExportHeaderProlog, "GOINTBITS", fmt.Sprint(8*p.IntSize), -1)
}

const gccExportHeaderProlog = `
/* Start of boilerplate cgo prologue.  */

#ifndef GO_CGO_PROLOGUE_H
#define GO_CGO_PROLOGUE_H

typedef signed char GoInt8;
typedef unsigned char GoUint8;
typedef short GoInt16;
typedef unsigned short GoUint16;
typedef int GoInt32;
typedef unsigned int GoUint32;
typedef long long GoInt64;
typedef unsigned long long GoUint64;
typedef GoIntGOINTBITS GoInt;
typedef GoUintGOINTBITS GoUint;
typedef __SIZE_TYPE__ GoUintptr;
typedef float GoFloat32;
typedef double GoFloat64;
typedef __complex float GoComplex64;
typedef __complex double GoComplex128;

// static assertion to make sure the file is being used on architecture
// at least with matching size of GoInt.
typedef char _check_for_GOINTBITS_bit_pointer_matching_GoInt[sizeof(void*)==GOINTBITS/8 ? 1:-1];

typedef struct { char *p; GoInt n; } GoString;
typedef void *GoMap;
typedef void *GoChan;
typedef struct { void *t; void *v; } GoInterface;
typedef struct { void *data; GoInt len; GoInt cap; } GoSlice;

#endif

/* End of boilerplate cgo prologue.  */

#ifdef __cplusplus
extern "C" {
#endif
`

// gccExportHeaderEpilog goes at the end of the generated header file.
const gccExportHeaderEpilog = `
#ifdef __cplusplus
}
#endif
`

// gccgoExportFileProlog is written to the _cgo_export.c file when
// using gccgo.
// We use weak declarations, and test the addresses, so that this code
// works with older versions of gccgo.
const gccgoExportFileProlog = `
extern _Bool runtime_iscgo __attribute__ ((weak));

static void GoInit(void) __attribute__ ((constructor));
static void GoInit(void) {
	if(&runtime_iscgo)
		runtime_iscgo = 1;
}

extern void _cgo_wait_runtime_init_done() __attribute__ ((weak));
`
