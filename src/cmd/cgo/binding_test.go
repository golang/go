// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestParseAndProcessBinding(t *testing.T) {
	t.Setenv("GOOS", "darwin")
	t.Setenv("GOARCH", "arm64")
	input := `package p

// #cgo LDFLAGS: ./libx.so
import "C"
import "structs"

//cgo:type S C.S
//cgo:type T T
//cgo:const MY_CONST C.MY_CONST
//cgo:const Other OTHER
//cgo:var _GlobalX C.GlobalX "libx.so"
//cgo:var staticY StaticY static
//cgo:func Add C.Add "libx.so"
//cgo:func _AddFloat AddFloat "./libx.so"
//cgo:func Puts puts libc
//cgo:func Strlen strlen libc
//cgo:func Helper helper static
//cgo:func _PassS PassS "libx.so"

type S struct {
	_ structs.HostLayout
	a int32
	b int16
	c int8
	_ [1]byte
}

type T struct {
	_ structs.HostLayout
	x int64
}

const (
	MY_CONST = 42
	Other    = 7
)

var _GlobalX int32

var staticY int32

func Add(a int32, b int32) int32

func _AddFloat(a float32, b float64) float64

func Puts(s *int8) int32

func Strlen(s *int8) uintptr

func Helper()

func _PassS(s S, t T) S
`

	p := newPackage(nil, token.NewFileSet())
	entry := p.parseBinding("binding.go", []byte(input))

	if entry.Package != "p" {
		t.Fatalf("Package = %q, want %q", entry.Package, "p")
	}
	if len(entry.LdFlags) != 1 || entry.LdFlags[0] != "./libx.so" {
		t.Fatalf("LdFlags = %v, want [./libx.so]", entry.LdFlags)
	}
	if sType := entry.TypeMap["_Ctype_S"]; sType.Size != 8 || sType.Align != 4 {
		t.Fatalf("TypeMap[_Ctype_S] = %+v, want struct of size 8 align 4", sType)
	}
	if len(entry.TypeGoToC) != 1 || entry.TypeGoToC["S"] != "S" {
		t.Fatalf("TypeGoToC = %+v, want only S", entry.TypeGoToC)
	}
	if len(entry.Vars) != 2 || entry.Vars[0].GoName != "_GlobalX" || entry.Vars[0].CName != "GlobalX" || !entry.Vars[0].BindC ||
		entry.Vars[1].GoName != "staticY" || !entry.Vars[1].Static || entry.Vars[1].BindC {
		t.Fatalf("Vars = %+v, want _GlobalX, staticY", entry.Vars)
	}
	if len(entry.ConstCToGo) != 1 || entry.ConstCToGo["MY_CONST"] != "MY_CONST" {
		t.Fatalf("ConstCToGo = %+v, want only MY_CONST", entry.ConstCToGo)
	}
	for _, name := range []string{"T", "OTHER", "StaticY", "AddFloat", "puts", "helper", "PassS"} {
		if !entry.GoOnlyCNames[name] {
			t.Errorf("GoOnlyCNames[%q] = false, want true", name)
		}
	}
	if len(entry.Funcs) != 6 {
		t.Fatalf("len(Funcs) = %d, want 6", len(entry.Funcs))
	}
	if fn := entry.Funcs[0]; fn.Name != "Add" || fn.CName != "Add" || !fn.BindC || fn.Library != "libx.so" || fn.Ret != "int32" {
		t.Errorf("Funcs[0] = %+v", fn)
	}
	if fn := entry.Funcs[1]; fn.GoName != "_AddFloat" || fn.CName != "AddFloat" || fn.BindC || fn.Library != "libx.so" || fn.Ret != "float64" {
		t.Errorf("Funcs[1] = %+v", fn)
	}
	if fn := entry.Funcs[2]; fn.GoName != "Puts" || fn.CName != "puts" || !fn.LibC || fn.Library != "" || fn.Static {
		t.Errorf("Funcs[2] = %+v", fn)
	}
	if fn := entry.Funcs[4]; fn.GoName != "Helper" || fn.CName != "helper" || !fn.Static || fn.Library != "" || fn.LibC {
		t.Errorf("Funcs[4] = %+v", fn)
	}

	tmpDir := t.TempDir()
	bindingFile := filepath.Join(tmpDir, "binding.go")
	if err := os.WriteFile(bindingFile, []byte(input), 0644); err != nil {
		t.Fatal(err)
	}
	mainFile := filepath.Join(tmpDir, "main.go")
	mainSrc := `package p

// #cgo noescape puts
// #cgo nocallback puts
import "C"

//export MyExport
func MyExport(a int32) int32 {
	return a + 1
}

func Call(x int32) int32 {
	C.GlobalX = C.MY_CONST
	cs := C.CString("hi")
	_ = cs
	_ = C.MyExport
	staticY = Other
	_ = C.OTHER
	return Add(x, 1) + int32(C.Add(C.int(x), 2))
}
`
	if err := os.WriteFile(mainFile, []byte(mainSrc), 0644); err != nil {
		t.Fatal(err)
	}

	origObjDir := *objDir
	origGoarch := goarch
	origGoos := goos
	*objDir = tmpDir + string(filepath.Separator)
	goarch = "arm64"
	goos = "darwin"
	defer func() {
		*objDir = origObjDir
		goarch = origGoarch
		goos = origGoos
	}()

	p.processBinding([]string{bindingFile, mainFile})

	gotypesBytes, err := os.ReadFile(filepath.Join(tmpDir, "_cgo_gotypes.go"))
	if err != nil {
		t.Fatalf("reading _cgo_gotypes.go: %v", err)
	}
	gotypes := string(gotypesBytes)
	for _, want := range []string{
		`func _Cgo_ptr(ptr unsafe.Pointer) unsafe.Pointer { return ptr }`,
		`//go:cgo_export_dynamic MyExport`,
		`//go:cgo_export_static MyExport`,
		`//go:linkname __cgofn_MyExport MyExport`,
		`var _Cfpvar_fp_MyExport unsafe.Pointer = unsafe.Pointer(&__cgofn_MyExport)`,
		`//go:linkname _cgoexp_`,
		`_MyExport(a unsafe.Pointer)`,
		`//go:cgo_import_dynamic _ _ "libx.so"`,
		`//go:cgo_import_dynamic Add Add`,
		`//go:cgo_import_dynamic AddFloat AddFloat`,
		`//go:cgo_import_dynamic puts puts`,
		`_var__GlobalX GlobalX`,
		`var _Cvar__GlobalX *int32 = (*int32)(unsafe.Pointer(&__cgo_var__GlobalX))`,
		`//go:cgo_import_static StaticY`,
		`//go:linkname __cgo_var_staticY StaticY`,
		`var _Cvar_staticY *int32 = (*int32)(unsafe.Pointer(&__cgo_var_staticY))`,
		`type _Ctype_S = S`,
		`//go:cgo_import_dynamic malloc malloc`,
		`func _Cfunc_CString`,
		`var _cgo_trampoline_Add byte`,
		`func _Cfunc_Add(p0 int32, p1 int32) int32`,
		`func Add(p0 int32, p1 int32) int32`,
		`func _AddFloat(p0 float32, p1 float64) float64`,
		`_Cgo_use(p0)`,
		`_Cgo_no_callback(true)`,
		`_Cgo_keepalive(p0)`,
		`//go:cgo_unsafe_args`,
	} {
		if !strings.Contains(gotypes, want) {
			t.Errorf("_cgo_gotypes.go missing %q:\n%s", want, gotypes)
		}
	}
	for _, notWant := range []string{
		// static symbols are not dynamically imported.
		`cgo_import_dynamic helper`,
		// AddFloat and T are bound without the C. prefix.
		`_Cfunc_AddFloat =`,
		`type _Ctype_T`,
	} {
		if strings.Contains(gotypes, notWant) {
			t.Errorf("_cgo_gotypes.go contains %q:\n%s", notWant, gotypes)
		}
	}
	for line := range strings.SplitSeq(gotypes, "\n") {
		if strings.Contains(line, "cgo_import_dynamic") && strings.Contains(line, "StaticY") {
			t.Errorf("_cgo_gotypes.go contains dynamic import for static var StaticY: %q", line)
		}
	}

	mainOut, err := os.ReadFile(filepath.Join(tmpDir, "main.cgobinding.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`(*_Cvar__GlobalX) = MY_CONST`,
		`(*_Cvar_staticY) = Other`,
		`_Cundef_OTHER`,
		`_ = _Cgo_ptr(_Cfpvar_fp_MyExport)`,
		`_Cfunc_Add(int32(x), 2)`,
		`_Cfunc_CString("hi")`,
	} {
		if !strings.Contains(string(mainOut), want) {
			t.Errorf("main.cgobinding.go missing %q:\n%s", want, mainOut)
		}
	}

	trampBytes, err := os.ReadFile(filepath.Join(tmpDir, "_cgo_trampolines.s"))
	if err != nil {
		t.Fatalf("reading _cgo_trampolines.s: %v", err)
	}
	tramp := string(trampBytes)
	for _, want := range []string{
		"_trampoline_Add(SB), NOSPLIT|NOFRAME, $0-0",
		"CALL\t_cgo_topofstack(SB)",
		"CALL\tAdd(SB)",
		"_trampoline__AddFloat(SB), NOSPLIT|NOFRAME, $0-0",
		"CALL\tAddFloat(SB)",
		"_trampoline_Helper(SB), NOSPLIT|NOFRAME, $0-0",
		"TEXT MyExport(SB)",
		"CALL\tcrosscall2(SB)",
		"CALL\thelper(SB)",
		"_trampoline__PassS(SB), NOSPLIT|NOFRAME, $0-0",
		"CALL\tPassS(SB)",
		"_trampoline__cgo_malloc(SB), NOSPLIT|NOFRAME, $0-0",
		"CALL\tmalloc(SB)",
	} {
		if !strings.Contains(tramp, want) {
			t.Errorf("_cgo_trampolines.s missing %q:\n%s", want, tramp)
		}
	}
}

func TestBindingLibcAndTargetTypes(t *testing.T) {
	origGoos := goos
	origGoarch := goarch
	defer func() {
		goos = origGoos
		goarch = origGoarch
	}()

	goos = "windows"
	goarch = "amd64"
	if gt, _ := primitiveCtypeToGo("_Ctype_long"); gt != "int32" {
		t.Errorf("windows/amd64 _Ctype_long = %q, want int32", gt)
	}
	if gt, _ := primitiveCtypeToGo("_Ctype_ulong"); gt != "uint32" {
		t.Errorf("windows/amd64 _Ctype_ulong = %q, want uint32", gt)
	}
	if gt, _ := primitiveCtypeToGo("_Ctype_size_t"); gt != "uint64" {
		t.Errorf("windows/amd64 _Ctype_size_t = %q, want uint64", gt)
	}
	if dll := windowsDLLName(""); dll != "msvcrt.dll" {
		t.Errorf("windowsDLLName(\"\") = %q, want msvcrt.dll", dll)
	}

	goos = "linux"
	goarch = "amd64"
	if isStandardC("libsystemd.so.0") {
		t.Errorf("linux isStandardC(libsystemd.so.0) = true, want false")
	}
	if gt, _ := primitiveCtypeToGo("_Ctype_long"); gt != "int64" {
		t.Errorf("linux/amd64 _Ctype_long = %q, want int64", gt)
	}
	if gt, _ := primitiveCtypeToGo("_Ctype_ulong"); gt != "uint64" {
		t.Errorf("linux/amd64 _Ctype_ulong = %q, want uint64", gt)
	}
	if gt, _ := primitiveCtypeToGo("_Ctype_size_t"); gt != "uint64" {
		t.Errorf("linux/amd64 _Ctype_size_t = %q, want uint64", gt)
	}
	if gt, _ := primitiveCtypeToGo("_Ctype_ssize_t"); gt != "int64" {
		t.Errorf("linux/amd64 _Ctype_ssize_t = %q, want int64", gt)
	}
	if gt, _ := primitiveCtypeToGo("_Ctype_uintptr_t"); gt != "uint64" {
		t.Errorf("linux/amd64 _Ctype_uintptr_t = %q, want uint64", gt)
	}

	// C.char is unsigned on linux/arm64 and signed on darwin/arm64 and linux/amd64.
	for _, tc := range []struct {
		os, arch string
		want     string
	}{
		{"linux", "arm64", "uint8"},
		{"linux", "riscv64", "uint8"},
		{"aix", "ppc64", "uint8"},
		{"darwin", "arm64", "int8"},
		{"windows", "arm64", "int8"},
		{"linux", "amd64", "int8"},
	} {
		goos, goarch = tc.os, tc.arch
		if got, _ := primitiveCtypeToGo("_Ctype_char"); got != tc.want {
			t.Errorf("%s/%s _Ctype_char = %q, want %q", tc.os, tc.arch, got, tc.want)
		}
	}

	tmpDir := t.TempDir()
	b1 := filepath.Join(tmpDir, "b1.go")
	b2 := filepath.Join(tmpDir, "b2.go")
	if err := os.WriteFile(b1, []byte("package p\nimport \"C\"\n//cgo:func cStrlen C.strlen libc\nfunc cStrlen(s *int8) uint64\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b2, []byte("package p\nimport \"C\"\nfunc cPuts(s *int8) int32\n//cgo:func cPuts C.puts libc\n"), 0644); err != nil {
		t.Fatal(err)
	}
	origObjDir := *objDir
	*objDir = tmpDir + string(filepath.Separator)
	defer func() { *objDir = origObjDir }()
	goos = "linux"
	goarch = "amd64"
	t.Setenv("GOOS", "linux")
	t.Setenv("GOARCH", "amd64")

	p := newPackage(nil, fset)
	p.processBinding([]string{b1, b2})
	gotypesBytes, err := os.ReadFile(filepath.Join(tmpDir, "_cgo_gotypes.go"))
	if err != nil {
		t.Fatal(err)
	}
	gotypes := string(gotypesBytes)
	if strings.Contains(gotypes, `cgo_import_dynamic _ _`) {
		t.Errorf("_cgo_gotypes.go should not import a library for libc symbols:\n%s", gotypes)
	}
	if !strings.Contains(gotypes, "//go:cgo_import_dynamic strlen strlen\n") || !strings.Contains(gotypes, "//go:cgo_import_dynamic puts puts\n") {
		t.Errorf("_cgo_gotypes.go missing 2-arg cgo_import_dynamic:\n%s", gotypes)
	}
}

// TestBindingVarWindows verifies the code generated for //cgo:var on
// Windows, where a dynamically imported variable is reached through its
// import address table slot, while a static variable is used directly.
func TestBindingVarWindows(t *testing.T) {
	tmpDir := t.TempDir()
	bindingFile := filepath.Join(tmpDir, "binding.go")
	if err := os.WriteFile(bindingFile, []byte(`package p

import "C"

//cgo:var _GlobalX C.GlobalX "libx.dll"
//cgo:var staticY StaticY static

var _GlobalX int32

var staticY int32

func Set() {
	C.GlobalX = 1
	staticY = 2
}
`), 0644); err != nil {
		t.Fatal(err)
	}

	origObjDir, origGoarch, origGoos := *objDir, goarch, goos
	defer func() { *objDir, goarch, goos = origObjDir, origGoarch, origGoos }()
	*objDir = tmpDir + string(filepath.Separator)
	goos, goarch = "windows", "amd64"
	t.Setenv("GOOS", "windows")
	t.Setenv("GOARCH", "amd64")

	p := newPackage(nil, token.NewFileSet())
	p.processBinding([]string{bindingFile})

	gotypesBytes, err := os.ReadFile(filepath.Join(tmpDir, "_cgo_gotypes.go"))
	if err != nil {
		t.Fatal(err)
	}
	gotypes := string(gotypesBytes)
	for _, want := range []string{
		`_var_addr__GlobalX GlobalX`,
		`//go:linkname _Cvar__GlobalX _cgo_`,
		"_var_addr__GlobalX\nvar _Cvar__GlobalX *int32\n",
		`//go:cgo_import_static StaticY`,
		`//go:linkname __cgo_var_staticY StaticY`,
		`var _Cvar_staticY *int32 = (*int32)(unsafe.Pointer(&__cgo_var_staticY))`,
	} {
		if !strings.Contains(gotypes, want) {
			t.Errorf("_cgo_gotypes.go missing %q:\n%s", want, gotypes)
		}
	}
	for _, notWant := range []string{`__cgo_var__GlobalX`, `__cgo_var_addr`, `**`} {
		if strings.Contains(gotypes, notWant) {
			t.Errorf("_cgo_gotypes.go contains %q:\n%s", notWant, gotypes)
		}
	}

	rewritten, err := os.ReadFile(filepath.Join(tmpDir, "binding.cgobinding.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`(*_Cvar__GlobalX) = 1`, `(*_Cvar_staticY) = 2`} {
		if !strings.Contains(string(rewritten), want) {
			t.Errorf("binding.cgobinding.go missing %q:\n%s", want, rewritten)
		}
	}
}

func TestParseCgoPragma(t *testing.T) {
	for _, tt := range []struct {
		src  string
		want cgoPragma
		ok   bool
	}{
		{`//cgo:func F C.f "libf.so"`, cgoPragma{Kind: "func", GoName: "F", CName: "f", BindC: true, Library: "libf.so"}, true},
		{`//cgo:func F f static`, cgoPragma{Kind: "func", GoName: "F", CName: "f", Static: true}, true},
		{`//cgo:func F f libc`, cgoPragma{Kind: "func", GoName: "F", CName: "f", LibC: true}, true},
		{`//cgo:func F f "static"`, cgoPragma{Kind: "func", GoName: "F", CName: "f", Library: "static"}, true},
		{`//cgo:var V C.v libc`, cgoPragma{Kind: "var", GoName: "V", CName: "v", BindC: true, LibC: true}, true},
		{`//cgo:type T C.struct_t`, cgoPragma{Kind: "type", GoName: "T", CName: "struct_t", BindC: true}, true},
		{`//cgo:const K K`, cgoPragma{Kind: "const", GoName: "K", CName: "K"}, true},
		{`//go:noinline`, cgoPragma{}, false},
		{`// cgo:func F f libc`, cgoPragma{}, false},
	} {
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, "x.go", "package p\n"+tt.src+"\n", parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		got, ok := parseCgoPragma(fset, file.Comments[0].List[0])
		got.Pos, got.TokPos, got.Text = token.Position{}, 0, ""
		if ok != tt.ok || got != tt.want {
			t.Errorf("parseCgoPragma(%q) = %+v, %v; want %+v, %v", tt.src, got, ok, tt.want, tt.ok)
		}
	}
}

func TestBindingScopeAndTrimpath(t *testing.T) {
	origGoos, origGoarch, origObjDir, origTrimpath, origLdflagsSet := goos, goarch, *objDir, *trimpath, ldflagsSet
	defer func() {
		goos, goarch, *objDir, *trimpath, ldflagsSet = origGoos, origGoarch, origObjDir, origTrimpath, origLdflagsSet
	}()
	goos, goarch = "linux", "amd64"
	ldflagsSet = false
	t.Setenv("GOOS", "linux")
	t.Setenv("GOARCH", "amd64")

	tmpDir := t.TempDir()
	*objDir = tmpDir + string(filepath.Separator)
	*trimpath = tmpDir + "=>/virtual"

	src := `package p

// #cgo LDFLAGS: -L${SRCDIR}/lib -lx
import "C"

//cgo:func Add C.Add "libx.so"
func Add(a int32, b int32) int32

type box struct{ Add int32 }

func UseC(x int32) int32 { return C.Add(C.int(x), 1) }

func ShadowParam(C box) int32 { return C.Add }

func ShadowLocal() int32 {
	C := box{Add: 3}
	return C.Add
}
`
	srcFile := filepath.Join(tmpDir, "b.go")
	if err := os.WriteFile(srcFile, []byte(src), 0644); err != nil {
		t.Fatal(err)
	}

	p := newPackage(nil, token.NewFileSet())
	p.processBinding([]string{srcFile})

	// ${SRCDIR} must expand from the untrimmed source directory using forward
	// slashes (matching go/build.expandSrcDir), after splitQuoted so spaces
	// and backslashes in the directory path are preserved.
	wantFlag := "-L" + filepath.ToSlash(tmpDir) + "/lib"
	if !slices.Equal(p.LdFlags, []string{wantFlag, "-lx"}) {
		t.Errorf("LdFlags = %q, want %q", p.LdFlags, []string{wantFlag, "-lx"})
	}
	spaceDir := filepath.Join(tmpDir, "dir with space")
	p2 := newPackage(nil, token.NewFileSet())
	entry2 := p2.parseBindingFiles([]bindingSource{{
		path:     "/virtual/b.go",
		origPath: filepath.Join(spaceDir, "b.go"),
		data:     []byte(src),
	}})
	wantSpace := []string{"-L" + filepath.ToSlash(spaceDir) + "/lib", "-lx"}
	if !slices.Equal(entry2.LdFlags, wantSpace) {
		t.Errorf("LdFlags with space in SRCDIR = %q, want %q", entry2.LdFlags, wantSpace)
	}
	if filepath.Separator == '/' {
		bsDir := filepath.Join(tmpDir, `a\b`)
		p3 := newPackage(nil, token.NewFileSet())
		entry3 := p3.parseBindingFiles([]bindingSource{{
			path:     "/virtual/b.go",
			origPath: filepath.Join(bsDir, "b.go"),
			data:     []byte(src),
		}})
		wantBS := []string{"-L" + bsDir + "/lib", "-lx"}
		if !slices.Equal(entry3.LdFlags, wantBS) {
			t.Errorf("LdFlags with backslash in SRCDIR = %q, want %q", entry3.LdFlags, wantBS)
		}
	}

	outBytes, err := os.ReadFile(filepath.Join(tmpDir, "b.cgobinding.go"))
	if err != nil {
		t.Fatal(err)
	}
	out := string(outBytes)
	for _, want := range []string{
		"//line /virtual/b.go:1:1\n",
		"return _Cfunc_Add(int32(x), 1)",
		"func ShadowParam(C box) int32 { return C.Add }",
		"C := box{Add: 3}\n\treturn C.Add",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("b.cgobinding.go missing %q:\n%s", want, out)
		}
	}
}

func TestBindingEdgeCases(t *testing.T) {
	origGoos, origGoarch, origObjDir := goos, goarch, *objDir
	defer func() {
		goos, goarch, *objDir = origGoos, origGoarch, origObjDir
	}()
	goos, goarch = "linux", "amd64"
	t.Setenv("GOOS", "linux")
	t.Setenv("GOARCH", "amd64")

	// TypeMap must not pollute Go type namespace with raw C name;
	// C.<const>, iota, and derived constants must work as array lengths;
	// aliased imports and inline HostLayout structs must emit valid
	// _cgo_gotypes.go with import "structs".
	input := `package p

import "C"
import s "structs"
import u "unsafe"

//cgo:type Foo C.Bar
//cgo:const N C.N
//cgo:const lowerC C.lower_c
//cgo:var gVar C.gVar "libx.so"
//cgo:func TakeInline TakeInline "libx.so"

const (
	_ = iota
	_
	_
	_
	N = iota
	lowerC = 2
	derived = C.lower_c * 2
	typedN Bar = 4
	derivedTyped = C.typed_n + 1
	derivedSizeof = u.Sizeof(Bar(0)) + C.lower_c
)

//cgo:const typedN C.typed_n

type Foo struct {
	_ s.HostLayout
	a int64
	b int64
}

type Bar int32

type MyMap map[int32]int32

var gVar int32

func TakeInline(x struct{ _ s.HostLayout; buf [C.N]byte; buf2 [derived]byte; buf3 [derivedTyped]byte; buf4 [derivedSizeof]byte; buf5 [int(u.Sizeof(Bar(0))) + int(derivedTyped)]byte; p u.Pointer }, b Bar)

func Use[gVar any](x gVar) MyMap {
	return MyMap{gVar: 1}
}

func UseMap() MyMap {
	return MyMap{gVar: 1}
}
`
	tmpDir := t.TempDir()
	srcFile := filepath.Join(tmpDir, "b.go")
	if err := os.WriteFile(srcFile, []byte(input), 0644); err != nil {
		t.Fatal(err)
	}
	*objDir = tmpDir + string(filepath.Separator)

	p := newPackage(nil, token.NewFileSet())
	entry := p.parseBinding(srcFile, []byte(input))
	if got := entry.TypeMap["_Ctype_Bar"]; got.Size != 16 {
		t.Fatalf("TypeMap[_Ctype_Bar].Size = %d, want 16", got.Size)
	}
	if got := entry.TypeMap["Bar"]; got.Size != 4 {
		t.Fatalf("TypeMap[Bar].Size = %d, want 4 (must not be overwritten by C.Bar)", got.Size)
	}

	p.processBinding([]string{srcFile})
	gotypesBytes, err := os.ReadFile(filepath.Join(tmpDir, "_cgo_gotypes.go"))
	if err != nil {
		t.Fatal(err)
	}
	gotypes := string(gotypesBytes)
	for _, want := range []string{
		`import "structs"`,
		`_ structs.HostLayout`,
		`buf [4]byte`,
		`buf2 [4]byte`,
		`buf3 [5]byte`,
		`buf4 [6]byte`,
		`buf5 [9]byte`,
		`p unsafe.Pointer`,
		`_Cgo_use(p0)`,
	} {
		if !strings.Contains(gotypes, want) {
			t.Errorf("_cgo_gotypes.go missing %q:\n%s", want, gotypes)
		}
	}
	rewrittenBytes, err := os.ReadFile(filepath.Join(tmpDir, "b.cgobinding.go"))
	if err != nil {
		t.Fatal(err)
	}
	rewritten := string(rewrittenBytes)
	if !strings.Contains(rewritten, "var _ u.Pointer") || !strings.Contains(rewritten, "var _ s.HostLayout") {
		t.Errorf("b.cgobinding.go missing aliased import keeps:\n%s", rewritten)
	}
	if !strings.Contains(rewritten, "func Use[gVar any](x gVar)") {
		t.Errorf("b.cgobinding.go rewrote shadowed type param gVar:\n%s", rewritten)
	}
	if !strings.Contains(rewritten, "return MyMap{(*_Cvar_gVar): 1}") {
		t.Errorf("b.cgobinding.go did not rewrite key in named map composite lit:\n%s", rewritten)
	}

	// evalBindingArrayLen must not panic on invalid constant expressions.
	for _, exprSrc := range []string{"^1.5", "!1", "&x", "1.5 | 2", "1 << 1.5", "-1", "1.5"} {
		e, err := parser.ParseExpr(exprSrc)
		if err != nil {
			t.Fatalf("ParseExpr(%q): %v", exprSrc, err)
		}
		if n, ok := p.evalBindingArrayLen(e, nil); ok {
			t.Errorf("evalBindingArrayLen(%q) = %d, want invalid", exprSrc, n)
		}
	}
	for _, tc := range []struct {
		expr string
		want int64
	}{
		{"0", 0},
		{"4", 4},
		{"4.0", 4},
		{"1 + 2*3", 7},
	} {
		e, err := parser.ParseExpr(tc.expr)
		if err != nil {
			t.Fatalf("ParseExpr(%q): %v", tc.expr, err)
		}
		if n, ok := p.evalBindingArrayLen(e, nil); !ok || n != tc.want {
			t.Errorf("evalBindingArrayLen(%q) = (%d, %v), want (%d, true)", tc.expr, n, ok, tc.want)
		}
	}
}

func TestBindingBuiltins(t *testing.T) {
	origGoos, origGoarch, origObjDir := goos, goarch, *objDir
	defer func() {
		goos, goarch, *objDir = origGoos, origGoarch, origObjDir
	}()

	// Non-C. Go funcs named CString or binding static malloc must not collide
	// with cgo builtins, and nocallback must still emit _Cgo_keepalive(&args),
	// while MayCallback with a return value pins args via //go:cgo_unsafe_args.
	goos, goarch = "linux", "arm64"
	t.Setenv("GOOS", "linux")
	t.Setenv("GOARCH", "arm64")
	tmpDir := t.TempDir()
	srcFile := filepath.Join(tmpDir, "b.go")
	input := `package p

// #cgo nocallback fast_fn
import "C"
import "unsafe"

//cgo:func CString my_cstr "libfoo.so"
//cgo:func customMalloc malloc static
//cgo:func FastFn fast_fn "libfoo.so"

func CString(x int32) int32
func customMalloc(n uint64) unsafe.Pointer
func FastFn(x int32) int32

func Use() {
	_ = C.CString("hi")
	_ = CString(1)
	_ = customMalloc(8)
	_ = FastFn(2)
}
`
	if err := os.WriteFile(srcFile, []byte(input), 0644); err != nil {
		t.Fatal(err)
	}
	*objDir = tmpDir + string(filepath.Separator)

	p := newPackage(nil, token.NewFileSet())
	p.processBinding([]string{srcFile})

	gotypesBytes, err := os.ReadFile(filepath.Join(tmpDir, "_cgo_gotypes.go"))
	if err != nil {
		t.Fatal(err)
	}
	gotypes := string(gotypesBytes)
	for _, want := range []string{
		`type _Ctype_char = uint8`,
		`_libc_malloc malloc`,
		`func _Cfunc__go_CString(`,
		`func _Cfunc_CString(`,
		`//go:cgo_unsafe_args`,
		`func _cgo_call__go_CString(args _cgo_arg__go_CString) int32`,
		`_Cgo_keepalive(&args)`,
	} {
		if !strings.Contains(gotypes, want) {
			t.Errorf("_cgo_gotypes.go missing %q:\n%s", want, gotypes)
		}
	}
}

func TestDynloadBinding(t *testing.T) {
	t.Setenv("GOOS", "linux")
	t.Setenv("GOARCH", "amd64")
	tmpDir := t.TempDir()
	src := filepath.Join(tmpDir, "lib.go")
	if err := os.WriteFile(src, []byte(`package p

import "C"
import cgo "runtime/cgo"
import _ "runtime/cgo"

//cgo:dynload Lib
//cgo:func Lib.Compute compute
//cgo:var Lib.Counter counter
//cgo:func Lib.Reset compute_reset

//go:noinline
func helper() int { return 1 }

type Lib struct {
	cgo.Library
	name string
}

func (*Lib) Compute(a, b int32) int32

func (*Lib) Counter() *int32

func (l *Lib) Reset()

func (l *Lib) Name() string { return l.name }
`), 0644); err != nil {
		t.Fatal(err)
	}
	origObjDir, origGoos, origGoarch := *objDir, goos, goarch
	*objDir = tmpDir + string(filepath.Separator)
	goos, goarch = "linux", "amd64"
	defer func() { *objDir, goos, goarch = origObjDir, origGoos, origGoarch }()

	p := newPackage(nil, token.NewFileSet())
	p.processBinding([]string{src})

	gotypes, err := os.ReadFile(filepath.Join(tmpDir, "_cgo_gotypes.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`//go:linkname _cgo_register_library runtime/cgo.registerLibrary`,
		`//go:linkname _cgo_library_symbol runtime/cgo.librarySymbol`,
		`func _cgo_register_libraries() bool {
	_cgo_register_library((*Lib)(nil), []string{"compute", "counter", "compute_reset"})
	return true
}

var _ = _cgo_register_libraries()`,
		`func (_cgo_recv *Lib) Counter() *int32 {
	return (*int32)(_cgo_library_symbol(&_cgo_recv.Library, 1))
}`,
		`type _cgo_arg__dyn3_Lib_Compute struct {
	fn uintptr
	p0 int32`,
		`func (_cgo_recv *Lib) Compute(p0 int32, p1 int32) int32 {
	return _Cfunc__dyn3_Lib_Compute(uintptr(_cgo_library_symbol(&_cgo_recv.Library, 0)), p0, p1)
}`,
		`func (_cgo_recv *Lib) Reset() {
	_Cfunc__dyn3_Lib_Reset(uintptr(_cgo_library_symbol(&_cgo_recv.Library, 2)))
}`,
	} {
		if !strings.Contains(string(gotypes), want) {
			t.Errorf("_cgo_gotypes.go missing %q:\n%s", want, gotypes)
		}
	}
	if strings.Contains(string(gotypes), "cgo_import_dynamic") {
		t.Errorf("_cgo_gotypes.go has dynamic imports for //cgo:dynload symbols:\n%s", gotypes)
	}

	// Registering the //cgo:dynload types is the first declaration in each
	// rewritten file, so that it happens before any other package
	// initialization. It goes on the line of the last import, so that the
	// first declaration keeps its directives.
	rewritten, err := os.ReadFile(filepath.Join(tmpDir, "lib.cgobinding.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"\nimport _ \"runtime/cgo\"; var _ = _cgo_register_libraries()\n",
		"\n//go:noinline\nfunc helper() int { return 1 }\n",
	} {
		if !strings.Contains(string(rewritten), want) {
			t.Errorf("lib.cgobinding.go missing %q:\n%s", want, rewritten)
		}
	}

	out, err := os.ReadFile(filepath.Join(tmpDir, "lib.cgobinding.go"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), "Compute(a, b int32)") || strings.Contains(string(out), "Counter() *int32") || !strings.Contains(string(out), "Name() string") {
		t.Errorf("lib.cgobinding.go should keep only unbound methods:\n%s", out)
	}

	tramp, err := os.ReadFile(filepath.Join(tmpDir, "_cgo_trampolines.s"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"_trampoline__dyn3_Lib_Compute(SB)", "MOVQ\t0(R12), R11", "CALL\tR11"} {
		if !strings.Contains(string(tramp), want) {
			t.Errorf("_cgo_trampolines.s missing %q:\n%s", want, tramp)
		}
	}
}

func TestBindingLdflagsPreamble(t *testing.T) {
	origGoos, origGoarch := goos, goarch
	defer func() { goos, goarch = origGoos, origGoarch }()
	t.Setenv("GOOS", "darwin")
	t.Setenv("GOARCH", "arm64")

	input := `package p

// #cgo LDFLAGS: -L${SRCDIR}/lib
// #cgo darwin LDFLAGS: -lmac
// #cgo linux LDFLAGS: -llinux
import "C"

// #cgo LDFLAGS: -lnotpreamble

//cgo:func F f libc
func F()
`
	dir := filepath.Join("some", "dir")
	p := newPackage(nil, fset)
	goos, goarch = "darwin", "arm64"
	entry := p.parseBinding(filepath.Join(dir, "binding.go"), []byte(input))
	want := []string{"-L" + filepath.ToSlash(dir) + "/lib", "-lmac"}
	if !slices.Equal(entry.LdFlags, want) {
		t.Errorf("LdFlags = %q, want %q", entry.LdFlags, want)
	}
}

func TestDynLibName(t *testing.T) {
	origGoos := goos
	defer func() { goos = origGoos }()
	for _, tt := range []struct {
		goos, lib, want string
	}{
		{"darwin", "/usr/lib/libobjc.A.dylib", "/usr/lib/libobjc.A.dylib"},
		{"darwin", "@rpath/libfoo.dylib", "@rpath/libfoo.dylib"},
		{"ios", "@loader_path/../lib/libfoo.dylib", "@loader_path/../lib/libfoo.dylib"},
		{"darwin", "./libfoo.dylib", "libfoo.dylib"},
		{"darwin", "lib/libfoo.dylib", "libfoo.dylib"},
		{"linux", "/usr/lib/libfoo.so.1", "libfoo.so.1"},
		{"linux", "./libx.so", "libx.so"},
	} {
		goos = tt.goos
		if got := dynLibName(tt.lib); got != tt.want {
			t.Errorf("%s: dynLibName(%q) = %q, want %q", tt.goos, tt.lib, got, tt.want)
		}
	}
}

func TestCTypeError(t *testing.T) {
	types := map[string]BindingType{
		"_Ctype_struct_ok":     {Name: "_Ctype_struct_ok", Def: "struct { _ structs.HostLayout; a int32; p *byte }"},
		"_Ctype_struct_bad":    {Name: "_Ctype_struct_bad", Def: "struct { _ structs.HostLayout; s []byte }"},
		"_Ctype_struct_rec":    {Name: "_Ctype_struct_rec", Def: "struct { _ structs.HostLayout; next *_Ctype_struct_rec }"},
		"_Ctype_struct_nohl":   {Name: "_Ctype_struct_nohl", Def: "struct { a int32 }"},
		"_Ctype_struct_opaque": {Name: "_Ctype_struct_opaque", Def: "struct { _ _cgopackage.Incomplete }"},
		"_Ctype_struct_inner":  {Name: "_Ctype_struct_inner", Def: "struct { _ structs.HostLayout; p *_Ctype_struct_nohl }"},
	}
	for _, tt := range []struct {
		typ  string
		want string // substring of the error, or "" for no error
	}{
		{"int32", ""},
		{"*int8", ""},
		{"unsafe.Pointer", ""},
		{"uintptr", ""},
		{"bool", ""},
		{"complex128", ""},
		{"[0]byte", ""},
		{"[4]float32", ""},
		{"struct{}", ""},
		{"struct { _ structs.HostLayout; x int32; y [2]uint8 }", ""},
		{"struct { _ structs.HostLayout; s struct { _ structs.HostLayout; x int32 } }", ""},
		{"[2]struct { _ structs.HostLayout; x int32 }", ""},
		{"*struct { _ structs.HostLayout; x int32 }", ""},
		{"**struct { _ structs.HostLayout; x int32 }", ""},
		{"_Ctype_struct_ok", ""},
		{"*_Ctype_struct_rec", ""},
		{"_Ctype_struct_unknown", ""},
		{"*_Ctype_struct_opaque", ""},
		{"*struct { _ _cgopackage.Incomplete }", ""},
		{"string", "Go string"},
		{"*string", "Go string"},
		{"[]byte", "Go slice"},
		{"*[]int32", "Go slice"},
		{"[2][]byte", "Go slice"},
		{"map[int32]int32", "Go map"},
		{"*map[int32]int32", "Go map"},
		{"chan int32", "Go channel"},
		{"[3]chan int32", "Go channel"},
		{"func()", "Go func value"},
		{"*func()", "Go func value"},
		{"interface{}", "Go interface"},
		{"[1]interface{}", "Go interface"},
		{"any", "Go interface"},
		{"error", "Go interface"},
		{"L[int32]", "generic"},
		{"*M[int32, int64]", "generic"},
		{"struct { _ structs.HostLayout; s string }", "Go string"},
		{"struct { _ structs.HostLayout; f func() }", "Go func value"},
		{"[2]struct { _ structs.HostLayout; m map[int32]int32 }", "Go map"},
		{"*struct { _ structs.HostLayout; s []byte }", "Go slice"},
		{"...int32", "variadic"},
		{"_Ctype_struct_bad", "Go slice"},
		{"*_Ctype_struct_bad", "Go slice"},
		{"struct { x int32 }", "structs.HostLayout"},
		{"struct { _ structs.HostLayout; s struct { x int32 } }", "structs.HostLayout"},
		{"[2]struct { x int32 }", "structs.HostLayout"},
		{"*struct { x int32 }", "structs.HostLayout"},
		{"**struct { x int32 }", "structs.HostLayout"},
		{"struct { _ structs.HostLayout; p *struct { x int32 } }", "structs.HostLayout"},
		{"_Ctype_struct_nohl", "structs.HostLayout"},
		{"*_Ctype_struct_nohl", "structs.HostLayout"},
		{"_Ctype_struct_inner", "structs.HostLayout"},
		{"_Ctype_struct_opaque", "incomplete"},
		{"struct { _ _cgopackage.Incomplete }", "incomplete"},
		{"[1]_Ctype_struct_opaque", "incomplete"},
		{"struct { _ structs.HostLayout; o _Ctype_struct_opaque }", "incomplete"},
	} {
		got := cTypeError(tt.typ, types)
		if (got == "") != (tt.want == "") || !strings.Contains(got, tt.want) {
			t.Errorf("cTypeError(%q) = %q, want error containing %q", tt.typ, got, tt.want)
		}
	}
}

// TestBindingNamedTypeChecks verifies that package-local and //cgo:type
// struct types shared with C must embed structs.HostLayout recursively,
// through nested structs, arrays, and pointers, and that incomplete
// types may only be used behind pointers.
func TestBindingNamedTypeChecks(t *testing.T) {
	origGoos, origGoarch := goos, goarch
	defer func() { goos, goarch = origGoos, origGoarch }()
	goos, goarch = "linux", "amd64"
	t.Setenv("GOOS", "linux")
	t.Setenv("GOARCH", "amd64")

	input := `package p

import "C"
import "runtime/cgo"
import "structs"

//cgo:type Node C.node
//cgo:type Opaque C.opaque
//cgo:func Walk walk "libx.so"

type Node struct {
	_    structs.HostLayout
	next *Node
	val  int32
	kids [2]Pair
}

type Pair struct {
	_    structs.HostLayout
	a, b int32
}

type Opaque struct{ _ cgo.Incomplete }

func Walk(n *Node, o *Opaque, p Pair) int32
`
	p := newPackage(nil, token.NewFileSet())
	entry := p.parseBinding("binding.go", []byte(input))
	if node := entry.TypeMap["Node"]; node.Size != 32 || node.Align != 8 || !node.HasPointers {
		t.Errorf("TypeMap[Node] = %+v, want size 32 align 8 with pointers", node)
	}
	if len(entry.Funcs) != 1 {
		t.Fatalf("len(Funcs) = %d, want 1", len(entry.Funcs))
	}
	for _, tt := range []struct {
		typ, want string
	}{
		{"Node", ""},
		{"*Node", ""},
		{"*Opaque", ""},
		{"[2]Pair", ""},
		{"Opaque", "incomplete"},
		{"struct { _ structs.HostLayout; o Opaque }", "incomplete"},
	} {
		if got := cTypeError(tt.typ, entry.TypeMap); (got == "") != (tt.want == "") || !strings.Contains(got, tt.want) {
			t.Errorf("cTypeError(%q) = %q, want error containing %q", tt.typ, got, tt.want)
		}
	}
	if got := cDeclTypeError("Opaque", entry.TypeMap); got != "" {
		t.Errorf("cDeclTypeError(Opaque) = %q, want no error", got)
	}

	types := map[string]BindingType{
		"Inner":    {Name: "Inner", Def: "struct { a int32 }"},
		"Outer":    {Name: "Outer", Def: "struct { _ structs.HostLayout; in Inner }"},
		"PtrOuter": {Name: "PtrOuter", Def: "struct { _ structs.HostLayout; in *Inner }"},
		"ArrOuter": {Name: "ArrOuter", Def: "struct { _ structs.HostLayout; in [2]Inner }"},
		"Self":     {Name: "Self", Def: "struct { _ structs.HostLayout; next *Self; m map[int32]int32 }"},
		"Alias":    {Name: "Alias", Def: "PtrOuter"},
	}
	for _, tt := range []struct {
		typ, want string
	}{
		{"Inner", "structs.HostLayout"},
		{"*Inner", "structs.HostLayout"},
		{"Outer", "structs.HostLayout"},
		{"PtrOuter", "structs.HostLayout"},
		{"ArrOuter", "structs.HostLayout"},
		{"Alias", "structs.HostLayout"},
		{"*Self", "Go map"},
	} {
		if got := cTypeError(tt.typ, types); !strings.Contains(got, tt.want) {
			t.Errorf("cTypeError(%q) = %q, want error containing %q", tt.typ, got, tt.want)
		}
		if got := cDeclTypeError(tt.typ, types); !strings.Contains(got, tt.want) {
			t.Errorf("cDeclTypeError(%q) = %q, want error containing %q", tt.typ, got, tt.want)
		}
	}
}

// TestInspectTypeGoOnly verifies that the ABI layout code panics on Go
// types that have no C equivalent, which cTypeError rejects first.
func TestInspectTypeGoOnly(t *testing.T) {
	cfg := getABI("linux", "amd64")
	for _, typ := range []string{
		"string",
		"any",
		"error",
		"[]byte",
		"[2][]int32",
		"struct { _ structs.HostLayout; s []byte }",
		"map[int32]int32",
		"chan int32",
		"func()",
		"interface{}",
		"L[int32]",
		"struct { _ structs.HostLayout; m map[int32]int32 }",
		"[2]func()",
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("inspectType(%q) did not panic", typ)
				}
			}()
			cfg.inspectType(typ, nil)
		}()
	}
}

func TestBindingPointerChecks(t *testing.T) {
	origGoos, origGoarch, origObjDir := goos, goarch, *objDir
	defer func() { goos, goarch, *objDir = origGoos, origGoarch, origObjDir }()
	t.Setenv("GOOS", "linux")
	t.Setenv("GOARCH", "amd64")

	tmpDir := t.TempDir()
	write := func(name, src string) string {
		path := filepath.Join(tmpDir, name)
		if err := os.WriteFile(path, []byte(src), 0644); err != nil {
			t.Fatal(err)
		}
		return path
	}
	bindings := write("bindings.go", `package p

import "C"
import "unsafe"

//cgo:func Fill C.fill "libx.so"
func Fill(p unsafe.Pointer, n int32) int32

//cgo:func Get get "libx.so"
func Get(pp **int32)

//cgo:func Len C.len "libx.so"
func Len(s *byte) int32

type conn struct {
	next   *conn
	outLen int32
}

func use(c *conn, s []*int32, b []byte) {
	n := Fill(unsafe.Pointer(&c.outLen), 4)
	_ = n
	C.fill(unsafe.Pointer(&s[1]), 1)
	Get(&s[0])
	defer Get(nil)
	go Get(&s[0])
	Fill(unsafe.Pointer(c), C.int(Len(&b[0])))
	_ = C.len(&b[0])
	_ = C.fill(nil, 1)
	_ = (C.fill)(unsafe.Pointer(&c.outLen), 4)
	f := C.fill
	_ = f
	g := Fill
	_ = g
}
`)
	shadow := write("shadow.go", `package p

import "unsafe"

func g(c *conn, Get func(**int32)) {
	Get(nil)
	Fill(unsafe.Pointer(c), 1)
}
`)
	*objDir = tmpDir + string(filepath.Separator)
	goos, goarch = "linux", "amd64"
	p := newPackage(nil, fset)
	p.processBinding([]string{bindings, shadow})

	read := func(name string) string {
		data, err := os.ReadFile(filepath.Join(tmpDir, name))
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	gotypes := read("_cgo_gotypes.go")
	for _, want := range []string{
		"//go:linkname _cgoCheckPointer runtime.cgoCheckPointer\n//go:noescape\nfunc _cgoCheckPointer(interface{}, interface{})\n",
		"func _Cgocheck_Fill(p0 unsafe.Pointer, p1 int32) int32 {\n\t_cgoCheckPointer(p0, nil)\n\treturn _Cfunc_Fill(p0, p1)\n}\n",
		"func Fill(p0 unsafe.Pointer, p1 int32) int32 {\n\treturn _Cgocheck_Fill(p0, p1)\n}\n",
		"func _Cgocheck_Get(p0 **int32) {\n\t_cgoCheckPointer(p0, nil)\n\t_Cfunc_Get(p0)\n}\n",
		// *byte cannot point to a Go pointer, so Len needs no checks.
		"func Len(p0 *byte) int32 {\n\treturn _Cfunc_Len(p0)\n}\n",
	} {
		if !strings.Contains(gotypes, want) {
			t.Errorf("_cgo_gotypes.go missing %q:\n%s", want, gotypes)
		}
	}
	if strings.Contains(gotypes, "_Cgocheck_Len") {
		t.Errorf("_cgo_gotypes.go has unneeded _Cgocheck_Len:\n%s", gotypes)
	}

	got := read("bindings.cgobinding.go")
	for _, want := range []string{
		// &x checks only x.
		"func() (_cgoa _cgo_arg_Fill) { _cgoBase0 := &c.outLen; _cgoa.p0 = unsafe.Pointer(_cgoBase0); _cgoa.p1 = 4; _cgoCheckPointer(_cgoBase0, 0 == 0); _cgoa.ret = _Cfunc_Fill(_cgoa.p0, _cgoa.p1); return }().ret\n",
		// &s[i] checks the elements of s.
		// The result of a call statement is discarded.
		"func() { var _cgoa _cgo_arg_Fill; _cgoIndex0 := &s; _cgoa.p0 = unsafe.Pointer(&(*_cgoIndex0)[1]); _cgoa.p1 = 1; _cgoCheckPointer(_cgoa.p0, *_cgoIndex0); _Cfunc_Fill(_cgoa.p0, _cgoa.p1) }()\n",
		"func() { var _cgoa _cgo_arg_Get; _cgoIndex0 := &s; _cgoa.p0 = &(*_cgoIndex0)[0]; _cgoCheckPointer(_cgoa.p0, *_cgoIndex0); _Cfunc_Get(_cgoa.p0) }()\n",
		// A nil argument needs no check.
		"defer Get(nil)\n",
		// go and defer evaluate the arguments immediately.
		"go func() func() { var _cgoa _cgo_arg_Get; _cgoIndex0 := &s; _cgoa.p0 = &(*_cgoIndex0)[0]; return func() { _cgoCheckPointer(_cgoa.p0, *_cgoIndex0); _Cfunc_Get(_cgoa.p0) } }()()\n",
		// Other arguments check the whole allocation; nested calls are
		// rewritten too.
		"func() { var _cgoa _cgo_arg_Fill; _cgoa.p0 = unsafe.Pointer(c); _cgoa.p1 = int32(Len(&b[0])); _cgoCheckPointer(_cgoa.p0, nil); _Cfunc_Fill(_cgoa.p0, _cgoa.p1) }()\n",
		"_ = _Cfunc_Len(&b[0])\n",
		// A C call with only nil pointer args stays a direct call to _Cgocheck_.
		"_ = _Cgocheck_Fill(nil, 1)\n",
		// A parenthesized C call is still rewritten as a call.
		"_ = func() (_cgoa _cgo_arg_Fill) { _cgoBase0 := &c.outLen; _cgoa.p0 = unsafe.Pointer(_cgoBase0); _cgoa.p1 = 4; _cgoCheckPointer(_cgoBase0, 0 == 0); _cgoa.ret = _Cfunc_Fill(_cgoa.p0, _cgoa.p1); return }().ret\n",
		// A C function used in non-call context is a non-assignable C function pointer.
		"f := _Cgo_ptr(_Cfpvar_fp_fill)\n",
		// The bound Go function used as a value checks its arguments via func Fill.
		"g := Fill\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("bindings.cgobinding.go missing %q:\n%s", want, got)
		}
	}

	// Calls through shadowed names are left alone; Fill is not rewritten in a
	// file that declares a local Get, since calls of Get may not refer to the
	// bound function, but calls of Fill are.
	got = read("shadow.cgobinding.go")
	for _, want := range []string{
		"\tGet(nil)\n",
		"func() { var _cgoa _cgo_arg_Fill; _cgoa.p0 = unsafe.Pointer(c); _cgoa.p1 = 1; _cgoCheckPointer(_cgoa.p0, nil); _Cfunc_Fill(_cgoa.p0, _cgoa.p1) }()\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("shadow.cgobinding.go missing %q:\n%s", want, got)
		}
	}
}

func TestBindingHasPointer(t *testing.T) {
	types := map[string]BindingType{
		"_Ctype_struct_node": {Name: "_Ctype_struct_node", Def: "struct { next *_Ctype_struct_node; v int32 }"},
		"_Ctype_struct_flat": {Name: "_Ctype_struct_flat", Def: "struct { a [4]byte; b float64 }"},
	}
	for _, tt := range []struct {
		typ  string
		want bool
	}{
		{"int32", false},
		{"*byte", false},
		{"*[16]float64", false},
		{"**byte", true},
		{"unsafe.Pointer", true},
		{"struct { p *byte }", false},
		{"struct { p **byte }", true},
		{"*_Ctype_struct_flat", false},
		{"*_Ctype_struct_node", true},
		{"_Ctype_struct_node", true},
		{"_Ctype_struct_flat", false},
		{"*_Ctype_struct_unknown", true},
	} {
		if got := bindingHasPointer(tt.typ, types, true); got != tt.want {
			t.Errorf("bindingHasPointer(%q, top) = %v, want %v", tt.typ, got, tt.want)
		}
	}
}

// TestIsAnonCtype verifies which _Ctype_ names -gen-binding treats as the
// names cgo invents for untagged C structs, unions, and classes.
func TestIsAnonCtype(t *testing.T) {
	for name, want := range map[string]bool{
		"_Ctype_struct___0":   true,
		"_Ctype_union___12":   true,
		"_Ctype_class___3":    true,
		"_Ctype_struct___":    false,
		"_Ctype_struct___foo": false,
		"_Ctype_struct_S":     false,
		"_Ctype_union_U":      false,
		"_Ctype_U":            false,
		"_Ctype_int":          false,
	} {
		if got := isAnonCtype(name); got != want {
			t.Errorf("isAnonCtype(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestBindingFuncPtrAndExportCall(t *testing.T) {
	origGoos, origGoarch, origObjDir := goos, goarch, *objDir
	defer func() { goos, goarch, *objDir = origGoos, origGoarch, origObjDir }()

	src := `package p

import "C"
import "unsafe"

//cgo:func DynFn C.dyn_fn "libx.so"
//cgo:func StaticFn C.static_fn static
//cgo:func UnusedFn C.unused_fn static

func DynFn(a int32) int32
func StaticFn(a int32) int32
func UnusedFn(a int32) int32

//export ExpBoth
func ExpBoth(a int32) int32 { return a + 1 }

//export ExpPtr
func ExpPtr(p unsafe.Pointer) int32 { return 0 }

//export ExpFPOnly
func ExpFPOnly(a int32) int32 { return a + 2 }

type box struct{ x int32 }

func Use(b *box) {
	_ = C.dyn_fn(1)
	_ = C.dyn_fn
	_ = C.static_fn
	_ = C.ExpBoth
	_ = C.ExpBoth(2)
	_ = (C.ExpBoth)(3)
	_ = C.ExpPtr(unsafe.Pointer(&b.x))
	_ = C.ExpPtr(nil)
	_ = C.ExpFPOnly
}
`
	run := func(targetOS, targetArch string) (gotypes, rewritten, tramp string) {
		t.Helper()
		tmpDir := t.TempDir()
		srcFile := filepath.Join(tmpDir, "p.go")
		if err := os.WriteFile(srcFile, []byte(src), 0644); err != nil {
			t.Fatal(err)
		}
		*objDir = tmpDir + string(filepath.Separator)
		goos, goarch = targetOS, targetArch
		t.Setenv("GOOS", targetOS)
		t.Setenv("GOARCH", targetArch)

		p := newPackage(nil, token.NewFileSet())
		p.processBinding([]string{srcFile})

		read := func(name string) string {
			b, err := os.ReadFile(filepath.Join(tmpDir, name))
			if err != nil {
				t.Fatalf("%s/%s: reading %s: %v", targetOS, targetArch, name, err)
			}
			return string(b)
		}
		return read("_cgo_gotypes.go"), read("p.cgobinding.go"), read("_cgo_trampolines.s")
	}

	gotypes, rewritten, tramp := run("linux", "amd64")
	for _, want := range []string{
		`func _Cgo_ptr(ptr unsafe.Pointer) unsafe.Pointer { return ptr }`,
		`//go:cgo_import_static static_fn`,
		`//go:linkname __cgo_fp_dyn_fn dyn_fn`,
		`var _Cfpvar_fp_dyn_fn unsafe.Pointer = unsafe.Pointer(&__cgo_fp_dyn_fn)`,
		`//go:linkname __cgo_fp_static_fn static_fn`,
		`var _Cfpvar_fp_static_fn unsafe.Pointer = unsafe.Pointer(&__cgo_fp_static_fn)`,
		`var _Cfpvar_fp_ExpBoth unsafe.Pointer = unsafe.Pointer(&__cgofn_ExpBoth)`,
		`func _cgo_call_ExpBoth(args _cgo_arg_ExpBoth) int32`,
		`func _Cfunc_ExpBoth(p0 int32) int32`,
		`func _Cfunc_ExpPtr(p0 unsafe.Pointer) int32`,
		`func _Cgocheck_ExpPtr(p0 unsafe.Pointer) int32`,
		`var _Cfpvar_fp_ExpFPOnly unsafe.Pointer = unsafe.Pointer(&__cgofn_ExpFPOnly)`,
	} {
		if !strings.Contains(gotypes, want) {
			t.Errorf("linux/amd64 _cgo_gotypes.go missing %q:\n%s", want, gotypes)
		}
	}
	if strings.Count(gotypes, "type _cgo_arg_ExpBoth struct") != 1 {
		t.Errorf("linux/amd64 _cgo_gotypes.go has %d definitions of _cgo_arg_ExpBoth, want 1:\n%s",
			strings.Count(gotypes, "type _cgo_arg_ExpBoth struct"), gotypes)
	}
	for _, notWant := range []string{
		`_Cfpvar_fp_unused_fn`,
		`cgo_import_static unused_fn`,
		`func _Cfunc_ExpFPOnly`,
		`func ExpBoth(`,
		`func ExpPtr(`,
	} {
		if strings.Contains(gotypes, notWant) {
			t.Errorf("linux/amd64 _cgo_gotypes.go contains unexpected %q:\n%s", notWant, gotypes)
		}
	}
	for _, want := range []string{
		`_ = _Cfunc_DynFn(1)`,
		`_ = _Cgo_ptr(_Cfpvar_fp_dyn_fn)`,
		`_ = _Cgo_ptr(_Cfpvar_fp_static_fn)`,
		`_ = _Cgo_ptr(_Cfpvar_fp_ExpBoth)`,
		`_ = _Cfunc_ExpBoth(2)`,
		`_ = (_Cfunc_ExpBoth)(3)`,
		`_cgoBase0 := &b.x; _cgoa.p0 = unsafe.Pointer(_cgoBase0); _cgoCheckPointer(_cgoBase0, 0 == 0); _cgoa.ret = _Cfunc_ExpPtr(_cgoa.p0)`,
		`_ = _Cgocheck_ExpPtr(nil)`,
		`_ = _Cgo_ptr(_Cfpvar_fp_ExpFPOnly)`,
	} {
		if !strings.Contains(rewritten, want) {
			t.Errorf("linux/amd64 p.cgobinding.go missing %q:\n%s", want, rewritten)
		}
	}
	for _, want := range []string{
		"TEXT ExpBoth(SB)",
		"_trampoline_ExpBoth(SB)",
		"CALL\tExpBoth(SB)",
		"TEXT ExpPtr(SB)",
		"_trampoline_ExpPtr(SB)",
		"CALL\tExpPtr(SB)",
		"TEXT ExpFPOnly(SB)",
	} {
		if !strings.Contains(tramp, want) {
			t.Errorf("linux/amd64 _cgo_trampolines.s missing %q:\n%s", want, tramp)
		}
	}
	if strings.Contains(tramp, "_trampoline_ExpFPOnly") {
		t.Errorf("linux/amd64 _cgo_trampolines.s emitted unused trampoline for ExpFPOnly:\n%s", tramp)
	}

	// On Windows, a dynamically imported function symbol is an IAT slot holding
	// the function pointer, whereas a static symbol and an //export symbol are
	// direct code symbols.
	winTypes, _, winTramp := run("windows", "amd64")
	for _, want := range []string{
		"//go:linkname _Cfpvar_fp_dyn_fn dyn_fn\nvar _Cfpvar_fp_dyn_fn unsafe.Pointer\n",
		"//go:cgo_import_static static_fn\n",
		"//go:linkname __cgo_fp_static_fn static_fn\nvar __cgo_fp_static_fn byte\nvar _Cfpvar_fp_static_fn unsafe.Pointer = unsafe.Pointer(&__cgo_fp_static_fn)\n",
		"var _Cfpvar_fp_ExpBoth unsafe.Pointer = unsafe.Pointer(&__cgofn_ExpBoth)\n",
	} {
		if !strings.Contains(winTypes, want) {
			t.Errorf("windows/amd64 _cgo_gotypes.go missing %q:\n%s", want, winTypes)
		}
	}
	if strings.Contains(winTypes, "__cgo_fp_dyn_fn") {
		t.Errorf("windows/amd64 _cgo_gotypes.go should link _Cfpvar_fp_dyn_fn directly to IAT slot:\n%s", winTypes)
	}
	if !strings.Contains(winTramp, "CALL\tExpBoth(SB)") {
		t.Errorf("windows/amd64 _cgo_trampolines.s should call //export ExpBoth directly:\n%s", winTramp)
	}

	// On AIX, a dynamically imported function symbol is an XCOFF TOC slot
	// (__cgo_imp_<sym>) holding the pointer to the function descriptor.
	aixDir := t.TempDir()
	*objDir = aixDir + string(filepath.Separator)
	goos, goarch = "aix", "ppc64"
	t.Setenv("GOOS", "aix")
	t.Setenv("GOARCH", "ppc64")
	aixPkg := newPackage(nil, token.NewFileSet())
	aixEntry := aixPkg.parseBinding("p.go", []byte(src))
	aixEntry.UsedFuncPtrs["dyn_fn"] = true
	aixEntry.UsedFuncPtrs["static_fn"] = true
	aixPkg.writeGoTypes(aixEntry, nil)
	aixBytes, err := os.ReadFile(filepath.Join(aixDir, "_cgo_gotypes.go"))
	if err != nil {
		t.Fatal(err)
	}
	aixTypes := string(aixBytes)
	for _, want := range []string{
		"//go:cgo_import_dynamic __cgo_imp_dyn_fn dyn_fn",
		"//go:linkname _Cfpvar_fp_dyn_fn __cgo_imp_dyn_fn\nvar _Cfpvar_fp_dyn_fn unsafe.Pointer\n",
		"//go:cgo_import_static static_fn\n",
		"//go:linkname __cgo_fp_static_fn static_fn\nvar __cgo_fp_static_fn byte\nvar _Cfpvar_fp_static_fn unsafe.Pointer = unsafe.Pointer(&__cgo_fp_static_fn)\n",
	} {
		if !strings.Contains(aixTypes, want) {
			t.Errorf("aix/ppc64 _cgo_gotypes.go missing %q:\n%s", want, aixTypes)
		}
	}
	if strings.Contains(aixTypes, "__cgo_fp_dyn_fn") {
		t.Errorf("aix/ppc64 _cgo_gotypes.go should link _Cfpvar_fp_dyn_fn directly to TOC slot:\n%s", aixTypes)
	}
}
