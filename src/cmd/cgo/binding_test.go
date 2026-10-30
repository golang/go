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

//cgo:func Add C.Add "libx.so"
//cgo:func _AddFloat AddFloat "./libx.so"
//cgo:func Puts puts libc
//cgo:func Helper helper static

func Add(a int32, b int32) int32

func _AddFloat(a float32, b float64) float64

func Puts(s *int8) int32

func Helper()
`

	p := newPackage(nil, fset)
	entry := p.parseBinding("binding.go", []byte(input))

	if entry.Package != "p" {
		t.Fatalf("Package = %q, want %q", entry.Package, "p")
	}
	if len(entry.LdFlags) != 1 || entry.LdFlags[0] != "./libx.so" {
		t.Fatalf("LdFlags = %v, want [./libx.so]", entry.LdFlags)
	}
	if len(entry.Funcs) != 4 {
		t.Fatalf("len(Funcs) = %d, want 4", len(entry.Funcs))
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
	if fn := entry.Funcs[3]; fn.GoName != "Helper" || fn.CName != "helper" || !fn.Static || fn.Library != "" || fn.LibC {
		t.Errorf("Funcs[3] = %+v", fn)
	}

	tmpDir := t.TempDir()
	bindingFile := filepath.Join(tmpDir, "binding.go")
	if err := os.WriteFile(bindingFile, []byte(input), 0644); err != nil {
		t.Fatal(err)
	}
	mainFile := filepath.Join(tmpDir, "main.go")
	mainSrc := `package p

import "C"

func Call(x int32) int32 {
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
		`//go:cgo_import_dynamic _ _ "libx.so"`,
		`//go:cgo_import_dynamic Add Add`,
		`//go:cgo_import_dynamic AddFloat AddFloat`,
		`//go:cgo_import_dynamic puts puts`,
		`var _cgo_trampoline_Add byte`,
		`func _Cfunc_Add(p0 int32, p1 int32) int32`,
		`func Add(p0 int32, p1 int32) int32`,
		`func _AddFloat(p0 float32, p1 float64) float64`,
		`//go:linkname _Cgo_keepalive runtime.cgoKeepAlive`,
		"\t\t_Cgo_keepalive(&args)\n",
	} {
		if !strings.Contains(gotypes, want) {
			t.Errorf("_cgo_gotypes.go missing %q:\n%s", want, gotypes)
		}
	}
	for _, notWant := range []string{
		// static symbols are not dynamically imported.
		`cgo_import_dynamic helper`,
		// AddFloat is bound without the C. prefix.
		`_Cfunc_AddFloat =`,
	} {
		if strings.Contains(gotypes, notWant) {
			t.Errorf("_cgo_gotypes.go contains %q:\n%s", notWant, gotypes)
		}
	}

	trampBytes, err := os.ReadFile(filepath.Join(tmpDir, "_cgo_trampolines.s"))
	if err != nil {
		t.Fatalf("reading _cgo_trampolines.s: %v", err)
	}
	tramp := string(trampBytes)
	for _, want := range []string{
		"_trampoline_Add(SB), NOSPLIT|NOFRAME, $0-0",
		"CALL\tAdd(SB)",
		"_trampoline__AddFloat(SB), NOSPLIT|NOFRAME, $0-0",
		"CALL\tAddFloat(SB)",
		"_trampoline_Helper(SB), NOSPLIT|NOFRAME, $0-0",
		"CALL\thelper(SB)",
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
		{`//go:noinline`, cgoPragma{}, false},
		{`// cgo:func F f libc`, cgoPragma{}, false},
	} {
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, "x.go", "package p\n"+tt.src+"\n", parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		got, ok := parseCgoPragma(fset, file.Comments[0].List[0])
		got.Pos = token.Position{}
		if ok != tt.ok || got != tt.want {
			t.Errorf("parseCgoPragma(%q) = %+v, %v; want %+v, %v", tt.src, got, ok, tt.want, tt.ok)
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
	want := []string{"-L" + dir + "/lib", "-lmac"}
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
		"_Ctype_struct_ok":  {Name: "_Ctype_struct_ok", Def: "struct { a int32; p *byte }"},
		"_Ctype_struct_bad": {Name: "_Ctype_struct_bad", Def: "struct { s []byte }"},
		"_Ctype_struct_rec": {Name: "_Ctype_struct_rec", Def: "struct { next *_Ctype_struct_rec }"},
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
		{"struct { x int32; y [2]uint8 }", ""},
		{"_Ctype_struct_ok", ""},
		{"*_Ctype_struct_rec", ""},
		{"_Ctype_struct_unknown", ""},
		{"string", "Go string"},
		{"*string", "Go string"},
		{"[]byte", "Go slice"},
		{"[2][]byte", "Go slice"},
		{"map[int32]int32", "Go map"},
		{"chan int32", "Go channel"},
		{"func()", "Go func value"},
		{"interface{}", "Go interface"},
		{"any", "Go interface"},
		{"error", "Go interface"},
		{"struct { s string }", "Go string"},
		{"...int32", "variadic"},
		{"_Ctype_struct_bad", "Go slice"},
		{"*_Ctype_struct_bad", "Go slice"},
	} {
		got := cTypeError(tt.typ, types)
		if (got == "") != (tt.want == "") || !strings.Contains(got, tt.want) {
			t.Errorf("cTypeError(%q) = %q, want error containing %q", tt.typ, got, tt.want)
		}
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
	f := C.fill
	_ = f
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
		// A C function used as a value checks its arguments.
		"f := _Cgocheck_Fill\n",
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
