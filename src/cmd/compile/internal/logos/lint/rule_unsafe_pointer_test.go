// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestUnsafePointer(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		src  string
		want string
	}{
		{"local pointer return", `import "unsafe"; func f() unsafe.Pointer { var x int; return unsafe.Pointer(&x) }`, ""},
		{"allocated pointer return", `import "unsafe"; func f() unsafe.Pointer { return unsafe.Pointer(new(int)) }`, ""},
		{"integer address", `import "unsafe"; func f() uintptr { var x int; u := uintptr(unsafe.Pointer(&x)); return u }`, ""},
		{"immediate round trip", `import "unsafe"; func f(p unsafe.Pointer) unsafe.Pointer { return unsafe.Pointer(uintptr(p)) }`, ""},
		{"add offset", `import "unsafe"; func f(p unsafe.Pointer, n uintptr) unsafe.Pointer { return unsafe.Pointer(uintptr(p) + n) }`, ""},
		{"offset first", `import "unsafe"; func f(p unsafe.Pointer, n uintptr) unsafe.Pointer { return unsafe.Pointer(n + uintptr(p)) }`, ""},
		{"subtract offset", `import "unsafe"; func f(p unsafe.Pointer, n uintptr) unsafe.Pointer { return unsafe.Pointer(uintptr(p) - n) }`, ""},
		{"alignment", `import "unsafe"; func f(p unsafe.Pointer) unsafe.Pointer { return unsafe.Pointer(uintptr(p) &^ 7) }`, ""},
		{"nested arithmetic", `import "unsafe"; func f(p unsafe.Pointer, n uintptr) unsafe.Pointer { return unsafe.Pointer((8 + uintptr(p)) + n*4) }`, ""},
		{"unsafe add", `import "unsafe"; func f(p unsafe.Pointer, n uintptr) unsafe.Pointer { return unsafe.Add(p, n) }`, ""},
		{"slice view", `import "unsafe"; func f(b []byte) []byte { return unsafe.Slice(unsafe.SliceData(b), len(b)) }`, ""},
		{"string view", `import "unsafe"; func f(s string) string { return unsafe.String(unsafe.StringData(s), len(s)) }`, ""},
		{"layout constants", `import "unsafe"; func f() uintptr { var x struct{ a int; b byte }; return unsafe.Sizeof(x) + unsafe.Alignof(x) + unsafe.Offsetof(x.b) }`, ""},
		{"aliases and parentheses", `import u "unsafe"; type Pointer = u.Pointer; type address uintptr; func f(p Pointer) Pointer { return (Pointer)((address)(p) + 4) }`, ""},
		{"saved address", `import "unsafe"; func f(p unsafe.Pointer) unsafe.Pointer { u := uintptr(p); return unsafe.Pointer(u) }`, "uintptr"},
		{"integer parameter", `import "unsafe"; func f(u uintptr) unsafe.Pointer { return unsafe.Pointer(u) }`, "uintptr"},
		{"integer constant", `import "unsafe"; func f() unsafe.Pointer { return unsafe.Pointer(uintptr(1)) }`, "uintptr"},
		{"named integer", `import "unsafe"; type address uintptr; type Pointer = unsafe.Pointer; func f(u address) Pointer { return Pointer(u) }`, "uintptr"},
		{"two pointers", `import "unsafe"; func f(p, q unsafe.Pointer) unsafe.Pointer { return unsafe.Pointer(uintptr(p) + uintptr(q)) }`, "uintptr"},
		{"scaled second pointer", `import "unsafe"; func f(p, q unsafe.Pointer) unsafe.Pointer { return unsafe.Pointer(uintptr(p) + uintptr(q)*2) }`, "uintptr"},
		{"converted second pointer", `import "unsafe"; func f(p, q unsafe.Pointer) unsafe.Pointer { return unsafe.Pointer(uintptr(p) + uintptr(uint64(uintptr(q)))) }`, "uintptr"},
		{"converted scalar offset", `import "unsafe"; func f(p unsafe.Pointer, n uint32) unsafe.Pointer { return unsafe.Pointer(uintptr(p) + uintptr(n)) }`, ""},
		{"pointer on right of subtract", `import "unsafe"; func f(p unsafe.Pointer, n uintptr) unsafe.Pointer { return unsafe.Pointer(n - uintptr(p)) }`, "uintptr"},
		{"scaled pointer", `import "unsafe"; func f(p unsafe.Pointer) unsafe.Pointer { return unsafe.Pointer(uintptr(p) * 2) }`, "uintptr"},
		{"shifted pointer", `import "unsafe"; func f(p unsafe.Pointer) unsafe.Pointer { return unsafe.Pointer(uintptr(p) << 1) }`, "uintptr"},
		{"pointer function", `import "unsafe"; func Pointer(uintptr) unsafe.Pointer; func f(u uintptr) unsafe.Pointer { return Pointer(u) }`, ""},
		{"shadowed uintptr", `import "unsafe"; type address uintptr; func f(p unsafe.Pointer) unsafe.Pointer { uintptr := func(unsafe.Pointer) address { return 0 }; return unsafe.Pointer(uintptr(p)) }`, "uintptr"},
		{"reflect immediate", `import ("reflect"; "unsafe"); func f(v reflect.Value) unsafe.Pointer { return unsafe.Pointer(v.Pointer()) }`, ""},
		{"reflect address immediate", `import ("reflect"; "unsafe"); func f(v reflect.Value) unsafe.Pointer { return unsafe.Pointer((v.UnsafeAddr)()) }`, ""},
		{"reflect method expression", `import ("reflect"; "unsafe"); func f(v reflect.Value) unsafe.Pointer { return unsafe.Pointer(reflect.Value.Pointer(v)) }`, ""},
		{"reflect saved", `import ("reflect"; "unsafe"); func f(v reflect.Value) unsafe.Pointer { u := v.Pointer(); return unsafe.Pointer(u) }`, "uintptr"},
		{"reflect lookalike", `import "unsafe"; type Value int; func (Value) Pointer() uintptr { return 0 }; func f(v Value) unsafe.Pointer { return unsafe.Pointer(v.Pointer()) }`, "uintptr"},
		{"header data pointer", `import ("reflect"; "unsafe"); func f(h *reflect.SliceHeader) unsafe.Pointer { return unsafe.Pointer(h.Data) }`, ""},
		{"real header overlay", `import ("reflect"; "unsafe"); func f(s []int) unsafe.Pointer { h := (*reflect.SliceHeader)(unsafe.Pointer(&s)); return unsafe.Pointer(h.Data) }`, ""},
		{"header data value", `import ("reflect"; "unsafe"); func f(h reflect.SliceHeader) unsafe.Pointer { return unsafe.Pointer(h.Data) }`, "uintptr"},
		{"header address", `import "reflect"; func f() *reflect.SliceHeader { return &reflect.SliceHeader{} }`, "reflect.SliceHeader"},
		{"header copy", `import "reflect"; func f(h *reflect.StringHeader) reflect.StringHeader { return *h }`, "reflect.StringHeader"},
		{"header alias", `import r "reflect"; type Header = r.StringHeader; func f() *Header { return &Header{} }`, "reflect.StringHeader"},
		{"header lookalike", `type SliceHeader struct { Data uintptr; Len, Cap int }; func f() *SliceHeader { return &SliceHeader{} }`, ""},
	}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "unsafe.go"
		if kind == KindLgo {
			name = "unsafe.lgo"
		}
		for _, tt := range tests {
			t.Run(name+"/"+tt.name, func(t *testing.T) {
				in := check(t, kind, name, "package p\n"+tt.src+"\n")
				got := runRule(in, policyOf(in), collectDirectives(in), arenaPointerEscape)
				if tt.want == "" {
					if len(got) != 0 {
						t.Fatalf("valid unsafe operation reported: %+v", got)
					}
					return
				}
				if len(got) != 1 || got[0].Severity != Deny || !strings.Contains(got[0].Message, tt.want) {
					t.Errorf("results = %+v, want one deny containing %q", got, tt.want)
				}
			})
		}
	}
}

func TestUnsafePointerCrossFileAlias(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, kind := range []Kind{KindGo, KindLgo} {
		ext, mode := ".go", syntax.CheckBranches
		if kind == KindLgo {
			ext, mode = ".lgo", mode|syntax.LogosMode
		}
		t.Run(ext, func(t *testing.T) {
			sources := []string{
				"package p\nimport \"unsafe\"\ntype Pointer = unsafe.Pointer\ntype address uintptr\n",
				"package p\nfunc good(p Pointer) Pointer { return Pointer(address(p) + 4) }\nfunc bad(u address) Pointer { return Pointer(u) }\n",
			}
			var files []*syntax.File
			for i, name := range []string{"alias", "consumer"} {
				file, err := syntax.Parse(syntax.NewFileBase(name+ext), strings.NewReader(sources[i]), nil, nil, mode)
				if err != nil {
					t.Fatal(err)
				}
				files = append(files, file)
			}
			info := &types2.Info{StoreTypesInSyntax: true, Uses: make(map[*syntax.Name]types2.Object)}
			conf := &types2.Config{Importer: &lintImporter{pkgs: make(map[string]*types2.Package)}, Logos: kind == KindLgo}
			pkg, err := conf.Check("p", files, info)
			if err != nil {
				t.Fatal(err)
			}
			in := &Input{Pkg: pkg, Info: info, Files: files, Kind: kind, LintCommand: true}
			got := runRule(in, policyOf(in), collectDirectives(in), arenaPointerEscape)
			if len(got) != 1 || got[0].Pos.RelFilename() != "consumer"+ext || got[0].Pos.Line() != 3 || got[0].Severity != Deny {
				t.Fatalf("results = %+v, want consumer line 3 deny", got)
			}
		})
	}
}

func TestUnsafeLintCommand(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	forbid, err := ParsePolicy([]byte(`{"version":1,"groups":{"correctness":"forbid"}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "unsafe.go"
		if kind == KindLgo {
			name = "unsafe.lgo"
		}
		for _, lintCommand := range []bool{false, true} {
			for _, suppressed := range []bool{false, true} {
				for _, policy := range []*Policy{nil, forbid} {
					comment := "// ordinary unsafe conversion"
					if suppressed {
						comment = "//logos:allow(correctness::arena_pointer_escape)"
					}
					src := "package p\nimport \"unsafe\"\n" + comment + "\nfunc f(u uintptr) unsafe.Pointer { return unsafe.Pointer(u) }\n"
					in := check(t, kind, name, src)
					in.LintCommand, in.Policy = lintCommand, policy
					var want []string
					if lintCommand {
						if policy != nil && kind == KindLgo {
							if suppressed {
								want = append(want, "3 correctness::allow_forbidden_rule forbid")
							}
							want = append(want, "4 correctness::arena_pointer_escape forbid")
						} else if !suppressed {
							want = append(want, "4 correctness::arena_pointer_escape deny")
						}
					}
					if got := summarize(Run(in)); !slices.Equal(got, want) {
						t.Errorf("%s lint=%v suppressed=%v explicitPolicy=%v: got %v, want %v", name, lintCommand, suppressed, policy != nil, got, want)
					}
				}
			}
		}
	}
}
