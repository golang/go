// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"internal/goexperiment"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// Quoted method helpers are generated after C selectors have acquired
// their checked _Ctype identities. Ejection must spell C.int again:
// cmd/cgo rejects source identifiers in its private generated namespace.
func TestEjectQuotedCgoTypes(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	imports := "\"encoding/json\"\n\"fmt\"\n"
	decls := `
type JSONNumber[T any] int
func (n *JSONNumber[T]) UnmarshalJSON([]byte) error { *n = 7; return nil }
type TextNumber[T any] int
func (n *TextNumber[T]) UnmarshalText([]byte) error { *n = 9; return nil }
`
	fields := "J *JSONNumber[C.int] `json:\"j,string\"`\nT *TextNumber[C.int] `json:\"t,string\"`\n"
	init := "J: new(JSONNumber[C.int]), T: new(TextNumber[C.int])"
	values := "int(*got.J), int(*got.T)"
	want := "quoted: 7 9\n"
	if goexperiment.JSONv2 {
		imports += "\"encoding/json/jsontext\"\n"
		decls += `
type StreamNumber[T any] int
func (n *StreamNumber[T]) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if _, err := dec.ReadValue(); err != nil { return err }
	*n = 11
	return nil
}
`
		fields += "S StreamNumber[C.int] `json:\"s,string\"`\nP *StreamNumber[C.int] `json:\"p,string\"`\n"
		init += ", P: new(StreamNumber[C.int])"
		values += ", int(got.S), int(*got.P)"
		want = "quoted: 7 9 11 11\n"
	}
	src := []byte("package main\nimport \"C\"\nimport (\n" + imports + ")\n" + decls +
		"type Packet struct {\n" + fields + "}\nfunc main() {\ngot := Packet{" + init + "}\n" +
		"if err := json.Unmarshal([]byte(`{\"j\":\"1\",\"t\":\"\\\"hello\\\"\",\"s\":\"2\",\"p\":\"3\"}`), &got); err != nil { panic(err) }\n" +
		"fmt.Println(\"quoted:\", " + values + ")\n}\n")
	run := func(label, work string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(work, "go.mod"), []byte("module example.com/quotedcgo\n\ngo 1.27\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		cmd := testenv.Command(t, testenv.GoToolPath(t), "run", ".")
		cmd.Dir = work
		cmd.Env = append(os.Environ(), "GOPROXY=off", "GOTOOLCHAIN=local", "CGO_ENABLED=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s cgo: %v\n%s", label, err, out)
		}
		if string(out) != want {
			t.Fatalf("%s cgo output = %q, want %q", label, out, want)
		}
	}
	control := t.TempDir()
	if err := os.WriteFile(filepath.Join(control, "quoted.go"), src, 0o644); err != nil {
		t.Fatal(err)
	}
	run("ordinary Go", control)
	base := syntax.NewFileBase("quoted.lgo")
	file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
	gotypes := []byte("package main\ntype _Ctype_int int32\n")
	generated := ejectParse(t, syntax.NewFileBase("_cgo_gotypes.go"), gotypes, 0)
	files := []*syntax.File{file, generated}
	info := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Instances:  make(map[*syntax.Name]types2.Instance),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Implicits:  make(map[syntax.Node]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
	}
	conf := &types2.Config{
		Importer: &ejectImporter{pkgs: make(map[string]*types2.Package)},
		Logos:    true,
		Error: func(err error) {
			if !types2.LogosWarnOnly(err) {
				t.Error(err)
			}
		},
	}
	types2.LogosEnableCgo(conf)
	pkg, _ := conf.Check("example.com/quotedcgo", files, info)
	if t.Failed() {
		t.FailNow()
	}
	sources := map[*syntax.File][]byte{file: src, generated: gotypes}
	_, edits := Lower(pkg, info, files, sources)
	work := t.TempDir()
	if err := Eject(work, []*syntax.File{file}, sources, edits); err != nil {
		t.Fatal(err)
	}
	run("ejected", work)
}
