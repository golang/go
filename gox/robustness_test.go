package gox

import (
	"bytes"
	"context"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestOrdinaryGoCorpus(t *testing.T) {
	for _, name := range []string{"go/ast/ast.go", "go/parser/parser.go", "go/scanner/scanner.go", "fmt/print.go", "net/http/server.go"} {
		src, e := os.ReadFile(filepath.Join(runtime.GOROOT(), "src", name))
		if e != nil {
			t.Fatal(e)
		}
		got, e := TransformFile(name, src, Options{BareJSX: true})
		if e != nil {
			t.Fatal(e)
		}
		normalize := func(b []byte) string {
			fs := token.NewFileSet()
			f, e := parser.ParseFile(fs, name, b, parser.ParseComments|parser.SkipObjectResolution)
			if e != nil {
				t.Fatal(e)
			}
			var out bytes.Buffer
			printer.Fprint(&out, fs, f)
			return out.String()
		}
		if normalize(src) != normalize(got.Source) {
			t.Errorf("ordinary Go AST changed: %s", name)
		}
	}
}

func TestPackageHygiene(t *testing.T) {
	files := map[string][]byte{"a.gox": []byte(`package p;func A()string{return F"{1}"}`), "b.gox": []byte(`package p;func B()string{return F"{2}"}`)}
	out, e := TransformPackage(context.Background(), files, Options{})
	if e != nil {
		t.Fatal(e)
	}
	names := map[string]bool{}
	for path, r := range out.Files {
		f, e := parser.ParseFile(token.NewFileSet(), path, r.Source, 0)
		if e != nil {
			t.Fatal(e)
		}
		for _, d := range f.Decls {
			if d, ok := d.(*ast.FuncDecl); ok {
				if names[d.Name.Name] {
					t.Fatal("duplicate helper", d.Name.Name)
				}
				names[d.Name.Name] = true
			}
		}
	}
}
func TestContextualTypes(t *testing.T) {
	for _, typ := range []string{"[]int", "*int", "map[string]int", "struct { X int }", "interface { M() }"} {
		if _, e := TransformFile("type.gox", []byte("package p;var x=do /*type*/ "+typ+"{return nil}"), Options{}); e != nil {
			t.Errorf("%s: %v", typ, e)
		}
	}
}
func TestHookAfterConditional(t *testing.T) {
	_, e := TransformFile("bad.gox", []byte(`package p;component C(){if true{return <a/>};n,_:=useState(0);return <a>{F"{n}"}</a>}`), Options{BareJSX: true})
	if e == nil || !strings.Contains(e.Error(), "hooks must") {
		t.Fatal(e)
	}
}
func FuzzTransform(f *testing.F) {
	for _, s := range []string{`package p;var x=F"{F"{1}"}"`, `package p;var x=<a>{"x"}</a>`, `package p;func f(){func g(){};g()}`, `package p;var x=do int{return 1}`} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 16384 {
			t.Skip()
		}
		_, _ = TransformFile("fuzz.gox", []byte(s), Options{BareJSX: true})
	})
}

func TestPackageHookShadow(t *testing.T) {
	files := map[string][]byte{
		"a.gox": []byte(`package p;component C(){if true {useState()};return <a/>}`),
		"b.go":  []byte(`package p;func useState(){}`),
	}
	r, e := TransformPackage(context.Background(), files, Options{BareJSX: true})
	if e != nil {
		t.Fatal(e)
	}
	if strings.Contains(string(r.Files["a.gox"].Source), ".UseState(") {
		t.Fatal("rewrote ordinary package binding")
	}
}
func TestSourceMapping(t *testing.T) {
	r, e := TransformFile("source.gox", []byte("package p\nfunc F()string {\nreturn F\"{1}\"\n}\n"), Options{SourcePositions: true})
	if e != nil {
		t.Fatal(e)
	}
	if len(r.SourceMap) == 0 || !bytes.Contains(r.Source, []byte("//line source.gox:")) {
		t.Fatal("missing mappings")
	}
}
