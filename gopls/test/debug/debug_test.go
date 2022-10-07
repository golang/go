// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug_test

// Provide 'static type checking' of the templates. This guards against changes is various
// gopls datastructures causing template execution to fail. The checking is done by
// the github.com/jba/templatecheck pacakge. Before that is run, the test checks that
// its list of templates and their arguments corresponds to the arguments in
// calls to render(). The test assumes that all uses of templates are done through render().

import (
	"go/ast"
	"html/template"
	"log"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/jba/templatecheck"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

type tdata struct {
	tmpl *template.Template
	data interface{} // a value of the needed type
}

var templates = map[string]tdata{
	"MainTmpl":    {debug.MainTmpl, &debug.Instance{}},
	"DebugTmpl":   {debug.DebugTmpl, nil},
	"RPCTmpl":     {debug.RPCTmpl, &debug.Rpcs{}},
	"TraceTmpl":   {debug.TraceTmpl, debug.TraceResults{}},
	"CacheTmpl":   {debug.CacheTmpl, &cache.Cache{}},
	"SessionTmpl": {debug.SessionTmpl, &cache.Session{}},
	"ViewTmpl":    {debug.ViewTmpl, &cache.View{}},
	"ClientTmpl":  {debug.ClientTmpl, &debug.Client{}},
	"ServerTmpl":  {debug.ServerTmpl, &debug.Server{}},
	//"FileTmpl":    {FileTmpl, source.Overlay{}}, // need to construct a source.Overlay in init
	"InfoTmpl":   {debug.InfoTmpl, "something"},
	"MemoryTmpl": {debug.MemoryTmpl, runtime.MemStats{}},
}

// construct a source.Overlay for fileTmpl
type fakeOverlay struct{}

func (fakeOverlay) Version() int32 {
	return 0
}
func (fakeOverlay) Session() string {
	return ""
}
func (fakeOverlay) VersionedFileIdentity() source.VersionedFileIdentity {
	return source.VersionedFileIdentity{}
}
func (fakeOverlay) FileIdentity() source.FileIdentity {
	return source.FileIdentity{}
}
func (fakeOverlay) Kind() source.FileKind {
	return 0
}
func (fakeOverlay) Read() ([]byte, error) {
	return nil, nil
}
func (fakeOverlay) Saved() bool {
	return true
}
func (fakeOverlay) URI() span.URI {
	return ""
}

var _ source.Overlay = fakeOverlay{}

func init() {
	log.SetFlags(log.Lshortfile)
	var v fakeOverlay
	templates["FileTmpl"] = tdata{debug.FileTmpl, v}
}

func TestTemplates(t *testing.T) {
	if runtime.GOOS == "android" {
		t.Skip("this test is not supported for Android")
	}
	cfg := &packages.Config{
		Mode: packages.NeedTypesInfo | packages.LoadAllSyntax, // figure out what's necessary PJW
	}
	pkgs, err := packages.Load(cfg, "golang.org/x/tools/gopls/internal/lsp/debug")
	if err != nil {
		t.Fatal(err)
	}
	if len(pkgs) != 1 {
		t.Fatalf("expected a single package, but got %d", len(pkgs))
	}
	p := pkgs[0]
	if len(p.Errors) != 0 {
		t.Fatalf("compiler error, e.g. %v", p.Errors[0])
	}
	// find the calls to render in serve.go
	tree := treeOf(p, "serve.go")
	if tree == nil {
		t.Fatalf("found no syntax tree for %s", "serve.go")
	}
	renders := callsOf(p, tree, "render")
	if len(renders) == 0 {
		t.Fatalf("found no calls to render")
	}
	var found = make(map[string]bool)
	for _, r := range renders {
		if len(r.Args) != 2 {
			// template, func
			t.Fatalf("got %d args, expected 2", len(r.Args))
		}
		t0, ok := p.TypesInfo.Types[r.Args[0]]
		if !ok || !t0.IsValue() || t0.Type.String() != "*html/template.Template" {
			t.Fatalf("no type info for template")
		}
		if id, ok := r.Args[0].(*ast.Ident); !ok {
			t.Errorf("expected *ast.Ident, got %T", r.Args[0])
		} else {
			found[id.Name] = true
		}
	}
	// make sure found and templates have the same templates
	for k := range found {
		if _, ok := templates[k]; !ok {
			t.Errorf("code has template %s, but test does not", k)
		}
	}
	for k := range templates {
		if _, ok := found[k]; !ok {
			t.Errorf("test has template %s, code does not", k)
		}
	}
	// now check all the known templates, in alphabetic order, for determinacy
	keys := []string{}
	for k := range templates {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := templates[k]
		// the FuncMap is an annoyance; should not be necessary
		if err := templatecheck.CheckHTML(v.tmpl, v.data); err != nil {
			t.Errorf("%s: %v", k, err)
		}
	}
}

func callsOf(p *packages.Package, tree *ast.File, name string) []*ast.CallExpr {
	var ans []*ast.CallExpr
	f := func(n ast.Node) bool {
		x, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if y, ok := x.Fun.(*ast.Ident); ok {
			if y.Name == name {
				ans = append(ans, x)
			}
		}
		return true
	}
	ast.Inspect(tree, f)
	return ans
}
func treeOf(p *packages.Package, fname string) *ast.File {
	for _, tree := range p.Syntax {
		loc := tree.Package
		pos := p.Fset.PositionFor(loc, false)
		if strings.HasSuffix(pos.Filename, fname) {
			return tree
		}
	}
	return nil
}
