// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"strings"
	"sync"
	"testing"
	"text/template/parse"
)

func TestAddParseTreeHTML(t *testing.T) {
	root := Must(New("root").Parse(`{{define "a"}} {{.}} {{template "b"}} {{.}} "></a>{{end}}`))
	tree, err := parse.Parse("t", `{{define "b"}}<a href="{{end}}`, "", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	added := Must(root.AddParseTree("b", tree["b"]))
	b := new(bytes.Buffer)
	err = added.ExecuteTemplate(b, "a", "1>0")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := b.String(), ` 1&gt;0 <a href=" 1%3e0 "></a>`; got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestClone(t *testing.T) {
	// The {{.}} will be executed with data "<i>*/" in different contexts.
	// In the t0 template, it will be in a text context.
	// In the t1 template, it will be in a URL context.
	// In the t2 template, it will be in a JavaScript context.
	// In the t3 template, it will be in a CSS context.
	const tmpl = `{{define "a"}}{{template "lhs"}}{{.}}{{template "rhs"}}{{end}}`
	b := new(bytes.Buffer)

	// Create an incomplete template t0.
	t0 := Must(New("t0").Parse(tmpl))

	// Clone t0 as t1.
	t1 := Must(t0.Clone())
	Must(t1.Parse(`{{define "lhs"}} <a href=" {{end}}`))
	Must(t1.Parse(`{{define "rhs"}} "></a> {{end}}`))

	// Execute t1.
	b.Reset()
	if err := t1.ExecuteTemplate(b, "a", "<i>*/"); err != nil {
		t.Fatal(err)
	}
	if got, want := b.String(), ` <a href=" %3ci%3e*/ "></a> `; got != want {
		t.Errorf("t1: got %q want %q", got, want)
	}

	// Clone t0 as t2.
	t2 := Must(t0.Clone())
	Must(t2.Parse(`{{define "lhs"}} <p onclick="javascript: {{end}}`))
	Must(t2.Parse(`{{define "rhs"}} "></p> {{end}}`))

	// Execute t2.
	b.Reset()
	if err := t2.ExecuteTemplate(b, "a", "<i>*/"); err != nil {
		t.Fatal(err)
	}
	if got, want := b.String(), ` <p onclick="javascript: &#34;\u003ci\u003e*/&#34; "></p> `; got != want {
		t.Errorf("t2: got %q want %q", got, want)
	}

	// Clone t0 as t3, but do not execute t3 yet.
	t3 := Must(t0.Clone())
	Must(t3.Parse(`{{define "lhs"}} <style> {{end}}`))
	Must(t3.Parse(`{{define "rhs"}} </style> {{end}}`))

	// Complete t0.
	Must(t0.Parse(`{{define "lhs"}} ( {{end}}`))
	Must(t0.Parse(`{{define "rhs"}} ) {{end}}`))

	// Clone t0 as t4. Redefining the "lhs" template should not fail.
	t4 := Must(t0.Clone())
	if _, err := t4.Parse(`{{define "lhs"}} OK {{end}}`); err != nil {
		t.Errorf(`redefine "lhs": got err %v want nil`, err)
	}
	// Cloning t1 should fail as it has been executed.
	if _, err := t1.Clone(); err == nil {
		t.Error("cloning t1: got nil err want non-nil")
	}
	// Redefining the "lhs" template in t1 should fail as it has been executed.
	if _, err := t1.Parse(`{{define "lhs"}} OK {{end}}`); err == nil {
		t.Error(`redefine "lhs": got nil err want non-nil`)
	}

	// Execute t0.
	b.Reset()
	if err := t0.ExecuteTemplate(b, "a", "<i>*/"); err != nil {
		t.Fatal(err)
	}
	if got, want := b.String(), ` ( &lt;i&gt;*/ ) `; got != want {
		t.Errorf("t0: got %q want %q", got, want)
	}

	// Clone t0. This should fail, as t0 has already executed.
	if _, err := t0.Clone(); err == nil {
		t.Error(`t0.Clone(): got nil err want non-nil`)
	}

	// Similarly, cloning sub-templates should fail.
	if _, err := t0.Lookup("a").Clone(); err == nil {
		t.Error(`t0.Lookup("a").Clone(): got nil err want non-nil`)
	}
	if _, err := t0.Lookup("lhs").Clone(); err == nil {
		t.Error(`t0.Lookup("lhs").Clone(): got nil err want non-nil`)
	}

	// Execute t3.
	b.Reset()
	if err := t3.ExecuteTemplate(b, "a", "<i>*/"); err != nil {
		t.Fatal(err)
	}
	if got, want := b.String(), ` <style> ZgotmplZ </style> `; got != want {
		t.Errorf("t3: got %q want %q", got, want)
	}
}

func TestTemplates(t *testing.T) {
	names := []string{"t0", "a", "lhs", "rhs"}
	// Some template definitions borrowed from TestClone.
	const tmpl = `
		{{define "a"}}{{template "lhs"}}{{.}}{{template "rhs"}}{{end}}
		{{define "lhs"}} <a href=" {{end}}
		{{define "rhs"}} "></a> {{end}}`
	t0 := Must(New("t0").Parse(tmpl))
	templates := t0.Templates()
	if len(templates) != len(names) {
		t.Errorf("expected %d templates; got %d", len(names), len(templates))
	}
	for _, name := range names {
		found := false
		for _, tmpl := range templates {
			if name == tmpl.text.Name() {
				found = true
				break
			}
		}
		if !found {
			t.Error("could not find template", name)
		}
	}
}

// This used to crash; https://golang.org/issue/3281
func TestCloneCrash(t *testing.T) {
	t1 := New("all")
	Must(t1.New("t1").Parse(`{{define "foo"}}foo{{end}}`))
	t1.Clone()
}

// Ensure that this guarantee from the docs is upheld:
// "Further calls to Parse in the copy will add templates
// to the copy but not to the original."
func TestCloneThenParse(t *testing.T) {
	t0 := Must(New("t0").Parse(`{{define "a"}}{{template "embedded"}}{{end}}`))
	t1 := Must(t0.Clone())
	Must(t1.Parse(`{{define "embedded"}}t1{{end}}`))
	if len(t0.Templates())+1 != len(t1.Templates()) {
		t.Error("adding a template to a clone added it to the original")
	}
	// double check that the embedded template isn't available in the original
	err := t0.ExecuteTemplate(ioutil.Discard, "a", nil)
	if err == nil {
		t.Error("expected 'no such template' error")
	}
}

// https://golang.org/issue/5980
func TestFuncMapWorksAfterClone(t *testing.T) {
	funcs := FuncMap{"customFunc": func() (string, error) {
		return "", errors.New("issue5980")
	}}

	// get the expected error output (no clone)
	uncloned := Must(New("").Funcs(funcs).Parse("{{customFunc}}"))
	wantErr := uncloned.Execute(ioutil.Discard, nil)

	// toClone must be the same as uncloned. It has to be recreated from scratch,
	// since cloning cannot occur after execution.
	toClone := Must(New("").Funcs(funcs).Parse("{{customFunc}}"))
	cloned := Must(toClone.Clone())
	gotErr := cloned.Execute(ioutil.Discard, nil)

	if wantErr.Error() != gotErr.Error() {
		t.Errorf("clone error message mismatch want %q got %q", wantErr, gotErr)
	}
}

// https://golang.org/issue/16101
func TestTemplateCloneExecuteRace(t *testing.T) {
	const (
		input   = `<title>{{block "a" .}}a{{end}}</title><body>{{block "b" .}}b{{end}}<body>`
		overlay = `{{define "b"}}A{{end}}`
	)
	outer := Must(New("outer").Parse(input))
	tmpl := Must(Must(outer.Clone()).Parse(overlay))

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				if err := tmpl.Execute(ioutil.Discard, "data"); err != nil {
					panic(err)
				}
			}
		}()
	}
	wg.Wait()
}

func TestTemplateCloneLookup(t *testing.T) {
	// Template.escape makes an assumption that the template associated
	// with t.Name() is t. Check that this holds.
	tmpl := Must(New("x").Parse("a"))
	tmpl = Must(tmpl.Clone())
	if tmpl.Lookup(tmpl.Name()) != tmpl {
		t.Error("after Clone, tmpl.Lookup(tmpl.Name()) != tmpl")
	}
}

func TestCloneGrowth(t *testing.T) {
	tmpl := Must(New("root").Parse(`<title>{{block "B". }}Arg{{end}}</title>`))
	tmpl = Must(tmpl.Clone())
	Must(tmpl.Parse(`{{define "B"}}Text{{end}}`))
	for i := 0; i < 10; i++ {
		tmpl.Execute(ioutil.Discard, nil)
	}
	if len(tmpl.DefinedTemplates()) > 200 {
		t.Fatalf("too many templates: %v", len(tmpl.DefinedTemplates()))
	}
}

// https://golang.org/issue/17735
func TestCloneRedefinedName(t *testing.T) {
	const base = `
{{ define "a" -}}<title>{{ template "b" . -}}</title>{{ end -}}
{{ define "b" }}{{ end -}}
`
	const page = `{{ template "a" . }}`

	t1 := Must(New("a").Parse(base))

	for i := 0; i < 2; i++ {
		t2 := Must(t1.Clone())
		t2 = Must(t2.New(fmt.Sprintf("%d", i)).Parse(page))
		err := t2.Execute(ioutil.Discard, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
}

// Issue 24791.
func TestClonePipe(t *testing.T) {
	a := Must(New("a").Parse(`{{define "a"}}{{range $v := .A}}{{$v}}{{end}}{{end}}`))
	data := struct{ A []string }{A: []string{"hi"}}
	b := Must(a.Clone())
	var buf strings.Builder
	if err := b.Execute(&buf, &data); err != nil {
		t.Fatal(err)
	}
	if got, want := buf.String(), "hi"; got != want {
		t.Errorf("got %q want %q", got, want)
	}
}
