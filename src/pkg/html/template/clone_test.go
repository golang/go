// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"testing"
	"text/template/parse"
)

func TestAddParseTree(t *testing.T) {
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

	// Clone t0 as t4. Redefining the "lhs" template should fail.
	t4 := Must(t0.Clone())
	if _, err := t4.Parse(`{{define "lhs"}} FAIL {{end}}`); err == nil {
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
