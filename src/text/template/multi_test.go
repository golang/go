// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

// Tests for multiple-template parsing and execution.

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"text/template/parse"
)

const (
	noError  = true
	hasError = false
)

type multiParseTest struct {
	name    string
	input   string
	ok      bool
	names   []string
	results []string
}

var multiParseTests = []multiParseTest{
	{"empty", "", noError,
		nil,
		nil},
	{"one", `{{define "foo"}} FOO {{end}}`, noError,
		[]string{"foo"},
		[]string{" FOO "}},
	{"two", `{{define "foo"}} FOO {{end}}{{define "bar"}} BAR {{end}}`, noError,
		[]string{"foo", "bar"},
		[]string{" FOO ", " BAR "}},
	// errors
	{"missing end", `{{define "foo"}} FOO `, hasError,
		nil,
		nil},
	{"malformed name", `{{define "foo}} FOO `, hasError,
		nil,
		nil},
}

func TestMultiParse(t *testing.T) {
	for _, test := range multiParseTests {
		template, err := New("root").Parse(test.input)
		switch {
		case err == nil && !test.ok:
			t.Errorf("%q: expected error; got none", test.name)
			continue
		case err != nil && test.ok:
			t.Errorf("%q: unexpected error: %v", test.name, err)
			continue
		case err != nil && !test.ok:
			// expected error, got one
			if *debug {
				fmt.Printf("%s: %s\n\t%s\n", test.name, test.input, err)
			}
			continue
		}
		if template == nil {
			continue
		}
		if len(template.tmpl) != len(test.names)+1 { // +1 for root
			t.Errorf("%s: wrong number of templates; wanted %d got %d", test.name, len(test.names), len(template.tmpl))
			continue
		}
		for i, name := range test.names {
			tmpl, ok := template.tmpl[name]
			if !ok {
				t.Errorf("%s: can't find template %q", test.name, name)
				continue
			}
			result := tmpl.Root.String()
			if result != test.results[i] {
				t.Errorf("%s=(%q): got\n\t%v\nexpected\n\t%v", test.name, test.input, result, test.results[i])
			}
		}
	}
}

var multiExecTests = []execTest{
	{"empty", "", "", nil, true},
	{"text", "some text", "some text", nil, true},
	{"invoke x", `{{template "x" .SI}}`, "TEXT", tVal, true},
	{"invoke x no args", `{{template "x"}}`, "TEXT", tVal, true},
	{"invoke dot int", `{{template "dot" .I}}`, "17", tVal, true},
	{"invoke dot []int", `{{template "dot" .SI}}`, "[3 4 5]", tVal, true},
	{"invoke dotV", `{{template "dotV" .U}}`, "v", tVal, true},
	{"invoke nested int", `{{template "nested" .I}}`, "17", tVal, true},
	{"variable declared by template", `{{template "nested" $x:=.SI}},{{index $x 1}}`, "[3 4 5],4", tVal, true},

	// User-defined function: test argument evaluator.
	{"testFunc literal", `{{oneArg "joe"}}`, "oneArg=joe", tVal, true},
	{"testFunc .", `{{oneArg .}}`, "oneArg=joe", "joe", true},
}

// These strings are also in testdata/*.
const multiText1 = `
	{{define "x"}}TEXT{{end}}
	{{define "dotV"}}{{.V}}{{end}}
`

const multiText2 = `
	{{define "dot"}}{{.}}{{end}}
	{{define "nested"}}{{template "dot" .}}{{end}}
`

func TestMultiExecute(t *testing.T) {
	// Declare a couple of templates first.
	template, err := New("root").Parse(multiText1)
	if err != nil {
		t.Fatalf("parse error for 1: %s", err)
	}
	_, err = template.Parse(multiText2)
	if err != nil {
		t.Fatalf("parse error for 2: %s", err)
	}
	testExecute(multiExecTests, template, t)
}

func TestParseFiles(t *testing.T) {
	_, err := ParseFiles("DOES NOT EXIST")
	if err == nil {
		t.Error("expected error for non-existent file; got none")
	}
	template := New("root")
	_, err = template.ParseFiles("testdata/file1.tmpl", "testdata/file2.tmpl")
	if err != nil {
		t.Fatalf("error parsing files: %v", err)
	}
	testExecute(multiExecTests, template, t)
}

func TestParseGlob(t *testing.T) {
	_, err := ParseGlob("DOES NOT EXIST")
	if err == nil {
		t.Error("expected error for non-existent file; got none")
	}
	_, err = New("error").ParseGlob("[x")
	if err == nil {
		t.Error("expected error for bad pattern; got none")
	}
	template := New("root")
	_, err = template.ParseGlob("testdata/file*.tmpl")
	if err != nil {
		t.Fatalf("error parsing files: %v", err)
	}
	testExecute(multiExecTests, template, t)
}

func TestParseFS(t *testing.T) {
	fs := os.DirFS("testdata")

	{
		_, err := ParseFS(fs, "DOES NOT EXIST")
		if err == nil {
			t.Error("expected error for non-existent file; got none")
		}
	}

	{
		template := New("root")
		_, err := template.ParseFS(fs, "file1.tmpl", "file2.tmpl")
		if err != nil {
			t.Fatalf("error parsing files: %v", err)
		}
		testExecute(multiExecTests, template, t)
	}

	{
		template := New("root")
		_, err := template.ParseFS(fs, "file*.tmpl")
		if err != nil {
			t.Fatalf("error parsing files: %v", err)
		}
		testExecute(multiExecTests, template, t)
	}
}

// In these tests, actual content (not just template definitions) comes from the parsed files.

var templateFileExecTests = []execTest{
	{"test", `{{template "tmpl1.tmpl"}}{{template "tmpl2.tmpl"}}`, "template1\n\ny\ntemplate2\n\nx\n", 0, true},
}

func TestParseFilesWithData(t *testing.T) {
	template, err := New("root").ParseFiles("testdata/tmpl1.tmpl", "testdata/tmpl2.tmpl")
	if err != nil {
		t.Fatalf("error parsing files: %v", err)
	}
	testExecute(templateFileExecTests, template, t)
}

func TestParseGlobWithData(t *testing.T) {
	template, err := New("root").ParseGlob("testdata/tmpl*.tmpl")
	if err != nil {
		t.Fatalf("error parsing files: %v", err)
	}
	testExecute(templateFileExecTests, template, t)
}

const (
	cloneText1 = `{{define "a"}}{{template "b"}}{{template "c"}}{{end}}`
	cloneText2 = `{{define "b"}}b{{end}}`
	cloneText3 = `{{define "c"}}root{{end}}`
	cloneText4 = `{{define "c"}}clone{{end}}`
)

func TestClone(t *testing.T) {
	// Create some templates and clone the root.
	root, err := New("root").Parse(cloneText1)
	if err != nil {
		t.Fatal(err)
	}
	_, err = root.Parse(cloneText2)
	if err != nil {
		t.Fatal(err)
	}
	clone := Must(root.Clone())
	// Add variants to both.
	_, err = root.Parse(cloneText3)
	if err != nil {
		t.Fatal(err)
	}
	_, err = clone.Parse(cloneText4)
	if err != nil {
		t.Fatal(err)
	}
	// Verify that the clone is self-consistent.
	for k, v := range clone.tmpl {
		if k == clone.name && v.tmpl[k] != clone {
			t.Error("clone does not contain root")
		}
		if v != v.tmpl[v.name] {
			t.Errorf("clone does not contain self for %q", k)
		}
	}
	// Execute root.
	var b strings.Builder
	err = root.ExecuteTemplate(&b, "a", 0)
	if err != nil {
		t.Fatal(err)
	}
	if b.String() != "broot" {
		t.Errorf("expected %q got %q", "broot", b.String())
	}
	// Execute copy.
	b.Reset()
	err = clone.ExecuteTemplate(&b, "a", 0)
	if err != nil {
		t.Fatal(err)
	}
	if b.String() != "bclone" {
		t.Errorf("expected %q got %q", "bclone", b.String())
	}
}

func TestAddParseTree(t *testing.T) {
	// Create some templates.
	root, err := New("root").Parse(cloneText1)
	if err != nil {
		t.Fatal(err)
	}
	_, err = root.Parse(cloneText2)
	if err != nil {
		t.Fatal(err)
	}
	// Add a new parse tree.
	tree, err := parse.Parse("cloneText3", cloneText3, "", "", nil, builtins())
	if err != nil {
		t.Fatal(err)
	}
	added, err := root.AddParseTree("c", tree["c"])
	if err != nil {
		t.Fatal(err)
	}
	// Execute.
	var b strings.Builder
	err = added.ExecuteTemplate(&b, "a", 0)
	if err != nil {
		t.Fatal(err)
	}
	if b.String() != "broot" {
		t.Errorf("expected %q got %q", "broot", b.String())
	}
}

// Issue 7032
func TestAddParseTreeToUnparsedTemplate(t *testing.T) {
	master := "{{define \"master\"}}{{end}}"
	tmpl := New("master")
	tree, err := parse.Parse("master", master, "", "", nil)
	if err != nil {
		t.Fatalf("unexpected parse err: %v", err)
	}
	masterTree := tree["master"]
	tmpl.AddParseTree("master", masterTree) // used to panic
}

func TestRedefinition(t *testing.T) {
	var tmpl *Template
	var err error
	if tmpl, err = New("tmpl1").Parse(`{{define "test"}}foo{{end}}`); err != nil {
		t.Fatalf("parse 1: %v", err)
	}
	if _, err = tmpl.Parse(`{{define "test"}}bar{{end}}`); err != nil {
		t.Fatalf("got error %v, expected nil", err)
	}
	if _, err = tmpl.New("tmpl2").Parse(`{{define "test"}}bar{{end}}`); err != nil {
		t.Fatalf("got error %v, expected nil", err)
	}
}

// Issue 10879
func TestEmptyTemplateCloneCrash(t *testing.T) {
	t1 := New("base")
	t1.Clone() // used to panic
}

// Issue 10910, 10926
func TestTemplateLookUp(t *testing.T) {
	t1 := New("foo")
	if t1.Lookup("foo") != nil {
		t.Error("Lookup returned non-nil value for undefined template foo")
	}
	t1.New("bar")
	if t1.Lookup("bar") != nil {
		t.Error("Lookup returned non-nil value for undefined template bar")
	}
	t1.Parse(`{{define "foo"}}test{{end}}`)
	if t1.Lookup("foo") == nil {
		t.Error("Lookup returned nil value for defined template")
	}
}

func TestNew(t *testing.T) {
	// template with same name already exists
	t1, _ := New("test").Parse(`{{define "test"}}foo{{end}}`)
	t2 := t1.New("test")

	if t1.common != t2.common {
		t.Errorf("t1 & t2 didn't share common struct; got %v != %v", t1.common, t2.common)
	}
	if t1.Tree == nil {
		t.Error("defined template got nil Tree")
	}
	if t2.Tree != nil {
		t.Error("undefined template got non-nil Tree")
	}

	containsT1 := false
	for _, tmpl := range t1.Templates() {
		if tmpl == t2 {
			t.Error("Templates included undefined template")
		}
		if tmpl == t1 {
			containsT1 = true
		}
	}
	if !containsT1 {
		t.Error("Templates didn't include defined template")
	}
}

func TestParse(t *testing.T) {
	// In multiple calls to Parse with the same receiver template, only one call
	// can contain text other than space, comments, and template definitions
	t1 := New("test")
	if _, err := t1.Parse(`{{define "test"}}{{end}}`); err != nil {
		t.Fatalf("parsing test: %s", err)
	}
	if _, err := t1.Parse(`{{define "test"}}{{/* this is a comment */}}{{end}}`); err != nil {
		t.Fatalf("parsing test: %s", err)
	}
	if _, err := t1.Parse(`{{define "test"}}foo{{end}}`); err != nil {
		t.Fatalf("parsing test: %s", err)
	}
}

func TestEmptyTemplate(t *testing.T) {
	cases := []struct {
		defn []string
		in   string
		want string
	}{
		{[]string{"x", "y"}, "", "y"},
		{[]string{""}, "once", ""},
		{[]string{"", ""}, "twice", ""},
		{[]string{"{{.}}", "{{.}}"}, "twice", "twice"},
		{[]string{"{{/* a comment */}}", "{{/* a comment */}}"}, "comment", ""},
		{[]string{"{{.}}", ""}, "twice", ""},
	}

	for i, c := range cases {
		root := New("root")

		var (
			m   *Template
			err error
		)
		for _, d := range c.defn {
			m, err = root.New(c.in).Parse(d)
			if err != nil {
				t.Fatal(err)
			}
		}
		buf := &strings.Builder{}
		if err := m.Execute(buf, c.in); err != nil {
			t.Error(i, err)
			continue
		}
		if buf.String() != c.want {
			t.Errorf("expected string %q: got %q", c.want, buf.String())
		}
	}
}

// Issue 19249 was a regression in 1.8 caused by the handling of empty
// templates added in that release, which got different answers depending
// on the order templates appeared in the internal map.
func TestIssue19294(t *testing.T) {
	// The empty block in "xhtml" should be replaced during execution
	// by the contents of "stylesheet", but if the internal map associating
	// names with templates is built in the wrong order, the empty block
	// looks non-empty and this doesn't happen.
	var inlined = map[string]string{
		"stylesheet": `{{define "stylesheet"}}stylesheet{{end}}`,
		"xhtml":      `{{block "stylesheet" .}}{{end}}`,
	}
	all := []string{"stylesheet", "xhtml"}
	for i := 0; i < 100; i++ {
		res, err := New("title.xhtml").Parse(`{{template "xhtml" .}}`)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range all {
			_, err := res.New(name).Parse(inlined[name])
			if err != nil {
				t.Fatal(err)
			}
		}
		var buf strings.Builder
		res.Execute(&buf, 0)
		if buf.String() != "stylesheet" {
			t.Fatalf("iteration %d: got %q; expected %q", i, buf.String(), "stylesheet")
		}
	}
}

// Issue 48436
func TestAddToZeroTemplate(t *testing.T) {
	tree, err := parse.Parse("c", cloneText3, "", "", nil, builtins())
	if err != nil {
		t.Fatal(err)
	}
	var tmpl Template
	tmpl.AddParseTree("x", tree["c"])
}
