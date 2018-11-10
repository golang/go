// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template_test

import (
	"bytes"
	. "html/template"
	"strings"
	"testing"
)

func TestTemplateClone(t *testing.T) {
	// https://golang.org/issue/12996
	orig := New("name")
	clone, err := orig.Clone()
	if err != nil {
		t.Fatal(err)
	}
	if len(clone.Templates()) != len(orig.Templates()) {
		t.Fatalf("Invalid length of t.Clone().Templates()")
	}

	const want = "stuff"
	parsed := Must(clone.Parse(want))
	var buf bytes.Buffer
	err = parsed.Execute(&buf, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := buf.String(); got != want {
		t.Fatalf("got %q; want %q", got, want)
	}
}

func TestRedefineNonEmptyAfterExecution(t *testing.T) {
	c := newTestCase(t)
	c.mustParse(c.root, `foo`)
	c.mustExecute(c.root, nil, "foo")
	c.mustNotParse(c.root, `bar`)
}

func TestRedefineEmptyAfterExecution(t *testing.T) {
	c := newTestCase(t)
	c.mustParse(c.root, ``)
	c.mustExecute(c.root, nil, "")
	c.mustNotParse(c.root, `foo`)
	c.mustExecute(c.root, nil, "")
}

func TestRedefineAfterNonExecution(t *testing.T) {
	c := newTestCase(t)
	c.mustParse(c.root, `{{if .}}<{{template "X"}}>{{end}}{{define "X"}}foo{{end}}`)
	c.mustExecute(c.root, 0, "")
	c.mustNotParse(c.root, `{{define "X"}}bar{{end}}`)
	c.mustExecute(c.root, 1, "&lt;foo>")
}

func TestRedefineAfterNamedExecution(t *testing.T) {
	c := newTestCase(t)
	c.mustParse(c.root, `<{{template "X" .}}>{{define "X"}}foo{{end}}`)
	c.mustExecute(c.root, nil, "&lt;foo>")
	c.mustNotParse(c.root, `{{define "X"}}bar{{end}}`)
	c.mustExecute(c.root, nil, "&lt;foo>")
}

func TestRedefineNestedByNameAfterExecution(t *testing.T) {
	c := newTestCase(t)
	c.mustParse(c.root, `{{define "X"}}foo{{end}}`)
	c.mustExecute(c.lookup("X"), nil, "foo")
	c.mustNotParse(c.root, `{{define "X"}}bar{{end}}`)
	c.mustExecute(c.lookup("X"), nil, "foo")
}

func TestRedefineNestedByTemplateAfterExecution(t *testing.T) {
	c := newTestCase(t)
	c.mustParse(c.root, `{{define "X"}}foo{{end}}`)
	c.mustExecute(c.lookup("X"), nil, "foo")
	c.mustNotParse(c.lookup("X"), `bar`)
	c.mustExecute(c.lookup("X"), nil, "foo")
}

func TestRedefineSafety(t *testing.T) {
	c := newTestCase(t)
	c.mustParse(c.root, `<html><a href="{{template "X"}}">{{define "X"}}{{end}}`)
	c.mustExecute(c.root, nil, `<html><a href="">`)
	// Note: Every version of Go prior to Go 1.8 accepted the redefinition of "X"
	// on the next line, but luckily kept it from being used in the outer template.
	// Now we reject it, which makes clearer that we're not going to use it.
	c.mustNotParse(c.root, `{{define "X"}}" bar="baz{{end}}`)
	c.mustExecute(c.root, nil, `<html><a href="">`)
}

func TestRedefineTopUse(t *testing.T) {
	c := newTestCase(t)
	c.mustParse(c.root, `{{template "X"}}{{.}}{{define "X"}}{{end}}`)
	c.mustExecute(c.root, 42, `42`)
	c.mustNotParse(c.root, `{{define "X"}}<script>{{end}}`)
	c.mustExecute(c.root, 42, `42`)
}

func TestRedefineOtherParsers(t *testing.T) {
	c := newTestCase(t)
	c.mustParse(c.root, ``)
	c.mustExecute(c.root, nil, ``)
	if _, err := c.root.ParseFiles("no.template"); err == nil || !strings.Contains(err.Error(), "Execute") {
		t.Errorf("ParseFiles: %v\nwanted error about already having Executed", err)
	}
	if _, err := c.root.ParseGlob("*.no.template"); err == nil || !strings.Contains(err.Error(), "Execute") {
		t.Errorf("ParseGlob: %v\nwanted error about already having Executed", err)
	}
	if _, err := c.root.AddParseTree("t1", c.root.Tree); err == nil || !strings.Contains(err.Error(), "Execute") {
		t.Errorf("AddParseTree: %v\nwanted error about already having Executed", err)
	}
}

type testCase struct {
	t    *testing.T
	root *Template
}

func newTestCase(t *testing.T) *testCase {
	return &testCase{
		t:    t,
		root: New("root"),
	}
}

func (c *testCase) lookup(name string) *Template {
	return c.root.Lookup(name)
}

func (c *testCase) mustParse(t *Template, text string) {
	_, err := t.Parse(text)
	if err != nil {
		c.t.Fatalf("parse: %v", err)
	}
}

func (c *testCase) mustNotParse(t *Template, text string) {
	_, err := t.Parse(text)
	if err == nil {
		c.t.Fatalf("parse: unexpected success")
	}
}

func (c *testCase) mustExecute(t *Template, val interface{}, want string) {
	var buf bytes.Buffer
	err := t.Execute(&buf, val)
	if err != nil {
		c.t.Fatalf("execute: %v", err)
	}
	if buf.String() != want {
		c.t.Fatalf("template output:\n%s\nwant:\n%s", buf.String(), want)
	}
}
