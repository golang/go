// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lazytemplate is a thin wrapper over text/template, allowing the use
// of global template variables without forcing them to be parsed at init.
package lazytemplate

import (
	"io"
	"os"
	"strings"
	"sync"
	"text/template"
)

// Template is a wrapper around text/template.Template, where the underlying
// template will be parsed the first time it is needed.
type Template struct {
	name, text string

	once sync.Once
	tmpl *template.Template
}

func (r *Template) tp() *template.Template {
	r.once.Do(r.build)
	return r.tmpl
}

func (r *Template) build() {
	r.tmpl = template.Must(template.New(r.name).Parse(r.text))
	r.name, r.text = "", ""
}

func (r *Template) Execute(w io.Writer, data any) error {
	return r.tp().Execute(w, data)
}

var inTest = len(os.Args) > 0 && strings.HasSuffix(strings.TrimSuffix(os.Args[0], ".exe"), ".test")

// New creates a new lazy template, delaying the parsing work until it is first
// needed. If the code is being run as part of tests, the template parsing will
// happen immediately.
func New(name, text string) *Template {
	lt := &Template{name: name, text: text}
	if inTest {
		// In tests, always parse the templates early.
		lt.tp()
	}
	return lt
}
