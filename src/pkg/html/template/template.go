// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"io"
	"path/filepath"
	"text/template"
)

// Set is a specialized template.Set that produces a safe HTML document
// fragment.
type Set struct {
	escaped map[string]bool
	template.Set
}

// Template is a specialized template.Template that produces a safe HTML
// document fragment.
type Template struct {
	escaped bool
	*template.Template
}

// Execute applies the named template to the specified data object, writing
// the output to wr.
func (s *Set) Execute(wr io.Writer, name string, data interface{}) error {
	if !s.escaped[name] {
		if err := escapeSet(&s.Set, name); err != nil {
			return err
		}
		if s.escaped == nil {
			s.escaped = make(map[string]bool)
		}
		s.escaped[name] = true
	}
	return s.Set.Execute(wr, name, data)
}

// Parse parses a string into a set of named templates.  Parse may be called
// multiple times for a given set, adding the templates defined in the string
// to the set.  If a template is redefined, the element in the set is
// overwritten with the new definition.
func (set *Set) Parse(src string) (*Set, error) {
	set.escaped = nil
	s, err := set.Set.Parse(src)
	if err != nil {
		return nil, err
	}
	if s != &(set.Set) {
		panic("allocated new set")
	}
	return set, nil
}

// Parse parses the template definition string to construct an internal
// representation of the template for execution.
func (tmpl *Template) Parse(src string) (*Template, error) {
	tmpl.escaped = false
	t, err := tmpl.Template.Parse(src)
	if err != nil {
		return nil, err
	}
	tmpl.Template = t
	return tmpl, nil
}

// Execute applies a parsed template to the specified data object,
// writing the output to wr.
func (t *Template) Execute(wr io.Writer, data interface{}) error {
	if !t.escaped {
		if err := escape(t.Template); err != nil {
			return err
		}
		t.escaped = true
	}
	return t.Template.Execute(wr, data)
}

// New allocates a new HTML template with the given name.
func New(name string) *Template {
	return &Template{false, template.New(name)}
}

// Must panics if err is non-nil in the same way as template.Must.
func Must(t *Template, err error) *Template {
	t.Template = template.Must(t.Template, err)
	return t
}

// ParseFile creates a new Template and parses the template definition from
// the named file.  The template name is the base name of the file.
func ParseFile(filename string) (*Template, error) {
	t, err := template.ParseFile(filename)
	if err != nil {
		return nil, err
	}
	return &Template{false, t}, nil
}

// ParseFile reads the template definition from a file and parses it to
// construct an internal representation of the template for execution.
// The returned template will be nil if an error occurs.
func (tmpl *Template) ParseFile(filename string) (*Template, error) {
	t, err := tmpl.Template.ParseFile(filename)
	if err != nil {
		return nil, err
	}
	tmpl.Template = t
	return tmpl, nil
}

// SetMust panics if the error is non-nil just like template.SetMust.
func SetMust(s *Set, err error) *Set {
	if err != nil {
		template.SetMust(&(s.Set), err)
	}
	return s
}

// ParseFiles parses the named files into a set of named templates.
// Each file must be parseable by itself.
// If an error occurs, parsing stops and the returned set is nil.
func (set *Set) ParseFiles(filenames ...string) (*Set, error) {
	s, err := set.Set.ParseFiles(filenames...)
	if err != nil {
		return nil, err
	}
	if s != &(set.Set) {
		panic("allocated new set")
	}
	return set, nil
}

// ParseSetFiles creates a new Set and parses the set definition from the
// named files. Each file must be individually parseable.
func ParseSetFiles(filenames ...string) (*Set, error) {
	set := new(Set)
	s, err := set.Set.ParseFiles(filenames...)
	if err != nil {
		return nil, err
	}
	if s != &(set.Set) {
		panic("allocated new set")
	}
	return set, nil
}

// ParseGlob parses the set definition from the files identified by the
// pattern. The pattern is processed by filepath.Glob and must match at
// least one file.
// If an error occurs, parsing stops and the returned set is nil.
func (s *Set) ParseGlob(pattern string) (*Set, error) {
	filenames, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	if len(filenames) == 0 {
		return nil, fmt.Errorf("pattern matches no files: %#q", pattern)
	}
	return s.ParseFiles(filenames...)
}

// ParseSetGlob creates a new Set and parses the set definition from the
// files identified by the pattern. The pattern is processed by filepath.Glob
// and must match at least one file.
func ParseSetGlob(pattern string) (*Set, error) {
	set, err := new(Set).ParseGlob(pattern)
	if err != nil {
		return nil, err
	}
	return set, nil
}

// Functions and methods to parse stand-alone template files into a set.

// ParseTemplateFiles parses the named template files and adds them to the set
// in the same way as template.ParseTemplateFiles but ensures that templates
// with upper-case names are contextually-autoescaped.
func (set *Set) ParseTemplateFiles(filenames ...string) (*Set, error) {
	s, err := set.Set.ParseTemplateFiles(filenames...)
	if err != nil {
		return nil, err
	}
	if s != &(set.Set) {
		panic("new set allocated")
	}
	return set, nil
}

// ParseTemplateGlob parses the template files matched by the
// patern and adds them to the set. Each template will be named
// the base name of its file.
// Unlike with ParseGlob, each file should be a stand-alone template
// definition suitable for Template.Parse (not Set.Parse); that is, the
// file does not contain {{define}} clauses. ParseTemplateGlob is
// therefore equivalent to calling the ParseFile function to create
// individual templates, which are then added to the set.
// Each file must be parseable by itself.
// If an error occurs, parsing stops and the returned set is nil.
func (s *Set) ParseTemplateGlob(pattern string) (*Set, error) {
	filenames, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	return s.ParseTemplateFiles(filenames...)
}

// ParseTemplateFiles creates a set by parsing the named files,
// each of which defines a single template. Each template will be
// named the base name of its file.
// Unlike with ParseFiles, each file should be a stand-alone template
// definition suitable for Template.Parse (not Set.Parse); that is, the
// file does not contain {{define}} clauses. ParseTemplateFiles is
// therefore equivalent to calling the ParseFile function to create
// individual templates, which are then added to the set.
// Each file must be parseable by itself. Parsing stops if an error is
// encountered.
func ParseTemplateFiles(filenames ...string) (*Set, error) {
	return new(Set).ParseTemplateFiles(filenames...)
}

// ParseTemplateGlob creates a set by parsing the files matched
// by the pattern, each of which defines a single template. The pattern
// is processed by filepath.Glob and must match at least one file. Each
// template will be named the base name of its file.
// Unlike with ParseGlob, each file should be a stand-alone template
// definition suitable for Template.Parse (not Set.Parse); that is, the
// file does not contain {{define}} clauses. ParseTemplateGlob is
// therefore equivalent to calling the ParseFile function to create
// individual templates, which are then added to the set.
// Each file must be parseable by itself. Parsing stops if an error is
// encountered.
func ParseTemplateGlob(pattern string) (*Set, error) {
	return new(Set).ParseTemplateGlob(pattern)
}
