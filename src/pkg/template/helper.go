// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Helper functions to make constructing templates and sets easier.

package template

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

// Functions and methods to parse a single template.

// Must is a helper that wraps a call to a function returning (*Template, os.Error)
// and panics if the error is non-nil. It is intended for use in variable initializations
// such as
//	var t = template.Must(template.New("name").Parse("text"))
func Must(t *Template, err os.Error) *Template {
	if err != nil {
		panic(err)
	}
	return t
}

// ParseFile creates a new Template and parses the template definition from
// the named file.  The template name is the base name of the file.
func ParseFile(filename string) (*Template, os.Error) {
	t := New(filepath.Base(filename))
	return t.ParseFile(filename)
}

// parseFileInSet creates a new Template and parses the template
// definition from the named file. The template name is the base name
// of the file. It also adds the template to the set. Function bindings are
// checked against those in the set.
func parseFileInSet(filename string, set *Set) (*Template, os.Error) {
	t := New(filepath.Base(filename))
	return t.parseFileInSet(filename, set)
}

// ParseFile reads the template definition from a file and parses it to
// construct an internal representation of the template for execution.
// The returned template will be nil if an error occurs.
func (t *Template) ParseFile(filename string) (*Template, os.Error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return t.Parse(string(b))
}

// parseFileInSet is the same as ParseFile except that function bindings
// are checked against those in the set and the template is added
// to the set.
// The returned template will be nil if an error occurs.
func (t *Template) parseFileInSet(filename string, set *Set) (*Template, os.Error) {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return t.ParseInSet(string(b), set)
}

// Functions and methods to parse a set.

// SetMust is a helper that wraps a call to a function returning (*Set, os.Error)
// and panics if the error is non-nil. It is intended for use in variable initializations
// such as
//	var s = template.SetMust(template.ParseSetFiles("file"))
func SetMust(s *Set, err os.Error) *Set {
	if err != nil {
		panic(err)
	}
	return s
}

// ParseFiles parses the named files into a set of named templates.
// Each file must be parseable by itself.
// If an error occurs, parsing stops and the returned set is nil.
func (s *Set) ParseFiles(filenames ...string) (*Set, os.Error) {
	for _, filename := range filenames {
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			return nil, err
		}
		_, err = s.Parse(string(b))
		if err != nil {
			return nil, err
		}
	}
	return s, nil
}

// ParseSetFiles creates a new Set and parses the set definition from the
// named files. Each file must be individually parseable.
func ParseSetFiles(filenames ...string) (*Set, os.Error) {
	s := new(Set)
	for _, filename := range filenames {
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			return nil, err
		}
		_, err = s.Parse(string(b))
		if err != nil {
			return nil, err
		}
	}
	return s, nil
}

// ParseGlob parses the set definition from the files identified by the
// pattern.  The pattern is processed by filepath.Glob and must match at
// least one file.
// If an error occurs, parsing stops and the returned set is nil.
func (s *Set) ParseGlob(pattern string) (*Set, os.Error) {
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
func ParseSetGlob(pattern string) (*Set, os.Error) {
	set, err := new(Set).ParseGlob(pattern)
	if err != nil {
		return nil, err
	}
	return set, nil
}

// Functions and methods to parse stand-alone template files into a set.

// ParseTemplateFiles parses the named template files and adds
// them to the set. Each template will be named the base name of
// its file.
// Unlike with ParseFiles, each file should be a stand-alone template
// definition suitable for Template.Parse (not Set.Parse); that is, the
// file does not contain {{define}} clauses. ParseTemplateFiles is
// therefore equivalent to calling the ParseFile function to create
// individual templates, which are then added to the set.
// Each file must be parseable by itself.
// If an error occurs, parsing stops and the returned set is nil.
func (s *Set) ParseTemplateFiles(filenames ...string) (*Set, os.Error) {
	for _, filename := range filenames {
		_, err := parseFileInSet(filename, s)
		if err != nil {
			return nil, err
		}
	}
	return s, nil
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
func (s *Set) ParseTemplateGlob(pattern string) (*Set, os.Error) {
	filenames, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	for _, filename := range filenames {
		_, err := parseFileInSet(filename, s)
		if err != nil {
			return nil, err
		}
	}
	return s, nil
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
func ParseTemplateFiles(filenames ...string) (*Set, os.Error) {
	set := new(Set)
	set.init()
	for _, filename := range filenames {
		t, err := ParseFile(filename)
		if err != nil {
			return nil, err
		}
		if err := set.add(t); err != nil {
			return nil, err
		}
	}
	return set, nil
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
func ParseTemplateGlob(pattern string) (*Set, os.Error) {
	set := new(Set)
	filenames, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	if len(filenames) == 0 {
		return nil, fmt.Errorf("pattern matches no files: %#q", pattern)
	}
	for _, filename := range filenames {
		t, err := ParseFile(filename)
		if err != nil {
			return nil, err
		}
		if err := set.add(t); err != nil {
			return nil, err
		}
	}
	return set, nil
}
