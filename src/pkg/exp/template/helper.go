// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Helper functions to make constructing templates and sets easier.

package template

import (
	"io/ioutil"
	"os"
	"path/filepath"
)

// MustParse parses the template definition string to construct an internal
// representation of the template for execution.
// It panics if the template cannot be parsed.
func (t *Template) MustParse(text string) *Template {
	if err := t.Parse(text); err != nil {
		panic(err)
	}
	return t
}

// ParseFile reads the template definition from a file and parses it to
// construct an internal representation of the template for execution.
func (t *Template) ParseFile(filename string) os.Error {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	return t.Parse(string(b))
}

// MustParseFile reads the template definition from a file and parses it to
// construct an internal representation of the template for execution.
// It panics if the file cannot be read or the template cannot be parsed.
func (t *Template) MustParseFile(filename string) *Template {
	if err := t.ParseFile(filename); err != nil {
		panic(err)
	}
	return t
}

// ParseFile creates a new Template and parses the template definition from
// the named file.  The template name is the base name of the file.
func ParseFile(filename string) (*Template, os.Error) {
	t := New(filepath.Base(filename))
	return t, t.ParseFile(filename)
}

// MustParseFile creates a new Template and parses the template definition
// from the named file.  The template name is the base name of the file.
// It panics if the file cannot be read or the template cannot be parsed.
func MustParseFile(filename string) *Template {
	return New(filepath.Base(filename)).MustParseFile(filename)
}

// MustParse parses a string into a set of named templates.
// It panics if the set cannot be parsed.
func (s *Set) MustParse(text string) *Set {
	if err := s.Parse(text); err != nil {
		panic(err)
	}
	return s
}

// ParseFile parses the named file into a set of named templates.
func (s *Set) ParseFile(filename string) os.Error {
	b, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	return s.Parse(string(b))
}

// MustParseFile parses the named file into a set of named templates.
// It panics if the file cannot be read or the set cannot be parsed.
func (s *Set) MustParseFile(filename string) *Set {
	if err := s.ParseFile(filename); err != nil {
		panic(err)
	}
	return s
}

// ParseSetFile creates a new Set and parses the set definition from the
// named file.
func ParseSetFile(filename string) (*Set, os.Error) {
	s := new(Set)
	return s, s.ParseFile(filename)
}

// MustParseSetFile creates a new Set and parses the set definition from the
// named file.
// It panics if the file cannot be read or the set cannot be parsed.
func MustParseSetFile(filename string) *Set {
	return new(Set).MustParseFile(filename)
}
