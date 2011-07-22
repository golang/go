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

// ParseFile parses the named files into a set of named templates.
// Each file must be parseable by itself. Parsing stops if an error is
// encountered.
func (s *Set) ParseFile(filenames ...string) os.Error {
	for _, filename := range filenames {
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			return err
		}
		err = s.Parse(string(b))
		if err != nil {
			return err
		}
	}
	return nil
}

// MustParseFile parses the named file into a set of named templates.
// Each file must be parseable by itself.
// MustParseFile panics if any file cannot be read or parsed.
func (s *Set) MustParseFile(filenames ...string) *Set {
	err := s.ParseFile(filenames...)
	if err != nil {
		panic(err)
	}
	return s
}

// ParseSetFile creates a new Set and parses the set definition from the
// named files. Each file must be individually parseable.
func ParseSetFile(filenames ...string) (set *Set, err os.Error) {
	s := new(Set)
	var b []byte
	for _, filename := range filenames {
		b, err = ioutil.ReadFile(filename)
		if err != nil {
			return
		}
		err = s.Parse(string(b))
		if err != nil {
			return
		}
	}
	return s, nil
}

// MustParseSetFile creates a new Set and parses the set definition from the
// named files. Each file must be individually parseable.
// MustParseSetFile panics if any file cannot be read or parsed.
func MustParseSetFile(filenames ...string) *Set {
	s, err := ParseSetFile(filenames...)
	if err != nil {
		panic(err)
	}
	return s
}

// ParseFiles parses the set definition from the files identified by the
// pattern.  The pattern is processed by filepath.Glob and must match at
// least one file.
func (s *Set) ParseFiles(pattern string) os.Error {
	filenames, err := filepath.Glob(pattern)
	if err != nil {
		return err
	}
	if len(filenames) == 0 {
		return fmt.Errorf("pattern matches no files: %#q", pattern)
	}
	return s.ParseFile(filenames...)
}

// ParseSetFiles creates a new Set and parses the set definition from the
// files identified by the pattern. The pattern is processed by filepath.Glob
// and must match at least one file.
func ParseSetFiles(pattern string) (*Set, os.Error) {
	set := new(Set)
	err := set.ParseFiles(pattern)
	if err != nil {
		return nil, err
	}
	return set, nil
}

// MustParseSetFiles creates a new Set and parses the set definition from the
// files identified by the pattern. The pattern is processed by filepath.Glob.
// MustParseSetFiles panics if the pattern is invalid or a matched file cannot be
// read or parsed.
func MustParseSetFiles(pattern string) *Set {
	set, err := ParseSetFiles(pattern)
	if err != nil {
		panic(err)
	}
	return set
}
