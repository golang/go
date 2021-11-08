// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Helper functions to make constructing templates easier.

package template

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
)

// Functions and methods to parse templates.

// Must is a helper that wraps a call to a function returning (*Template, error)
// and panics if the error is non-nil. It is intended for use in variable
// initializations such as
//	var t = template.Must(template.New("name").Parse("text"))
func Must(t *Template, err error) *Template {
	if err != nil {
		panic(err)
	}
	return t
}

// ParseFiles creates a new Template and parses the template definitions from
// the named files. The returned template's name will have the base name and
// parsed contents of the first file. There must be at least one file.
// If an error occurs, parsing stops and the returned *Template is nil.
//
// When parsing multiple files with the same name in different directories,
// the last one mentioned will be the one that results.
// For instance, ParseFiles("a/foo", "b/foo") stores "b/foo" as the template
// named "foo", while "a/foo" is unavailable.
func ParseFiles(filenames ...string) (*Template, error) {
	return parseFiles(nil, readFileOS, filenames...)
}

// ParseFiles parses the named files and associates the resulting templates with
// t. If an error occurs, parsing stops and the returned template is nil;
// otherwise it is t. There must be at least one file.
// Since the templates created by ParseFiles are named by the base
// names of the argument files, t should usually have the name of one
// of the (base) names of the files. If it does not, depending on t's
// contents before calling ParseFiles, t.Execute may fail. In that
// case use t.ExecuteTemplate to execute a valid template.
//
// When parsing multiple files with the same name in different directories,
// the last one mentioned will be the one that results.
func (t *Template) ParseFiles(filenames ...string) (*Template, error) {
	t.init()
	return parseFiles(t, readFileOS, filenames...)
}

// parseFiles is the helper for the method and function. If the argument
// template is nil, it is created from the first file.
func parseFiles(t *Template, readFile func(string) (string, []byte, error), filenames ...string) (*Template, error) {
	if len(filenames) == 0 {
		// Not really a problem, but be consistent.
		return nil, fmt.Errorf("template: no files named in call to ParseFiles")
	}
	for _, filename := range filenames {
		name, b, err := readFile(filename)
		if err != nil {
			return nil, err
		}
		s := string(b)
		// First template becomes return value if not already defined,
		// and we use that one for subsequent New calls to associate
		// all the templates together. Also, if this file has the same name
		// as t, this file becomes the contents of t, so
		//  t, err := New(name).Funcs(xxx).ParseFiles(name)
		// works. Otherwise we create a new template associated with t.
		var tmpl *Template
		if t == nil {
			t = New(name)
		}
		if name == t.Name() {
			tmpl = t
		} else {
			tmpl = t.New(name)
		}
		_, err = tmpl.Parse(s)
		if err != nil {
			return nil, err
		}
	}
	return t, nil
}

// ParseGlob creates a new Template and parses the template definitions from
// the files identified by the pattern. The files are matched according to the
// semantics of filepath.Match, and the pattern must match at least one file.
// The returned template will have the (base) name and (parsed) contents of the
// first file matched by the pattern. ParseGlob is equivalent to calling
// ParseFiles with the list of files matched by the pattern.
//
// When parsing multiple files with the same name in different directories,
// the last one mentioned will be the one that results.
func ParseGlob(pattern string) (*Template, error) {
	return parseGlob(nil, pattern)
}

// ParseGlob parses the template definitions in the files identified by the
// pattern and associates the resulting templates with t. The files are matched
// according to the semantics of filepath.Match, and the pattern must match at
// least one file. ParseGlob is equivalent to calling t.ParseFiles with the
// list of files matched by the pattern.
//
// When parsing multiple files with the same name in different directories,
// the last one mentioned will be the one that results.
func (t *Template) ParseGlob(pattern string) (*Template, error) {
	t.init()
	return parseGlob(t, pattern)
}

// parseGlob is the implementation of the function and method ParseGlob.
func parseGlob(t *Template, pattern string) (*Template, error) {
	filenames, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	if len(filenames) == 0 {
		return nil, fmt.Errorf("template: pattern matches no files: %#q", pattern)
	}
	return parseFiles(t, readFileOS, filenames...)
}

// ParseFS is like ParseFiles or ParseGlob but reads from the file system fsys
// instead of the host operating system's file system.
// It accepts a list of glob patterns.
// (Note that most file names serve as glob patterns matching only themselves.)
func ParseFS(fsys fs.FS, patterns ...string) (*Template, error) {
	return parseFS(nil, fsys, patterns)
}

// ParseFS is like ParseFiles or ParseGlob but reads from the file system fsys
// instead of the host operating system's file system.
// It accepts a list of glob patterns.
// (Note that most file names serve as glob patterns matching only themselves.)
func (t *Template) ParseFS(fsys fs.FS, patterns ...string) (*Template, error) {
	t.init()
	return parseFS(t, fsys, patterns)
}

func parseFS(t *Template, fsys fs.FS, patterns []string) (*Template, error) {
	var filenames []string
	for _, pattern := range patterns {
		list, err := fs.Glob(fsys, pattern)
		if err != nil {
			return nil, err
		}
		if len(list) == 0 {
			return nil, fmt.Errorf("template: pattern matches no files: %#q", pattern)
		}
		filenames = append(filenames, list...)
	}
	return parseFiles(t, readFileFS(fsys), filenames...)
}

func readFileOS(file string) (name string, b []byte, err error) {
	name = filepath.Base(file)
	b, err = os.ReadFile(file)
	return
}

func readFileFS(fsys fs.FS) func(string) (string, []byte, error) {
	return func(file string) (name string, b []byte, err error) {
		name = path.Base(file)
		b, err = fs.ReadFile(fsys, file)
		return
	}
}
