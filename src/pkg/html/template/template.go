// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"
	"sync"
	"text/template"
	"text/template/parse"
)

// Template is a specialized template.Template that produces a safe HTML
// document fragment.
type Template struct {
	escaped bool
	// We could embed the text/template field, but it's safer not to because
	// we need to keep our version of the name space and the underlying
	// template's in sync.
	text       *template.Template
	*nameSpace // common to all associated templates
}

// nameSpace is the data structure shared by all templates in an association.
type nameSpace struct {
	mu  sync.Mutex
	set map[string]*Template
}

// Execute applies a parsed template to the specified data object,
// writing the output to wr.
func (t *Template) Execute(wr io.Writer, data interface{}) (err error) {
	t.nameSpace.mu.Lock()
	if !t.escaped {
		if err = escapeTemplates(t, t.Name()); err != nil {
			t.escaped = true
		}
	}
	t.nameSpace.mu.Unlock()
	if err != nil {
		return
	}
	return t.text.Execute(wr, data)
}

// ExecuteTemplate applies the template associated with t that has the given name
// to the specified data object and writes the output to wr.
func (t *Template) ExecuteTemplate(wr io.Writer, name string, data interface{}) (err error) {
	t.nameSpace.mu.Lock()
	tmpl := t.set[name]
	if tmpl == nil {
		t.nameSpace.mu.Unlock()
		return fmt.Errorf("template: no template %q associated with template %q", name, t.Name())
	}
	if !tmpl.escaped {
		err = escapeTemplates(tmpl, name)
	}
	t.nameSpace.mu.Unlock()
	if err != nil {
		return
	}
	return tmpl.text.ExecuteTemplate(wr, name, data)
}

// Parse parses a string into a template. Nested template definitions
// will be associated with the top-level template t. Parse may be
// called multiple times to parse definitions of templates to associate
// with t. It is an error if a resulting template is non-empty (contains
// content other than template definitions) and would replace a
// non-empty template with the same name.  (In multiple calls to Parse
// with the same receiver template, only one call can contain text
// other than space, comments, and template definitions.)
func (t *Template) Parse(src string) (*Template, error) {
	t.nameSpace.mu.Lock()
	t.escaped = false
	t.nameSpace.mu.Unlock()
	ret, err := t.text.Parse(src)
	if err != nil {
		return nil, err
	}
	// In general, all the named templates might have changed underfoot.
	// Regardless, some new ones may have been defined.
	// The template.Template set has been updated; update ours.
	t.nameSpace.mu.Lock()
	defer t.nameSpace.mu.Unlock()
	for _, v := range ret.Templates() {
		name := v.Name()
		tmpl := t.set[name]
		if tmpl == nil {
			tmpl = t.new(name)
		}
		tmpl.escaped = false
		tmpl.text = v
	}
	return t, nil
}

// AddParseTree is unimplemented.
func (t *Template) AddParseTree(name string, tree *parse.Tree) error {
	return fmt.Errorf("html/template: AddParseTree unimplemented")
}

// Clone is unimplemented.
func (t *Template) Clone(name string) error {
	return fmt.Errorf("html/template: Add unimplemented")
}

// New allocates a new HTML template with the given name.
func New(name string) *Template {
	tmpl := &Template{
		false,
		template.New(name),
		&nameSpace{
			set: make(map[string]*Template),
		},
	}
	tmpl.set[name] = tmpl
	return tmpl
}

// New allocates a new HTML template associated with the given one
// and with the same delimiters. The association, which is transitive,
// allows one template to invoke another with a {{template}} action.
func (t *Template) New(name string) *Template {
	t.nameSpace.mu.Lock()
	defer t.nameSpace.mu.Unlock()
	return t.new(name)
}

// new is the implementation of New, without the lock.
func (t *Template) new(name string) *Template {
	tmpl := &Template{
		false,
		t.text.New(name),
		t.nameSpace,
	}
	tmpl.set[name] = tmpl
	return tmpl
}

// Name returns the name of the template.
func (t *Template) Name() string {
	return t.text.Name()
}

// Funcs adds the elements of the argument map to the template's function map.
// It panics if a value in the map is not a function with appropriate return
// type. However, it is legal to overwrite elements of the map. The return
// value is the template, so calls can be chained.
func (t *Template) Funcs(funcMap template.FuncMap) *Template {
	t.text.Funcs(funcMap)
	return t
}

// Delims sets the action delimiters to the specified strings, to be used in
// subsequent calls to Parse, ParseFiles, or ParseGlob. Nested template
// definitions will inherit the settings. An empty delimiter stands for the
// corresponding default: {{ or }}.
// The return value is the template, so calls can be chained.
func (t *Template) Delims(left, right string) *Template {
	t.text.Delims(left, right)
	return t
}

// Lookup returns the template with the given name that is associated with t,
// or nil if there is no such template.
func (t *Template) Lookup(name string) *Template {
	t.nameSpace.mu.Lock()
	defer t.nameSpace.mu.Unlock()
	return t.set[name]
}

// Must panics if err is non-nil in the same way as template.Must.
func Must(t *Template, err error) *Template {
	t.text = template.Must(t.text, err)
	return t
}

// ParseFiles creates a new Template and parses the template definitions from
// the named files. The returned template's name will have the (base) name and
// (parsed) contents of the first file. There must be at least one file.
// If an error occurs, parsing stops and the returned *Template is nil.
func ParseFiles(filenames ...string) (*Template, error) {
	return parseFiles(nil, filenames...)
}

// ParseFiles parses the named files and associates the resulting templates with
// t. If an error occurs, parsing stops and the returned template is nil;
// otherwise it is t. There must be at least one file.
func (t *Template) ParseFiles(filenames ...string) (*Template, error) {
	return parseFiles(t, filenames...)
}

// parseFiles is the helper for the method and function. If the argument
// template is nil, it is created from the first file.
func parseFiles(t *Template, filenames ...string) (*Template, error) {
	if len(filenames) == 0 {
		// Not really a problem, but be consistent.
		return nil, fmt.Errorf("template: no files named in call to ParseFiles")
	}
	for _, filename := range filenames {
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			return nil, err
		}
		s := string(b)
		name := filepath.Base(filename)
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

// ParseGlob creates a new Template and parses the template definitions from the
// files identified by the pattern, which must match at least one file. The
// returned template will have the (base) name and (parsed) contents of the
// first file matched by the pattern. ParseGlob is equivalent to calling
// ParseFiles with the list of files matched by the pattern.
func ParseGlob(pattern string) (*Template, error) {
	return parseGlob(nil, pattern)
}

// ParseGlob parses the template definitions in the files identified by the
// pattern and associates the resulting templates with t. The pattern is
// processed by filepath.Glob and must match at least one file. ParseGlob is
// equivalent to calling t.ParseFiles with the list of files matched by the
// pattern.
func (t *Template) ParseGlob(pattern string) (*Template, error) {
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
	return parseFiles(t, filenames...)
}
