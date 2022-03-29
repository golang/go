// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sync"
	"text/template"
	"text/template/parse"
)

// Template is a specialized Template from "text/template" that produces a safe
// HTML document fragment.
type Template struct {
	// Sticky error if escaping fails, or escapeOK if succeeded.
	escapeErr error
	// We could embed the text/template field, but it's safer not to because
	// we need to keep our version of the name space and the underlying
	// template's in sync.
	text *template.Template
	// The underlying template's parse tree, updated to be HTML-safe.
	Tree       *parse.Tree
	*nameSpace // common to all associated templates
}

// escapeOK is a sentinel value used to indicate valid escaping.
var escapeOK = fmt.Errorf("template escaped correctly")

// nameSpace is the data structure shared by all templates in an association.
type nameSpace struct {
	mu      sync.Mutex
	set     map[string]*Template
	escaped bool
	esc     escaper
}

// Templates returns a slice of the templates associated with t, including t
// itself.
func (t *Template) Templates() []*Template {
	ns := t.nameSpace
	ns.mu.Lock()
	defer ns.mu.Unlock()
	// Return a slice so we don't expose the map.
	m := make([]*Template, 0, len(ns.set))
	for _, v := range ns.set {
		m = append(m, v)
	}
	return m
}

// Option sets options for the template. Options are described by
// strings, either a simple string or "key=value". There can be at
// most one equals sign in an option string. If the option string
// is unrecognized or otherwise invalid, Option panics.
//
// Known options:
//
// missingkey: Control the behavior during execution if a map is
// indexed with a key that is not present in the map.
//	"missingkey=default" or "missingkey=invalid"
//		The default behavior: Do nothing and continue execution.
//		If printed, the result of the index operation is the string
//		"<no value>".
//	"missingkey=zero"
//		The operation returns the zero value for the map type's element.
//	"missingkey=error"
//		Execution stops immediately with an error.
//
func (t *Template) Option(opt ...string) *Template {
	t.text.Option(opt...)
	return t
}

// checkCanParse checks whether it is OK to parse templates.
// If not, it returns an error.
func (t *Template) checkCanParse() error {
	if t == nil {
		return nil
	}
	t.nameSpace.mu.Lock()
	defer t.nameSpace.mu.Unlock()
	if t.nameSpace.escaped {
		return fmt.Errorf("html/template: cannot Parse after Execute")
	}
	return nil
}

// escape escapes all associated templates.
func (t *Template) escape() error {
	t.nameSpace.mu.Lock()
	defer t.nameSpace.mu.Unlock()
	t.nameSpace.escaped = true
	if t.escapeErr == nil {
		if t.Tree == nil {
			return fmt.Errorf("template: %q is an incomplete or empty template", t.Name())
		}
		if err := escapeTemplate(t, t.text.Root, t.Name()); err != nil {
			return err
		}
	} else if t.escapeErr != escapeOK {
		return t.escapeErr
	}
	return nil
}

// Execute applies a parsed template to the specified data object,
// writing the output to wr.
// If an error occurs executing the template or writing its output,
// execution stops, but partial results may already have been written to
// the output writer.
// A template may be executed safely in parallel, although if parallel
// executions share a Writer the output may be interleaved.
func (t *Template) Execute(wr io.Writer, data any) error {
	if err := t.escape(); err != nil {
		return err
	}
	return t.text.Execute(wr, data)
}

// ExecuteTemplate applies the template associated with t that has the given
// name to the specified data object and writes the output to wr.
// If an error occurs executing the template or writing its output,
// execution stops, but partial results may already have been written to
// the output writer.
// A template may be executed safely in parallel, although if parallel
// executions share a Writer the output may be interleaved.
func (t *Template) ExecuteTemplate(wr io.Writer, name string, data any) error {
	tmpl, err := t.lookupAndEscapeTemplate(name)
	if err != nil {
		return err
	}
	return tmpl.text.Execute(wr, data)
}

// lookupAndEscapeTemplate guarantees that the template with the given name
// is escaped, or returns an error if it cannot be. It returns the named
// template.
func (t *Template) lookupAndEscapeTemplate(name string) (tmpl *Template, err error) {
	t.nameSpace.mu.Lock()
	defer t.nameSpace.mu.Unlock()
	t.nameSpace.escaped = true
	tmpl = t.set[name]
	if tmpl == nil {
		return nil, fmt.Errorf("html/template: %q is undefined", name)
	}
	if tmpl.escapeErr != nil && tmpl.escapeErr != escapeOK {
		return nil, tmpl.escapeErr
	}
	if tmpl.text.Tree == nil || tmpl.text.Root == nil {
		return nil, fmt.Errorf("html/template: %q is an incomplete template", name)
	}
	if t.text.Lookup(name) == nil {
		panic("html/template internal error: template escaping out of sync")
	}
	if tmpl.escapeErr == nil {
		err = escapeTemplate(tmpl, tmpl.text.Root, name)
	}
	return tmpl, err
}

// DefinedTemplates returns a string listing the defined templates,
// prefixed by the string "; defined templates are: ". If there are none,
// it returns the empty string. Used to generate an error message.
func (t *Template) DefinedTemplates() string {
	return t.text.DefinedTemplates()
}

// Parse parses text as a template body for t.
// Named template definitions ({{define ...}} or {{block ...}} statements) in text
// define additional templates associated with t and are removed from the
// definition of t itself.
//
// Templates can be redefined in successive calls to Parse,
// before the first use of Execute on t or any associated template.
// A template definition with a body containing only white space and comments
// is considered empty and will not replace an existing template's body.
// This allows using Parse to add new named template definitions without
// overwriting the main template body.
func (t *Template) Parse(text string) (*Template, error) {
	if err := t.checkCanParse(); err != nil {
		return nil, err
	}

	ret, err := t.text.Parse(text)
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
		tmpl.text = v
		tmpl.Tree = v.Tree
	}
	return t, nil
}

// AddParseTree creates a new template with the name and parse tree
// and associates it with t.
//
// It returns an error if t or any associated template has already been executed.
func (t *Template) AddParseTree(name string, tree *parse.Tree) (*Template, error) {
	if err := t.checkCanParse(); err != nil {
		return nil, err
	}

	t.nameSpace.mu.Lock()
	defer t.nameSpace.mu.Unlock()
	text, err := t.text.AddParseTree(name, tree)
	if err != nil {
		return nil, err
	}
	ret := &Template{
		nil,
		text,
		text.Tree,
		t.nameSpace,
	}
	t.set[name] = ret
	return ret, nil
}

// Clone returns a duplicate of the template, including all associated
// templates. The actual representation is not copied, but the name space of
// associated templates is, so further calls to Parse in the copy will add
// templates to the copy but not to the original. Clone can be used to prepare
// common templates and use them with variant definitions for other templates
// by adding the variants after the clone is made.
//
// It returns an error if t has already been executed.
func (t *Template) Clone() (*Template, error) {
	t.nameSpace.mu.Lock()
	defer t.nameSpace.mu.Unlock()
	if t.escapeErr != nil {
		return nil, fmt.Errorf("html/template: cannot Clone %q after it has executed", t.Name())
	}
	textClone, err := t.text.Clone()
	if err != nil {
		return nil, err
	}
	ns := &nameSpace{set: make(map[string]*Template)}
	ns.esc = makeEscaper(ns)
	ret := &Template{
		nil,
		textClone,
		textClone.Tree,
		ns,
	}
	ret.set[ret.Name()] = ret
	for _, x := range textClone.Templates() {
		name := x.Name()
		src := t.set[name]
		if src == nil || src.escapeErr != nil {
			return nil, fmt.Errorf("html/template: cannot Clone %q after it has executed", t.Name())
		}
		x.Tree = x.Tree.Copy()
		ret.set[name] = &Template{
			nil,
			x,
			x.Tree,
			ret.nameSpace,
		}
	}
	// Return the template associated with the name of this template.
	return ret.set[ret.Name()], nil
}

// New allocates a new HTML template with the given name.
func New(name string) *Template {
	ns := &nameSpace{set: make(map[string]*Template)}
	ns.esc = makeEscaper(ns)
	tmpl := &Template{
		nil,
		template.New(name),
		nil,
		ns,
	}
	tmpl.set[name] = tmpl
	return tmpl
}

// New allocates a new HTML template associated with the given one
// and with the same delimiters. The association, which is transitive,
// allows one template to invoke another with a {{template}} action.
//
// If a template with the given name already exists, the new HTML template
// will replace it. The existing template will be reset and disassociated with
// t.
func (t *Template) New(name string) *Template {
	t.nameSpace.mu.Lock()
	defer t.nameSpace.mu.Unlock()
	return t.new(name)
}

// new is the implementation of New, without the lock.
func (t *Template) new(name string) *Template {
	tmpl := &Template{
		nil,
		t.text.New(name),
		nil,
		t.nameSpace,
	}
	if existing, ok := tmpl.set[name]; ok {
		emptyTmpl := New(existing.Name())
		*existing = *emptyTmpl
	}
	tmpl.set[name] = tmpl
	return tmpl
}

// Name returns the name of the template.
func (t *Template) Name() string {
	return t.text.Name()
}

// FuncMap is the type of the map defining the mapping from names to
// functions. Each function must have either a single return value, or two
// return values of which the second has type error. In that case, if the
// second (error) argument evaluates to non-nil during execution, execution
// terminates and Execute returns that error. FuncMap has the same base type
// as FuncMap in "text/template", copied here so clients need not import
// "text/template".
type FuncMap map[string]any

// Funcs adds the elements of the argument map to the template's function map.
// It must be called before the template is parsed.
// It panics if a value in the map is not a function with appropriate return
// type. However, it is legal to overwrite elements of the map. The return
// value is the template, so calls can be chained.
func (t *Template) Funcs(funcMap FuncMap) *Template {
	t.text.Funcs(template.FuncMap(funcMap))
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

// Must is a helper that wraps a call to a function returning (*Template, error)
// and panics if the error is non-nil. It is intended for use in variable initializations
// such as
//	var t = template.Must(template.New("name").Parse("html"))
func Must(t *Template, err error) *Template {
	if err != nil {
		panic(err)
	}
	return t
}

// ParseFiles creates a new Template and parses the template definitions from
// the named files. The returned template's name will have the (base) name and
// (parsed) contents of the first file. There must be at least one file.
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
//
// When parsing multiple files with the same name in different directories,
// the last one mentioned will be the one that results.
//
// ParseFiles returns an error if t or any associated template has already been executed.
func (t *Template) ParseFiles(filenames ...string) (*Template, error) {
	return parseFiles(t, readFileOS, filenames...)
}

// parseFiles is the helper for the method and function. If the argument
// template is nil, it is created from the first file.
func parseFiles(t *Template, readFile func(string) (string, []byte, error), filenames ...string) (*Template, error) {
	if err := t.checkCanParse(); err != nil {
		return nil, err
	}

	if len(filenames) == 0 {
		// Not really a problem, but be consistent.
		return nil, fmt.Errorf("html/template: no files named in call to ParseFiles")
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
//
// ParseGlob returns an error if t or any associated template has already been executed.
func (t *Template) ParseGlob(pattern string) (*Template, error) {
	return parseGlob(t, pattern)
}

// parseGlob is the implementation of the function and method ParseGlob.
func parseGlob(t *Template, pattern string) (*Template, error) {
	if err := t.checkCanParse(); err != nil {
		return nil, err
	}
	filenames, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	if len(filenames) == 0 {
		return nil, fmt.Errorf("html/template: pattern matches no files: %#q", pattern)
	}
	return parseFiles(t, readFileOS, filenames...)
}

// IsTrue reports whether the value is 'true', in the sense of not the zero of its type,
// and whether the value has a meaningful truth value. This is the definition of
// truth used by if and other such actions.
func IsTrue(val any) (truth, ok bool) {
	return template.IsTrue(val)
}

// ParseFS is like ParseFiles or ParseGlob but reads from the file system fs
// instead of the host operating system's file system.
// It accepts a list of glob patterns.
// (Note that most file names serve as glob patterns matching only themselves.)
func ParseFS(fs fs.FS, patterns ...string) (*Template, error) {
	return parseFS(nil, fs, patterns)
}

// ParseFS is like ParseFiles or ParseGlob but reads from the file system fs
// instead of the host operating system's file system.
// It accepts a list of glob patterns.
// (Note that most file names serve as glob patterns matching only themselves.)
func (t *Template) ParseFS(fs fs.FS, patterns ...string) (*Template, error) {
	return parseFS(t, fs, patterns)
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
