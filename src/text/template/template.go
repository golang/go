// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"reflect"
	"sync"
	"text/template/parse"
)

// common holds the information shared by related templates.
type common struct {
	tmpl   map[string]*Template // Map from name to defined templates.
	muTmpl sync.RWMutex         // protects tmpl
	option option
	// We use two maps, one for parsing and one for execution.
	// This separation makes the API cleaner since it doesn't
	// expose reflection to the client.
	muFuncs    sync.RWMutex // protects parseFuncs and execFuncs
	parseFuncs FuncMap
	execFuncs  map[string]reflect.Value
}

// Template is the representation of a parsed template. The *parse.Tree
// field is exported only for use by html/template and should be treated
// as unexported by all other clients.
type Template struct {
	name string
	*parse.Tree
	*common
	leftDelim  string
	rightDelim string
}

// New allocates a new, undefined template with the given name.
func New(name string) *Template {
	t := &Template{
		name: name,
	}
	t.init()
	return t
}

// Name returns the name of the template.
func (t *Template) Name() string {
	return t.name
}

// New allocates a new, undefined template associated with the given one and with the same
// delimiters. The association, which is transitive, allows one template to
// invoke another with a {{template}} action.
//
// Because associated templates share underlying data, template construction
// cannot be done safely in parallel. Once the templates are constructed, they
// can be executed in parallel.
func (t *Template) New(name string) *Template {
	t.init()
	nt := &Template{
		name:       name,
		common:     t.common,
		leftDelim:  t.leftDelim,
		rightDelim: t.rightDelim,
	}
	return nt
}

// init guarantees that t has a valid common structure.
func (t *Template) init() {
	if t.common == nil {
		c := new(common)
		c.tmpl = make(map[string]*Template)
		c.parseFuncs = make(FuncMap)
		c.execFuncs = make(map[string]reflect.Value)
		t.common = c
	}
}

// Clone returns a duplicate of the template, including all associated
// templates. The actual representation is not copied, but the name space of
// associated templates is, so further calls to Parse in the copy will add
// templates to the copy but not to the original. Clone can be used to prepare
// common templates and use them with variant definitions for other templates
// by adding the variants after the clone is made.
func (t *Template) Clone() (*Template, error) {
	nt := t.copy(nil)
	nt.init()
	if t.common == nil {
		return nt, nil
	}
	t.muTmpl.RLock()
	defer t.muTmpl.RUnlock()
	for k, v := range t.tmpl {
		if k == t.name {
			nt.tmpl[t.name] = nt
			continue
		}
		// The associated templates share nt's common structure.
		tmpl := v.copy(nt.common)
		nt.tmpl[k] = tmpl
	}
	t.muFuncs.RLock()
	defer t.muFuncs.RUnlock()
	for k, v := range t.parseFuncs {
		nt.parseFuncs[k] = v
	}
	for k, v := range t.execFuncs {
		nt.execFuncs[k] = v
	}
	return nt, nil
}

// copy returns a shallow copy of t, with common set to the argument.
func (t *Template) copy(c *common) *Template {
	return &Template{
		name:       t.name,
		Tree:       t.Tree,
		common:     c,
		leftDelim:  t.leftDelim,
		rightDelim: t.rightDelim,
	}
}

// AddParseTree associates the argument parse tree with the template t, giving
// it the specified name. If the template has not been defined, this tree becomes
// its definition. If it has been defined and already has that name, the existing
// definition is replaced; otherwise a new template is created, defined, and returned.
func (t *Template) AddParseTree(name string, tree *parse.Tree) (*Template, error) {
	t.init()
	t.muTmpl.Lock()
	defer t.muTmpl.Unlock()
	nt := t
	if name != t.name {
		nt = t.New(name)
	}
	// Even if nt == t, we need to install it in the common.tmpl map.
	if t.associate(nt, tree) || nt.Tree == nil {
		nt.Tree = tree
	}
	return nt, nil
}

// Templates returns a slice of defined templates associated with t.
func (t *Template) Templates() []*Template {
	if t.common == nil {
		return nil
	}
	// Return a slice so we don't expose the map.
	t.muTmpl.RLock()
	defer t.muTmpl.RUnlock()
	m := make([]*Template, 0, len(t.tmpl))
	for _, v := range t.tmpl {
		m = append(m, v)
	}
	return m
}

// Delims sets the action delimiters to the specified strings, to be used in
// subsequent calls to Parse, ParseFiles, or ParseGlob. Nested template
// definitions will inherit the settings. An empty delimiter stands for the
// corresponding default: {{ or }}.
// The return value is the template, so calls can be chained.
func (t *Template) Delims(left, right string) *Template {
	t.init()
	t.leftDelim = left
	t.rightDelim = right
	return t
}

// Funcs adds the elements of the argument map to the template's function map.
// It must be called before the template is parsed.
// It panics if a value in the map is not a function with appropriate return
// type or if the name cannot be used syntactically as a function in a template.
// It is legal to overwrite elements of the map. The return value is the template,
// so calls can be chained.
func (t *Template) Funcs(funcMap FuncMap) *Template {
	t.init()
	t.muFuncs.Lock()
	defer t.muFuncs.Unlock()
	addValueFuncs(t.execFuncs, funcMap)
	addFuncs(t.parseFuncs, funcMap)
	return t
}

// Lookup returns the template with the given name that is associated with t.
// It returns nil if there is no such template or the template has no definition.
func (t *Template) Lookup(name string) *Template {
	if t.common == nil {
		return nil
	}
	t.muTmpl.RLock()
	defer t.muTmpl.RUnlock()
	return t.tmpl[name]
}

// Parse parses text as a template body for t.
// Named template definitions ({{define ...}} or {{block ...}} statements) in text
// define additional templates associated with t and are removed from the
// definition of t itself.
//
// Templates can be redefined in successive calls to Parse.
// A template definition with a body containing only white space and comments
// is considered empty and will not replace an existing template's body.
// This allows using Parse to add new named template definitions without
// overwriting the main template body.
func (t *Template) Parse(text string) (*Template, error) {
	t.init()
	t.muFuncs.RLock()
	trees, err := parse.Parse(t.name, text, t.leftDelim, t.rightDelim, t.parseFuncs, builtins())
	t.muFuncs.RUnlock()
	if err != nil {
		return nil, err
	}
	// Add the newly parsed trees, including the one for t, into our common structure.
	for name, tree := range trees {
		if _, err := t.AddParseTree(name, tree); err != nil {
			return nil, err
		}
	}
	return t, nil
}

// associate installs the new template into the group of templates associated
// with t. The two are already known to share the common structure.
// The boolean return value reports whether to store this tree as t.Tree.
func (t *Template) associate(new *Template, tree *parse.Tree) bool {
	if new.common != t.common {
		panic("internal error: associate not common")
	}
	if old := t.tmpl[new.name]; old != nil && parse.IsEmptyTree(tree.Root) && old.Tree != nil {
		// If a template by that name exists,
		// don't replace it with an empty template.
		return false
	}
	t.tmpl[new.name] = new
	return true
}
