// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"io"
	"reflect"
	"template/parse"
)

// Set holds a set of related templates that can refer to one another by name.
// The zero value represents an empty set.
// A template may be a member of multiple sets.
type Set struct {
	tmpl       map[string]*Template
	leftDelim  string
	rightDelim string
	parseFuncs FuncMap
	execFuncs  map[string]reflect.Value
}

func (s *Set) init() {
	if s.tmpl == nil {
		s.tmpl = make(map[string]*Template)
		s.parseFuncs = make(FuncMap)
		s.execFuncs = make(map[string]reflect.Value)
	}
}

// Delims sets the action delimiters, to be used in a subsequent
// parse, to the specified strings.
// An empty delimiter stands for the corresponding default: {{ or }}.
// The return value is the set, so calls can be chained.
func (s *Set) Delims(left, right string) *Set {
	s.leftDelim = left
	s.rightDelim = right
	return s
}

// Funcs adds the elements of the argument map to the set's function map.  It
// panics if a value in the map is not a function with appropriate return
// type.
// The return value is the set, so calls can be chained.
func (s *Set) Funcs(funcMap FuncMap) *Set {
	s.init()
	addValueFuncs(s.execFuncs, funcMap)
	addFuncs(s.parseFuncs, funcMap)
	return s
}

// Add adds the argument templates to the set. It panics if two templates
// with the same name are added or if a template is already a member of
// a set.
// The return value is the set, so calls can be chained.
func (s *Set) Add(templates ...*Template) *Set {
	for _, t := range templates {
		if err := s.add(t); err != nil {
			panic(err)
		}
	}
	return s
}

// add adds the argument template to the set.
func (s *Set) add(t *Template) error {
	s.init()
	if t.set != nil {
		return fmt.Errorf("template: %q already in a set", t.name)
	}
	if _, ok := s.tmpl[t.name]; ok {
		return fmt.Errorf("template: %q already defined in set", t.name)
	}
	s.tmpl[t.name] = t
	t.set = s
	return nil
}

// Template returns the template with the given name in the set,
// or nil if there is no such template.
func (s *Set) Template(name string) *Template {
	return s.tmpl[name]
}

// FuncMap returns the set's function map.
func (s *Set) FuncMap() FuncMap {
	return s.parseFuncs
}

// Execute applies the named template to the specified data object, writing
// the output to wr.
func (s *Set) Execute(wr io.Writer, name string, data interface{}) error {
	tmpl := s.tmpl[name]
	if tmpl == nil {
		return fmt.Errorf("template: no template %q in set", name)
	}
	return tmpl.Execute(wr, data)
}

// Parse parses a string into a set of named templates.  Parse may be called
// multiple times for a given set, adding the templates defined in the string
// to the set.  If a template is redefined, the element in the set is
// overwritten with the new definition.
func (s *Set) Parse(text string) (*Set, error) {
	trees, err := parse.Set(text, s.leftDelim, s.rightDelim, s.parseFuncs, builtins)
	if err != nil {
		return nil, err
	}
	s.init()
	for name, tree := range trees {
		tmpl := New(name)
		tmpl.Tree = tree
		tmpl.addToSet(s)
		s.tmpl[name] = tmpl
	}
	return s, nil
}
