// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"io"
	"os"
	"reflect"
	"runtime"
	"strconv"
)

// Set holds a set of related templates that can refer to one another by name.
// The zero value represents an empty set.
// A template may be a member of multiple sets.
type Set struct {
	tmpl  map[string]*Template
	funcs map[string]reflect.Value
}

func (s *Set) init() {
	if s.tmpl == nil {
		s.tmpl = make(map[string]*Template)
		s.funcs = make(map[string]reflect.Value)
	}
}

// Funcs adds the elements of the argument map to the set's function map.  It
// panics if a value in the map is not a function with appropriate return
// type.
// The return value is the set, so calls can be chained.
func (s *Set) Funcs(funcMap FuncMap) *Set {
	s.init()
	addFuncs(s.funcs, funcMap)
	return s
}

// Add adds the argument templates to the set. It panics if two templates
// with the same name are added or if a template is already a member of
// a set.
// The return value is the set, so calls can be chained.
func (s *Set) Add(templates ...*Template) *Set {
	s.init()
	for _, t := range templates {
		if t.set != nil {
			panic(fmt.Errorf("template: %q already in a set", t.name))
		}
		if _, ok := s.tmpl[t.name]; ok {
			panic(fmt.Errorf("template: %q already defined in set", t.name))
		}
		s.tmpl[t.name] = t
		t.set = s
	}
	return s
}

// Template returns the template with the given name in the set,
// or nil if there is no such template.
func (s *Set) Template(name string) *Template {
	return s.tmpl[name]
}

// Execute applies the named template to the specified data object, writing
// the output to wr.
func (s *Set) Execute(wr io.Writer, name string, data interface{}) os.Error {
	tmpl := s.tmpl[name]
	if tmpl == nil {
		return fmt.Errorf("template: no template %q in set", name)
	}
	return tmpl.Execute(wr, data)
}

// recover is the handler that turns panics into returns from the top
// level of Parse.
func (s *Set) recover(errp *os.Error) {
	e := recover()
	if e != nil {
		if _, ok := e.(runtime.Error); ok {
			panic(e)
		}
		s.tmpl = nil
		*errp = e.(os.Error)
	}
	return
}

// Parse parses a string into a set of named templates.  Parse may be called
// multiple times for a given set, adding the templates defined in the string
// to the set.  If a template is redefined, the element in the set is
// overwritten with the new definition.
func (s *Set) Parse(text string) (err os.Error) {
	s.init()
	defer s.recover(&err)
	lex := lex("set", text)
	const context = "define clause"
	for {
		t := New("set") // name will be updated once we know it.
		t.startParse(s, lex)
		// Expect EOF or "{{ define name }}".
		if t.atEOF() {
			return
		}
		t.expect(itemLeftDelim, context)
		t.expect(itemDefine, context)
		name := t.expect(itemString, context)
		t.name, err = strconv.Unquote(name.val)
		if err != nil {
			t.error(err)
		}
		t.expect(itemRightDelim, context)
		end := t.parse(false)
		if end == nil {
			t.errorf("unexpected EOF in %s", context)
		}
		if end.typ() != nodeEnd {
			t.errorf("unexpected %s in %s", end, context)
		}
		t.stopParse()
		t.addToSet(s)
		s.tmpl[t.name] = t
	}
	return nil
}
