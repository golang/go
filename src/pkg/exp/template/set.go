// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"os"
	"reflect"
	"runtime"
	"strconv"
)

// Set holds a set of related templates that can refer to one another by name.
// A template may be a member of multiple sets.
type Set struct {
	tmpl  map[string]*Template
	funcs map[string]reflect.Value
}

// NewSet allocates a new, empty template set.
func NewSet() *Set {
	return &Set{
		tmpl:  make(map[string]*Template),
		funcs: make(map[string]reflect.Value),
	}
}

// Funcs adds to the set's function map the elements of the
// argument map.   It panics if a value in the map is not a function
// with appropriate return type.
// The return value is the set, so calls can be chained.
func (s *Set) Funcs(funcMap FuncMap) *Set {
	addFuncs(s.funcs, funcMap)
	return s
}

// Add adds the argument templates to the set. It panics if the call
// attempts to reuse a name defined in the template.
// The return value is the set, so calls can be chained.
func (s *Set) Add(templates ...*Template) *Set {
	for _, t := range templates {
		if _, ok := s.tmpl[t.name]; ok {
			panic(fmt.Errorf("template: %q already defined in set", t.name))
		}
		s.tmpl[t.name] = t
	}
	return s
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

// Parse parses the file into a set of named templates.
func (s *Set) Parse(text string) (err os.Error) {
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
		t.expect(itemLeftMeta, context)
		t.expect(itemDefine, context)
		name := t.expect(itemString, context)
		t.name, err = strconv.Unquote(name.val)
		if err != nil {
			t.error(err)
		}
		t.expect(itemRightMeta, context)
		end := t.parse(false)
		if end == nil {
			t.errorf("unexpected EOF in %s", context)
		}
		if end.typ() != nodeEnd {
			t.errorf("unexpected %s in %s", end, context)
		}
		t.stopParse()
		s.tmpl[t.name] = t
	}
	return nil
}
