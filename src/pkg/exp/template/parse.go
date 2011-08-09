// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"exp/template/parse"
	"os"
	"reflect"
)

// Template is the representation of a parsed template.
type Template struct {
	name string
	*parse.Tree
	funcs map[string]reflect.Value
	set   *Set // can be nil.
}

// Name returns the name of the template.
func (t *Template) Name() string {
	return t.Tree.Name
}

// Parsing.

// New allocates a new template with the given name.
func New(name string) *Template {
	return &Template{
		name:  name,
		funcs: make(map[string]reflect.Value),
	}
}

// Funcs adds the elements of the argument map to the template's function
// map.  It panics if a value in the map is not a function with appropriate
// return type.
// The return value is the template, so calls can be chained.
func (t *Template) Funcs(funcMap FuncMap) *Template {
	addFuncs(t.funcs, funcMap)
	return t
}

// Parse parses the template definition string to construct an internal
// representation of the template for execution.
func (t *Template) Parse(s string) (tmpl *Template, err os.Error) {
	t.Tree, err = parse.New(t.name).Parse(s, t.funcs, builtins)
	if err != nil {
		return nil, err
	}
	return t, nil
}

// ParseInSet parses the template definition string to construct an internal
// representation of the template for execution. It also adds the template
// to the set.
// Function bindings are checked against those in the set.
func (t *Template) ParseInSet(s string, set *Set) (tmpl *Template, err os.Error) {
	var setFuncs map[string]reflect.Value
	if set != nil {
		setFuncs = set.funcs
	}
	t.Tree, err = parse.New(t.name).Parse(s, t.funcs, setFuncs, builtins)
	if err != nil {
		return nil, err
	}
	t.addToSet(set)
	return t, nil
}

// addToSet adds the template to the set, verifying it's not being double-assigned.
func (t *Template) addToSet(set *Set) {
	if set == nil || t.set == set {
		return
	}
	// If double-assigned, Add will panic and we will turn that into an error.
	set.Add(t)
}
