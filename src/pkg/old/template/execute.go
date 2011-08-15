// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code to execute a parsed template.

package template

import (
	"bytes"
	"io"
	"reflect"
	"strings"
)

// Internal state for executing a Template.  As we evaluate the struct,
// the data item descends into the fields associated with sections, etc.
// Parent is used to walk upwards to find variables higher in the tree.
type state struct {
	parent *state          // parent in hierarchy
	data   reflect.Value   // the driver data for this section etc.
	wr     io.Writer       // where to send output
	buf    [2]bytes.Buffer // alternating buffers used when chaining formatters
}

func (parent *state) clone(data reflect.Value) *state {
	return &state{parent: parent, data: data, wr: parent.wr}
}

// Evaluate interfaces and pointers looking for a value that can look up the name, via a
// struct field, method, or map key, and return the result of the lookup.
func (t *Template) lookup(st *state, v reflect.Value, name string) reflect.Value {
	for v.IsValid() {
		typ := v.Type()
		if n := v.Type().NumMethod(); n > 0 {
			for i := 0; i < n; i++ {
				m := typ.Method(i)
				mtyp := m.Type
				if m.Name == name && mtyp.NumIn() == 1 && mtyp.NumOut() == 1 {
					if !isExported(name) {
						t.execError(st, t.linenum, "name not exported: %s in type %s", name, st.data.Type())
					}
					return v.Method(i).Call(nil)[0]
				}
			}
		}
		switch av := v; av.Kind() {
		case reflect.Ptr:
			v = av.Elem()
		case reflect.Interface:
			v = av.Elem()
		case reflect.Struct:
			if !isExported(name) {
				t.execError(st, t.linenum, "name not exported: %s in type %s", name, st.data.Type())
			}
			return av.FieldByName(name)
		case reflect.Map:
			if v := av.MapIndex(reflect.ValueOf(name)); v.IsValid() {
				return v
			}
			return reflect.Zero(typ.Elem())
		default:
			return reflect.Value{}
		}
	}
	return v
}

// indirectPtr returns the item numLevels levels of indirection below the value.
// It is forgiving: if the value is not a pointer, it returns it rather than giving
// an error.  If the pointer is nil, it is returned as is.
func indirectPtr(v reflect.Value, numLevels int) reflect.Value {
	for i := numLevels; v.IsValid() && i > 0; i++ {
		if p := v; p.Kind() == reflect.Ptr {
			if p.IsNil() {
				return v
			}
			v = p.Elem()
		} else {
			break
		}
	}
	return v
}

// Walk v through pointers and interfaces, extracting the elements within.
func indirect(v reflect.Value) reflect.Value {
loop:
	for v.IsValid() {
		switch av := v; av.Kind() {
		case reflect.Ptr:
			v = av.Elem()
		case reflect.Interface:
			v = av.Elem()
		default:
			break loop
		}
	}
	return v
}

// If the data for this template is a struct, find the named variable.
// Names of the form a.b.c are walked down the data tree.
// The special name "@" (the "cursor") denotes the current data.
// The value coming in (st.data) might need indirecting to reach
// a struct while the return value is not indirected - that is,
// it represents the actual named field. Leading stars indicate
// levels of indirection to be applied to the value.
func (t *Template) findVar(st *state, s string) reflect.Value {
	data := st.data
	flattenedName := strings.TrimLeft(s, "*")
	numStars := len(s) - len(flattenedName)
	s = flattenedName
	if s == "@" {
		return indirectPtr(data, numStars)
	}
	for _, elem := range strings.Split(s, ".") {
		// Look up field; data must be a struct or map.
		data = t.lookup(st, data, elem)
		if !data.IsValid() {
			return reflect.Value{}
		}
	}
	return indirectPtr(data, numStars)
}

// Is there no data to look at?
func empty(v reflect.Value) bool {
	v = indirect(v)
	if !v.IsValid() {
		return true
	}
	switch v.Kind() {
	case reflect.Bool:
		return v.Bool() == false
	case reflect.String:
		return v.String() == ""
	case reflect.Struct:
		return false
	case reflect.Map:
		return false
	case reflect.Array:
		return v.Len() == 0
	case reflect.Slice:
		return v.Len() == 0
	}
	return false
}

// Look up a variable or method, up through the parent if necessary.
func (t *Template) varValue(name string, st *state) reflect.Value {
	field := t.findVar(st, name)
	if !field.IsValid() {
		if st.parent == nil {
			t.execError(st, t.linenum, "name not found: %s in type %s", name, st.data.Type())
		}
		return t.varValue(name, st.parent)
	}
	return field
}

func (t *Template) format(wr io.Writer, fmt string, val []interface{}, v *variableElement, st *state) {
	fn := t.formatter(fmt)
	if fn == nil {
		t.execError(st, v.linenum, "missing formatter %s for variable", fmt)
	}
	fn(wr, fmt, val...)
}

// Evaluate a variable, looking up through the parent if necessary.
// If it has a formatter attached ({var|formatter}) run that too.
func (t *Template) writeVariable(v *variableElement, st *state) {
	// Resolve field names
	val := make([]interface{}, len(v.args))
	for i, arg := range v.args {
		if name, ok := arg.(fieldName); ok {
			val[i] = t.varValue(string(name), st).Interface()
		} else {
			val[i] = arg
		}
	}
	for i, fmt := range v.fmts[:len(v.fmts)-1] {
		b := &st.buf[i&1]
		b.Reset()
		t.format(b, fmt, val, v, st)
		val = val[0:1]
		val[0] = b.Bytes()
	}
	t.format(st.wr, v.fmts[len(v.fmts)-1], val, v, st)
}

// Execute element i.  Return next index to execute.
func (t *Template) executeElement(i int, st *state) int {
	switch elem := t.elems[i].(type) {
	case *textElement:
		st.wr.Write(elem.text)
		return i + 1
	case *literalElement:
		st.wr.Write(elem.text)
		return i + 1
	case *variableElement:
		t.writeVariable(elem, st)
		return i + 1
	case *sectionElement:
		t.executeSection(elem, st)
		return elem.end
	case *repeatedElement:
		t.executeRepeated(elem, st)
		return elem.end
	}
	e := t.elems[i]
	t.execError(st, 0, "internal error: bad directive in execute: %v %T\n", reflect.ValueOf(e).Interface(), e)
	return 0
}

// Execute the template.
func (t *Template) execute(start, end int, st *state) {
	for i := start; i < end; {
		i = t.executeElement(i, st)
	}
}

// Execute a .section
func (t *Template) executeSection(s *sectionElement, st *state) {
	// Find driver data for this section.  It must be in the current struct.
	field := t.varValue(s.field, st)
	if !field.IsValid() {
		t.execError(st, s.linenum, ".section: cannot find field %s in %s", s.field, st.data.Type())
	}
	st = st.clone(field)
	start, end := s.start, s.or
	if !empty(field) {
		// Execute the normal block.
		if end < 0 {
			end = s.end
		}
	} else {
		// Execute the .or block.  If it's missing, do nothing.
		start, end = s.or, s.end
		if start < 0 {
			return
		}
	}
	for i := start; i < end; {
		i = t.executeElement(i, st)
	}
}

// Return the result of calling the Iter method on v, or nil.
func iter(v reflect.Value) reflect.Value {
	for j := 0; j < v.Type().NumMethod(); j++ {
		mth := v.Type().Method(j)
		fv := v.Method(j)
		ft := fv.Type()
		// TODO(rsc): NumIn() should return 0 here, because ft is from a curried FuncValue.
		if mth.Name != "Iter" || ft.NumIn() != 1 || ft.NumOut() != 1 {
			continue
		}
		ct := ft.Out(0)
		if ct.Kind() != reflect.Chan ||
			ct.ChanDir()&reflect.RecvDir == 0 {
			continue
		}
		return fv.Call(nil)[0]
	}
	return reflect.Value{}
}

// Execute a .repeated section
func (t *Template) executeRepeated(r *repeatedElement, st *state) {
	// Find driver data for this section.  It must be in the current struct.
	field := t.varValue(r.field, st)
	if !field.IsValid() {
		t.execError(st, r.linenum, ".repeated: cannot find field %s in %s", r.field, st.data.Type())
	}
	field = indirect(field)

	start, end := r.start, r.or
	if end < 0 {
		end = r.end
	}
	if r.altstart >= 0 {
		end = r.altstart
	}
	first := true

	// Code common to all the loops.
	loopBody := func(newst *state) {
		// .alternates between elements
		if !first && r.altstart >= 0 {
			for i := r.altstart; i < r.altend; {
				i = t.executeElement(i, newst)
			}
		}
		first = false
		for i := start; i < end; {
			i = t.executeElement(i, newst)
		}
	}

	if array := field; array.Kind() == reflect.Array || array.Kind() == reflect.Slice {
		for j := 0; j < array.Len(); j++ {
			loopBody(st.clone(array.Index(j)))
		}
	} else if m := field; m.Kind() == reflect.Map {
		for _, key := range m.MapKeys() {
			loopBody(st.clone(m.MapIndex(key)))
		}
	} else if ch := iter(field); ch.IsValid() {
		for {
			e, ok := ch.Recv()
			if !ok {
				break
			}
			loopBody(st.clone(e))
		}
	} else {
		t.execError(st, r.linenum, ".repeated: cannot repeat %s (type %s)",
			r.field, field.Type())
	}

	if first {
		// Empty. Execute the .or block, once.  If it's missing, do nothing.
		start, end := r.or, r.end
		if start >= 0 {
			newst := st.clone(field)
			for i := start; i < end; {
				i = t.executeElement(i, newst)
			}
		}
		return
	}
}

// A valid delimiter must contain no space and be non-empty.
func validDelim(d []byte) bool {
	if len(d) == 0 {
		return false
	}
	for _, c := range d {
		if isSpace(c) {
			return false
		}
	}
	return true
}
