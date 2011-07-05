// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"io"
	"fmt"
	"reflect"
	"strings"
)

// FuncMap is the type of the map defining the mapping from names to functions.
// Each function must have either a single return value, or two return values of
// which the second has type os.Error.
type FuncMap map[string]interface{}

var funcs = map[string]reflect.Value{
	"printf": reflect.ValueOf(fmt.Sprintf),
	"html":   reflect.ValueOf(HTMLEscaper),
	"and":    reflect.ValueOf(and),
	"or":     reflect.ValueOf(or),
	"not":    reflect.ValueOf(not),
}

// addFuncs adds to values the functions in funcs, converting them to reflect.Values.
func addFuncs(values map[string]reflect.Value, funcMap FuncMap) {
	for name, fn := range funcMap {
		v := reflect.ValueOf(fn)
		if v.Kind() != reflect.Func {
			panic("value for " + name + " not a function")
		}
		if !goodFunc(v.Type()) {
			panic(fmt.Errorf("can't handle multiple results from method/function %q", name))
		}
		values[name] = v
	}
}

// goodFunc checks that the function or method has the right result signature.
func goodFunc(typ reflect.Type) bool {
	// We allow functions with 1 result or 2 results where the second is an os.Error.
	switch {
	case typ.NumOut() == 1:
		return true
	case typ.NumOut() == 2 && typ.Out(1) == osErrorType:
		return true
	}
	return false
}

// findFunction looks for a function in the template, set, and global map.
func findFunction(name string, tmpl *Template, set *Set) (reflect.Value, bool) {
	if tmpl != nil {
		if fn := tmpl.funcs[name]; fn.IsValid() {
			return fn, true
		}
	}
	if set != nil {
		if fn := set.funcs[name]; fn.IsValid() {
			return fn, true
		}
	}
	if fn := funcs[name]; fn.IsValid() {
		return fn, true
	}
	return reflect.Value{}, false
}

// Boolean logic.

// and returns the Boolean AND of its arguments.
func and(arg0 interface{}, args ...interface{}) (truth bool) {
	truth, _ = isTrue(reflect.ValueOf(arg0))
	for i := 0; truth && i < len(args); i++ {
		truth, _ = isTrue(reflect.ValueOf(args[i]))
	}
	return
}

// or returns the Boolean OR of its arguments.
func or(arg0 interface{}, args ...interface{}) (truth bool) {
	truth, _ = isTrue(reflect.ValueOf(arg0))
	for i := 0; !truth && i < len(args); i++ {
		truth, _ = isTrue(reflect.ValueOf(args[i]))
	}
	return
}

// not returns the Boolean negation of its argument.
func not(arg interface{}) (truth bool) {
	truth, _ = isTrue(reflect.ValueOf(arg))
	return !truth
}

// HTML escaping.

var (
	escQuot = []byte("&#34;") // shorter than "&quot;"
	escApos = []byte("&#39;") // shorter than "&apos;"
	escAmp  = []byte("&amp;")
	escLt   = []byte("&lt;")
	escGt   = []byte("&gt;")
)

// HTMLEscape writes to w the escaped HTML equivalent of the plain text data b.
func HTMLEscape(w io.Writer, b []byte) {
	last := 0
	for i, c := range b {
		var esc []byte
		switch c {
		case '"':
			esc = escQuot
		case '\'':
			esc = escApos
		case '&':
			esc = escAmp
		case '<':
			esc = escLt
		case '>':
			esc = escGt
		default:
			continue
		}
		w.Write(b[last:i])
		w.Write(esc)
		last = i + 1
	}
	w.Write(b[last:])
}

// HTMLEscapeString returns the escaped HTML equivalent of the plain text data s.
func HTMLEscapeString(s string) string {
	// Avoid allocation if we can.
	if strings.IndexAny(s, `'"&<>`) < 0 {
		return s
	}
	var b bytes.Buffer
	HTMLEscape(&b, []byte(s))
	return b.String()
}

// HTMLEscaper returns the escaped HTML equivalent of the textual
// representation of its arguments.
func HTMLEscaper(args ...interface{}) string {
	ok := false
	var s string
	if len(args) == 1 {
		s, ok = args[0].(string)
	}
	if !ok {
		s = fmt.Sprint(args...)
	}
	return HTMLEscapeString(s)
}
