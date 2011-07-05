// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"fmt"
	"reflect"
)

// FuncMap is the type of the map defining the mapping from names to functions.
// Each function must have either a single return value, or two return values of
// which the second has type os.Error.
type FuncMap map[string]interface{}

var funcs = map[string]reflect.Value{
	"printf": reflect.ValueOf(fmt.Sprintf),
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
