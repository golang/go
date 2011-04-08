// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Formatting of reflection types and values for debugging.
// Not defined as methods so they do not need to be linked into most binaries;
// the functions are not used by the library itself, only in tests.

package reflect_test

import (
	. "reflect"
	"strconv"
)

// valueToString returns a textual representation of the reflection value val.
// For debugging only.
func valueToString(val Value) string {
	var str string
	if !val.IsValid() {
		return "<zero Value>"
	}
	typ := val.Type()
	switch val.Kind() {
	case Int, Int8, Int16, Int32, Int64:
		return strconv.Itoa64(val.Int())
	case Uint, Uint8, Uint16, Uint32, Uint64, Uintptr:
		return strconv.Uitoa64(val.Uint())
	case Float32, Float64:
		return strconv.Ftoa64(val.Float(), 'g', -1)
	case Complex64, Complex128:
		c := val.Complex()
		return strconv.Ftoa64(real(c), 'g', -1) + "+" + strconv.Ftoa64(imag(c), 'g', -1) + "i"
	case String:
		return val.String()
	case Bool:
		if val.Bool() {
			return "true"
		} else {
			return "false"
		}
	case Ptr:
		v := val
		str = typ.String() + "("
		if v.IsNil() {
			str += "0"
		} else {
			str += "&" + valueToString(v.Elem())
		}
		str += ")"
		return str
	case Array, Slice:
		v := val
		str += typ.String()
		str += "{"
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				str += ", "
			}
			str += valueToString(v.Index(i))
		}
		str += "}"
		return str
	case Map:
		t := typ
		str = t.String()
		str += "{"
		str += "<can't iterate on maps>"
		str += "}"
		return str
	case Chan:
		str = typ.String()
		return str
	case Struct:
		t := typ
		v := val
		str += t.String()
		str += "{"
		for i, n := 0, v.NumField(); i < n; i++ {
			if i > 0 {
				str += ", "
			}
			str += valueToString(v.Field(i))
		}
		str += "}"
		return str
	case Interface:
		return typ.String() + "(" + valueToString(val.Elem()) + ")"
	case Func:
		v := val
		return typ.String() + "(" + strconv.Uitoa64(uint64(v.Pointer())) + ")"
	default:
		panic("valueToString: can't print type " + typ.String())
	}
	return "valueToString: can't happen"
}
