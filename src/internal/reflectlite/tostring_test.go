// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Formatting of reflection types and values for debugging.
// Not defined as methods so they do not need to be linked into most binaries;
// the functions are not used by the library itself, only in tests.

package reflectlite_test

import (
	. "internal/reflectlite"
	"reflect"
	"strconv"
)

// valueToString returns a textual representation of the reflection value val.
// For debugging only.
func valueToString(v Value) string {
	return valueToStringImpl(reflect.ValueOf(ToInterface(v)))
}

func valueToStringImpl(val reflect.Value) string {
	var str string
	if !val.IsValid() {
		return "<zero Value>"
	}
	typ := val.Type()
	switch val.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(val.Int(), 10)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.FormatUint(val.Uint(), 10)
	case reflect.Float32, reflect.Float64:
		return strconv.FormatFloat(val.Float(), 'g', -1, 64)
	case reflect.Complex64, reflect.Complex128:
		c := val.Complex()
		return strconv.FormatFloat(real(c), 'g', -1, 64) + "+" + strconv.FormatFloat(imag(c), 'g', -1, 64) + "i"
	case reflect.String:
		return val.String()
	case reflect.Bool:
		if val.Bool() {
			return "true"
		} else {
			return "false"
		}
	case reflect.Pointer:
		v := val
		str = typ.String() + "("
		if v.IsNil() {
			str += "0"
		} else {
			str += "&" + valueToStringImpl(v.Elem())
		}
		str += ")"
		return str
	case reflect.Array, reflect.Slice:
		v := val
		str += typ.String()
		str += "{"
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				str += ", "
			}
			str += valueToStringImpl(v.Index(i))
		}
		str += "}"
		return str
	case reflect.Map:
		str += typ.String()
		str += "{"
		str += "<can't iterate on maps>"
		str += "}"
		return str
	case reflect.Chan:
		str = typ.String()
		return str
	case reflect.Struct:
		t := typ
		v := val
		str += t.String()
		str += "{"
		for i, n := 0, v.NumField(); i < n; i++ {
			if i > 0 {
				str += ", "
			}
			str += valueToStringImpl(v.Field(i))
		}
		str += "}"
		return str
	case reflect.Interface:
		return typ.String() + "(" + valueToStringImpl(val.Elem()) + ")"
	case reflect.Func:
		return typ.String() + "(arg)"
	default:
		panic("valueToString: can't print type " + typ.String())
	}
}
