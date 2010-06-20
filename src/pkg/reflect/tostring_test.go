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
	if val == nil {
		return "<nil>"
	}
	typ := val.Type()
	switch val := val.(type) {
	case *IntValue:
		return strconv.Itoa64(val.Get())
	case *UintValue:
		return strconv.Uitoa64(val.Get())
	case *FloatValue:
		return strconv.Ftoa64(float64(val.Get()), 'g', -1)
	case *ComplexValue:
		c := val.Get()
		return strconv.Ftoa64(float64(real(c)), 'g', -1) + "+" + strconv.Ftoa64(float64(imag(c)), 'g', -1) + "i"
	case *StringValue:
		return val.Get()
	case *BoolValue:
		if val.Get() {
			return "true"
		} else {
			return "false"
		}
	case *PtrValue:
		v := val
		str = typ.String() + "("
		if v.IsNil() {
			str += "0"
		} else {
			str += "&" + valueToString(v.Elem())
		}
		str += ")"
		return str
	case ArrayOrSliceValue:
		v := val
		str += typ.String()
		str += "{"
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				str += ", "
			}
			str += valueToString(v.Elem(i))
		}
		str += "}"
		return str
	case *MapValue:
		t := typ.(*MapType)
		str = t.String()
		str += "{"
		str += "<can't iterate on maps>"
		str += "}"
		return str
	case *ChanValue:
		str = typ.String()
		return str
	case *StructValue:
		t := typ.(*StructType)
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
	case *InterfaceValue:
		return typ.String() + "(" + valueToString(val.Elem()) + ")"
	case *FuncValue:
		v := val
		return typ.String() + "(" + strconv.Itoa64(int64(v.Get())) + ")"
	default:
		panic("valueToString: can't print type " + typ.String())
	}
	return "valueToString: can't happen"
}
