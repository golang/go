// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reflection library.
// Formatting of reflection types and values for debugging.
// Not defined as methods so they do not need to be linked into most binaries;
// the functions are not used by the library itself, only in tests.

package reflect

import (
	"reflect";
	"strconv";
)

func typeToString(typ Type, expand bool) string
func valueToString(val Value) string

func doubleQuote(s string) string {
	out := "\"";
	for i := 0; i < len(s); i++ {
		c := s[i];
		switch c {
		case '\n':
			out += `\n`;
		case '\t':
			out += `\t`;
		case '\x00':
			out += `\x00`;
		case '"':
			out += `\"`;
		case '\\':
			out += `\\`;
		default:
			out += string(c);
		}
	}
	out += "\"";
	return out;
}

type hasFields interface {
	Field(i int)	(name string, typ Type, tag string, offset int);
	Len()	int;
}

func typeFieldsToString(t hasFields, sep string) string {
	var str string;
	for i := 0; i < t.Len(); i++ {
		str1, typ, tag, offset := t.Field(i);
		if str1 != "" {
			str1 += " "
		}
		str1 += typeToString(typ, false);
		if tag != "" {
			str1 += " " + doubleQuote(tag);
		}
		if i < t.Len() - 1 {
			str1 += sep + " ";
		}
		str += str1;
	}
	return str;
}

// typeToString returns a textual representation of typ.  The expand
// flag specifies whether to expand the contents of type names; if false,
// the name itself is used as the representation.
// Meant for debugging only; typ.String() serves for most purposes.
func typeToString(typ Type, expand bool) string {
	var str string;
	if name := typ.Name(); !expand && name != "" {
		return name
	}
	switch(typ.Kind()) {
	case MissingKind:
		return "$missing$";
	case IntKind, Int8Kind, Int16Kind, Int32Kind, Int64Kind,
	     UintKind, Uint8Kind, Uint16Kind, Uint32Kind, Uint64Kind,
	     FloatKind, Float32Kind, Float64Kind, Float80Kind,
	     StringKind,
	     DotDotDotKind:
		return typ.Name();
	case PtrKind:
		p := typ.(PtrType);
		return "*" + typeToString(p.Sub(), false);
	case ArrayKind:
		a := typ.(ArrayType);
		if a.IsSlice() {
			str = "[]"
		} else {
			str = "[" + strconv.Itoa64(int64(a.Len())) +  "]"
		}
		return str + typeToString(a.Elem(), false);
	case MapKind:
		m := typ.(MapType);
		str = "map[" + typeToString(m.Key(), false) + "]";
		return str + typeToString(m.Elem(), false);
	case ChanKind:
		c := typ.(ChanType);
		switch c.Dir() {
		case RecvDir:
			str = "<-chan";
		case SendDir:
			str = "chan<-";
		case BothDir:
			str = "chan";
		default:
			panicln("reflect.typeToString: unknown chan direction");
		}
		return str + typeToString(c.Elem(), false);
	case StructKind:
		return "struct{" + typeFieldsToString(typ.(StructType), ";") + "}";
	case InterfaceKind:
		return "interface{" + typeFieldsToString(typ.(InterfaceType), ";") + "}";
	case FuncKind:
		f := typ.(FuncType);
		str = "(" + typeFieldsToString(f.In(), ",") + ")";
		if f.Out() != nil {
			str += "(" + typeFieldsToString(f.Out(), ",") + ")";
		}
		return str;
	default:
		panicln("reflect.typeToString: can't print type ", typ.Kind());
	}
	return "reflect.typeToString: can't happen";
}

// TODO: want an unsigned one too
func integer(v int64) string {
	return strconv.Itoa64(v);
}

// valueToString returns a textual representation of the reflection value val.
// For debugging only.
func valueToString(val Value) string {
	var str string;
	typ := val.Type();
	switch(val.Kind()) {
	case MissingKind:
		return "missing";
	case IntKind:
		return integer(int64(val.(IntValue).Get()));
	case Int8Kind:
		return integer(int64(val.(Int8Value).Get()));
	case Int16Kind:
		return integer(int64(val.(Int16Value).Get()));
	case Int32Kind:
		return integer(int64(val.(Int32Value).Get()));
	case Int64Kind:
		return integer(int64(val.(Int64Value).Get()));
	case UintKind:
		return integer(int64(val.(UintValue).Get()));
	case Uint8Kind:
		return integer(int64(val.(Uint8Value).Get()));
	case Uint16Kind:
		return integer(int64(val.(Uint16Value).Get()));
	case Uint32Kind:
		return integer(int64(val.(Uint32Value).Get()));
	case Uint64Kind:
		return integer(int64(val.(Uint64Value).Get()));
	case FloatKind:
		if strconv.FloatSize == 32 {
			return strconv.Ftoa32(float32(val.(FloatValue).Get()), 'g', -1);
		} else {
			return strconv.Ftoa64(float64(val.(FloatValue).Get()), 'g', -1);
		}
	case Float32Kind:
		return strconv.Ftoa32(val.(Float32Value).Get(), 'g', -1);
	case Float64Kind:
		return strconv.Ftoa64(val.(Float64Value).Get(), 'g', -1);
	case Float80Kind:
		return "float80";
	case StringKind:
		return val.(StringValue).Get();
	case BoolKind:
		if val.(BoolValue).Get() {
			return "true"
		} else {
			return "false"
		}
	case PtrKind:
		v := val.(PtrValue);
		return typeToString(typ, false) + "(" + integer(int64(uintptr(v.Get()))) + ")";
	case ArrayKind:
		t := typ.(ArrayType);
		v := val.(ArrayValue);
		str += typeToString(t, false);
		str += "{";
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				str += ", "
			}
			str += valueToString(v.Elem(i));
		}
		str += "}";
		return str;
	case MapKind:
		t := typ.(MapType);
		v := val.(MapValue);
		str = typeToString(t, false);
		str += "{";
		str += "<can't iterate on maps>";
		str += "}";
		return str;
	case ChanKind:
		str = typeToString(typ, false);
		return str;
	case StructKind:
		t := typ.(StructType);
		v := val.(StructValue);
		str += typeToString(t, false);
		str += "{";
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				str += ", "
			}
			str += valueToString(v.Field(i));
		}
		str += "}";
		return str;
	case InterfaceKind:
		return "can't print interfaces yet";
	case FuncKind:
		return "can't print funcs yet";
	default:
		panicln("reflect.valueToString: can't print type ", val.Kind());
	}
	return "reflect.valueToString: can't happen";
}
