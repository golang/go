// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reflection library.
// Formatting of types and values for debugging.

package reflect

import (
	"reflect";
	"strings";
)

export func TypeToString(typ Type, expand bool) string
export func ValueToString(val Value) string

func DoubleQuote(s string) string {
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

type HasFields interface {
	Field(i int)	(name string, typ Type, tag string, offset uint64);
	Len()	int;
}

func TypeFieldsToString(t HasFields, sep string) string {
	var str string;
	for i := 0; i < t.Len(); i++ {
		str1, typ, tag, offset := t.Field(i);
		str1 +=  " " + TypeToString(typ, false);
		if tag != "" {
			str1 += " " + DoubleQuote(tag);
		}
		if i < t.Len() - 1 {
			str1 += sep + " ";
		}
		str += str1;
	}
	return str;
}

func TypeToString(typ Type, expand bool) string {
	var str string;
	if name := typ.Name(); !expand && name != "" {
		return name
	}
	switch(typ.Kind()) {
	case MissingKind:
		return "$missing$";
	case IntKind, Int8Kind, Int16Kind, Int32Kind, Int64Kind,
	     UintKind, Uint8Kind, Uint16Kind, Uint32Kind, Uint64Kind,
	     FloatKind, Float32Kind, Float64Kind, Float80Kind:
	     StringKind:
		return typ.Name();
	case PtrKind:
		p := typ.(PtrType);
		return "*" + TypeToString(p.Sub(), false);
	case ArrayKind:
		a := typ.(ArrayType);
		if a.Open() {
			str = "[]"
		} else {
			str = "[" + strings.ltoa(int64(a.Len())) +  "]"
		}
		return str + TypeToString(a.Elem(), false);
	case MapKind:
		m := typ.(MapType);
		str = "map[" + TypeToString(m.Key(), false) + "]";
		return str + TypeToString(m.Elem(), false);
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
			panicln("reflect.TypeToString: unknown chan direction");
		}
		return str + TypeToString(c.Elem(), false);
	case StructKind:
		return "struct{" + TypeFieldsToString(typ, ";") + "}";
	case InterfaceKind:
		return "interface{" + TypeFieldsToString(typ, ";") + "}";
	case FuncKind:
		f := typ.(FuncType);
		str = "(" + TypeFieldsToString(f.In(), ",") + ")";
		if f.Out() != nil {
			str += "(" + TypeFieldsToString(f.Out(), ",") + ")";
		}
		return str;
	default:
		panicln("reflect.TypeToString: can't print type ", typ.Kind());
	}
	return "reflect.TypeToString: can't happen";
}

// TODO: want an unsigned one too
func integer(v int64) string {
	return strings.ltoa(v);
}

func floatingpoint(v float64) string {
	return strings.dtoa(v);
}

func ValueToString(val Value) string {
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
		return floatingpoint(float64(val.(FloatValue).Get()));
	case Float32Kind:
		return floatingpoint(float64(val.(Float32Value).Get()));
	case Float64Kind:
		return floatingpoint(float64(val.(Float64Value).Get()));
	case Float80Kind:
		return "float80";
	case StringKind:
		return val.(StringValue).Get();
	case PtrKind:
		v := val.(PtrValue);
		return TypeToString(typ, false) + "(" + integer(int64(v.Get())) + ")";
	case ArrayKind:
		t := typ.(ArrayType);
		v := val.(ArrayValue);
		str += TypeToString(t, false);
		str += "{";
		for i := uint64(0); i < v.Len(); i++ {
			if i > 0 {
				str += ", "
			}
			str += ValueToString(v.Elem(i));
		}
		str += "}";
		return str;
	case MapKind:
		t := typ.(MapType);
		v := val.(ArrayValue);
		str = TypeToString(t, false);
		str += "{";
		str += "<can't iterate on maps>";
		str += "}";
		return str;
	case ChanKind:
		return "can't print chans yet";
	case StructKind:
		t := typ.(StructType);
		v := val.(StructValue);
		str += TypeToString(t, false);
		str += "{";
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				str += ", "
			}
			str += ValueToString(v.Field(i));
		}
		str += "}";
		return str;
	case InterfaceKind:
		return "can't print interfaces yet";
	case FuncKind:
		return "can't print funcs yet";
	default:
		panicln("reflect.ValueToString: can't print type ", val.Kind());
	}
	return "reflect.ValueToString: can't happen";
}
