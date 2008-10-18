// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reflection library.
// Formatting of types for debugging.

package reflect

import (
	"reflect";
	"strings";
)

export func TypeToString(typ Type) string
export func ValueToString(val Value) string

func TypeFieldsToString(t Type, sep string) string {
	s := t.(StructType);
	var str string;
	for i := 0; i < s.Len(); i++ {
		str1, t := s.Field(i);
		str1 +=  " " + TypeToString(t);
		if i < s.Len() - 1 {
			str1 += sep + " ";
		}
		str += str1;
	}
	return str;
}

func TypeToString(typ Type) string {
	var str string;
	switch(typ.Kind()) {
	case MissingKind:
		return "missing";
	case Int8Kind:
		return "int8";
	case Int16Kind:
		return "int16";
	case Int32Kind:
		return "int32";
	case Int64Kind:
		return "int64";
	case Uint8Kind:
		return "uint8";
	case Uint16Kind:
		return "uint16";
	case Uint32Kind:
		return "uint32";
	case Uint64Kind:
		return "uint64";
	case Float32Kind:
		return "float32";
	case Float64Kind:
		return "float64";
	case Float80Kind:
		return "float80";
	case StringKind:
		return "string";
	case PtrKind:
		p := typ.(PtrType);
		return "*" + TypeToString(p.Sub());
	case ArrayKind:
		a := typ.(ArrayType);
		if a.Len() < 0 {
			str = "[]"
		} else {
			str = "[" + strings.itoa(a.Len()) +  "]"
		}
		return str + TypeToString(a.Elem());
	case MapKind:
		m := typ.(MapType);
		str = "map[" + TypeToString(m.Key()) + "]";
		return str + TypeToString(m.Elem());
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
		return str + TypeToString(c.Elem());
	case StructKind:
		return "struct{" + TypeFieldsToString(typ, ";") + "}";
	case InterfaceKind:
		return "interface{" + TypeFieldsToString(typ, ";") + "}";
	case FuncKind:
		f := typ.(FuncType);
		str = "func";
		str += "(" + TypeFieldsToString(f.In(), ",") + ")";
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
	return strings.itol(v);
}

func ValueToString(val Value) string {
	var str string;
	typ := val.Type();
	switch(val.Kind()) {
	case MissingKind:
		return "missing";
	case Int8Kind:
		return integer(int64(val.(Int8Value).Get()));
	case Int16Kind:
		return integer(int64(val.(Int16Value).Get()));
	case Int32Kind:
		return integer(int64(val.(Int32Value).Get()));
	case Int64Kind:
		return integer(int64(val.(Int64Value).Get()));
	case Uint8Kind:
		return integer(int64(val.(Uint8Value).Get()));
	case Uint16Kind:
		return integer(int64(val.(Uint16Value).Get()));
	case Uint32Kind:
		return integer(int64(val.(Uint32Value).Get()));
	case Uint64Kind:
		return integer(int64(val.(Uint64Value).Get()));
	case Float32Kind:
		return "float32";
	case Float64Kind:
		return "float64";
	case Float80Kind:
		return "float80";
	case StringKind:
		return val.(StringValue).Get();
	case PtrKind:
		p := typ.(PtrType);
		return ValueToString(p.Sub());
	default:
		panicln("reflect.ValueToString: can't print type ", val.Kind());
	}
	return "reflect.ValueToString: can't happen";
}
