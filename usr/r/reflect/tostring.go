// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"reflect";
	"strings";
)

export func ToString(typ Type) string

func FieldsToString(t Type) string {
	s := t.(StructType);
	var str string;
	for i := 0; i < s.Len(); i++ {
		str1, t := s.Field(i);
		str1 +=  " " + ToString(t);
		if i < s.Len() - 1 {
			str1 += "; ";
		}
		str += str1;
	}
	return str;
}

func ToString(typ Type) string {
	var str string;
	switch(typ.Kind()) {
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
		return "*" + ToString(p.Sub());
	case ArrayKind:
		a := typ.(ArrayType);
		if a.Len() < 0 {
			str = "[]"
		} else {
			str = "[" + strings.itoa(a.Len()) +  "]"
		}
		return str + ToString(a.Elem());
	case MapKind:
		m := typ.(MapType);
		str = "map[" + ToString(m.Key()) + "]";
		return str + ToString(m.Elem());
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
			panicln("reflect.ToString: unknown chan direction");
		}
		return str + ToString(c.Elem());
	case StructKind:
		return "struct{" + FieldsToString(typ) + "}";
	case FuncKind:
		f := typ.(FuncType);
		str = "func";
		if f.Receiver() != nil {
			str += "(" + FieldsToString(f.Receiver()) + ")";
		}
		str += "(" + FieldsToString(f.In()) + ")";
		if f.Out() != nil {
			str += "(" + FieldsToString(f.Out()) + ")";
		}
		return str;
	default:
		panicln("reflect.ToString: can't print type ", typ.Kind());
	}
	return "reflect.ToString: can't happen";
}
