// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"reflect"
)

// Implemented as a function rather than a method to keep the
// Type interface small.  TODO: should this return a string?
export func Print(typ Type) {
	switch(typ.Kind()) {
	case Int8Kind:
		print("int8");
	case Int16Kind:
		print("int16");
	case Int32Kind:
		print("int32");
	case Int64Kind:
		print("int64");
	case Uint8Kind:
		print("uint8");
	case Uint16Kind:
		print("uint16");
	case Uint32Kind:
		print("uint32");
	case Uint64Kind:
		print("uint64");
	case Float32Kind:
		print("float32");
	case Float64Kind:
		print("float64");
	case Float80Kind:
		print("float80");
	case StringKind:
		print("string");
	case PtrKind:
		p := typ.(PtrType);
		print("*");
		Print(p.Sub());
	case ArrayKind:
		a := typ.(ArrayType);
		if a.Len() >= 0 {
			print("[", a.Len(), "]")
		} else {
			print("[]")
		}
		Print(a.Elem());
	case MapKind:
		m := typ.(MapType);
		print("map[");
		Print(m.Key());
		print("]");
		Print(m.Elem());
	case ChanKind:
		c := typ.(ChanType);
		switch c.Dir() {
		case RecvDir:
			print("<-chan");
		case SendDir:
			print("chan<-");
		case BothDir:
			print("chan");
		default:
			panicln("reflect.Print: unknown chan direction");
		}
		Print(c.Elem());
	case StructKind:
		s := typ.(StructType);
		print("struct{");
		for i := 0; i < s.Len(); i++ {
			n, t := s.Field(i);
			print(n, " ");
			Print(t);
			if i < s.Len() - 1 {
				print("; ");
			}
		}
		print("}");
	case FuncKind:
		f := typ.(FuncType);
		print("func ");
		if f.Receiver() != nil {
			print("(");
			Print(f.Receiver());
			print(")");
		}
		print("(");
		Print(f.In());
		print(")");
		if f.Out() != nil {
			print("(");
			Print(f.Out());
			print(")");
		}
	default:
		panicln("can't print type ", typ.Kind());
	}
}
