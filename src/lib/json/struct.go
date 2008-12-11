// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Marshalling and unmarshalling of
// JSON data into Go structs using reflection.

package json

import (
	"json";
	"reflect";
)

type StructBuilder struct {
	val reflect.Value
}

var nobuilder *StructBuilder

func SetFloat(v reflect.Value, f float64) {
	switch v.Kind() {
	case reflect.FloatKind:
		v.(reflect.FloatValue).Set(float(f));
	case reflect.Float32Kind:
		v.(reflect.Float32Value).Set(float32(f));
	case reflect.Float64Kind:
		v.(reflect.Float64Value).Set(float64(f));
	}
}

func SetInt(v reflect.Value, i int64) {
	switch v.Kind() {
	case reflect.IntKind:
		v.(reflect.IntValue).Set(int(i));
	case reflect.Int8Kind:
		v.(reflect.Int8Value).Set(int8(i));
	case reflect.Int16Kind:
		v.(reflect.Int16Value).Set(int16(i));
	case reflect.Int32Kind:
		v.(reflect.Int32Value).Set(int32(i));
	case reflect.Int64Kind:
		v.(reflect.Int64Value).Set(int64(i));
	case reflect.UintKind:
		v.(reflect.UintValue).Set(uint(i));
	case reflect.Uint8Kind:
		v.(reflect.Uint8Value).Set(uint8(i));
	case reflect.Uint16Kind:
		v.(reflect.Uint16Value).Set(uint16(i));
	case reflect.Uint32Kind:
		v.(reflect.Uint32Value).Set(uint32(i));
	case reflect.Uint64Kind:
		v.(reflect.Uint64Value).Set(uint64(i));
	}
}

func (b *StructBuilder) Int64(i int64) {
	if b == nil {
		return
	}
	v := b.val;
	switch v.Kind() {
	case reflect.FloatKind, reflect.Float32Kind, reflect.Float64Kind:
		SetFloat(v, float64(i));
	default:
		SetInt(v, i);
	}
}

func (b *StructBuilder) Uint64(i uint64) {
	if b == nil {
		return
	}
	v := b.val;
	switch v.Kind() {
	case reflect.FloatKind, reflect.Float32Kind, reflect.Float64Kind:
		SetFloat(v, float64(i));
	default:
		SetInt(v, int64(i));
	}
}

func (b *StructBuilder) Float64(f float64) {
	if b == nil {
		return
	}
	v := b.val;
	switch v.Kind() {
	case reflect.FloatKind, reflect.Float32Kind, reflect.Float64Kind:
		SetFloat(v, f);
	default:
		SetInt(v, int64(f));
	}
}

func (b *StructBuilder) Null() {
}

func (b *StructBuilder) String(s string) {
	if b == nil {
		return
	}
	if v := b.val; v.Kind() == reflect.StringKind {
		v.(reflect.StringValue).Set(s);
	}
}

func (b *StructBuilder) Bool(tf bool) {
	if b == nil {
		return
	}
	if v := b.val; v.Kind() == reflect.BoolKind {
		v.(reflect.BoolValue).Set(tf);
	}
}

func (b *StructBuilder) Array() {
	if b == nil {
		return
	}
	if v := b.val; v.Kind() == reflect.PtrKind {
		pv := v.(reflect.PtrValue);
		psubtype := pv.Type().(reflect.PtrType).Sub();
		if pv.Get() == nil && psubtype.Kind() == reflect.ArrayKind {
			av := reflect.NewOpenArrayValue(psubtype, 0, 8);
			pv.SetSub(av);
		}
	}
}

func (b *StructBuilder) Elem(i int) Builder {
	if b == nil || i < 0 {
		return nobuilder
	}
	v := b.val;
	if v.Kind() == reflect.PtrKind {
		// If we have a pointer to an array, allocate or grow
		// the array as necessary.  Then set v to the array itself.
		pv := v.(reflect.PtrValue);
		psub := pv.Sub();
		if psub.Kind() == reflect.ArrayKind {
			av := psub.(reflect.ArrayValue);
			if i > av.Cap() {
				n := av.Cap();
				if n < 8 {
					n = 8
				}
				for n <= i {
					n *= 2
				}
				av1 := reflect.NewOpenArrayValue(av.Type(), av.Len(), n);
				reflect.CopyArray(av1, av, av.Len());
				pv.SetSub(av1);
				av = av1;
			}
		}
		v = psub;
	}
	if v.Kind() == reflect.ArrayKind {
		// Array was grown above, or is fixed size.
		av := v.(reflect.ArrayValue);
		if av.Len() <= i && i < av.Cap() {
			av.SetLen(i+1);
		}
		if i < av.Len() {
			return &StructBuilder{ av.Elem(i) }
		}
	}
	return nobuilder
}

func (b *StructBuilder) Map() {
	if b == nil {
		return
	}
	if v := b.val; v.Kind() == reflect.PtrKind {
		pv := v.(reflect.PtrValue);
		if pv.Get() == nil {
			pv.SetSub(reflect.NewInitValue(pv.Type().(reflect.PtrType).Sub()))
		}
	}
}

func (b *StructBuilder) Key(k string) Builder {
	if b == nil {
		return nobuilder
	}
	v := b.val;
	if v.Kind() == reflect.PtrKind {
		v = v.(reflect.PtrValue).Sub();
	}
	if v.Kind() == reflect.StructKind {
		sv := v.(reflect.StructValue);
		t := v.Type().(reflect.StructType);
		for i := 0; i < t.Len(); i++ {
			name, typ, tag, off := t.Field(i);
			if k == name {
				return &StructBuilder{ sv.Field(i) }
			}
		}
	}
	return nobuilder
}

export func Unmarshal(s string, val interface{}) (ok bool, errtok string) {
	var errindx int;
	var val1 interface{};
	b := &StructBuilder{ reflect.NewValue(val) };
	ok, errindx, errtok = Parse(s, b);
	if !ok {
		return false, errtok
	}
	return true, ""
}
