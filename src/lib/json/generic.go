// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generic JSON representation.

package json

import (
	"array";
	"fmt";
	"math";
	"json";
	"strconv";
	"strings";
)

export const (
	StringKind = iota;
	NumberKind;
	MapKind;		// JSON term is "Object", but in Go, it's a map
	ArrayKind;
	BoolKind;
	NullKind;
)

export type Json interface {
	Kind() int;
	String() string;
	Number() float64;
	Bool() bool;
	Get(s string) Json;
	Elem(i int) Json;
	Len() int;
}

export func JsonToString(j Json) string {
	if j == nil {
		return "null"
	}
	if j.Kind() == StringKind {
		return Quote(j.String())
	}
	return j.String()
}

type Null struct { }
export var null Json = &Null{}
func (*Null) Kind() int { return NullKind }
func (*Null) String() string { return "null" }
func (*Null) Number() float64 { return 0 }
func (*Null) Bool() bool { return false }
func (*Null) Get(s string) Json { return null }
func (*Null) Elem(int) Json { return null }
func (*Null) Len() int { return 0 }

type String struct { s string; Null }
func (j *String) Kind() int { return StringKind }
func (j *String) String() string { return j.s }

type Number struct { f float64; Null }
func (j *Number) Kind() int { return NumberKind }
func (j *Number) Number() float64 { return j.f }
func (j *Number) String() string {
	if math.Floor(j.f) == j.f {
		return fmt.sprintf("%.0f", j.f);
	}
	return fmt.sprintf("%g", j.f);
}

type Array struct { a *array.Array; Null }
func (j *Array) Kind() int { return ArrayKind }
func (j *Array) Len() int { return j.a.Len() }
func (j *Array) Elem(i int) Json {
	if i < 0 || i >= j.a.Len() {
		return null
	}
	return j.a.At(i)
}
func (j *Array) String() string {
	s := "[";
	for i := 0; i < j.a.Len(); i++ {
		if i > 0 {
			s += ",";
		}
		s += JsonToString(j.a.At(i).(Json));
	}
	s += "]";
	return s;
}

type Bool struct { b bool; Null }
func (j *Bool) Kind() int { return BoolKind }
func (j *Bool) Bool() bool { return j.b }
func (j *Bool) String() string {
	if j.b {
		return "true"
	}
	return "false"
}

type Map struct { m map[string]Json; Null }
func (j *Map) Kind() int { return MapKind }
func (j *Map) Get(s string) Json {
	if j.m == nil {
		return null
	}
	v, ok := j.m[s];
	if !ok {
		return null
	}
	return v;
}
func (j *Map) String() string {
	s := "{";
	first := true;
	for k,v := range j.m {
		if first {
			first = false;
		} else {
			s += ",";
		}
		s += Quote(k);
		s += ":";
		s += JsonToString(v);
	}
	s += "}";
	return s;
}

export func Walk(j Json, path string) Json {
	for len(path) > 0 {
		var elem string;
		if i := strings.index(path, "/"); i >= 0 {
			elem = path[0:i];
			path = path[i+1:len(path)];
		} else {
			elem = path;
			path = "";
		}
		switch j.Kind() {
		case ArrayKind:
			indx, err := strconv.atoi(elem);
			if err != nil {
				return null
			}
			j = j.Elem(indx);
		case MapKind:
			j = j.Get(elem);
		default:
			return null
		}
	}
	return j
}

export func Equal(a, b Json) bool {
	switch {
	case a == nil && b == nil:
		return true;
	case a == nil || b == nil:
		return false;
	case a.Kind() != b.Kind():
		return false;
	}

	switch a.Kind() {
	case NullKind:
		return true;
	case StringKind:
		return a.String() == b.String();
	case NumberKind:
		return a.Number() == b.Number();
	case BoolKind:
		return a.Bool() == b.Bool();
	case ArrayKind:
		if a.Len() != b.Len() {
			return false;
		}
		for i := 0; i < a.Len(); i++ {
			if !Equal(a.Elem(i), b.Elem(i)) {
				return false;
			}
		}
		return true;
	case MapKind:
		m := a.(*Map).m;
		if len(m) != len(b.(*Map).m) {
			return false;
		}
		for k,v := range m {
			if !Equal(v, b.Get(k)) {
				return false;
			}
		}
		return true;
	}

	// invalid kind
	return false;
}


// Parse builder for Json objects.

type JsonBuilder struct {
	// either writing to *ptr
	ptr *Json;

	// or to a[i] (can't set ptr = &a[i])
	a *array.Array;
	i int;

	// or to m[k] (can't set ptr = &m[k])
	m map[string] Json;
	k string;
}

func (b *JsonBuilder) Put(j Json) {
	switch {
	case b.ptr != nil:
		*b.ptr = j;
	case b.a != nil:
		b.a.Set(b.i, j);
	case b.m != nil:
		b.m[b.k] = j;
	}
}

func (b *JsonBuilder) Get() Json {
	switch {
	case b.ptr != nil:
		return *b.ptr;
	case b.a != nil:
		return b.a.At(b.i);
	case b.m != nil:
		return b.m[b.k];
	}
	return nil
}

func (b *JsonBuilder) Float64(f float64) {
	b.Put(&Number{f, Null{}})
}

func (b *JsonBuilder) Int64(i int64) {
	b.Float64(float64(i))
}

func (b *JsonBuilder) Uint64(i uint64) {
	b.Float64(float64(i))
}

func (b *JsonBuilder) Bool(tf bool) {
	b.Put(&Bool{tf, Null{}})
}

func (b *JsonBuilder) Null() {
	b.Put(null)
}

func (b *JsonBuilder) String(s string) {
	b.Put(&String{s, Null{}})
}


func (b *JsonBuilder) Array() {
	b.Put(&Array{array.New(0), Null{}})
}

func (b *JsonBuilder) Map() {
	b.Put(&Map{make(map[string]Json), Null{}})
}

func (b *JsonBuilder) Elem(i int) Builder {
	bb := new(JsonBuilder);
	bb.a = b.Get().(*Array).a;
	bb.i = i;
	for i >= bb.a.Len() {
		bb.a.Push(null)
	}
	return bb
}

func (b *JsonBuilder) Key(k string) Builder {
	bb := new(JsonBuilder);
	bb.m = b.Get().(*Map).m;
	bb.k = k;
	bb.m[k] = null;
	return bb
}

export func StringToJson(s string) (json Json, ok bool, errtok string) {
	var errindx int;
	var j Json;
	b := new(JsonBuilder);
	b.ptr = &j;
	ok, errindx, errtok = Parse(s, b);
	if !ok {
		return nil, false, errtok
	}
	return j, true, ""
}
