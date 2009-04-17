// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generic representation of JSON objects.

package json

import (
	"container/vector";
	"fmt";
	"json";
	"math";
	"strconv";
	"strings";
)

// Integers identifying the data type in the Json interface.
const (
	StringKind = iota;
	NumberKind;
	MapKind;		// JSON term is "Object", but in Go, it's a map
	ArrayKind;
	BoolKind;
	NullKind;
)

// The Json interface is implemented by all JSON objects.
type Json interface {
	Kind() int;		// StringKind, NumberKind, etc.
	String() string;	// a string form (any kind)
	Number() float64;	// numeric form (NumberKind)
	Bool() bool;		// boolean (BoolKind)
	Get(s string) Json;	// field lookup (MapKind)
	Elem(i int) Json;	// element lookup (ArrayKind)
	Len() int;		// length (ArrayKind)
}

// JsonToString returns the textual JSON syntax representation
// for the JSON object j.
//
// JsonToString differs from j.String() in the handling
// of string objects.  If j represents the string abc,
// j.String() == `abc`, but JsonToString(j) == `"abc"`.
func JsonToString(j Json) string {
	if j == nil {
		return "null"
	}
	if j.Kind() == StringKind {
		return Quote(j.String())
	}
	return j.String()
}

type _Null struct { }

// Null is the JSON object representing the null data object.
var Null Json = &_Null{}

func (*_Null) Kind() int { return NullKind }
func (*_Null) String() string { return "null" }
func (*_Null) Number() float64 { return 0 }
func (*_Null) Bool() bool { return false }
func (*_Null) Get(s string) Json { return Null }
func (*_Null) Elem(int) Json { return Null }
func (*_Null) Len() int { return 0 }

type _String struct { s string; _Null }
func (j *_String) Kind() int { return StringKind }
func (j *_String) String() string { return j.s }

type _Number struct { f float64; _Null }
func (j *_Number) Kind() int { return NumberKind }
func (j *_Number) Number() float64 { return j.f }
func (j *_Number) String() string {
	if math.Floor(j.f) == j.f {
		return fmt.Sprintf("%.0f", j.f);
	}
	return fmt.Sprintf("%g", j.f);
}

type _Array struct { a *vector.Vector; _Null }
func (j *_Array) Kind() int { return ArrayKind }
func (j *_Array) Len() int { return j.a.Len() }
func (j *_Array) Elem(i int) Json {
	if i < 0 || i >= j.a.Len() {
		return Null
	}
	return j.a.At(i).(Json)
}
func (j *_Array) String() string {
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

type _Bool struct { b bool; _Null }
func (j *_Bool) Kind() int { return BoolKind }
func (j *_Bool) Bool() bool { return j.b }
func (j *_Bool) String() string {
	if j.b {
		return "true"
	}
	return "false"
}

type _Map struct { m map[string]Json; _Null }
func (j *_Map) Kind() int { return MapKind }
func (j *_Map) Get(s string) Json {
	if j.m == nil {
		return Null
	}
	v, ok := j.m[s];
	if !ok {
		return Null
	}
	return v;
}
func (j *_Map) String() string {
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

// Walk evaluates path relative to the JSON object j.
// Path is taken as a sequence of slash-separated field names
// or numbers that can be used to index into JSON map and
// array objects.
//
// For example, if j is the JSON object for
// {"abc": [true, false]}, then Walk(j, "abc/1") returns the
// JSON object for true.
func Walk(j Json, path string) Json {
	for len(path) > 0 {
		var elem string;
		if i := strings.Index(path, "/"); i >= 0 {
			elem = path[0:i];
			path = path[i+1:len(path)];
		} else {
			elem = path;
			path = "";
		}
		switch j.Kind() {
		case ArrayKind:
			indx, err := strconv.Atoi(elem);
			if err != nil {
				return Null
			}
			j = j.Elem(indx);
		case MapKind:
			j = j.Get(elem);
		default:
			return Null
		}
	}
	return j
}

// Equal returns whether a and b are indistinguishable JSON objects.
func Equal(a, b Json) bool {
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
		m := a.(*_Map).m;
		if len(m) != len(b.(*_Map).m) {
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


// Parse builder for JSON objects.

type _JsonBuilder struct {
	// either writing to *ptr
	ptr *Json;

	// or to a[i] (can't set ptr = &a[i])
	a *vector.Vector;
	i int;

	// or to m[k] (can't set ptr = &m[k])
	m map[string] Json;
	k string;
}

func (b *_JsonBuilder) Put(j Json) {
	switch {
	case b.ptr != nil:
		*b.ptr = j;
	case b.a != nil:
		b.a.Set(b.i, j);
	case b.m != nil:
		b.m[b.k] = j;
	}
}

func (b *_JsonBuilder) Get() Json {
	switch {
	case b.ptr != nil:
		return *b.ptr;
	case b.a != nil:
		return b.a.At(b.i).(Json);
	case b.m != nil:
		return b.m[b.k];
	}
	return nil
}

func (b *_JsonBuilder) Float64(f float64) {
	b.Put(&_Number{f, _Null{}})
}

func (b *_JsonBuilder) Int64(i int64) {
	b.Float64(float64(i))
}

func (b *_JsonBuilder) Uint64(i uint64) {
	b.Float64(float64(i))
}

func (b *_JsonBuilder) Bool(tf bool) {
	b.Put(&_Bool{tf, _Null{}})
}

func (b *_JsonBuilder) Null() {
	b.Put(Null)
}

func (b *_JsonBuilder) String(s string) {
	b.Put(&_String{s, _Null{}})
}


func (b *_JsonBuilder) Array() {
	b.Put(&_Array{vector.New(0), _Null{}})
}

func (b *_JsonBuilder) Map() {
	b.Put(&_Map{make(map[string]Json), _Null{}})
}

func (b *_JsonBuilder) Elem(i int) Builder {
	bb := new(_JsonBuilder);
	bb.a = b.Get().(*_Array).a;
	bb.i = i;
	for i >= bb.a.Len() {
		bb.a.Push(Null)
	}
	return bb
}

func (b *_JsonBuilder) Key(k string) Builder {
	bb := new(_JsonBuilder);
	bb.m = b.Get().(*_Map).m;
	bb.k = k;
	bb.m[k] = Null;
	return bb
}

// StringToJson parses the string s as a JSON-syntax string
// and returns the generic JSON object representation.
// On success, StringToJson returns with ok set to true and errtok empty.
// If StringToJson encounters a syntax error, it returns with
// ok set to false and errtok set to a fragment of the offending syntax.
func StringToJson(s string) (json Json, ok bool, errtok string) {
	var errindx int;
	var j Json;
	b := new(_JsonBuilder);
	b.ptr = &j;
	ok, errindx, errtok = Parse(s, b);
	if !ok {
		return nil, false, errtok
	}
	return j, true, ""
}

// BUG(rsc): StringToJson should return an os.Error instead of a bool.
