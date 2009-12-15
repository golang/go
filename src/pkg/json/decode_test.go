// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"container/vector"
	"reflect"
	"testing"
)

func TestDecodeInt64(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.Int64(-15)
	assertResult(t, nb.Data(), float64(-15))
}

func TestDecodeUint64(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.Uint64(15)
	assertResult(t, nb.Data(), float64(15))
}

func TestDecodeFloat64(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.Float64(3.14159)
	assertResult(t, nb.Data(), float64(3.14159))
}

func TestDecodeString(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.String("Some string")
	assertResult(t, nb.Data(), "Some string")
}

func TestDecodeBool(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.Bool(true)
	assertResult(t, nb.Data(), true)
}

func TestDecodeNull(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.Null()
	assertResult(t, nb.Data(), nil)
}

func TestDecodeEmptyArray(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.Array()
	assertResult(t, nb.Data(), []interface{}{})
}

func TestDecodeEmptyMap(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.Map()
	assertResult(t, nb.Data(), map[string]interface{}{})
}

func TestDecodeFlushElem(t *testing.T) {
	testVec := new(vector.Vector).Resize(2, 2)
	nb := newDecoder(testVec, 1)
	nb.Float64(3.14159)
	nb.Flush()
	assertResult(t, testVec.Data(), []interface{}{nil, float64(3.14159)})
}

func TestDecodeFlushKey(t *testing.T) {
	testMap := make(map[string]interface{})
	nb := newDecoder(testMap, "key")
	nb.Float64(3.14159)
	nb.Flush()
	assertResult(t, testMap, map[string]interface{}{"key": float64(3.14159)})
}

// Elem() and Key() are hard to test in isolation because all they do
// is create a new, properly initialized, decoder, and modify state of
// the underlying decoder.  I'm testing them through already tested
// Array(), String(), and Flush().

func TestDecodeElem(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.Array()
	var b Builder = nb.Elem(0)
	b.String("0")
	b.Flush()
	assertResult(t, nb.Data(), []interface{}{"0"})
}

func TestDecodeKey(t *testing.T) {
	nb := newDecoder(nil, nil)
	nb.Map()
	var b Builder = nb.Key("a")
	b.String("0")
	b.Flush()
	assertResult(t, nb.Data(), map[string]interface{}{"a": "0"})
}

func assertResult(t *testing.T, results, expected interface{}) {
	if !reflect.DeepEqual(results, expected) {
		t.Fatalf("have %T(%#v) want %T(%#v)", results, results, expected, expected)
	}
}

type decodeTest struct {
	s string
	r interface{}
}

var tests = []decodeTest{
	decodeTest{`null`, nil},
	decodeTest{`true`, true},
	decodeTest{`false`, false},
	decodeTest{`"abc"`, "abc"},
	decodeTest{`123`, float64(123)},
	decodeTest{`0.1`, float64(0.1)},
	decodeTest{`1e-10`, float64(1e-10)},
	decodeTest{`[]`, []interface{}{}},
	decodeTest{`[1,2,3,4]`, []interface{}{float64(1), float64(2), float64(3), float64(4)}},
	decodeTest{`[1,2,"abc",null,true,false]`, []interface{}{float64(1), float64(2), "abc", nil, true, false}},
	decodeTest{`{}`, map[string]interface{}{}},
	decodeTest{`{"a":1}`, map[string]interface{}{"a": float64(1)}},
	decodeTest{`"q\u0302"`, "q\u0302"},
}

func TestDecode(t *testing.T) {
	for _, test := range tests {
		if val, err := Decode(test.s); err != nil || !reflect.DeepEqual(val, test.r) {
			t.Errorf("Decode(%#q) = %v, %v want %v, nil", test.s, val, err, test.r)
		}
	}
}
