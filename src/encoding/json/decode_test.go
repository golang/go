// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"encoding"
	"fmt"
	"image"
	"reflect"
	"strings"
	"testing"
	"time"
)

type T struct {
	X string
	Y int
	Z int `json:"-"`
}

type U struct {
	Alphabet string `json:"alpha"`
}

type V struct {
	F1 interface{}
	F2 int32
	F3 Number
}

// ifaceNumAsFloat64/ifaceNumAsNumber are used to test unmarshaling with and
// without UseNumber
var ifaceNumAsFloat64 = map[string]interface{}{
	"k1": float64(1),
	"k2": "s",
	"k3": []interface{}{float64(1), float64(2.0), float64(3e-3)},
	"k4": map[string]interface{}{"kk1": "s", "kk2": float64(2)},
}

var ifaceNumAsNumber = map[string]interface{}{
	"k1": Number("1"),
	"k2": "s",
	"k3": []interface{}{Number("1"), Number("2.0"), Number("3e-3")},
	"k4": map[string]interface{}{"kk1": "s", "kk2": Number("2")},
}

type tx struct {
	x int
}

// A type that can unmarshal itself.

type unmarshaler struct {
	T bool
}

func (u *unmarshaler) UnmarshalJSON(b []byte) error {
	*u = unmarshaler{true} // All we need to see that UnmarshalJSON is called.
	return nil
}

type ustruct struct {
	M unmarshaler
}

type unmarshalerText struct {
	T bool
}

// needed for re-marshaling tests
func (u *unmarshalerText) MarshalText() ([]byte, error) {
	return []byte(""), nil
}

func (u *unmarshalerText) UnmarshalText(b []byte) error {
	*u = unmarshalerText{true} // All we need to see that UnmarshalText is called.
	return nil
}

var _ encoding.TextUnmarshaler = (*unmarshalerText)(nil)

type ustructText struct {
	M unmarshalerText
}

var (
	um0, um1 unmarshaler // target2 of unmarshaling
	ump      = &um1
	umtrue   = unmarshaler{true}
	umslice  = []unmarshaler{{true}}
	umslicep = new([]unmarshaler)
	umstruct = ustruct{unmarshaler{true}}

	um0T, um1T unmarshalerText // target2 of unmarshaling
	umpT       = &um1T
	umtrueT    = unmarshalerText{true}
	umsliceT   = []unmarshalerText{{true}}
	umslicepT  = new([]unmarshalerText)
	umstructT  = ustructText{unmarshalerText{true}}
)

// Test data structures for anonymous fields.

type Point struct {
	Z int
}

type Top struct {
	Level0 int
	Embed0
	*Embed0a
	*Embed0b `json:"e,omitempty"` // treated as named
	Embed0c  `json:"-"`           // ignored
	Loop
	Embed0p // has Point with X, Y, used
	Embed0q // has Point with Z, used
}

type Embed0 struct {
	Level1a int // overridden by Embed0a's Level1a with json tag
	Level1b int // used because Embed0a's Level1b is renamed
	Level1c int // used because Embed0a's Level1c is ignored
	Level1d int // annihilated by Embed0a's Level1d
	Level1e int `json:"x"` // annihilated by Embed0a.Level1e
}

type Embed0a struct {
	Level1a int `json:"Level1a,omitempty"`
	Level1b int `json:"LEVEL1B,omitempty"`
	Level1c int `json:"-"`
	Level1d int // annihilated by Embed0's Level1d
	Level1f int `json:"x"` // annihilated by Embed0's Level1e
}

type Embed0b Embed0

type Embed0c Embed0

type Embed0p struct {
	image.Point
}

type Embed0q struct {
	Point
}

type Loop struct {
	Loop1 int `json:",omitempty"`
	Loop2 int `json:",omitempty"`
	*Loop
}

// From reflect test:
// The X in S6 and S7 annihilate, but they also block the X in S8.S9.
type S5 struct {
	S6
	S7
	S8
}

type S6 struct {
	X int
}

type S7 S6

type S8 struct {
	S9
}

type S9 struct {
	X int
	Y int
}

// From reflect test:
// The X in S11.S6 and S12.S6 annihilate, but they also block the X in S13.S8.S9.
type S10 struct {
	S11
	S12
	S13
}

type S11 struct {
	S6
}

type S12 struct {
	S6
}

type S13 struct {
	S8
}

type unmarshalTest struct {
	in        string
	ptr       interface{}
	out       interface{}
	err       error
	useNumber bool
}

type Ambig struct {
	// Given "hello", the first match should win.
	First  int `json:"HELLO"`
	Second int `json:"Hello"`
}

type XYZ struct {
	X interface{}
	Y interface{}
	Z interface{}
}

var unmarshalTests = []unmarshalTest{
	// basic types
	{in: `true`, ptr: new(bool), out: true},
	{in: `1`, ptr: new(int), out: 1},
	{in: `1.2`, ptr: new(float64), out: 1.2},
	{in: `-5`, ptr: new(int16), out: int16(-5)},
	{in: `2`, ptr: new(Number), out: Number("2"), useNumber: true},
	{in: `2`, ptr: new(Number), out: Number("2")},
	{in: `2`, ptr: new(interface{}), out: float64(2.0)},
	{in: `2`, ptr: new(interface{}), out: Number("2"), useNumber: true},
	{in: `"a\u1234"`, ptr: new(string), out: "a\u1234"},
	{in: `"http:\/\/"`, ptr: new(string), out: "http://"},
	{in: `"g-clef: \uD834\uDD1E"`, ptr: new(string), out: "g-clef: \U0001D11E"},
	{in: `"invalid: \uD834x\uDD1E"`, ptr: new(string), out: "invalid: \uFFFDx\uFFFD"},
	{in: "null", ptr: new(interface{}), out: nil},
	{in: `{"X": [1,2,3], "Y": 4}`, ptr: new(T), out: T{Y: 4}, err: &UnmarshalTypeError{"array", reflect.TypeOf(""), 7}},
	{in: `{"x": 1}`, ptr: new(tx), out: tx{}},
	{in: `{"F1":1,"F2":2,"F3":3}`, ptr: new(V), out: V{F1: float64(1), F2: int32(2), F3: Number("3")}},
	{in: `{"F1":1,"F2":2,"F3":3}`, ptr: new(V), out: V{F1: Number("1"), F2: int32(2), F3: Number("3")}, useNumber: true},
	{in: `{"k1":1,"k2":"s","k3":[1,2.0,3e-3],"k4":{"kk1":"s","kk2":2}}`, ptr: new(interface{}), out: ifaceNumAsFloat64},
	{in: `{"k1":1,"k2":"s","k3":[1,2.0,3e-3],"k4":{"kk1":"s","kk2":2}}`, ptr: new(interface{}), out: ifaceNumAsNumber, useNumber: true},

	// raw values with whitespace
	{in: "\n true ", ptr: new(bool), out: true},
	{in: "\t 1 ", ptr: new(int), out: 1},
	{in: "\r 1.2 ", ptr: new(float64), out: 1.2},
	{in: "\t -5 \n", ptr: new(int16), out: int16(-5)},
	{in: "\t \"a\\u1234\" \n", ptr: new(string), out: "a\u1234"},

	// Z has a "-" tag.
	{in: `{"Y": 1, "Z": 2}`, ptr: new(T), out: T{Y: 1}},

	{in: `{"alpha": "abc", "alphabet": "xyz"}`, ptr: new(U), out: U{Alphabet: "abc"}},
	{in: `{"alpha": "abc"}`, ptr: new(U), out: U{Alphabet: "abc"}},
	{in: `{"alphabet": "xyz"}`, ptr: new(U), out: U{}},

	// syntax errors
	{in: `{"X": "foo", "Y"}`, err: &SyntaxError{"invalid character '}' after object key", 17}},
	{in: `[1, 2, 3+]`, err: &SyntaxError{"invalid character '+' after array element", 9}},
	{in: `{"X":12x}`, err: &SyntaxError{"invalid character 'x' after object key:value pair", 8}, useNumber: true},

	// raw value errors
	{in: "\x01 42", err: &SyntaxError{"invalid character '\\x01' looking for beginning of value", 1}},
	{in: " 42 \x01", err: &SyntaxError{"invalid character '\\x01' after top-level value", 5}},
	{in: "\x01 true", err: &SyntaxError{"invalid character '\\x01' looking for beginning of value", 1}},
	{in: " false \x01", err: &SyntaxError{"invalid character '\\x01' after top-level value", 8}},
	{in: "\x01 1.2", err: &SyntaxError{"invalid character '\\x01' looking for beginning of value", 1}},
	{in: " 3.4 \x01", err: &SyntaxError{"invalid character '\\x01' after top-level value", 6}},
	{in: "\x01 \"string\"", err: &SyntaxError{"invalid character '\\x01' looking for beginning of value", 1}},
	{in: " \"string\" \x01", err: &SyntaxError{"invalid character '\\x01' after top-level value", 11}},

	// array tests
	{in: `[1, 2, 3]`, ptr: new([3]int), out: [3]int{1, 2, 3}},
	{in: `[1, 2, 3]`, ptr: new([1]int), out: [1]int{1}},
	{in: `[1, 2, 3]`, ptr: new([5]int), out: [5]int{1, 2, 3, 0, 0}},

	// empty array to interface test
	{in: `[]`, ptr: new([]interface{}), out: []interface{}{}},
	{in: `null`, ptr: new([]interface{}), out: []interface{}(nil)},
	{in: `{"T":[]}`, ptr: new(map[string]interface{}), out: map[string]interface{}{"T": []interface{}{}}},
	{in: `{"T":null}`, ptr: new(map[string]interface{}), out: map[string]interface{}{"T": interface{}(nil)}},

	// composite tests
	{in: allValueIndent, ptr: new(All), out: allValue},
	{in: allValueCompact, ptr: new(All), out: allValue},
	{in: allValueIndent, ptr: new(*All), out: &allValue},
	{in: allValueCompact, ptr: new(*All), out: &allValue},
	{in: pallValueIndent, ptr: new(All), out: pallValue},
	{in: pallValueCompact, ptr: new(All), out: pallValue},
	{in: pallValueIndent, ptr: new(*All), out: &pallValue},
	{in: pallValueCompact, ptr: new(*All), out: &pallValue},

	// unmarshal interface test
	{in: `{"T":false}`, ptr: &um0, out: umtrue}, // use "false" so test will fail if custom unmarshaler is not called
	{in: `{"T":false}`, ptr: &ump, out: &umtrue},
	{in: `[{"T":false}]`, ptr: &umslice, out: umslice},
	{in: `[{"T":false}]`, ptr: &umslicep, out: &umslice},
	{in: `{"M":{"T":false}}`, ptr: &umstruct, out: umstruct},

	// UnmarshalText interface test
	{in: `"X"`, ptr: &um0T, out: umtrueT}, // use "false" so test will fail if custom unmarshaler is not called
	{in: `"X"`, ptr: &umpT, out: &umtrueT},
	{in: `["X"]`, ptr: &umsliceT, out: umsliceT},
	{in: `["X"]`, ptr: &umslicepT, out: &umsliceT},
	{in: `{"M":"X"}`, ptr: &umstructT, out: umstructT},

	{
		in: `{
			"Level0": 1,
			"Level1b": 2,
			"Level1c": 3,
			"x": 4,
			"Level1a": 5,
			"LEVEL1B": 6,
			"e": {
				"Level1a": 8,
				"Level1b": 9,
				"Level1c": 10,
				"Level1d": 11,
				"x": 12
			},
			"Loop1": 13,
			"Loop2": 14,
			"X": 15,
			"Y": 16,
			"Z": 17
		}`,
		ptr: new(Top),
		out: Top{
			Level0: 1,
			Embed0: Embed0{
				Level1b: 2,
				Level1c: 3,
			},
			Embed0a: &Embed0a{
				Level1a: 5,
				Level1b: 6,
			},
			Embed0b: &Embed0b{
				Level1a: 8,
				Level1b: 9,
				Level1c: 10,
				Level1d: 11,
				Level1e: 12,
			},
			Loop: Loop{
				Loop1: 13,
				Loop2: 14,
			},
			Embed0p: Embed0p{
				Point: image.Point{X: 15, Y: 16},
			},
			Embed0q: Embed0q{
				Point: Point{Z: 17},
			},
		},
	},
	{
		in:  `{"hello": 1}`,
		ptr: new(Ambig),
		out: Ambig{First: 1},
	},

	{
		in:  `{"X": 1,"Y":2}`,
		ptr: new(S5),
		out: S5{S8: S8{S9: S9{Y: 2}}},
	},
	{
		in:  `{"X": 1,"Y":2}`,
		ptr: new(S10),
		out: S10{S13: S13{S8: S8{S9: S9{Y: 2}}}},
	},

	// invalid UTF-8 is coerced to valid UTF-8.
	{
		in:  "\"hello\xffworld\"",
		ptr: new(string),
		out: "hello\ufffdworld",
	},
	{
		in:  "\"hello\xc2\xc2world\"",
		ptr: new(string),
		out: "hello\ufffd\ufffdworld",
	},
	{
		in:  "\"hello\xc2\xffworld\"",
		ptr: new(string),
		out: "hello\ufffd\ufffdworld",
	},
	{
		in:  "\"hello\\ud800world\"",
		ptr: new(string),
		out: "hello\ufffdworld",
	},
	{
		in:  "\"hello\\ud800\\ud800world\"",
		ptr: new(string),
		out: "hello\ufffd\ufffdworld",
	},
	{
		in:  "\"hello\\ud800\\ud800world\"",
		ptr: new(string),
		out: "hello\ufffd\ufffdworld",
	},
	{
		in:  "\"hello\xed\xa0\x80\xed\xb0\x80world\"",
		ptr: new(string),
		out: "hello\ufffd\ufffd\ufffd\ufffd\ufffd\ufffdworld",
	},

	// issue 8305
	{
		in:  `{"2009-11-10T23:00:00Z": "hello world"}`,
		ptr: &map[time.Time]string{},
		err: &UnmarshalTypeError{"object", reflect.TypeOf(map[time.Time]string{}), 1},
	},
}

func TestMarshal(t *testing.T) {
	b, err := Marshal(allValue)
	if err != nil {
		t.Fatalf("Marshal allValue: %v", err)
	}
	if string(b) != allValueCompact {
		t.Errorf("Marshal allValueCompact")
		diff(t, b, []byte(allValueCompact))
		return
	}

	b, err = Marshal(pallValue)
	if err != nil {
		t.Fatalf("Marshal pallValue: %v", err)
	}
	if string(b) != pallValueCompact {
		t.Errorf("Marshal pallValueCompact")
		diff(t, b, []byte(pallValueCompact))
		return
	}
}

var badUTF8 = []struct {
	in, out string
}{
	{"hello\xffworld", `"hello\ufffdworld"`},
	{"", `""`},
	{"\xff", `"\ufffd"`},
	{"\xff\xff", `"\ufffd\ufffd"`},
	{"a\xffb", `"a\ufffdb"`},
	{"\xe6\x97\xa5\xe6\x9c\xac\xff\xaa\x9e", `"日本\ufffd\ufffd\ufffd"`},
}

func TestMarshalBadUTF8(t *testing.T) {
	for _, tt := range badUTF8 {
		b, err := Marshal(tt.in)
		if string(b) != tt.out || err != nil {
			t.Errorf("Marshal(%q) = %#q, %v, want %#q, nil", tt.in, b, err, tt.out)
		}
	}
}

func TestMarshalNumberZeroVal(t *testing.T) {
	var n Number
	out, err := Marshal(n)
	if err != nil {
		t.Fatal(err)
	}
	outStr := string(out)
	if outStr != "0" {
		t.Fatalf("Invalid zero val for Number: %q", outStr)
	}
}

func TestMarshalEmbeds(t *testing.T) {
	top := &Top{
		Level0: 1,
		Embed0: Embed0{
			Level1b: 2,
			Level1c: 3,
		},
		Embed0a: &Embed0a{
			Level1a: 5,
			Level1b: 6,
		},
		Embed0b: &Embed0b{
			Level1a: 8,
			Level1b: 9,
			Level1c: 10,
			Level1d: 11,
			Level1e: 12,
		},
		Loop: Loop{
			Loop1: 13,
			Loop2: 14,
		},
		Embed0p: Embed0p{
			Point: image.Point{X: 15, Y: 16},
		},
		Embed0q: Embed0q{
			Point: Point{Z: 17},
		},
	}
	b, err := Marshal(top)
	if err != nil {
		t.Fatal(err)
	}
	want := "{\"Level0\":1,\"Level1b\":2,\"Level1c\":3,\"Level1a\":5,\"LEVEL1B\":6,\"e\":{\"Level1a\":8,\"Level1b\":9,\"Level1c\":10,\"Level1d\":11,\"x\":12},\"Loop1\":13,\"Loop2\":14,\"X\":15,\"Y\":16,\"Z\":17}"
	if string(b) != want {
		t.Errorf("Wrong marshal result.\n got: %q\nwant: %q", b, want)
	}
}

func TestUnmarshal(t *testing.T) {
	for i, tt := range unmarshalTests {
		var scan scanner
		in := []byte(tt.in)
		if err := checkValid(in, &scan); err != nil {
			if !reflect.DeepEqual(err, tt.err) {
				t.Errorf("#%d: checkValid: %#v", i, err)
				continue
			}
		}
		if tt.ptr == nil {
			continue
		}

		// v = new(right-type)
		v := reflect.New(reflect.TypeOf(tt.ptr).Elem())
		dec := NewDecoder(bytes.NewReader(in))
		if tt.useNumber {
			dec.UseNumber()
		}
		if err := dec.Decode(v.Interface()); !reflect.DeepEqual(err, tt.err) {
			t.Errorf("#%d: %v, want %v", i, err, tt.err)
			continue
		} else if err != nil {
			continue
		}
		if !reflect.DeepEqual(v.Elem().Interface(), tt.out) {
			t.Errorf("#%d: mismatch\nhave: %#+v\nwant: %#+v", i, v.Elem().Interface(), tt.out)
			data, _ := Marshal(v.Elem().Interface())
			println(string(data))
			data, _ = Marshal(tt.out)
			println(string(data))
			continue
		}

		// Check round trip.
		if tt.err == nil {
			enc, err := Marshal(v.Interface())
			if err != nil {
				t.Errorf("#%d: error re-marshaling: %v", i, err)
				continue
			}
			vv := reflect.New(reflect.TypeOf(tt.ptr).Elem())
			dec = NewDecoder(bytes.NewReader(enc))
			if tt.useNumber {
				dec.UseNumber()
			}
			if err := dec.Decode(vv.Interface()); err != nil {
				t.Errorf("#%d: error re-unmarshaling %#q: %v", i, enc, err)
				continue
			}
			if !reflect.DeepEqual(v.Elem().Interface(), vv.Elem().Interface()) {
				t.Errorf("#%d: mismatch\nhave: %#+v\nwant: %#+v", i, v.Elem().Interface(), vv.Elem().Interface())
				t.Errorf("     In: %q", strings.Map(noSpace, string(in)))
				t.Errorf("Marshal: %q", strings.Map(noSpace, string(enc)))
				continue
			}
		}
	}
}

func TestUnmarshalMarshal(t *testing.T) {
	initBig()
	var v interface{}
	if err := Unmarshal(jsonBig, &v); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	b, err := Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !bytes.Equal(jsonBig, b) {
		t.Errorf("Marshal jsonBig")
		diff(t, b, jsonBig)
		return
	}
}

var numberTests = []struct {
	in       string
	i        int64
	intErr   string
	f        float64
	floatErr string
}{
	{in: "-1.23e1", intErr: "strconv.ParseInt: parsing \"-1.23e1\": invalid syntax", f: -1.23e1},
	{in: "-12", i: -12, f: -12.0},
	{in: "1e1000", intErr: "strconv.ParseInt: parsing \"1e1000\": invalid syntax", floatErr: "strconv.ParseFloat: parsing \"1e1000\": value out of range"},
}

// Independent of Decode, basic coverage of the accessors in Number
func TestNumberAccessors(t *testing.T) {
	for _, tt := range numberTests {
		n := Number(tt.in)
		if s := n.String(); s != tt.in {
			t.Errorf("Number(%q).String() is %q", tt.in, s)
		}
		if i, err := n.Int64(); err == nil && tt.intErr == "" && i != tt.i {
			t.Errorf("Number(%q).Int64() is %d", tt.in, i)
		} else if (err == nil && tt.intErr != "") || (err != nil && err.Error() != tt.intErr) {
			t.Errorf("Number(%q).Int64() wanted error %q but got: %v", tt.in, tt.intErr, err)
		}
		if f, err := n.Float64(); err == nil && tt.floatErr == "" && f != tt.f {
			t.Errorf("Number(%q).Float64() is %g", tt.in, f)
		} else if (err == nil && tt.floatErr != "") || (err != nil && err.Error() != tt.floatErr) {
			t.Errorf("Number(%q).Float64() wanted error %q but got: %v", tt.in, tt.floatErr, err)
		}
	}
}

func TestLargeByteSlice(t *testing.T) {
	s0 := make([]byte, 2000)
	for i := range s0 {
		s0[i] = byte(i)
	}
	b, err := Marshal(s0)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var s1 []byte
	if err := Unmarshal(b, &s1); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !bytes.Equal(s0, s1) {
		t.Errorf("Marshal large byte slice")
		diff(t, s0, s1)
	}
}

type Xint struct {
	X int
}

func TestUnmarshalInterface(t *testing.T) {
	var xint Xint
	var i interface{} = &xint
	if err := Unmarshal([]byte(`{"X":1}`), &i); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if xint.X != 1 {
		t.Fatalf("Did not write to xint")
	}
}

func TestUnmarshalPtrPtr(t *testing.T) {
	var xint Xint
	pxint := &xint
	if err := Unmarshal([]byte(`{"X":1}`), &pxint); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if xint.X != 1 {
		t.Fatalf("Did not write to xint")
	}
}

func TestEscape(t *testing.T) {
	const input = `"foobar"<html>` + " [\u2028 \u2029]"
	const expected = `"\"foobar\"\u003chtml\u003e [\u2028 \u2029]"`
	b, err := Marshal(input)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if s := string(b); s != expected {
		t.Errorf("Encoding of [%s]:\n got [%s]\nwant [%s]", input, s, expected)
	}
}

// WrongString is a struct that's misusing the ,string modifier.
type WrongString struct {
	Message string `json:"result,string"`
}

type wrongStringTest struct {
	in, err string
}

var wrongStringTests = []wrongStringTest{
	{`{"result":"x"}`, `json: invalid use of ,string struct tag, trying to unmarshal "x" into string`},
	{`{"result":"foo"}`, `json: invalid use of ,string struct tag, trying to unmarshal "foo" into string`},
	{`{"result":"123"}`, `json: invalid use of ,string struct tag, trying to unmarshal "123" into string`},
	{`{"result":123}`, `json: invalid use of ,string struct tag, trying to unmarshal unquoted value into string`},
}

// If people misuse the ,string modifier, the error message should be
// helpful, telling the user that they're doing it wrong.
func TestErrorMessageFromMisusedString(t *testing.T) {
	for n, tt := range wrongStringTests {
		r := strings.NewReader(tt.in)
		var s WrongString
		err := NewDecoder(r).Decode(&s)
		got := fmt.Sprintf("%v", err)
		if got != tt.err {
			t.Errorf("%d. got err = %q, want %q", n, got, tt.err)
		}
	}
}

func noSpace(c rune) rune {
	if isSpace(c) {
		return -1
	}
	return c
}

type All struct {
	Bool    bool
	Int     int
	Int8    int8
	Int16   int16
	Int32   int32
	Int64   int64
	Uint    uint
	Uint8   uint8
	Uint16  uint16
	Uint32  uint32
	Uint64  uint64
	Uintptr uintptr
	Float32 float32
	Float64 float64

	Foo  string `json:"bar"`
	Foo2 string `json:"bar2,dummyopt"`

	IntStr int64 `json:",string"`

	PBool    *bool
	PInt     *int
	PInt8    *int8
	PInt16   *int16
	PInt32   *int32
	PInt64   *int64
	PUint    *uint
	PUint8   *uint8
	PUint16  *uint16
	PUint32  *uint32
	PUint64  *uint64
	PUintptr *uintptr
	PFloat32 *float32
	PFloat64 *float64

	String  string
	PString *string

	Map   map[string]Small
	MapP  map[string]*Small
	PMap  *map[string]Small
	PMapP *map[string]*Small

	EmptyMap map[string]Small
	NilMap   map[string]Small

	Slice   []Small
	SliceP  []*Small
	PSlice  *[]Small
	PSliceP *[]*Small

	EmptySlice []Small
	NilSlice   []Small

	StringSlice []string
	ByteSlice   []byte

	Small   Small
	PSmall  *Small
	PPSmall **Small

	Interface  interface{}
	PInterface *interface{}

	unexported int
}

type Small struct {
	Tag string
}

var allValue = All{
	Bool:    true,
	Int:     2,
	Int8:    3,
	Int16:   4,
	Int32:   5,
	Int64:   6,
	Uint:    7,
	Uint8:   8,
	Uint16:  9,
	Uint32:  10,
	Uint64:  11,
	Uintptr: 12,
	Float32: 14.1,
	Float64: 15.1,
	Foo:     "foo",
	Foo2:    "foo2",
	IntStr:  42,
	String:  "16",
	Map: map[string]Small{
		"17": {Tag: "tag17"},
		"18": {Tag: "tag18"},
	},
	MapP: map[string]*Small{
		"19": {Tag: "tag19"},
		"20": nil,
	},
	EmptyMap:    map[string]Small{},
	Slice:       []Small{{Tag: "tag20"}, {Tag: "tag21"}},
	SliceP:      []*Small{{Tag: "tag22"}, nil, {Tag: "tag23"}},
	EmptySlice:  []Small{},
	StringSlice: []string{"str24", "str25", "str26"},
	ByteSlice:   []byte{27, 28, 29},
	Small:       Small{Tag: "tag30"},
	PSmall:      &Small{Tag: "tag31"},
	Interface:   5.2,
}

var pallValue = All{
	PBool:      &allValue.Bool,
	PInt:       &allValue.Int,
	PInt8:      &allValue.Int8,
	PInt16:     &allValue.Int16,
	PInt32:     &allValue.Int32,
	PInt64:     &allValue.Int64,
	PUint:      &allValue.Uint,
	PUint8:     &allValue.Uint8,
	PUint16:    &allValue.Uint16,
	PUint32:    &allValue.Uint32,
	PUint64:    &allValue.Uint64,
	PUintptr:   &allValue.Uintptr,
	PFloat32:   &allValue.Float32,
	PFloat64:   &allValue.Float64,
	PString:    &allValue.String,
	PMap:       &allValue.Map,
	PMapP:      &allValue.MapP,
	PSlice:     &allValue.Slice,
	PSliceP:    &allValue.SliceP,
	PPSmall:    &allValue.PSmall,
	PInterface: &allValue.Interface,
}

var allValueIndent = `{
	"Bool": true,
	"Int": 2,
	"Int8": 3,
	"Int16": 4,
	"Int32": 5,
	"Int64": 6,
	"Uint": 7,
	"Uint8": 8,
	"Uint16": 9,
	"Uint32": 10,
	"Uint64": 11,
	"Uintptr": 12,
	"Float32": 14.1,
	"Float64": 15.1,
	"bar": "foo",
	"bar2": "foo2",
	"IntStr": "42",
	"PBool": null,
	"PInt": null,
	"PInt8": null,
	"PInt16": null,
	"PInt32": null,
	"PInt64": null,
	"PUint": null,
	"PUint8": null,
	"PUint16": null,
	"PUint32": null,
	"PUint64": null,
	"PUintptr": null,
	"PFloat32": null,
	"PFloat64": null,
	"String": "16",
	"PString": null,
	"Map": {
		"17": {
			"Tag": "tag17"
		},
		"18": {
			"Tag": "tag18"
		}
	},
	"MapP": {
		"19": {
			"Tag": "tag19"
		},
		"20": null
	},
	"PMap": null,
	"PMapP": null,
	"EmptyMap": {},
	"NilMap": null,
	"Slice": [
		{
			"Tag": "tag20"
		},
		{
			"Tag": "tag21"
		}
	],
	"SliceP": [
		{
			"Tag": "tag22"
		},
		null,
		{
			"Tag": "tag23"
		}
	],
	"PSlice": null,
	"PSliceP": null,
	"EmptySlice": [],
	"NilSlice": null,
	"StringSlice": [
		"str24",
		"str25",
		"str26"
	],
	"ByteSlice": "Gxwd",
	"Small": {
		"Tag": "tag30"
	},
	"PSmall": {
		"Tag": "tag31"
	},
	"PPSmall": null,
	"Interface": 5.2,
	"PInterface": null
}`

var allValueCompact = strings.Map(noSpace, allValueIndent)

var pallValueIndent = `{
	"Bool": false,
	"Int": 0,
	"Int8": 0,
	"Int16": 0,
	"Int32": 0,
	"Int64": 0,
	"Uint": 0,
	"Uint8": 0,
	"Uint16": 0,
	"Uint32": 0,
	"Uint64": 0,
	"Uintptr": 0,
	"Float32": 0,
	"Float64": 0,
	"bar": "",
	"bar2": "",
        "IntStr": "0",
	"PBool": true,
	"PInt": 2,
	"PInt8": 3,
	"PInt16": 4,
	"PInt32": 5,
	"PInt64": 6,
	"PUint": 7,
	"PUint8": 8,
	"PUint16": 9,
	"PUint32": 10,
	"PUint64": 11,
	"PUintptr": 12,
	"PFloat32": 14.1,
	"PFloat64": 15.1,
	"String": "",
	"PString": "16",
	"Map": null,
	"MapP": null,
	"PMap": {
		"17": {
			"Tag": "tag17"
		},
		"18": {
			"Tag": "tag18"
		}
	},
	"PMapP": {
		"19": {
			"Tag": "tag19"
		},
		"20": null
	},
	"EmptyMap": null,
	"NilMap": null,
	"Slice": null,
	"SliceP": null,
	"PSlice": [
		{
			"Tag": "tag20"
		},
		{
			"Tag": "tag21"
		}
	],
	"PSliceP": [
		{
			"Tag": "tag22"
		},
		null,
		{
			"Tag": "tag23"
		}
	],
	"EmptySlice": null,
	"NilSlice": null,
	"StringSlice": null,
	"ByteSlice": null,
	"Small": {
		"Tag": ""
	},
	"PSmall": null,
	"PPSmall": {
		"Tag": "tag31"
	},
	"Interface": null,
	"PInterface": 5.2
}`

var pallValueCompact = strings.Map(noSpace, pallValueIndent)

func TestRefUnmarshal(t *testing.T) {
	type S struct {
		// Ref is defined in encode_test.go.
		R0 Ref
		R1 *Ref
		R2 RefText
		R3 *RefText
	}
	want := S{
		R0: 12,
		R1: new(Ref),
		R2: 13,
		R3: new(RefText),
	}
	*want.R1 = 12
	*want.R3 = 13

	var got S
	if err := Unmarshal([]byte(`{"R0":"ref","R1":"ref","R2":"ref","R3":"ref"}`), &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

// Test that the empty string doesn't panic decoding when ,string is specified
// Issue 3450
func TestEmptyString(t *testing.T) {
	type T2 struct {
		Number1 int `json:",string"`
		Number2 int `json:",string"`
	}
	data := `{"Number1":"1", "Number2":""}`
	dec := NewDecoder(strings.NewReader(data))
	var t2 T2
	err := dec.Decode(&t2)
	if err == nil {
		t.Fatal("Decode: did not return error")
	}
	if t2.Number1 != 1 {
		t.Fatal("Decode: did not set Number1")
	}
}

// Test that a null for ,string is not replaced with the previous quoted string (issue 7046).
// It should also not be an error (issue 2540, issue 8587).
func TestNullString(t *testing.T) {
	type T struct {
		A int  `json:",string"`
		B int  `json:",string"`
		C *int `json:",string"`
	}
	data := []byte(`{"A": "1", "B": null, "C": null}`)
	var s T
	s.B = 1
	s.C = new(int)
	*s.C = 2
	err := Unmarshal(data, &s)
	if err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if s.B != 1 || s.C != nil {
		t.Fatalf("after Unmarshal, s.B=%d, s.C=%p, want 1, nil", s.B, s.C)
	}
}

func intp(x int) *int {
	p := new(int)
	*p = x
	return p
}

func intpp(x *int) **int {
	pp := new(*int)
	*pp = x
	return pp
}

var interfaceSetTests = []struct {
	pre  interface{}
	json string
	post interface{}
}{
	{"foo", `"bar"`, "bar"},
	{"foo", `2`, 2.0},
	{"foo", `true`, true},
	{"foo", `null`, nil},

	{nil, `null`, nil},
	{new(int), `null`, nil},
	{(*int)(nil), `null`, nil},
	{new(*int), `null`, new(*int)},
	{(**int)(nil), `null`, nil},
	{intp(1), `null`, nil},
	{intpp(nil), `null`, intpp(nil)},
	{intpp(intp(1)), `null`, intpp(nil)},
}

func TestInterfaceSet(t *testing.T) {
	for _, tt := range interfaceSetTests {
		b := struct{ X interface{} }{tt.pre}
		blob := `{"X":` + tt.json + `}`
		if err := Unmarshal([]byte(blob), &b); err != nil {
			t.Errorf("Unmarshal %#q: %v", blob, err)
			continue
		}
		if !reflect.DeepEqual(b.X, tt.post) {
			t.Errorf("Unmarshal %#q into %#v: X=%#v, want %#v", blob, tt.pre, b.X, tt.post)
		}
	}
}

// JSON null values should be ignored for primitives and string values instead of resulting in an error.
// Issue 2540
func TestUnmarshalNulls(t *testing.T) {
	jsonData := []byte(`{
		"Bool"    : null,
		"Int"     : null,
		"Int8"    : null,
		"Int16"   : null,
		"Int32"   : null,
		"Int64"   : null,
		"Uint"    : null,
		"Uint8"   : null,
		"Uint16"  : null,
		"Uint32"  : null,
		"Uint64"  : null,
		"Float32" : null,
		"Float64" : null,
		"String"  : null}`)

	nulls := All{
		Bool:    true,
		Int:     2,
		Int8:    3,
		Int16:   4,
		Int32:   5,
		Int64:   6,
		Uint:    7,
		Uint8:   8,
		Uint16:  9,
		Uint32:  10,
		Uint64:  11,
		Float32: 12.1,
		Float64: 13.1,
		String:  "14"}

	err := Unmarshal(jsonData, &nulls)
	if err != nil {
		t.Errorf("Unmarshal of null values failed: %v", err)
	}
	if !nulls.Bool || nulls.Int != 2 || nulls.Int8 != 3 || nulls.Int16 != 4 || nulls.Int32 != 5 || nulls.Int64 != 6 ||
		nulls.Uint != 7 || nulls.Uint8 != 8 || nulls.Uint16 != 9 || nulls.Uint32 != 10 || nulls.Uint64 != 11 ||
		nulls.Float32 != 12.1 || nulls.Float64 != 13.1 || nulls.String != "14" {

		t.Errorf("Unmarshal of null values affected primitives")
	}
}

func TestStringKind(t *testing.T) {
	type stringKind string

	var m1, m2 map[stringKind]int
	m1 = map[stringKind]int{
		"foo": 42,
	}

	data, err := Marshal(m1)
	if err != nil {
		t.Errorf("Unexpected error marshalling: %v", err)
	}

	err = Unmarshal(data, &m2)
	if err != nil {
		t.Errorf("Unexpected error unmarshalling: %v", err)
	}

	if !reflect.DeepEqual(m1, m2) {
		t.Error("Items should be equal after encoding and then decoding")
	}

}

var decodeTypeErrorTests = []struct {
	dest interface{}
	src  string
}{
	{new(string), `{"user": "name"}`}, // issue 4628.
	{new(error), `{}`},                // issue 4222
	{new(error), `[]`},
	{new(error), `""`},
	{new(error), `123`},
	{new(error), `true`},
}

func TestUnmarshalTypeError(t *testing.T) {
	for _, item := range decodeTypeErrorTests {
		err := Unmarshal([]byte(item.src), item.dest)
		if _, ok := err.(*UnmarshalTypeError); !ok {
			t.Errorf("expected type error for Unmarshal(%q, type %T): got %T",
				item.src, item.dest, err)
		}
	}
}

var unmarshalSyntaxTests = []string{
	"tru",
	"fals",
	"nul",
	"123e",
	`"hello`,
	`[1,2,3`,
	`{"key":1`,
	`{"key":1,`,
}

func TestUnmarshalSyntax(t *testing.T) {
	var x interface{}
	for _, src := range unmarshalSyntaxTests {
		err := Unmarshal([]byte(src), &x)
		if _, ok := err.(*SyntaxError); !ok {
			t.Errorf("expected syntax error for Unmarshal(%q): got %T", src, err)
		}
	}
}

// Test handling of unexported fields that should be ignored.
// Issue 4660
type unexportedFields struct {
	Name string
	m    map[string]interface{} `json:"-"`
	m2   map[string]interface{} `json:"abcd"`
}

func TestUnmarshalUnexported(t *testing.T) {
	input := `{"Name": "Bob", "m": {"x": 123}, "m2": {"y": 456}, "abcd": {"z": 789}}`
	want := &unexportedFields{Name: "Bob"}

	out := &unexportedFields{}
	err := Unmarshal([]byte(input), out)
	if err != nil {
		t.Errorf("got error %v, expected nil", err)
	}
	if !reflect.DeepEqual(out, want) {
		t.Errorf("got %q, want %q", out, want)
	}
}

// Time3339 is a time.Time which encodes to and from JSON
// as an RFC 3339 time in UTC.
type Time3339 time.Time

func (t *Time3339) UnmarshalJSON(b []byte) error {
	if len(b) < 2 || b[0] != '"' || b[len(b)-1] != '"' {
		return fmt.Errorf("types: failed to unmarshal non-string value %q as an RFC 3339 time", b)
	}
	tm, err := time.Parse(time.RFC3339, string(b[1:len(b)-1]))
	if err != nil {
		return err
	}
	*t = Time3339(tm)
	return nil
}

func TestUnmarshalJSONLiteralError(t *testing.T) {
	var t3 Time3339
	err := Unmarshal([]byte(`"0000-00-00T00:00:00Z"`), &t3)
	if err == nil {
		t.Fatalf("expected error; got time %v", time.Time(t3))
	}
	if !strings.Contains(err.Error(), "range") {
		t.Errorf("got err = %v; want out of range error", err)
	}
}

// Test that extra object elements in an array do not result in a
// "data changing underfoot" error.
// Issue 3717
func TestSkipArrayObjects(t *testing.T) {
	json := `[{}]`
	var dest [0]interface{}

	err := Unmarshal([]byte(json), &dest)
	if err != nil {
		t.Errorf("got error %q, want nil", err)
	}
}

// Test semantics of pre-filled struct fields and pre-filled map fields.
// Issue 4900.
func TestPrefilled(t *testing.T) {
	ptrToMap := func(m map[string]interface{}) *map[string]interface{} { return &m }

	// Values here change, cannot reuse table across runs.
	var prefillTests = []struct {
		in  string
		ptr interface{}
		out interface{}
	}{
		{
			in:  `{"X": 1, "Y": 2}`,
			ptr: &XYZ{X: float32(3), Y: int16(4), Z: 1.5},
			out: &XYZ{X: float64(1), Y: float64(2), Z: 1.5},
		},
		{
			in:  `{"X": 1, "Y": 2}`,
			ptr: ptrToMap(map[string]interface{}{"X": float32(3), "Y": int16(4), "Z": 1.5}),
			out: ptrToMap(map[string]interface{}{"X": float64(1), "Y": float64(2), "Z": 1.5}),
		},
	}

	for _, tt := range prefillTests {
		ptrstr := fmt.Sprintf("%v", tt.ptr)
		err := Unmarshal([]byte(tt.in), tt.ptr) // tt.ptr edited here
		if err != nil {
			t.Errorf("Unmarshal: %v", err)
		}
		if !reflect.DeepEqual(tt.ptr, tt.out) {
			t.Errorf("Unmarshal(%#q, %s): have %v, want %v", tt.in, ptrstr, tt.ptr, tt.out)
		}
	}
}

var invalidUnmarshalTests = []struct {
	v    interface{}
	want string
}{
	{nil, "json: Unmarshal(nil)"},
	{struct{}{}, "json: Unmarshal(non-pointer struct {})"},
	{(*int)(nil), "json: Unmarshal(nil *int)"},
}

func TestInvalidUnmarshal(t *testing.T) {
	buf := []byte(`{"a":"1"}`)
	for _, tt := range invalidUnmarshalTests {
		err := Unmarshal(buf, tt.v)
		if err == nil {
			t.Errorf("Unmarshal expecting error, got nil")
			continue
		}
		if got := err.Error(); got != tt.want {
			t.Errorf("Unmarshal = %q; want %q", got, tt.want)
		}
	}
}
