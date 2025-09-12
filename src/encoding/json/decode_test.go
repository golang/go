// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.jsonv2

package json

import (
	"bytes"
	"encoding"
	"errors"
	"fmt"
	"image"
	"io"
	"maps"
	"math"
	"math/big"
	"net"
	"reflect"
	"slices"
	"strconv"
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
	F1 any
	F2 int32
	F3 Number
	F4 *VOuter
}

type VOuter struct {
	V V
}

type W struct {
	S SS
}

type P struct {
	PP PP
}

type PP struct {
	T  T
	Ts []T
}

type SS string

func (*SS) UnmarshalJSON(data []byte) error {
	return &UnmarshalTypeError{Value: "number", Type: reflect.TypeFor[SS]()}
}

type TAlias T

func (tt *TAlias) UnmarshalJSON(data []byte) error {
	t := T{}
	if err := Unmarshal(data, &t); err != nil {
		return err
	}
	*tt = TAlias(t)
	return nil
}

type TOuter struct {
	T TAlias
}

// ifaceNumAsFloat64/ifaceNumAsNumber are used to test unmarshaling with and
// without UseNumber
var ifaceNumAsFloat64 = map[string]any{
	"k1": float64(1),
	"k2": "s",
	"k3": []any{float64(1), float64(2.0), float64(3e-3)},
	"k4": map[string]any{"kk1": "s", "kk2": float64(2)},
}

var ifaceNumAsNumber = map[string]any{
	"k1": Number("1"),
	"k2": "s",
	"k3": []any{Number("1"), Number("2.0"), Number("3e-3")},
	"k4": map[string]any{"kk1": "s", "kk2": Number("2")},
}

type tx struct {
	x int
}

type u8 uint8

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
	A, B string
}

// needed for re-marshaling tests
func (u unmarshalerText) MarshalText() ([]byte, error) {
	return []byte(u.A + ":" + u.B), nil
}

func (u *unmarshalerText) UnmarshalText(b []byte) error {
	pos := bytes.IndexByte(b, ':')
	if pos == -1 {
		return errors.New("missing separator")
	}
	u.A, u.B = string(b[:pos]), string(b[pos+1:])
	return nil
}

var _ encoding.TextUnmarshaler = (*unmarshalerText)(nil)

type ustructText struct {
	M unmarshalerText
}

// u8marshal is an integer type that can marshal/unmarshal itself.
type u8marshal uint8

func (u8 u8marshal) MarshalText() ([]byte, error) {
	return []byte(fmt.Sprintf("u%d", u8)), nil
}

var errMissingU8Prefix = errors.New("missing 'u' prefix")

func (u8 *u8marshal) UnmarshalText(b []byte) error {
	if !bytes.HasPrefix(b, []byte{'u'}) {
		return errMissingU8Prefix
	}
	n, err := strconv.Atoi(string(b[1:]))
	if err != nil {
		return err
	}
	*u8 = u8marshal(n)
	return nil
}

var _ encoding.TextUnmarshaler = (*u8marshal)(nil)

var (
	umtrue   = unmarshaler{true}
	umslice  = []unmarshaler{{true}}
	umstruct = ustruct{unmarshaler{true}}

	umtrueXY   = unmarshalerText{"x", "y"}
	umsliceXY  = []unmarshalerText{{"x", "y"}}
	umstructXY = ustructText{unmarshalerText{"x", "y"}}

	ummapXY = map[unmarshalerText]bool{{"x", "y"}: true}
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
	embed   // contains exported field
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

type embed struct {
	Q int
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

type Ambig struct {
	// Given "hello", the first match should win.
	First  int `json:"HELLO"`
	Second int `json:"Hello"`
}

type XYZ struct {
	X any
	Y any
	Z any
}

type unexportedWithMethods struct{}

func (unexportedWithMethods) F() {}

type byteWithMarshalJSON byte

func (b byteWithMarshalJSON) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"Z%.2x"`, byte(b))), nil
}

func (b *byteWithMarshalJSON) UnmarshalJSON(data []byte) error {
	if len(data) != 5 || data[0] != '"' || data[1] != 'Z' || data[4] != '"' {
		return fmt.Errorf("bad quoted string")
	}
	i, err := strconv.ParseInt(string(data[2:4]), 16, 8)
	if err != nil {
		return fmt.Errorf("bad hex")
	}
	*b = byteWithMarshalJSON(i)
	return nil
}

type byteWithPtrMarshalJSON byte

func (b *byteWithPtrMarshalJSON) MarshalJSON() ([]byte, error) {
	return byteWithMarshalJSON(*b).MarshalJSON()
}

func (b *byteWithPtrMarshalJSON) UnmarshalJSON(data []byte) error {
	return (*byteWithMarshalJSON)(b).UnmarshalJSON(data)
}

type byteWithMarshalText byte

func (b byteWithMarshalText) MarshalText() ([]byte, error) {
	return []byte(fmt.Sprintf(`Z%.2x`, byte(b))), nil
}

func (b *byteWithMarshalText) UnmarshalText(data []byte) error {
	if len(data) != 3 || data[0] != 'Z' {
		return fmt.Errorf("bad quoted string")
	}
	i, err := strconv.ParseInt(string(data[1:3]), 16, 8)
	if err != nil {
		return fmt.Errorf("bad hex")
	}
	*b = byteWithMarshalText(i)
	return nil
}

type byteWithPtrMarshalText byte

func (b *byteWithPtrMarshalText) MarshalText() ([]byte, error) {
	return byteWithMarshalText(*b).MarshalText()
}

func (b *byteWithPtrMarshalText) UnmarshalText(data []byte) error {
	return (*byteWithMarshalText)(b).UnmarshalText(data)
}

type intWithMarshalJSON int

func (b intWithMarshalJSON) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"Z%.2x"`, int(b))), nil
}

func (b *intWithMarshalJSON) UnmarshalJSON(data []byte) error {
	if len(data) != 5 || data[0] != '"' || data[1] != 'Z' || data[4] != '"' {
		return fmt.Errorf("bad quoted string")
	}
	i, err := strconv.ParseInt(string(data[2:4]), 16, 8)
	if err != nil {
		return fmt.Errorf("bad hex")
	}
	*b = intWithMarshalJSON(i)
	return nil
}

type intWithPtrMarshalJSON int

func (b *intWithPtrMarshalJSON) MarshalJSON() ([]byte, error) {
	return intWithMarshalJSON(*b).MarshalJSON()
}

func (b *intWithPtrMarshalJSON) UnmarshalJSON(data []byte) error {
	return (*intWithMarshalJSON)(b).UnmarshalJSON(data)
}

type intWithMarshalText int

func (b intWithMarshalText) MarshalText() ([]byte, error) {
	return []byte(fmt.Sprintf(`Z%.2x`, int(b))), nil
}

func (b *intWithMarshalText) UnmarshalText(data []byte) error {
	if len(data) != 3 || data[0] != 'Z' {
		return fmt.Errorf("bad quoted string")
	}
	i, err := strconv.ParseInt(string(data[1:3]), 16, 8)
	if err != nil {
		return fmt.Errorf("bad hex")
	}
	*b = intWithMarshalText(i)
	return nil
}

type intWithPtrMarshalText int

func (b *intWithPtrMarshalText) MarshalText() ([]byte, error) {
	return intWithMarshalText(*b).MarshalText()
}

func (b *intWithPtrMarshalText) UnmarshalText(data []byte) error {
	return (*intWithMarshalText)(b).UnmarshalText(data)
}

type mapStringToStringData struct {
	Data map[string]string `json:"data"`
}

type B struct {
	B bool `json:",string"`
}

type DoublePtr struct {
	I **int
	J **int
}

type NestedUnamed struct{ F struct{ V int } }

var unmarshalTests = []struct {
	CaseName
	in                    string
	ptr                   any // new(type)
	out                   any
	err                   error
	useNumber             bool
	golden                bool
	disallowUnknownFields bool
}{
	// basic types
	{CaseName: Name(""), in: `true`, ptr: new(bool), out: true},
	{CaseName: Name(""), in: `1`, ptr: new(int), out: 1},
	{CaseName: Name(""), in: `1.2`, ptr: new(float64), out: 1.2},
	{CaseName: Name(""), in: `-5`, ptr: new(int16), out: int16(-5)},
	{CaseName: Name(""), in: `2`, ptr: new(Number), out: Number("2"), useNumber: true},
	{CaseName: Name(""), in: `2`, ptr: new(Number), out: Number("2")},
	{CaseName: Name(""), in: `2`, ptr: new(any), out: float64(2.0)},
	{CaseName: Name(""), in: `2`, ptr: new(any), out: Number("2"), useNumber: true},
	{CaseName: Name(""), in: `"a\u1234"`, ptr: new(string), out: "a\u1234"},
	{CaseName: Name(""), in: `"http:\/\/"`, ptr: new(string), out: "http://"},
	{CaseName: Name(""), in: `"g-clef: \uD834\uDD1E"`, ptr: new(string), out: "g-clef: \U0001D11E"},
	{CaseName: Name(""), in: `"invalid: \uD834x\uDD1E"`, ptr: new(string), out: "invalid: \uFFFDx\uFFFD"},
	{CaseName: Name(""), in: "null", ptr: new(any), out: nil},
	{CaseName: Name(""), in: `{"X": [1,2,3], "Y": 4}`, ptr: new(T), out: T{Y: 4}, err: &UnmarshalTypeError{"array", reflect.TypeFor[string](), 7, "T", "X"}},
	{CaseName: Name(""), in: `{"X": 23}`, ptr: new(T), out: T{}, err: &UnmarshalTypeError{"number", reflect.TypeFor[string](), 8, "T", "X"}},
	{CaseName: Name(""), in: `{"x": 1}`, ptr: new(tx), out: tx{}},
	{CaseName: Name(""), in: `{"x": 1}`, ptr: new(tx), out: tx{}},
	{CaseName: Name(""), in: `{"x": 1}`, ptr: new(tx), err: fmt.Errorf("json: unknown field \"x\""), disallowUnknownFields: true},
	{CaseName: Name(""), in: `{"S": 23}`, ptr: new(W), out: W{}, err: &UnmarshalTypeError{"number", reflect.TypeFor[SS](), 0, "W", "S"}},
	{CaseName: Name(""), in: `{"T": {"X": 23}}`, ptr: new(TOuter), out: TOuter{}, err: &UnmarshalTypeError{"number", reflect.TypeFor[string](), 8, "TOuter", "T.X"}},
	{CaseName: Name(""), in: `{"F1":1,"F2":2,"F3":3}`, ptr: new(V), out: V{F1: float64(1), F2: int32(2), F3: Number("3")}},
	{CaseName: Name(""), in: `{"F1":1,"F2":2,"F3":3}`, ptr: new(V), out: V{F1: Number("1"), F2: int32(2), F3: Number("3")}, useNumber: true},
	{CaseName: Name(""), in: `{"k1":1,"k2":"s","k3":[1,2.0,3e-3],"k4":{"kk1":"s","kk2":2}}`, ptr: new(any), out: ifaceNumAsFloat64},
	{CaseName: Name(""), in: `{"k1":1,"k2":"s","k3":[1,2.0,3e-3],"k4":{"kk1":"s","kk2":2}}`, ptr: new(any), out: ifaceNumAsNumber, useNumber: true},

	// raw values with whitespace
	{CaseName: Name(""), in: "\n true ", ptr: new(bool), out: true},
	{CaseName: Name(""), in: "\t 1 ", ptr: new(int), out: 1},
	{CaseName: Name(""), in: "\r 1.2 ", ptr: new(float64), out: 1.2},
	{CaseName: Name(""), in: "\t -5 \n", ptr: new(int16), out: int16(-5)},
	{CaseName: Name(""), in: "\t \"a\\u1234\" \n", ptr: new(string), out: "a\u1234"},

	// Z has a "-" tag.
	{CaseName: Name(""), in: `{"Y": 1, "Z": 2}`, ptr: new(T), out: T{Y: 1}},
	{CaseName: Name(""), in: `{"Y": 1, "Z": 2}`, ptr: new(T), out: T{Y: 1}, err: fmt.Errorf("json: unknown field \"Z\""), disallowUnknownFields: true},

	{CaseName: Name(""), in: `{"alpha": "abc", "alphabet": "xyz"}`, ptr: new(U), out: U{Alphabet: "abc"}},
	{CaseName: Name(""), in: `{"alpha": "abc", "alphabet": "xyz"}`, ptr: new(U), out: U{Alphabet: "abc"}, err: fmt.Errorf("json: unknown field \"alphabet\""), disallowUnknownFields: true},
	{CaseName: Name(""), in: `{"alpha": "abc"}`, ptr: new(U), out: U{Alphabet: "abc"}},
	{CaseName: Name(""), in: `{"alphabet": "xyz"}`, ptr: new(U), out: U{}},
	{CaseName: Name(""), in: `{"alphabet": "xyz"}`, ptr: new(U), err: fmt.Errorf("json: unknown field \"alphabet\""), disallowUnknownFields: true},

	// syntax errors
	{CaseName: Name(""), in: ``, ptr: new(any), err: &SyntaxError{"unexpected end of JSON input", 0}},
	{CaseName: Name(""), in: " \n\r\t", ptr: new(any), err: &SyntaxError{"unexpected end of JSON input", 4}},
	{CaseName: Name(""), in: `[2, 3`, ptr: new(any), err: &SyntaxError{"unexpected end of JSON input", 5}},
	{CaseName: Name(""), in: `{"X": "foo", "Y"}`, err: &SyntaxError{"invalid character '}' after object key", 17}},
	{CaseName: Name(""), in: `[1, 2, 3+]`, err: &SyntaxError{"invalid character '+' after array element", 9}},
	{CaseName: Name(""), in: `{"X":12x}`, err: &SyntaxError{"invalid character 'x' after object key:value pair", 8}, useNumber: true},
	{CaseName: Name(""), in: `{"F3": -}`, ptr: new(V), err: &SyntaxError{"invalid character '}' in numeric literal", 9}},

	// raw value errors
	{CaseName: Name(""), in: "\x01 42", err: &SyntaxError{"invalid character '\\x01' looking for beginning of value", 1}},
	{CaseName: Name(""), in: " 42 \x01", err: &SyntaxError{"invalid character '\\x01' after top-level value", 5}},
	{CaseName: Name(""), in: "\x01 true", err: &SyntaxError{"invalid character '\\x01' looking for beginning of value", 1}},
	{CaseName: Name(""), in: " false \x01", err: &SyntaxError{"invalid character '\\x01' after top-level value", 8}},
	{CaseName: Name(""), in: "\x01 1.2", err: &SyntaxError{"invalid character '\\x01' looking for beginning of value", 1}},
	{CaseName: Name(""), in: " 3.4 \x01", err: &SyntaxError{"invalid character '\\x01' after top-level value", 6}},
	{CaseName: Name(""), in: "\x01 \"string\"", err: &SyntaxError{"invalid character '\\x01' looking for beginning of value", 1}},
	{CaseName: Name(""), in: " \"string\" \x01", err: &SyntaxError{"invalid character '\\x01' after top-level value", 11}},

	// array tests
	{CaseName: Name(""), in: `[1, 2, 3]`, ptr: new([3]int), out: [3]int{1, 2, 3}},
	{CaseName: Name(""), in: `[1, 2, 3]`, ptr: new([1]int), out: [1]int{1}},
	{CaseName: Name(""), in: `[1, 2, 3]`, ptr: new([5]int), out: [5]int{1, 2, 3, 0, 0}},
	{CaseName: Name(""), in: `[1, 2, 3]`, ptr: new(MustNotUnmarshalJSON), err: errors.New("MustNotUnmarshalJSON was used")},

	// empty array to interface test
	{CaseName: Name(""), in: `[]`, ptr: new([]any), out: []any{}},
	{CaseName: Name(""), in: `null`, ptr: new([]any), out: []any(nil)},
	{CaseName: Name(""), in: `{"T":[]}`, ptr: new(map[string]any), out: map[string]any{"T": []any{}}},
	{CaseName: Name(""), in: `{"T":null}`, ptr: new(map[string]any), out: map[string]any{"T": any(nil)}},

	// composite tests
	{CaseName: Name(""), in: allValueIndent, ptr: new(All), out: allValue},
	{CaseName: Name(""), in: allValueCompact, ptr: new(All), out: allValue},
	{CaseName: Name(""), in: allValueIndent, ptr: new(*All), out: &allValue},
	{CaseName: Name(""), in: allValueCompact, ptr: new(*All), out: &allValue},
	{CaseName: Name(""), in: pallValueIndent, ptr: new(All), out: pallValue},
	{CaseName: Name(""), in: pallValueCompact, ptr: new(All), out: pallValue},
	{CaseName: Name(""), in: pallValueIndent, ptr: new(*All), out: &pallValue},
	{CaseName: Name(""), in: pallValueCompact, ptr: new(*All), out: &pallValue},

	// unmarshal interface test
	{CaseName: Name(""), in: `{"T":false}`, ptr: new(unmarshaler), out: umtrue}, // use "false" so test will fail if custom unmarshaler is not called
	{CaseName: Name(""), in: `{"T":false}`, ptr: new(*unmarshaler), out: &umtrue},
	{CaseName: Name(""), in: `[{"T":false}]`, ptr: new([]unmarshaler), out: umslice},
	{CaseName: Name(""), in: `[{"T":false}]`, ptr: new(*[]unmarshaler), out: &umslice},
	{CaseName: Name(""), in: `{"M":{"T":"x:y"}}`, ptr: new(ustruct), out: umstruct},

	// UnmarshalText interface test
	{CaseName: Name(""), in: `"x:y"`, ptr: new(unmarshalerText), out: umtrueXY},
	{CaseName: Name(""), in: `"x:y"`, ptr: new(*unmarshalerText), out: &umtrueXY},
	{CaseName: Name(""), in: `["x:y"]`, ptr: new([]unmarshalerText), out: umsliceXY},
	{CaseName: Name(""), in: `["x:y"]`, ptr: new(*[]unmarshalerText), out: &umsliceXY},
	{CaseName: Name(""), in: `{"M":"x:y"}`, ptr: new(ustructText), out: umstructXY},

	// integer-keyed map test
	{
		CaseName: Name(""),
		in:       `{"-1":"a","0":"b","1":"c"}`,
		ptr:      new(map[int]string),
		out:      map[int]string{-1: "a", 0: "b", 1: "c"},
	},
	{
		CaseName: Name(""),
		in:       `{"0":"a","10":"c","9":"b"}`,
		ptr:      new(map[u8]string),
		out:      map[u8]string{0: "a", 9: "b", 10: "c"},
	},
	{
		CaseName: Name(""),
		in:       `{"-9223372036854775808":"min","9223372036854775807":"max"}`,
		ptr:      new(map[int64]string),
		out:      map[int64]string{math.MinInt64: "min", math.MaxInt64: "max"},
	},
	{
		CaseName: Name(""),
		in:       `{"18446744073709551615":"max"}`,
		ptr:      new(map[uint64]string),
		out:      map[uint64]string{math.MaxUint64: "max"},
	},
	{
		CaseName: Name(""),
		in:       `{"0":false,"10":true}`,
		ptr:      new(map[uintptr]bool),
		out:      map[uintptr]bool{0: false, 10: true},
	},

	// Check that MarshalText and UnmarshalText take precedence
	// over default integer handling in map keys.
	{
		CaseName: Name(""),
		in:       `{"u2":4}`,
		ptr:      new(map[u8marshal]int),
		out:      map[u8marshal]int{2: 4},
	},
	{
		CaseName: Name(""),
		in:       `{"2":4}`,
		ptr:      new(map[u8marshal]int),
		out:      map[u8marshal]int{},
		err:      errMissingU8Prefix,
	},

	// integer-keyed map errors
	{
		CaseName: Name(""),
		in:       `{"abc":"abc"}`,
		ptr:      new(map[int]string),
		out:      map[int]string{},
		err:      &UnmarshalTypeError{Value: "number abc", Type: reflect.TypeFor[int](), Offset: 2},
	},
	{
		CaseName: Name(""),
		in:       `{"256":"abc"}`,
		ptr:      new(map[uint8]string),
		out:      map[uint8]string{},
		err:      &UnmarshalTypeError{Value: "number 256", Type: reflect.TypeFor[uint8](), Offset: 2},
	},
	{
		CaseName: Name(""),
		in:       `{"128":"abc"}`,
		ptr:      new(map[int8]string),
		out:      map[int8]string{},
		err:      &UnmarshalTypeError{Value: "number 128", Type: reflect.TypeFor[int8](), Offset: 2},
	},
	{
		CaseName: Name(""),
		in:       `{"-1":"abc"}`,
		ptr:      new(map[uint8]string),
		out:      map[uint8]string{},
		err:      &UnmarshalTypeError{Value: "number -1", Type: reflect.TypeFor[uint8](), Offset: 2},
	},
	{
		CaseName: Name(""),
		in:       `{"F":{"a":2,"3":4}}`,
		ptr:      new(map[string]map[int]int),
		out:      map[string]map[int]int{"F": {3: 4}},
		err:      &UnmarshalTypeError{Value: "number a", Type: reflect.TypeFor[int](), Offset: 7},
	},
	{
		CaseName: Name(""),
		in:       `{"F":{"a":2,"3":4}}`,
		ptr:      new(map[string]map[uint]int),
		out:      map[string]map[uint]int{"F": {3: 4}},
		err:      &UnmarshalTypeError{Value: "number a", Type: reflect.TypeFor[uint](), Offset: 7},
	},

	// Map keys can be encoding.TextUnmarshalers.
	{CaseName: Name(""), in: `{"x:y":true}`, ptr: new(map[unmarshalerText]bool), out: ummapXY},
	// If multiple values for the same key exists, only the most recent value is used.
	{CaseName: Name(""), in: `{"x:y":false,"x:y":true}`, ptr: new(map[unmarshalerText]bool), out: ummapXY},

	{
		CaseName: Name(""),
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
			"Z": 17,
			"Q": 18
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
			embed: embed{
				Q: 18,
			},
		},
	},
	{
		CaseName: Name(""),
		in:       `{"hello": 1}`,
		ptr:      new(Ambig),
		out:      Ambig{First: 1},
	},

	{
		CaseName: Name(""),
		in:       `{"X": 1,"Y":2}`,
		ptr:      new(S5),
		out:      S5{S8: S8{S9: S9{Y: 2}}},
	},
	{
		CaseName:              Name(""),
		in:                    `{"X": 1,"Y":2}`,
		ptr:                   new(S5),
		out:                   S5{S8: S8{S9{Y: 2}}},
		err:                   fmt.Errorf("json: unknown field \"X\""),
		disallowUnknownFields: true,
	},
	{
		CaseName: Name(""),
		in:       `{"X": 1,"Y":2}`,
		ptr:      new(S10),
		out:      S10{S13: S13{S8: S8{S9: S9{Y: 2}}}},
	},
	{
		CaseName:              Name(""),
		in:                    `{"X": 1,"Y":2}`,
		ptr:                   new(S10),
		out:                   S10{S13: S13{S8{S9{Y: 2}}}},
		err:                   fmt.Errorf("json: unknown field \"X\""),
		disallowUnknownFields: true,
	},
	{
		CaseName: Name(""),
		in:       `{"I": 0, "I": null, "J": null}`,
		ptr:      new(DoublePtr),
		out:      DoublePtr{I: nil, J: nil},
	},

	// invalid UTF-8 is coerced to valid UTF-8.
	{
		CaseName: Name(""),
		in:       "\"hello\xffworld\"",
		ptr:      new(string),
		out:      "hello\ufffdworld",
	},
	{
		CaseName: Name(""),
		in:       "\"hello\xc2\xc2world\"",
		ptr:      new(string),
		out:      "hello\ufffd\ufffdworld",
	},
	{
		CaseName: Name(""),
		in:       "\"hello\xc2\xffworld\"",
		ptr:      new(string),
		out:      "hello\ufffd\ufffdworld",
	},
	{
		CaseName: Name(""),
		in:       "\"hello\\ud800world\"",
		ptr:      new(string),
		out:      "hello\ufffdworld",
	},
	{
		CaseName: Name(""),
		in:       "\"hello\\ud800\\ud800world\"",
		ptr:      new(string),
		out:      "hello\ufffd\ufffdworld",
	},
	{
		CaseName: Name(""),
		in:       "\"hello\\ud800\\ud800world\"",
		ptr:      new(string),
		out:      "hello\ufffd\ufffdworld",
	},
	{
		CaseName: Name(""),
		in:       "\"hello\xed\xa0\x80\xed\xb0\x80world\"",
		ptr:      new(string),
		out:      "hello\ufffd\ufffd\ufffd\ufffd\ufffd\ufffdworld",
	},

	// Used to be issue 8305, but time.Time implements encoding.TextUnmarshaler so this works now.
	{
		CaseName: Name(""),
		in:       `{"2009-11-10T23:00:00Z": "hello world"}`,
		ptr:      new(map[time.Time]string),
		out:      map[time.Time]string{time.Date(2009, 11, 10, 23, 0, 0, 0, time.UTC): "hello world"},
	},

	// issue 8305
	{
		CaseName: Name(""),
		in:       `{"2009-11-10T23:00:00Z": "hello world"}`,
		ptr:      new(map[Point]string),
		err:      &UnmarshalTypeError{Value: "object", Type: reflect.TypeFor[map[Point]string](), Offset: 1},
	},
	{
		CaseName: Name(""),
		in:       `{"asdf": "hello world"}`,
		ptr:      new(map[unmarshaler]string),
		err:      &UnmarshalTypeError{Value: "object", Type: reflect.TypeFor[map[unmarshaler]string](), Offset: 1},
	},

	// related to issue 13783.
	// Go 1.7 changed marshaling a slice of typed byte to use the methods on the byte type,
	// similar to marshaling a slice of typed int.
	// These tests check that, assuming the byte type also has valid decoding methods,
	// either the old base64 string encoding or the new per-element encoding can be
	// successfully unmarshaled. The custom unmarshalers were accessible in earlier
	// versions of Go, even though the custom marshaler was not.
	{
		CaseName: Name(""),
		in:       `"AQID"`,
		ptr:      new([]byteWithMarshalJSON),
		out:      []byteWithMarshalJSON{1, 2, 3},
	},
	{
		CaseName: Name(""),
		in:       `["Z01","Z02","Z03"]`,
		ptr:      new([]byteWithMarshalJSON),
		out:      []byteWithMarshalJSON{1, 2, 3},
		golden:   true,
	},
	{
		CaseName: Name(""),
		in:       `"AQID"`,
		ptr:      new([]byteWithMarshalText),
		out:      []byteWithMarshalText{1, 2, 3},
	},
	{
		CaseName: Name(""),
		in:       `["Z01","Z02","Z03"]`,
		ptr:      new([]byteWithMarshalText),
		out:      []byteWithMarshalText{1, 2, 3},
		golden:   true,
	},
	{
		CaseName: Name(""),
		in:       `"AQID"`,
		ptr:      new([]byteWithPtrMarshalJSON),
		out:      []byteWithPtrMarshalJSON{1, 2, 3},
	},
	{
		CaseName: Name(""),
		in:       `["Z01","Z02","Z03"]`,
		ptr:      new([]byteWithPtrMarshalJSON),
		out:      []byteWithPtrMarshalJSON{1, 2, 3},
		golden:   true,
	},
	{
		CaseName: Name(""),
		in:       `"AQID"`,
		ptr:      new([]byteWithPtrMarshalText),
		out:      []byteWithPtrMarshalText{1, 2, 3},
	},
	{
		CaseName: Name(""),
		in:       `["Z01","Z02","Z03"]`,
		ptr:      new([]byteWithPtrMarshalText),
		out:      []byteWithPtrMarshalText{1, 2, 3},
		golden:   true,
	},

	// ints work with the marshaler but not the base64 []byte case
	{
		CaseName: Name(""),
		in:       `["Z01","Z02","Z03"]`,
		ptr:      new([]intWithMarshalJSON),
		out:      []intWithMarshalJSON{1, 2, 3},
		golden:   true,
	},
	{
		CaseName: Name(""),
		in:       `["Z01","Z02","Z03"]`,
		ptr:      new([]intWithMarshalText),
		out:      []intWithMarshalText{1, 2, 3},
		golden:   true,
	},
	{
		CaseName: Name(""),
		in:       `["Z01","Z02","Z03"]`,
		ptr:      new([]intWithPtrMarshalJSON),
		out:      []intWithPtrMarshalJSON{1, 2, 3},
		golden:   true,
	},
	{
		CaseName: Name(""),
		in:       `["Z01","Z02","Z03"]`,
		ptr:      new([]intWithPtrMarshalText),
		out:      []intWithPtrMarshalText{1, 2, 3},
		golden:   true,
	},

	{CaseName: Name(""), in: `0.000001`, ptr: new(float64), out: 0.000001, golden: true},
	{CaseName: Name(""), in: `1e-7`, ptr: new(float64), out: 1e-7, golden: true},
	{CaseName: Name(""), in: `100000000000000000000`, ptr: new(float64), out: 100000000000000000000.0, golden: true},
	{CaseName: Name(""), in: `1e+21`, ptr: new(float64), out: 1e21, golden: true},
	{CaseName: Name(""), in: `-0.000001`, ptr: new(float64), out: -0.000001, golden: true},
	{CaseName: Name(""), in: `-1e-7`, ptr: new(float64), out: -1e-7, golden: true},
	{CaseName: Name(""), in: `-100000000000000000000`, ptr: new(float64), out: -100000000000000000000.0, golden: true},
	{CaseName: Name(""), in: `-1e+21`, ptr: new(float64), out: -1e21, golden: true},
	{CaseName: Name(""), in: `999999999999999900000`, ptr: new(float64), out: 999999999999999900000.0, golden: true},
	{CaseName: Name(""), in: `9007199254740992`, ptr: new(float64), out: 9007199254740992.0, golden: true},
	{CaseName: Name(""), in: `9007199254740993`, ptr: new(float64), out: 9007199254740992.0, golden: false},

	{
		CaseName: Name(""),
		in:       `{"V": {"F2": "hello"}}`,
		ptr:      new(VOuter),
		err: &UnmarshalTypeError{
			Value:  "string",
			Struct: "V",
			Field:  "V.F2",
			Type:   reflect.TypeFor[int32](),
			Offset: 20,
		},
	},
	{
		CaseName: Name(""),
		in:       `{"V": {"F4": {}, "F2": "hello"}}`,
		ptr:      new(VOuter),
		out:      VOuter{V: V{F4: &VOuter{}}},
		err: &UnmarshalTypeError{
			Value:  "string",
			Struct: "V",
			Field:  "V.F2",
			Type:   reflect.TypeFor[int32](),
			Offset: 30,
		},
	},

	{
		CaseName: Name(""),
		in:       `{"Level1a": "hello"}`,
		ptr:      new(Top),
		out:      Top{Embed0a: &Embed0a{}},
		err: &UnmarshalTypeError{
			Value:  "string",
			Struct: "Top",
			Field:  "Embed0a.Level1a",
			Type:   reflect.TypeFor[int](),
			Offset: 19,
		},
	},

	// issue 15146.
	// invalid inputs in wrongStringTests below.
	{CaseName: Name(""), in: `{"B":"true"}`, ptr: new(B), out: B{true}, golden: true},
	{CaseName: Name(""), in: `{"B":"false"}`, ptr: new(B), out: B{false}, golden: true},
	{CaseName: Name(""), in: `{"B": "maybe"}`, ptr: new(B), err: errors.New(`json: invalid use of ,string struct tag, trying to unmarshal "maybe" into bool`)},
	{CaseName: Name(""), in: `{"B": "tru"}`, ptr: new(B), err: errors.New(`json: invalid use of ,string struct tag, trying to unmarshal "tru" into bool`)},
	{CaseName: Name(""), in: `{"B": "False"}`, ptr: new(B), err: errors.New(`json: invalid use of ,string struct tag, trying to unmarshal "False" into bool`)},
	{CaseName: Name(""), in: `{"B": "null"}`, ptr: new(B), out: B{false}},
	{CaseName: Name(""), in: `{"B": "nul"}`, ptr: new(B), err: errors.New(`json: invalid use of ,string struct tag, trying to unmarshal "nul" into bool`)},
	{CaseName: Name(""), in: `{"B": [2, 3]}`, ptr: new(B), err: errors.New(`json: invalid use of ,string struct tag, trying to unmarshal unquoted value into bool`)},

	// additional tests for disallowUnknownFields
	{
		CaseName: Name(""),
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
			"Z": 17,
			"Q": 18,
			"extra": true
		}`,
		ptr: new(Top),
		out: Top{
			Level0: 1,
			Embed0: Embed0{
				Level1b: 2,
				Level1c: 3,
			},
			Embed0a: &Embed0a{Level1a: 5, Level1b: 6},
			Embed0b: &Embed0b{Level1a: 8, Level1b: 9, Level1c: 10, Level1d: 11, Level1e: 12},
			Loop: Loop{
				Loop1: 13,
				Loop2: 14,
				Loop:  nil,
			},
			Embed0p: Embed0p{
				Point: image.Point{
					X: 15,
					Y: 16,
				},
			},
			Embed0q: Embed0q{Point: Point{Z: 17}},
			embed:   embed{Q: 18},
		},
		err:                   fmt.Errorf("json: unknown field \"extra\""),
		disallowUnknownFields: true,
	},
	{
		CaseName: Name(""),
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
				"x": 12,
				"extra": null
			},
			"Loop1": 13,
			"Loop2": 14,
			"X": 15,
			"Y": 16,
			"Z": 17,
			"Q": 18
		}`,
		ptr: new(Top),
		out: Top{
			Level0: 1,
			Embed0: Embed0{
				Level1b: 2,
				Level1c: 3,
			},
			Embed0a: &Embed0a{Level1a: 5, Level1b: 6},
			Embed0b: &Embed0b{Level1a: 8, Level1b: 9, Level1c: 10, Level1d: 11, Level1e: 12},
			Loop: Loop{
				Loop1: 13,
				Loop2: 14,
				Loop:  nil,
			},
			Embed0p: Embed0p{
				Point: image.Point{
					X: 15,
					Y: 16,
				},
			},
			Embed0q: Embed0q{Point: Point{Z: 17}},
			embed:   embed{Q: 18},
		},
		err:                   fmt.Errorf("json: unknown field \"extra\""),
		disallowUnknownFields: true,
	},
	// issue 26444
	// UnmarshalTypeError without field & struct values
	{
		CaseName: Name(""),
		in:       `{"data":{"test1": "bob", "test2": 123}}`,
		ptr:      new(mapStringToStringData),
		out:      mapStringToStringData{map[string]string{"test1": "bob", "test2": ""}},
		err:      &UnmarshalTypeError{Value: "number", Type: reflect.TypeFor[string](), Offset: 37, Struct: "mapStringToStringData", Field: "data"},
	},
	{
		CaseName: Name(""),
		in:       `{"data":{"test1": 123, "test2": "bob"}}`,
		ptr:      new(mapStringToStringData),
		out:      mapStringToStringData{Data: map[string]string{"test1": "", "test2": "bob"}},
		err:      &UnmarshalTypeError{Value: "number", Type: reflect.TypeFor[string](), Offset: 21, Struct: "mapStringToStringData", Field: "data"},
	},

	// trying to decode JSON arrays or objects via TextUnmarshaler
	{
		CaseName: Name(""),
		in:       `[1, 2, 3]`,
		ptr:      new(MustNotUnmarshalText),
		err:      &UnmarshalTypeError{Value: "array", Type: reflect.TypeFor[*MustNotUnmarshalText](), Offset: 1},
	},
	{
		CaseName: Name(""),
		in:       `{"foo": "bar"}`,
		ptr:      new(MustNotUnmarshalText),
		err:      &UnmarshalTypeError{Value: "object", Type: reflect.TypeFor[*MustNotUnmarshalText](), Offset: 1},
	},
	// #22369
	{
		CaseName: Name(""),
		in:       `{"PP": {"T": {"Y": "bad-type"}}}`,
		ptr:      new(P),
		err: &UnmarshalTypeError{
			Value:  "string",
			Struct: "T",
			Field:  "PP.T.Y",
			Type:   reflect.TypeFor[int](),
			Offset: 29,
		},
	},
	{
		CaseName: Name(""),
		in:       `{"Ts": [{"Y": 1}, {"Y": 2}, {"Y": "bad-type"}]}`,
		ptr:      new(PP),
		out:      PP{Ts: []T{{Y: 1}, {Y: 2}, {Y: 0}}},
		err: &UnmarshalTypeError{
			Value:  "string",
			Struct: "T",
			Field:  "Ts.Y",
			Type:   reflect.TypeFor[int](),
			Offset: 44,
		},
	},
	// #14702
	{
		CaseName: Name(""),
		in:       `invalid`,
		ptr:      new(Number),
		err: &SyntaxError{
			msg:    "invalid character 'i' looking for beginning of value",
			Offset: 1,
		},
	},
	{
		CaseName: Name(""),
		in:       `"invalid"`,
		ptr:      new(Number),
		err:      fmt.Errorf("json: invalid number literal, trying to unmarshal %q into Number", `"invalid"`),
	},
	{
		CaseName: Name(""),
		in:       `{"A":"invalid"}`,
		ptr:      new(struct{ A Number }),
		err:      fmt.Errorf("json: invalid number literal, trying to unmarshal %q into Number", `"invalid"`),
	},
	{
		CaseName: Name(""),
		in:       `{"A":"invalid"}`,
		ptr: new(struct {
			A Number `json:",string"`
		}),
		err: fmt.Errorf("json: invalid use of ,string struct tag, trying to unmarshal %q into json.Number", `invalid`),
	},
	{
		CaseName: Name(""),
		in:       `{"A":"invalid"}`,
		ptr:      new(map[string]Number),
		out:      map[string]Number{},
		err:      fmt.Errorf("json: invalid number literal, trying to unmarshal %q into Number", `"invalid"`),
	},

	{
		CaseName: Name(""),
		in:       `5`,
		ptr:      new(Number),
		out:      Number("5"),
	},
	{
		CaseName: Name(""),
		in:       `"5"`,
		ptr:      new(Number),
		out:      Number("5"),
	},
	{
		CaseName: Name(""),
		in:       `{"N":5}`,
		ptr:      new(struct{ N Number }),
		out:      struct{ N Number }{"5"},
	},
	{
		CaseName: Name(""),
		in:       `{"N":"5"}`,
		ptr:      new(struct{ N Number }),
		out:      struct{ N Number }{"5"},
	},
	{
		CaseName: Name(""),
		in:       `{"N":5}`,
		ptr: new(struct {
			N Number `json:",string"`
		}),
		err: fmt.Errorf("json: invalid use of ,string struct tag, trying to unmarshal unquoted value into json.Number"),
	},
	{
		CaseName: Name(""),
		in:       `{"N":"5"}`,
		ptr: new(struct {
			N Number `json:",string"`
		}),
		out: struct {
			N Number `json:",string"`
		}{"5"},
	},

	// Verify that syntactic errors are immediately fatal,
	// while semantic errors are lazily reported
	// (i.e., allow processing to continue).
	{
		CaseName: Name(""),
		in:       `[1,2,true,4,5}`,
		ptr:      new([]int),
		err:      &SyntaxError{msg: "invalid character '}' after array element", Offset: 14},
	},
	{
		CaseName: Name(""),
		in:       `[1,2,true,4,5]`,
		ptr:      new([]int),
		out:      []int{1, 2, 0, 4, 5},
		err:      &UnmarshalTypeError{Value: "bool", Type: reflect.TypeFor[int](), Offset: 9},
	},

	{
		CaseName: Name("DashComma"),
		in:       `{"-":"hello"}`,
		ptr: new(struct {
			F string `json:"-,"`
		}),
		out: struct {
			F string `json:"-,"`
		}{"hello"},
	},
	{
		CaseName: Name("DashCommaOmitEmpty"),
		in:       `{"-":"hello"}`,
		ptr: new(struct {
			F string `json:"-,omitempty"`
		}),
		out: struct {
			F string `json:"-,omitempty"`
		}{"hello"},
	},

	{
		CaseName: Name("ErrorForNestedUnamed"),
		in:       `{"F":{"V":"s"}}`,
		ptr:      new(NestedUnamed),
		out:      NestedUnamed{},
		err:      &UnmarshalTypeError{Value: "string", Type: reflect.TypeFor[int](), Offset: 13, Field: "F.V"},
	},
	{
		CaseName: Name("ErrorInterface"),
		in:       `1`,
		ptr:      new(error),
		out:      error(nil),
		err:      &UnmarshalTypeError{Value: "number", Type: reflect.TypeFor[error](), Offset: 1},
	},
	{
		CaseName: Name("ErrorChan"),
		in:       `1`,
		ptr:      new(chan int),
		out:      (chan int)(nil),
		err:      &UnmarshalTypeError{Value: "number", Type: reflect.TypeFor[chan int](), Offset: 1},
	},
}

func TestMarshal(t *testing.T) {
	b, err := Marshal(allValue)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if string(b) != allValueCompact {
		t.Errorf("Marshal:")
		diff(t, b, []byte(allValueCompact))
		return
	}

	b, err = Marshal(pallValue)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if string(b) != pallValueCompact {
		t.Errorf("Marshal:")
		diff(t, b, []byte(pallValueCompact))
		return
	}
}

func TestMarshalInvalidUTF8(t *testing.T) {
	tests := []struct {
		CaseName
		in   string
		want string
	}{
		{Name(""), "hello\xffworld", `"hello\ufffdworld"`},
		{Name(""), "", `""`},
		{Name(""), "\xff", `"\ufffd"`},
		{Name(""), "\xff\xff", `"\ufffd\ufffd"`},
		{Name(""), "a\xffb", `"a\ufffdb"`},
		{Name(""), "\xe6\x97\xa5\xe6\x9c\xac\xff\xaa\x9e", `"日本\ufffd\ufffd\ufffd"`},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			got, err := Marshal(tt.in)
			if string(got) != tt.want || err != nil {
				t.Errorf("%s: Marshal(%q):\n\tgot:  (%q, %v)\n\twant: (%q, nil)", tt.Where, tt.in, got, err, tt.want)
			}
		})
	}
}

func TestMarshalNumberZeroVal(t *testing.T) {
	var n Number
	out, err := Marshal(n)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	got := string(out)
	if got != "0" {
		t.Fatalf("Marshal: got %s, want 0", got)
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
		embed: embed{
			Q: 18,
		},
	}
	got, err := Marshal(top)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	want := "{\"Level0\":1,\"Level1b\":2,\"Level1c\":3,\"Level1a\":5,\"LEVEL1B\":6,\"e\":{\"Level1a\":8,\"Level1b\":9,\"Level1c\":10,\"Level1d\":11,\"x\":12},\"Loop1\":13,\"Loop2\":14,\"X\":15,\"Y\":16,\"Z\":17,\"Q\":18}"
	if string(got) != want {
		t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

func equalError(a, b error) bool {
	isJSONError := func(err error) bool {
		switch err.(type) {
		case
			*InvalidUTF8Error,
			*InvalidUnmarshalError,
			*MarshalerError,
			*SyntaxError,
			*UnmarshalFieldError,
			*UnmarshalTypeError,
			*UnsupportedTypeError,
			*UnsupportedValueError:
			return true
		}
		return false
	}

	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if isJSONError(a) || isJSONError(b) {
		return reflect.DeepEqual(a, b) // safe for locally defined error types
	}
	return a.Error() == b.Error()
}

func TestUnmarshal(t *testing.T) {
	for _, tt := range unmarshalTests {
		t.Run(tt.Name, func(t *testing.T) {
			in := []byte(tt.in)
			var scan scanner
			if err := checkValid(in, &scan); err != nil {
				if !equalError(err, tt.err) {
					t.Fatalf("%s: checkValid error:\n\tgot  %#v\n\twant %#v", tt.Where, err, tt.err)
				}
			}
			if tt.ptr == nil {
				return
			}

			typ := reflect.TypeOf(tt.ptr)
			if typ.Kind() != reflect.Pointer {
				t.Fatalf("%s: unmarshalTest.ptr %T is not a pointer type", tt.Where, tt.ptr)
			}
			typ = typ.Elem()

			// v = new(right-type)
			v := reflect.New(typ)

			if !reflect.DeepEqual(tt.ptr, v.Interface()) {
				// There's no reason for ptr to point to non-zero data,
				// as we decode into new(right-type), so the data is
				// discarded.
				// This can easily mean tests that silently don't test
				// what they should. To test decoding into existing
				// data, see TestPrefilled.
				t.Fatalf("%s: unmarshalTest.ptr %#v is not a pointer to a zero value", tt.Where, tt.ptr)
			}

			dec := NewDecoder(bytes.NewReader(in))
			if tt.useNumber {
				dec.UseNumber()
			}
			if tt.disallowUnknownFields {
				dec.DisallowUnknownFields()
			}
			if tt.err != nil && strings.Contains(tt.err.Error(), "unexpected end of JSON input") {
				// In streaming mode, we expect EOF or ErrUnexpectedEOF instead.
				if strings.TrimSpace(tt.in) == "" {
					tt.err = io.EOF
				} else {
					tt.err = io.ErrUnexpectedEOF
				}
			}
			if err := dec.Decode(v.Interface()); !equalError(err, tt.err) {
				t.Fatalf("%s: Decode error:\n\tgot:  %v\n\twant: %v\n\n\tgot:  %#v\n\twant: %#v", tt.Where, err, tt.err, err, tt.err)
			} else if err != nil && tt.out == nil {
				// Initialize tt.out during an error where there are no mutations,
				// so the output is just the zero value of the input type.
				tt.out = reflect.Zero(v.Elem().Type()).Interface()
			}
			if got := v.Elem().Interface(); !reflect.DeepEqual(got, tt.out) {
				gotJSON, _ := Marshal(got)
				wantJSON, _ := Marshal(tt.out)
				t.Fatalf("%s: Decode:\n\tgot:  %#+v\n\twant: %#+v\n\n\tgotJSON:  %s\n\twantJSON: %s", tt.Where, got, tt.out, gotJSON, wantJSON)
			}

			// Check round trip also decodes correctly.
			if tt.err == nil {
				enc, err := Marshal(v.Interface())
				if err != nil {
					t.Fatalf("%s: Marshal error after roundtrip: %v", tt.Where, err)
				}
				if tt.golden && !bytes.Equal(enc, in) {
					t.Errorf("%s: Marshal:\n\tgot:  %s\n\twant: %s", tt.Where, enc, in)
				}
				vv := reflect.New(reflect.TypeOf(tt.ptr).Elem())
				dec = NewDecoder(bytes.NewReader(enc))
				if tt.useNumber {
					dec.UseNumber()
				}
				if err := dec.Decode(vv.Interface()); err != nil {
					t.Fatalf("%s: Decode(%#q) error after roundtrip: %v", tt.Where, enc, err)
				}
				if !reflect.DeepEqual(v.Elem().Interface(), vv.Elem().Interface()) {
					t.Fatalf("%s: Decode:\n\tgot:  %#+v\n\twant: %#+v\n\n\tgotJSON:  %s\n\twantJSON: %s",
						tt.Where, v.Elem().Interface(), vv.Elem().Interface(),
						stripWhitespace(string(enc)), stripWhitespace(string(in)))
				}
			}
		})
	}
}

func TestUnmarshalMarshal(t *testing.T) {
	initBig()
	var v any
	if err := Unmarshal(jsonBig, &v); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	b, err := Marshal(v)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if !bytes.Equal(jsonBig, b) {
		t.Errorf("Marshal:")
		diff(t, b, jsonBig)
		return
	}
}

// Independent of Decode, basic coverage of the accessors in Number
func TestNumberAccessors(t *testing.T) {
	tests := []struct {
		CaseName
		in       string
		i        int64
		intErr   string
		f        float64
		floatErr string
	}{
		{CaseName: Name(""), in: "-1.23e1", intErr: "strconv.ParseInt: parsing \"-1.23e1\": invalid syntax", f: -1.23e1},
		{CaseName: Name(""), in: "-12", i: -12, f: -12.0},
		{CaseName: Name(""), in: "1e1000", intErr: "strconv.ParseInt: parsing \"1e1000\": invalid syntax", floatErr: "strconv.ParseFloat: parsing \"1e1000\": value out of range"},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			n := Number(tt.in)
			if got := n.String(); got != tt.in {
				t.Errorf("%s: Number(%q).String() = %s, want %s", tt.Where, tt.in, got, tt.in)
			}
			if i, err := n.Int64(); err == nil && tt.intErr == "" && i != tt.i {
				t.Errorf("%s: Number(%q).Int64() = %d, want %d", tt.Where, tt.in, i, tt.i)
			} else if (err == nil && tt.intErr != "") || (err != nil && err.Error() != tt.intErr) {
				t.Errorf("%s: Number(%q).Int64() error:\n\tgot:  %v\n\twant: %v", tt.Where, tt.in, err, tt.intErr)
			}
			if f, err := n.Float64(); err == nil && tt.floatErr == "" && f != tt.f {
				t.Errorf("%s: Number(%q).Float64() = %g, want %g", tt.Where, tt.in, f, tt.f)
			} else if (err == nil && tt.floatErr != "") || (err != nil && err.Error() != tt.floatErr) {
				t.Errorf("%s: Number(%q).Float64() error:\n\tgot  %v\n\twant: %v", tt.Where, tt.in, err, tt.floatErr)
			}
		})
	}
}

func TestLargeByteSlice(t *testing.T) {
	s0 := make([]byte, 2000)
	for i := range s0 {
		s0[i] = byte(i)
	}
	b, err := Marshal(s0)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	var s1 []byte
	if err := Unmarshal(b, &s1); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if !bytes.Equal(s0, s1) {
		t.Errorf("Marshal:")
		diff(t, s0, s1)
	}
}

type Xint struct {
	X int
}

func TestUnmarshalInterface(t *testing.T) {
	var xint Xint
	var i any = &xint
	if err := Unmarshal([]byte(`{"X":1}`), &i); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if xint.X != 1 {
		t.Fatalf("xint.X = %d, want 1", xint.X)
	}
}

func TestUnmarshalPtrPtr(t *testing.T) {
	var xint Xint
	pxint := &xint
	if err := Unmarshal([]byte(`{"X":1}`), &pxint); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if xint.X != 1 {
		t.Fatalf("xint.X = %d, want 1", xint.X)
	}
}

func TestEscape(t *testing.T) {
	const input = `"foobar"<html>` + " [\u2028 \u2029]"
	const want = `"\"foobar\"\u003chtml\u003e [\u2028 \u2029]"`
	got, err := Marshal(input)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if string(got) != want {
		t.Errorf("Marshal(%#q):\n\tgot:  %s\n\twant: %s", input, got, want)
	}
}

// If people misuse the ,string modifier, the error message should be
// helpful, telling the user that they're doing it wrong.
func TestErrorMessageFromMisusedString(t *testing.T) {
	// WrongString is a struct that's misusing the ,string modifier.
	type WrongString struct {
		Message string `json:"result,string"`
	}
	tests := []struct {
		CaseName
		in, err string
	}{
		{Name(""), `{"result":"x"}`, `json: invalid use of ,string struct tag, trying to unmarshal "x" into string`},
		{Name(""), `{"result":"foo"}`, `json: invalid use of ,string struct tag, trying to unmarshal "foo" into string`},
		{Name(""), `{"result":"123"}`, `json: invalid use of ,string struct tag, trying to unmarshal "123" into string`},
		{Name(""), `{"result":123}`, `json: invalid use of ,string struct tag, trying to unmarshal unquoted value into string`},
		{Name(""), `{"result":"\""}`, `json: invalid use of ,string struct tag, trying to unmarshal "\"" into string`},
		{Name(""), `{"result":"\"foo"}`, `json: invalid use of ,string struct tag, trying to unmarshal "\"foo" into string`},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			r := strings.NewReader(tt.in)
			var s WrongString
			err := NewDecoder(r).Decode(&s)
			got := fmt.Sprintf("%v", err)
			if got != tt.err {
				t.Errorf("%s: Decode error:\n\tgot:  %s\n\twant: %s", tt.Where, got, tt.err)
			}
		})
	}
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

	IntStr     int64   `json:",string"`
	UintptrStr uintptr `json:",string"`

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

	Interface  any
	PInterface *any

	unexported int
}

type Small struct {
	Tag string
}

var allValue = All{
	Bool:       true,
	Int:        2,
	Int8:       3,
	Int16:      4,
	Int32:      5,
	Int64:      6,
	Uint:       7,
	Uint8:      8,
	Uint16:     9,
	Uint32:     10,
	Uint64:     11,
	Uintptr:    12,
	Float32:    14.1,
	Float64:    15.1,
	Foo:        "foo",
	Foo2:       "foo2",
	IntStr:     42,
	UintptrStr: 44,
	String:     "16",
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
	"UintptrStr": "44",
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

var allValueCompact = stripWhitespace(allValueIndent)

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
	"UintptrStr": "0",
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

var pallValueCompact = stripWhitespace(pallValueIndent)

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
		t.Fatalf("Unmarshal error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Unmarsha:\n\tgot:  %+v\n\twant: %+v", got, want)
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
	var got T2
	switch err := dec.Decode(&got); {
	case err == nil:
		t.Fatalf("Decode error: got nil, want non-nil")
	case got.Number1 != 1:
		t.Fatalf("Decode: got.Number1 = %d, want 1", got.Number1)
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
	switch err := Unmarshal(data, &s); {
	case err != nil:
		t.Fatalf("Unmarshal error: %v", err)
	case s.B != 1:
		t.Fatalf("Unmarshal: s.B = %d, want 1", s.B)
	case s.C != nil:
		t.Fatalf("Unmarshal: s.C = %d, want non-nil", s.C)
	}
}

func addr[T any](v T) *T {
	return &v
}

func TestInterfaceSet(t *testing.T) {
	errUnmarshal := &UnmarshalTypeError{Value: "object", Offset: 6, Type: reflect.TypeFor[int](), Field: "X"}
	tests := []struct {
		CaseName
		pre  any
		json string
		post any
	}{
		{Name(""), "foo", `"bar"`, "bar"},
		{Name(""), "foo", `2`, 2.0},
		{Name(""), "foo", `true`, true},
		{Name(""), "foo", `null`, nil},
		{Name(""), map[string]any{}, `true`, true},
		{Name(""), []string{}, `true`, true},

		{Name(""), any(nil), `null`, any(nil)},
		{Name(""), (*int)(nil), `null`, any(nil)},
		{Name(""), (*int)(addr(0)), `null`, any(nil)},
		{Name(""), (*int)(addr(1)), `null`, any(nil)},
		{Name(""), (**int)(nil), `null`, any(nil)},
		{Name(""), (**int)(addr[*int](nil)), `null`, (**int)(addr[*int](nil))},
		{Name(""), (**int)(addr(addr(1))), `null`, (**int)(addr[*int](nil))},
		{Name(""), (***int)(nil), `null`, any(nil)},
		{Name(""), (***int)(addr[**int](nil)), `null`, (***int)(addr[**int](nil))},
		{Name(""), (***int)(addr(addr[*int](nil))), `null`, (***int)(addr[**int](nil))},
		{Name(""), (***int)(addr(addr(addr(1)))), `null`, (***int)(addr[**int](nil))},

		{Name(""), any(nil), `2`, float64(2)},
		{Name(""), (int)(1), `2`, float64(2)},
		{Name(""), (*int)(nil), `2`, float64(2)},
		{Name(""), (*int)(addr(0)), `2`, (*int)(addr(2))},
		{Name(""), (*int)(addr(1)), `2`, (*int)(addr(2))},
		{Name(""), (**int)(nil), `2`, float64(2)},
		{Name(""), (**int)(addr[*int](nil)), `2`, (**int)(addr(addr(2)))},
		{Name(""), (**int)(addr(addr(1))), `2`, (**int)(addr(addr(2)))},
		{Name(""), (***int)(nil), `2`, float64(2)},
		{Name(""), (***int)(addr[**int](nil)), `2`, (***int)(addr(addr(addr(2))))},
		{Name(""), (***int)(addr(addr[*int](nil))), `2`, (***int)(addr(addr(addr(2))))},
		{Name(""), (***int)(addr(addr(addr(1)))), `2`, (***int)(addr(addr(addr(2))))},

		{Name(""), any(nil), `{}`, map[string]any{}},
		{Name(""), (int)(1), `{}`, map[string]any{}},
		{Name(""), (*int)(nil), `{}`, map[string]any{}},
		{Name(""), (*int)(addr(0)), `{}`, errUnmarshal},
		{Name(""), (*int)(addr(1)), `{}`, errUnmarshal},
		{Name(""), (**int)(nil), `{}`, map[string]any{}},
		{Name(""), (**int)(addr[*int](nil)), `{}`, errUnmarshal},
		{Name(""), (**int)(addr(addr(1))), `{}`, errUnmarshal},
		{Name(""), (***int)(nil), `{}`, map[string]any{}},
		{Name(""), (***int)(addr[**int](nil)), `{}`, errUnmarshal},
		{Name(""), (***int)(addr(addr[*int](nil))), `{}`, errUnmarshal},
		{Name(""), (***int)(addr(addr(addr(1)))), `{}`, errUnmarshal},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			b := struct{ X any }{tt.pre}
			blob := `{"X":` + tt.json + `}`
			if err := Unmarshal([]byte(blob), &b); err != nil {
				if wantErr, _ := tt.post.(error); equalError(err, wantErr) {
					return
				}
				t.Fatalf("%s: Unmarshal(%#q) error: %v", tt.Where, blob, err)
			}
			if !reflect.DeepEqual(b.X, tt.post) {
				t.Errorf("%s: Unmarshal(%#q):\n\tpre.X:  %#v\n\tgot.X:  %#v\n\twant.X: %#v", tt.Where, blob, tt.pre, b.X, tt.post)
			}
		})
	}
}

type NullTest struct {
	Bool      bool
	Int       int
	Int8      int8
	Int16     int16
	Int32     int32
	Int64     int64
	Uint      uint
	Uint8     uint8
	Uint16    uint16
	Uint32    uint32
	Uint64    uint64
	Float32   float32
	Float64   float64
	String    string
	PBool     *bool
	Map       map[string]string
	Slice     []string
	Interface any

	PRaw    *RawMessage
	PTime   *time.Time
	PBigInt *big.Int
	PText   *MustNotUnmarshalText
	PBuffer *bytes.Buffer // has methods, just not relevant ones
	PStruct *struct{}

	Raw    RawMessage
	Time   time.Time
	BigInt big.Int
	Text   MustNotUnmarshalText
	Buffer bytes.Buffer
	Struct struct{}
}

// JSON null values should be ignored for primitives and string values instead of resulting in an error.
// Issue 2540
func TestUnmarshalNulls(t *testing.T) {
	// Unmarshal docs:
	// The JSON null value unmarshals into an interface, map, pointer, or slice
	// by setting that Go value to nil. Because null is often used in JSON to mean
	// ``not present,'' unmarshaling a JSON null into any other Go type has no effect
	// on the value and produces no error.

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
				"String"  : null,
				"PBool": null,
				"Map": null,
				"Slice": null,
				"Interface": null,
				"PRaw": null,
				"PTime": null,
				"PBigInt": null,
				"PText": null,
				"PBuffer": null,
				"PStruct": null,
				"Raw": null,
				"Time": null,
				"BigInt": null,
				"Text": null,
				"Buffer": null,
				"Struct": null
			}`)
	nulls := NullTest{
		Bool:      true,
		Int:       2,
		Int8:      3,
		Int16:     4,
		Int32:     5,
		Int64:     6,
		Uint:      7,
		Uint8:     8,
		Uint16:    9,
		Uint32:    10,
		Uint64:    11,
		Float32:   12.1,
		Float64:   13.1,
		String:    "14",
		PBool:     new(bool),
		Map:       map[string]string{},
		Slice:     []string{},
		Interface: new(MustNotUnmarshalJSON),
		PRaw:      new(RawMessage),
		PTime:     new(time.Time),
		PBigInt:   new(big.Int),
		PText:     new(MustNotUnmarshalText),
		PStruct:   new(struct{}),
		PBuffer:   new(bytes.Buffer),
		Raw:       RawMessage("123"),
		Time:      time.Unix(123456789, 0),
		BigInt:    *big.NewInt(123),
	}

	before := nulls.Time.String()

	err := Unmarshal(jsonData, &nulls)
	if err != nil {
		t.Errorf("Unmarshal of null values failed: %v", err)
	}
	if !nulls.Bool || nulls.Int != 2 || nulls.Int8 != 3 || nulls.Int16 != 4 || nulls.Int32 != 5 || nulls.Int64 != 6 ||
		nulls.Uint != 7 || nulls.Uint8 != 8 || nulls.Uint16 != 9 || nulls.Uint32 != 10 || nulls.Uint64 != 11 ||
		nulls.Float32 != 12.1 || nulls.Float64 != 13.1 || nulls.String != "14" {
		t.Errorf("Unmarshal of null values affected primitives")
	}

	if nulls.PBool != nil {
		t.Errorf("Unmarshal of null did not clear nulls.PBool")
	}
	if nulls.Map != nil {
		t.Errorf("Unmarshal of null did not clear nulls.Map")
	}
	if nulls.Slice != nil {
		t.Errorf("Unmarshal of null did not clear nulls.Slice")
	}
	if nulls.Interface != nil {
		t.Errorf("Unmarshal of null did not clear nulls.Interface")
	}
	if nulls.PRaw != nil {
		t.Errorf("Unmarshal of null did not clear nulls.PRaw")
	}
	if nulls.PTime != nil {
		t.Errorf("Unmarshal of null did not clear nulls.PTime")
	}
	if nulls.PBigInt != nil {
		t.Errorf("Unmarshal of null did not clear nulls.PBigInt")
	}
	if nulls.PText != nil {
		t.Errorf("Unmarshal of null did not clear nulls.PText")
	}
	if nulls.PBuffer != nil {
		t.Errorf("Unmarshal of null did not clear nulls.PBuffer")
	}
	if nulls.PStruct != nil {
		t.Errorf("Unmarshal of null did not clear nulls.PStruct")
	}

	if string(nulls.Raw) != "null" {
		t.Errorf("Unmarshal of RawMessage null did not record null: %v", string(nulls.Raw))
	}
	if nulls.Time.String() != before {
		t.Errorf("Unmarshal of time.Time null set time to %v", nulls.Time.String())
	}
	if nulls.BigInt.String() != "123" {
		t.Errorf("Unmarshal of big.Int null set int to %v", nulls.BigInt.String())
	}
}

type MustNotUnmarshalJSON struct{}

func (x MustNotUnmarshalJSON) UnmarshalJSON(data []byte) error {
	return errors.New("MustNotUnmarshalJSON was used")
}

type MustNotUnmarshalText struct{}

func (x MustNotUnmarshalText) UnmarshalText(text []byte) error {
	return errors.New("MustNotUnmarshalText was used")
}

func TestStringKind(t *testing.T) {
	type stringKind string
	want := map[stringKind]int{"foo": 42}
	data, err := Marshal(want)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	var got map[stringKind]int
	err = Unmarshal(data, &got)
	if err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if !maps.Equal(got, want) {
		t.Fatalf("Marshal/Unmarshal mismatch:\n\tgot:  %v\n\twant: %v", got, want)
	}
}

// Custom types with []byte as underlying type could not be marshaled
// and then unmarshaled.
// Issue 8962.
func TestByteKind(t *testing.T) {
	type byteKind []byte
	want := byteKind("hello")
	data, err := Marshal(want)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	var got byteKind
	err = Unmarshal(data, &got)
	if err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("Marshal/Unmarshal mismatch:\n\tgot:  %v\n\twant: %v", got, want)
	}
}

// The fix for issue 8962 introduced a regression.
// Issue 12921.
func TestSliceOfCustomByte(t *testing.T) {
	type Uint8 uint8
	want := []Uint8("hello")
	data, err := Marshal(want)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	var got []Uint8
	err = Unmarshal(data, &got)
	if err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("Marshal/Unmarshal mismatch:\n\tgot:  %v\n\twant: %v", got, want)
	}
}

func TestUnmarshalTypeError(t *testing.T) {
	tests := []struct {
		CaseName
		dest any
		in   string
	}{
		{Name(""), new(string), `{"user": "name"}`}, // issue 4628.
		{Name(""), new(error), `{}`},                // issue 4222
		{Name(""), new(error), `[]`},
		{Name(""), new(error), `""`},
		{Name(""), new(error), `123`},
		{Name(""), new(error), `true`},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			err := Unmarshal([]byte(tt.in), tt.dest)
			if _, ok := err.(*UnmarshalTypeError); !ok {
				t.Errorf("%s: Unmarshal(%#q, %T):\n\tgot:  %T\n\twant: %T",
					tt.Where, tt.in, tt.dest, err, new(UnmarshalTypeError))
			}
		})
	}
}

func TestUnmarshalSyntax(t *testing.T) {
	var x any
	tests := []struct {
		CaseName
		in string
	}{
		{Name(""), "tru"},
		{Name(""), "fals"},
		{Name(""), "nul"},
		{Name(""), "123e"},
		{Name(""), `"hello`},
		{Name(""), `[1,2,3`},
		{Name(""), `{"key":1`},
		{Name(""), `{"key":1,`},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			err := Unmarshal([]byte(tt.in), &x)
			if _, ok := err.(*SyntaxError); !ok {
				t.Errorf("%s: Unmarshal(%#q, any):\n\tgot:  %T\n\twant: %T",
					tt.Where, tt.in, err, new(SyntaxError))
			}
		})
	}
}

// Test handling of unexported fields that should be ignored.
// Issue 4660
type unexportedFields struct {
	Name string
	m    map[string]any `json:"-"`
	m2   map[string]any `json:"abcd"`

	s []int `json:"-"`
}

func TestUnmarshalUnexported(t *testing.T) {
	input := `{"Name": "Bob", "m": {"x": 123}, "m2": {"y": 456}, "abcd": {"z": 789}, "s": [2, 3]}`
	want := &unexportedFields{Name: "Bob"}

	out := &unexportedFields{}
	err := Unmarshal([]byte(input), out)
	if err != nil {
		t.Errorf("Unmarshal error: %v", err)
	}
	if !reflect.DeepEqual(out, want) {
		t.Errorf("Unmarshal:\n\tgot:  %+v\n\twant: %+v", out, want)
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
	switch err := Unmarshal([]byte(`"0000-00-00T00:00:00Z"`), &t3); {
	case err == nil:
		t.Fatalf("Unmarshal error: got nil, want non-nil")
	case !strings.Contains(err.Error(), "range"):
		t.Errorf("Unmarshal error:\n\tgot:  %v\n\twant: out of range", err)
	}
}

// Test that extra object elements in an array do not result in a
// "data changing underfoot" error.
// Issue 3717
func TestSkipArrayObjects(t *testing.T) {
	json := `[{}]`
	var dest [0]any

	err := Unmarshal([]byte(json), &dest)
	if err != nil {
		t.Errorf("Unmarshal error: %v", err)
	}
}

// Test semantics of pre-filled data, such as struct fields, map elements,
// slices, and arrays.
// Issues 4900 and 8837, among others.
func TestPrefilled(t *testing.T) {
	// Values here change, cannot reuse table across runs.
	tests := []struct {
		CaseName
		in  string
		ptr any
		out any
	}{{
		CaseName: Name(""),
		in:       `{"X": 1, "Y": 2}`,
		ptr:      &XYZ{X: float32(3), Y: int16(4), Z: 1.5},
		out:      &XYZ{X: float64(1), Y: float64(2), Z: 1.5},
	}, {
		CaseName: Name(""),
		in:       `{"X": 1, "Y": 2}`,
		ptr:      &map[string]any{"X": float32(3), "Y": int16(4), "Z": 1.5},
		out:      &map[string]any{"X": float64(1), "Y": float64(2), "Z": 1.5},
	}, {
		CaseName: Name(""),
		in:       `[2]`,
		ptr:      &[]int{1},
		out:      &[]int{2},
	}, {
		CaseName: Name(""),
		in:       `[2, 3]`,
		ptr:      &[]int{1},
		out:      &[]int{2, 3},
	}, {
		CaseName: Name(""),
		in:       `[2, 3]`,
		ptr:      &[...]int{1},
		out:      &[...]int{2},
	}, {
		CaseName: Name(""),
		in:       `[3]`,
		ptr:      &[...]int{1, 2},
		out:      &[...]int{3, 0},
	}}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			ptrstr := fmt.Sprintf("%v", tt.ptr)
			err := Unmarshal([]byte(tt.in), tt.ptr) // tt.ptr edited here
			if err != nil {
				t.Errorf("%s: Unmarshal error: %v", tt.Where, err)
			}
			if !reflect.DeepEqual(tt.ptr, tt.out) {
				t.Errorf("%s: Unmarshal(%#q, %T):\n\tgot:  %v\n\twant: %v", tt.Where, tt.in, ptrstr, tt.ptr, tt.out)
			}
		})
	}
}

func TestInvalidUnmarshal(t *testing.T) {
	tests := []struct {
		CaseName
		in      string
		v       any
		wantErr error
	}{
		{Name(""), `{"a":"1"}`, nil, &InvalidUnmarshalError{}},
		{Name(""), `{"a":"1"}`, struct{}{}, &InvalidUnmarshalError{reflect.TypeFor[struct{}]()}},
		{Name(""), `{"a":"1"}`, (*int)(nil), &InvalidUnmarshalError{reflect.TypeFor[*int]()}},
		{Name(""), `123`, nil, &InvalidUnmarshalError{}},
		{Name(""), `123`, struct{}{}, &InvalidUnmarshalError{reflect.TypeFor[struct{}]()}},
		{Name(""), `123`, (*int)(nil), &InvalidUnmarshalError{reflect.TypeFor[*int]()}},
		{Name(""), `123`, new(net.IP), &UnmarshalTypeError{Value: "number", Type: reflect.TypeFor[*net.IP](), Offset: 3}},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			switch gotErr := Unmarshal([]byte(tt.in), tt.v); {
			case gotErr == nil:
				t.Fatalf("%s: Unmarshal error: got nil, want non-nil", tt.Where)
			case !reflect.DeepEqual(gotErr, tt.wantErr):
				t.Errorf("%s: Unmarshal error:\n\tgot:  %#v\n\twant: %#v", tt.Where, gotErr, tt.wantErr)
			}
		})
	}
}

// Test that string option is ignored for invalid types.
// Issue 9812.
func TestInvalidStringOption(t *testing.T) {
	num := 0
	item := struct {
		T time.Time         `json:",string"`
		M map[string]string `json:",string"`
		S []string          `json:",string"`
		A [1]string         `json:",string"`
		I any               `json:",string"`
		P *int              `json:",string"`
	}{M: make(map[string]string), S: make([]string, 0), I: num, P: &num}

	data, err := Marshal(item)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}

	err = Unmarshal(data, &item)
	if err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
}

// Test unmarshal behavior with regards to embedded unexported structs.
//
// (Issue 21357) If the embedded struct is a pointer and is unallocated,
// this returns an error because unmarshal cannot set the field.
//
// (Issue 24152) If the embedded struct is given an explicit name,
// ensure that the normal unmarshal logic does not panic in reflect.
//
// (Issue 28145) If the embedded struct is given an explicit name and has
// exported methods, don't cause a panic trying to get its value.
func TestUnmarshalEmbeddedUnexported(t *testing.T) {
	type (
		embed1 struct{ Q int }
		embed2 struct{ Q int }
		embed3 struct {
			Q int64 `json:",string"`
		}
		S1 struct {
			*embed1
			R int
		}
		S2 struct {
			*embed1
			Q int
		}
		S3 struct {
			embed1
			R int
		}
		S4 struct {
			*embed1
			embed2
		}
		S5 struct {
			*embed3
			R int
		}
		S6 struct {
			embed1 `json:"embed1"`
		}
		S7 struct {
			embed1 `json:"embed1"`
			embed2
		}
		S8 struct {
			embed1 `json:"embed1"`
			embed2 `json:"embed2"`
			Q      int
		}
		S9 struct {
			unexportedWithMethods `json:"embed"`
		}
	)

	tests := []struct {
		CaseName
		in  string
		ptr any
		out any
		err error
	}{{
		// Error since we cannot set S1.embed1, but still able to set S1.R.
		CaseName: Name(""),
		in:       `{"R":2,"Q":1}`,
		ptr:      new(S1),
		out:      &S1{R: 2},
		err:      fmt.Errorf("json: cannot set embedded pointer to unexported struct: json.embed1"),
	}, {
		// The top level Q field takes precedence.
		CaseName: Name(""),
		in:       `{"Q":1}`,
		ptr:      new(S2),
		out:      &S2{Q: 1},
	}, {
		// No issue with non-pointer variant.
		CaseName: Name(""),
		in:       `{"R":2,"Q":1}`,
		ptr:      new(S3),
		out:      &S3{embed1: embed1{Q: 1}, R: 2},
	}, {
		// No error since both embedded structs have field R, which annihilate each other.
		// Thus, no attempt is made at setting S4.embed1.
		CaseName: Name(""),
		in:       `{"R":2}`,
		ptr:      new(S4),
		out:      new(S4),
	}, {
		// Error since we cannot set S5.embed1, but still able to set S5.R.
		CaseName: Name(""),
		in:       `{"R":2,"Q":1}`,
		ptr:      new(S5),
		out:      &S5{R: 2},
		err:      fmt.Errorf("json: cannot set embedded pointer to unexported struct: json.embed3"),
	}, {
		// Issue 24152, ensure decodeState.indirect does not panic.
		CaseName: Name(""),
		in:       `{"embed1": {"Q": 1}}`,
		ptr:      new(S6),
		out:      &S6{embed1{1}},
	}, {
		// Issue 24153, check that we can still set forwarded fields even in
		// the presence of a name conflict.
		//
		// This relies on obscure behavior of reflect where it is possible
		// to set a forwarded exported field on an unexported embedded struct
		// even though there is a name conflict, even when it would have been
		// impossible to do so according to Go visibility rules.
		// Go forbids this because it is ambiguous whether S7.Q refers to
		// S7.embed1.Q or S7.embed2.Q. Since embed1 and embed2 are unexported,
		// it should be impossible for an external package to set either Q.
		//
		// It is probably okay for a future reflect change to break this.
		CaseName: Name(""),
		in:       `{"embed1": {"Q": 1}, "Q": 2}`,
		ptr:      new(S7),
		out:      &S7{embed1{1}, embed2{2}},
	}, {
		// Issue 24153, similar to the S7 case.
		CaseName: Name(""),
		in:       `{"embed1": {"Q": 1}, "embed2": {"Q": 2}, "Q": 3}`,
		ptr:      new(S8),
		out:      &S8{embed1{1}, embed2{2}, 3},
	}, {
		// Issue 228145, similar to the cases above.
		CaseName: Name(""),
		in:       `{"embed": {}}`,
		ptr:      new(S9),
		out:      &S9{},
	}}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			err := Unmarshal([]byte(tt.in), tt.ptr)
			if !equalError(err, tt.err) {
				t.Errorf("%s: Unmarshal error:\n\tgot:  %v\n\twant: %v", tt.Where, err, tt.err)
			}
			if !reflect.DeepEqual(tt.ptr, tt.out) {
				t.Errorf("%s: Unmarshal:\n\tgot:  %#+v\n\twant: %#+v", tt.Where, tt.ptr, tt.out)
			}
		})
	}
}

func TestUnmarshalErrorAfterMultipleJSON(t *testing.T) {
	tests := []struct {
		CaseName
		in  string
		err error
	}{{
		CaseName: Name(""),
		in:       `1 false null :`,
		err:      &SyntaxError{"invalid character ':' looking for beginning of value", 14},
	}, {
		CaseName: Name(""),
		in:       `1 [] [,]`,
		err:      &SyntaxError{"invalid character ',' looking for beginning of value", 7},
	}, {
		CaseName: Name(""),
		in:       `1 [] [true:]`,
		err:      &SyntaxError{"invalid character ':' after array element", 11},
	}, {
		CaseName: Name(""),
		in:       `1  {}    {"x"=}`,
		err:      &SyntaxError{"invalid character '=' after object key", 14},
	}, {
		CaseName: Name(""),
		in:       `falsetruenul#`,
		err:      &SyntaxError{"invalid character '#' in literal null (expecting 'l')", 13},
	}}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			dec := NewDecoder(strings.NewReader(tt.in))
			var err error
			for err == nil {
				var v any
				err = dec.Decode(&v)
			}
			if !reflect.DeepEqual(err, tt.err) {
				t.Errorf("%s: Decode error:\n\tgot:  %v\n\twant: %v", tt.Where, err, tt.err)
			}
		})
	}
}

type unmarshalPanic struct{}

func (unmarshalPanic) UnmarshalJSON([]byte) error { panic(0xdead) }

func TestUnmarshalPanic(t *testing.T) {
	defer func() {
		if got := recover(); !reflect.DeepEqual(got, 0xdead) {
			t.Errorf("panic() = (%T)(%v), want 0xdead", got, got)
		}
	}()
	Unmarshal([]byte("{}"), &unmarshalPanic{})
	t.Fatalf("Unmarshal should have panicked")
}

// The decoder used to hang if decoding into an interface pointing to its own address.
// See golang.org/issues/31740.
func TestUnmarshalRecursivePointer(t *testing.T) {
	var v any
	v = &v
	data := []byte(`{"a": "b"}`)

	if err := Unmarshal(data, v); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
}

type textUnmarshalerString string

func (m *textUnmarshalerString) UnmarshalText(text []byte) error {
	*m = textUnmarshalerString(strings.ToLower(string(text)))
	return nil
}

// Test unmarshal to a map, where the map key is a user defined type.
// See golang.org/issues/34437.
func TestUnmarshalMapWithTextUnmarshalerStringKey(t *testing.T) {
	var p map[textUnmarshalerString]string
	if err := Unmarshal([]byte(`{"FOO": "1"}`), &p); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}

	if _, ok := p["foo"]; !ok {
		t.Errorf(`key "foo" missing in map: %v`, p)
	}
}

func TestUnmarshalRescanLiteralMangledUnquote(t *testing.T) {
	// See golang.org/issues/38105.
	var p map[textUnmarshalerString]string
	if err := Unmarshal([]byte(`{"开源":"12345开源"}`), &p); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if _, ok := p["开源"]; !ok {
		t.Errorf(`key "开源" missing in map: %v`, p)
	}

	// See golang.org/issues/38126.
	type T struct {
		F1 string `json:"F1,string"`
	}
	wantT := T{"aaa\tbbb"}

	b, err := Marshal(wantT)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	var gotT T
	if err := Unmarshal(b, &gotT); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if gotT != wantT {
		t.Errorf("Marshal/Unmarshal roundtrip:\n\tgot:  %q\n\twant: %q", gotT, wantT)
	}

	// See golang.org/issues/39555.
	input := map[textUnmarshalerString]string{"FOO": "", `"`: ""}

	encoded, err := Marshal(input)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	var got map[textUnmarshalerString]string
	if err := Unmarshal(encoded, &got); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	want := map[textUnmarshalerString]string{"foo": "", `"`: ""}
	if !maps.Equal(got, want) {
		t.Errorf("Marshal/Unmarshal roundtrip:\n\tgot:  %q\n\twant: %q", gotT, wantT)
	}
}

func TestUnmarshalMaxDepth(t *testing.T) {
	tests := []struct {
		CaseName
		data        string
		errMaxDepth bool
	}{{
		CaseName:    Name("ArrayUnderMaxNestingDepth"),
		data:        `{"a":` + strings.Repeat(`[`, 10000-1) + strings.Repeat(`]`, 10000-1) + `}`,
		errMaxDepth: false,
	}, {
		CaseName:    Name("ArrayOverMaxNestingDepth"),
		data:        `{"a":` + strings.Repeat(`[`, 10000) + strings.Repeat(`]`, 10000) + `}`,
		errMaxDepth: true,
	}, {
		CaseName:    Name("ArrayOverStackDepth"),
		data:        `{"a":` + strings.Repeat(`[`, 3000000) + strings.Repeat(`]`, 3000000) + `}`,
		errMaxDepth: true,
	}, {
		CaseName:    Name("ObjectUnderMaxNestingDepth"),
		data:        `{"a":` + strings.Repeat(`{"a":`, 10000-1) + `0` + strings.Repeat(`}`, 10000-1) + `}`,
		errMaxDepth: false,
	}, {
		CaseName:    Name("ObjectOverMaxNestingDepth"),
		data:        `{"a":` + strings.Repeat(`{"a":`, 10000) + `0` + strings.Repeat(`}`, 10000) + `}`,
		errMaxDepth: true,
	}, {
		CaseName:    Name("ObjectOverStackDepth"),
		data:        `{"a":` + strings.Repeat(`{"a":`, 3000000) + `0` + strings.Repeat(`}`, 3000000) + `}`,
		errMaxDepth: true,
	}}

	targets := []struct {
		CaseName
		newValue func() any
	}{{
		CaseName: Name("unstructured"),
		newValue: func() any {
			var v any
			return &v
		},
	}, {
		CaseName: Name("typed named field"),
		newValue: func() any {
			v := struct {
				A any `json:"a"`
			}{}
			return &v
		},
	}, {
		CaseName: Name("typed missing field"),
		newValue: func() any {
			v := struct {
				B any `json:"b"`
			}{}
			return &v
		},
	}, {
		CaseName: Name("custom unmarshaler"),
		newValue: func() any {
			v := unmarshaler{}
			return &v
		},
	}}

	for _, tt := range tests {
		for _, target := range targets {
			t.Run(target.Name+"-"+tt.Name, func(t *testing.T) {
				err := Unmarshal([]byte(tt.data), target.newValue())
				if !tt.errMaxDepth {
					if err != nil {
						t.Errorf("%s: %s: Unmarshal error: %v", tt.Where, target.Where, err)
					}
				} else {
					if err == nil || !strings.Contains(err.Error(), "exceeded max depth") {
						t.Errorf("%s: %s: Unmarshal error:\n\tgot:  %v\n\twant: exceeded max depth", tt.Where, target.Where, err)
					}
				}
			})
		}
	}
}
