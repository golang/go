// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json_test

import (
	"errors"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"

	jsonv1 "encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
)

// NOTE: This file serves as a list of semantic differences between v1 and v2.
// Each test explains how v1 behaves, how v2 behaves, and
// a rationale for why the behavior was changed.

var jsonPackages = []struct {
	Version   string
	Marshal   func(any) ([]byte, error)
	Unmarshal func([]byte, any) error
}{
	{"v1", jsonv1.Marshal, jsonv1.Unmarshal},
	{"v2",
		func(in any) ([]byte, error) { return jsonv2.Marshal(in) },
		func(in []byte, out any) error { return jsonv2.Unmarshal(in, out) }},
}

// In v1, unmarshal matches struct fields using a case-insensitive match.
// In v2, unmarshal matches struct fields using a case-sensitive match.
//
// Case-insensitive matching is a surprising default and
// incurs significant performance cost when unmarshaling unknown fields.
// In v2, we can opt into v1-like behavior with the `case:ignore` tag option.
// The case-insensitive matching performed by v2 is looser than that of v1
// where it also ignores dashes and underscores.
// This allows v2 to match fields regardless of whether the name is in
// snake_case, camelCase, or kebab-case.
//
// Related issue:
//
//	https://go.dev/issue/14750
func TestCaseSensitivity(t *testing.T) {
	type Fields struct {
		FieldA bool
		FieldB bool `json:"fooBar"`
		FieldC bool `json:"fizzBuzz,case:ignore"` // `case:ignore` is used by v2 to explicitly enable case-insensitive matching
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal", json.Version), func(t *testing.T) {
			// This is a mapping from Go field names to JSON member names to
			// whether the JSON member name would match the Go field name.
			type goName = string
			type jsonName = string
			onlyV1 := json.Version == "v1"
			onlyV2 := json.Version == "v2"
			allMatches := map[goName]map[jsonName]bool{
				"FieldA": {
					"FieldA": true,   // exact match
					"fielda": onlyV1, // v1 is case-insensitive by default
					"fieldA": onlyV1, // v1 is case-insensitive by default
					"FIELDA": onlyV1, // v1 is case-insensitive by default
					"FieldB": false,
					"FieldC": false,
				},
				"FieldB": {
					"fooBar":   true,   // exact match for explicitly specified JSON name
					"FooBar":   onlyV1, // v1 is case-insensitive even if an explicit JSON name is provided
					"foobar":   onlyV1, // v1 is case-insensitive even if an explicit JSON name is provided
					"FOOBAR":   onlyV1, // v1 is case-insensitive even if an explicit JSON name is provided
					"fizzBuzz": false,
					"FieldA":   false,
					"FieldB":   false, // explicit JSON name means that the Go field name is not used for matching
					"FieldC":   false,
				},
				"FieldC": {
					"fizzBuzz":  true,   // exact match for explicitly specified JSON name
					"fizzbuzz":  true,   // v2 is case-insensitive due to `case:ignore` tag
					"FIZZBUZZ":  true,   // v2 is case-insensitive due to `case:ignore` tag
					"fizz_buzz": onlyV2, // case-insensitivity in v2 ignores dashes and underscores
					"fizz-buzz": onlyV2, // case-insensitivity in v2 ignores dashes and underscores
					"fooBar":    false,
					"FieldA":    false,
					"FieldC":    false, // explicit JSON name means that the Go field name is not used for matching
					"FieldB":    false,
				},
			}

			for goFieldName, matches := range allMatches {
				for jsonMemberName, wantMatch := range matches {
					in := `{"` + jsonMemberName + `":true}`
					var s Fields
					if err := json.Unmarshal([]byte(in), &s); err != nil {
						t.Fatalf("json.Unmarshal error: %v", err)
					}
					gotMatch := reflect.ValueOf(s).FieldByName(goFieldName).Bool()
					if gotMatch != wantMatch {
						t.Fatalf("%T.%s = %v, want %v", s, goFieldName, gotMatch, wantMatch)
					}
				}
			}
		})
	}
}

// In v1, the "omitempty" option specifies that a struct field is omitted
// when marshaling if it is an empty Go value, which is defined as
// false, 0, a nil pointer, a nil interface value, and
// any empty array, slice, map, or string.
//
// In v2, the "omitempty" option specifies that a struct field is omitted
// when marshaling if it is an empty JSON value, which is defined as
// a JSON null or empty JSON string, object, or array.
//
// In v2, we also provide the "omitzero" option which specifies that a field
// is omitted if it is the zero Go value or if it implements an "IsZero() bool"
// method that reports true. Together, "omitzero" and "omitempty" can cover
// all the prior use cases of the v1 definition of "omitempty".
// Note that "omitempty" is defined in terms of the Go type system in v1,
// but now defined in terms of the JSON type system in v2.
//
// Related issues:
//
//	https://go.dev/issue/11939
//	https://go.dev/issue/22480
//	https://go.dev/issue/29310
//	https://go.dev/issue/32675
//	https://go.dev/issue/45669
//	https://go.dev/issue/45787
//	https://go.dev/issue/50480
//	https://go.dev/issue/52803
func TestOmitEmptyOption(t *testing.T) {
	type Struct struct {
		Foo string  `json:",omitempty"`
		Bar []int   `json:",omitempty"`
		Baz *Struct `json:",omitempty"`
	}
	type Types struct {
		Bool       bool              `json:",omitempty"`
		StringA    string            `json:",omitempty"`
		StringB    string            `json:",omitempty"`
		BytesA     []byte            `json:",omitempty"`
		BytesB     []byte            `json:",omitempty"`
		BytesC     []byte            `json:",omitempty"`
		Int        int               `json:",omitempty"`
		MapA       map[string]string `json:",omitempty"`
		MapB       map[string]string `json:",omitempty"`
		MapC       map[string]string `json:",omitempty"`
		StructA    Struct            `json:",omitempty"`
		StructB    Struct            `json:",omitempty"`
		StructC    Struct            `json:",omitempty"`
		SliceA     []string          `json:",omitempty"`
		SliceB     []string          `json:",omitempty"`
		SliceC     []string          `json:",omitempty"`
		Array      [1]string         `json:",omitempty"`
		PointerA   *string           `json:",omitempty"`
		PointerB   *string           `json:",omitempty"`
		PointerC   *string           `json:",omitempty"`
		InterfaceA any               `json:",omitempty"`
		InterfaceB any               `json:",omitempty"`
		InterfaceC any               `json:",omitempty"`
		InterfaceD any               `json:",omitempty"`
	}

	something := "something"
	for _, json := range jsonPackages {
		t.Run(path.Join("Marshal", json.Version), func(t *testing.T) {
			in := Types{
				Bool:       false,
				StringA:    "",
				StringB:    something,
				BytesA:     nil,
				BytesB:     []byte{},
				BytesC:     []byte(something),
				Int:        0,
				MapA:       nil,
				MapB:       map[string]string{},
				MapC:       map[string]string{something: something},
				StructA:    Struct{},
				StructB:    Struct{Bar: []int{}, Baz: new(Struct)},
				StructC:    Struct{Foo: something},
				SliceA:     nil,
				SliceB:     []string{},
				SliceC:     []string{something},
				Array:      [1]string{something},
				PointerA:   nil,
				PointerB:   new(string),
				PointerC:   &something,
				InterfaceA: nil,
				InterfaceB: (*string)(nil),
				InterfaceC: new(string),
				InterfaceD: &something,
			}
			b, err := json.Marshal(in)
			if err != nil {
				t.Fatalf("json.Marshal error: %v", err)
			}
			var out map[string]any
			if err := json.Unmarshal(b, &out); err != nil {
				t.Fatalf("json.Unmarshal error: %v", err)
			}

			onlyV1 := json.Version == "v1"
			onlyV2 := json.Version == "v2"
			wantPresent := map[string]bool{
				"Bool":       onlyV2, // false is an empty Go bool, but is NOT an empty JSON value
				"StringA":    false,
				"StringB":    true,
				"BytesA":     false,
				"BytesB":     false,
				"BytesC":     true,
				"Int":        onlyV2, // 0 is an empty Go integer, but NOT an empty JSON value
				"MapA":       false,
				"MapB":       false,
				"MapC":       true,
				"StructA":    onlyV1, // Struct{} is NOT an empty Go value, but {} is an empty JSON value
				"StructB":    onlyV1, // Struct{...} is NOT an empty Go value, but {} is an empty JSON value
				"StructC":    true,
				"SliceA":     false,
				"SliceB":     false,
				"SliceC":     true,
				"Array":      true,
				"PointerA":   false,
				"PointerB":   onlyV1, // new(string) is NOT a nil Go pointer, but "" is an empty JSON value
				"PointerC":   true,
				"InterfaceA": false,
				"InterfaceB": onlyV1, // (*string)(nil) is NOT a nil Go interface, but null is an empty JSON value
				"InterfaceC": onlyV1, // new(string) is NOT a nil Go interface, but "" is an empty JSON value
				"InterfaceD": true,
			}
			for field, want := range wantPresent {
				_, got := out[field]
				if got != want {
					t.Fatalf("%T.%s = %v, want %v", in, field, got, want)
				}
			}
		})
	}
}

func addr[T any](v T) *T {
	return &v
}

// In v1, the "string" option specifies that Go strings, bools, and numeric
// values are encoded within a JSON string when marshaling and
// are unmarshaled from its native representation escaped within a JSON string.
// The "string" option is not applied recursively, and so does not affect
// strings, bools, and numeric values within a Go slice or map, but
// does have special handling to affect the underlying value within a pointer.
// When unmarshaling, the "string" option permits decoding from a JSON null
// escaped within a JSON string in some inconsistent cases.
//
// In v2, the "string" option specifies that only numeric values are encoded as
// a JSON number within a JSON string when marshaling and are unmarshaled
// from either a JSON number or a JSON string containing a JSON number.
// The "string" option is applied recursively to all numeric sub-values,
// and thus affects numeric values within a Go slice or map.
// There is no support for escaped JSON nulls within a JSON string.
//
// The main utility for stringifying JSON numbers is because JSON parsers
// often represents numbers as IEEE 754 floating-point numbers.
// This results in a loss of precision representing 64-bit integer values.
// Consequently, many JSON-based APIs actually requires that such values
// be encoded within a JSON string. Since the main utility of stringification
// is for numeric values, v2 limits the effect of the "string" option
// to just numeric Go types. According to all code known by the Go module proxy,
// there are close to zero usages of the "string" option on a Go string or bool.
//
// Regarding the recursive application of the "string" option,
// there have been a number of issues filed about users being surprised that
// the "string" option does not recursively affect numeric values
// within a composite type like a Go map, slice, or interface value.
// In v1, specifying the "string" option on composite type has no effect
// and so this would be a largely backwards compatible change.
//
// The ability to decode from a JSON null wrapped within a JSON string
// is removed in v2 because this behavior was surprising and inconsistent in v1.
//
// Related issues:
//
//	https://go.dev/issue/15624
//	https://go.dev/issue/20651
//	https://go.dev/issue/22177
//	https://go.dev/issue/32055
//	https://go.dev/issue/32117
//	https://go.dev/issue/50997
func TestStringOption(t *testing.T) {
	type Types struct {
		String     string              `json:",string"`
		Bool       bool                `json:",string"`
		Int        int                 `json:",string"`
		Float      float64             `json:",string"`
		Map        map[string]int      `json:",string"`
		Struct     struct{ Field int } `json:",string"`
		Slice      []int               `json:",string"`
		Array      [1]int              `json:",string"`
		PointerA   *int                `json:",string"`
		PointerB   *int                `json:",string"`
		PointerC   **int               `json:",string"`
		InterfaceA any                 `json:",string"`
		InterfaceB any                 `json:",string"`
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Marshal", json.Version), func(t *testing.T) {
			in := Types{
				String:     "string",
				Bool:       true,
				Int:        1,
				Float:      1,
				Map:        map[string]int{"Name": 1},
				Struct:     struct{ Field int }{1},
				Slice:      []int{1},
				Array:      [1]int{1},
				PointerA:   nil,
				PointerB:   addr(1),
				PointerC:   addr(addr(1)),
				InterfaceA: nil,
				InterfaceB: 1,
			}
			quote := func(s string) string {
				b, _ := jsontext.AppendQuote(nil, s)
				return string(b)
			}
			quoteOnlyV1 := func(s string) string {
				if json.Version == "v1" {
					s = quote(s)
				}
				return s
			}
			quoteOnlyV2 := func(s string) string {
				if json.Version == "v2" {
					s = quote(s)
				}
				return s
			}
			want := strings.Join([]string{
				`{`,
				`"String":` + quoteOnlyV1(`"string"`) + `,`, // in v1, Go strings are also stringified
				`"Bool":` + quoteOnlyV1("true") + `,`,       // in v1, Go bools are also stringified
				`"Int":` + quote("1") + `,`,
				`"Float":` + quote("1") + `,`,
				`"Map":{"Name":` + quoteOnlyV2("1") + `},`,     // in v2, numbers are recursively stringified
				`"Struct":{"Field":` + quoteOnlyV2("1") + `},`, // in v2, numbers are recursively stringified
				`"Slice":[` + quoteOnlyV2("1") + `],`,          // in v2, numbers are recursively stringified
				`"Array":[` + quoteOnlyV2("1") + `],`,          // in v2, numbers are recursively stringified
				`"PointerA":null,`,
				`"PointerB":` + quote("1") + `,`,       // in v1, numbers are stringified after a single pointer indirection
				`"PointerC":` + quoteOnlyV2("1") + `,`, // in v2, numbers are recursively stringified
				`"InterfaceA":null,`,
				`"InterfaceB":` + quoteOnlyV2("1") + ``, // in v2, numbers are recursively stringified
				`}`}, "")
			got, err := json.Marshal(in)
			if err != nil {
				t.Fatalf("json.Marshal error: %v", err)
			}
			if string(got) != want {
				t.Fatalf("json.Marshal = %s, want %s", got, want)
			}
		})
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal/Null", json.Version), func(t *testing.T) {
			var got Types
			err := json.Unmarshal([]byte(`{
				"Bool":     "null",
				"Int":      "null",
				"PointerA": "null"
			}`), &got)
			switch {
			case !reflect.DeepEqual(got, Types{}):
				t.Fatalf("json.Unmarshal = %v, want %v", got, Types{})
			case json.Version == "v1" && err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			case json.Version == "v2" && err == nil:
				t.Fatal("json.Unmarshal error is nil, want non-nil")
			}
		})

		t.Run(path.Join("Unmarshal/Bool", json.Version), func(t *testing.T) {
			var got Types
			want := map[string]Types{
				"v1": {Bool: true},
				"v2": {Bool: false},
			}[json.Version]
			err := json.Unmarshal([]byte(`{"Bool": "true"}`), &got)
			switch {
			case !reflect.DeepEqual(got, want):
				t.Fatalf("json.Unmarshal = %v, want %v", got, want)
			case json.Version == "v1" && err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			case json.Version == "v2" && err == nil:
				t.Fatal("json.Unmarshal error is nil, want non-nil")
			}
		})

		t.Run(path.Join("Unmarshal/Shallow", json.Version), func(t *testing.T) {
			var got Types
			want := Types{Int: 1, PointerB: addr(1)}
			err := json.Unmarshal([]byte(`{
				"Int":      "1",
				"PointerB": "1"
			}`), &got)
			switch {
			case !reflect.DeepEqual(got, want):
				t.Fatalf("json.Unmarshal = %v, want %v", got, want)
			case err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			}
		})

		t.Run(path.Join("Unmarshal/Deep", json.Version), func(t *testing.T) {
			var got Types
			want := map[string]Types{
				"v1": {
					Map:      map[string]int{"Name": 0},
					Slice:    []int{0},
					PointerC: addr(addr(0)),
				},
				"v2": {
					Map:      map[string]int{"Name": 1},
					Struct:   struct{ Field int }{1},
					Slice:    []int{1},
					Array:    [1]int{1},
					PointerC: addr(addr(1)),
				},
			}[json.Version]
			err := json.Unmarshal([]byte(`{
				"Map":      {"Name":"1"},
				"Struct":   {"Field":"1"},
				"Slice":    ["1"],
				"Array":    ["1"],
				"PointerC": "1"
			}`), &got)
			switch {
			case !reflect.DeepEqual(got, want):
				t.Fatalf("json.Unmarshal =\n%v, want\n%v", got, want)
			case json.Version == "v1" && err == nil:
				t.Fatal("json.Unmarshal error is nil, want non-nil")
			case json.Version == "v2" && err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			}
		})
	}
}

// In v1, nil slices and maps are marshaled as a JSON null.
// In v2, nil slices and maps are marshaled as an empty JSON object or array.
//
// Users of v2 can opt into the v1 behavior by setting
// the "format:emitnull" option in the `json` struct field tag:
//
//	struct {
//		S []string          `json:",format:emitnull"`
//		M map[string]string `json:",format:emitnull"`
//	}
//
// JSON is a language-agnostic data interchange format.
// The fact that maps and slices are nil-able in Go is a semantic detail of the
// Go language. We should avoid leaking such details to the JSON representation.
// When JSON implementations leak language-specific details,
// it complicates transition to/from languages with different type systems.
//
// Furthermore, consider two related Go types: string and []byte.
// It's an asymmetric oddity of v1 that zero values of string and []byte marshal
// as an empty JSON string for the former, while the latter as a JSON null.
// The non-zero values of those types always marshal as JSON strings.
//
// Related issues:
//
//	https://go.dev/issue/27589
//	https://go.dev/issue/37711
func TestNilSlicesAndMaps(t *testing.T) {
	type Composites struct {
		B []byte            // always encoded in v2 as a JSON string
		S []string          // always encoded in v2 as a JSON array
		M map[string]string // always encoded in v2 as a JSON object
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Marshal", json.Version), func(t *testing.T) {
			in := []Composites{
				{B: []byte(nil), S: []string(nil), M: map[string]string(nil)},
				{B: []byte{}, S: []string{}, M: map[string]string{}},
			}
			want := map[string]string{
				"v1": `[{"B":null,"S":null,"M":null},{"B":"","S":[],"M":{}}]`,
				"v2": `[{"B":"","S":[],"M":{}},{"B":"","S":[],"M":{}}]`, // v2 emits nil slices and maps as empty JSON objects and arrays
			}[json.Version]
			got, err := json.Marshal(in)
			if err != nil {
				t.Fatalf("json.Marshal error: %v", err)
			}
			if string(got) != want {
				t.Fatalf("json.Marshal = %s, want %s", got, want)
			}
		})
	}
}

// In v1, unmarshaling into a Go array permits JSON arrays with any length.
// In v2, unmarshaling into a Go array requires that the JSON array
// have the exact same number of elements as the Go array.
//
// Go arrays are often used because the exact length has significant meaning.
// Ignoring this detail seems like a mistake. Also, the v1 behavior leads to
// silent data loss when excess JSON array elements are discarded.
func TestArrays(t *testing.T) {
	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal/TooFew", json.Version), func(t *testing.T) {
			var got [2]int
			err := json.Unmarshal([]byte(`[1]`), &got)
			switch {
			case got != [2]int{1, 0}:
				t.Fatalf(`json.Unmarshal = %v, want [1 0]`, got)
			case json.Version == "v1" && err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			case json.Version == "v2" && err == nil:
				t.Fatal("json.Unmarshal error is nil, want non-nil")
			}
		})
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal/TooMany", json.Version), func(t *testing.T) {
			var got [2]int
			err := json.Unmarshal([]byte(`[1,2,3]`), &got)
			switch {
			case got != [2]int{1, 2}:
				t.Fatalf(`json.Unmarshal = %v, want [1 2]`, got)
			case json.Version == "v1" && err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			case json.Version == "v2" && err == nil:
				t.Fatal("json.Unmarshal error is nil, want non-nil")
			}
		})
	}
}

// In v1, byte arrays are treated as arrays of unsigned integers.
// In v2, byte arrays are treated as binary values (similar to []byte).
// This is to make the behavior of [N]byte and []byte more consistent.
//
// Users of v2 can opt into the v1 behavior by setting
// the "format:array" option in the `json` struct field tag:
//
//	struct {
//		B [32]byte `json:",format:array"`
//	}
func TestByteArrays(t *testing.T) {
	for _, json := range jsonPackages {
		t.Run(path.Join("Marshal", json.Version), func(t *testing.T) {
			in := [4]byte{1, 2, 3, 4}
			got, err := json.Marshal(in)
			if err != nil {
				t.Fatalf("json.Marshal error: %v", err)
			}
			want := map[string]string{
				"v1": `[1,2,3,4]`,
				"v2": `"AQIDBA=="`,
			}[json.Version]
			if string(got) != want {
				t.Fatalf("json.Marshal = %s, want %s", got, want)
			}
		})
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal", json.Version), func(t *testing.T) {
			in := map[string]string{
				"v1": `[1,2,3,4]`,
				"v2": `"AQIDBA=="`,
			}[json.Version]
			var got [4]byte
			err := json.Unmarshal([]byte(in), &got)
			switch {
			case err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			case got != [4]byte{1, 2, 3, 4}:
				t.Fatalf("json.Unmarshal = %v, want [1 2 3 4]", got)
			}
		})
	}
}

// CallCheck implements json.{Marshaler,Unmarshaler} on a pointer receiver.
type CallCheck string

// MarshalJSON always returns a JSON string with the literal "CALLED".
func (*CallCheck) MarshalJSON() ([]byte, error) {
	return []byte(`"CALLED"`), nil
}

// UnmarshalJSON always stores a string with the literal "CALLED".
func (v *CallCheck) UnmarshalJSON([]byte) error {
	*v = `CALLED`
	return nil
}

// In v1, the implementation is inconsistent about whether it calls
// MarshalJSON and UnmarshalJSON methods declared on pointer receivers
// when it has an unaddressable value (per reflect.Value.CanAddr) on hand.
// When marshaling, it never boxes the value on the heap to make it addressable,
// while it sometimes boxes values (e.g., for map entries) when unmarshaling.
//
// In v2, the implementation always calls MarshalJSON and UnmarshalJSON methods
// by boxing the value on the heap if necessary.
//
// The v1 behavior is surprising at best and buggy at worst.
// Unfortunately, it cannot be changed without breaking existing usages.
//
// Related issues:
//
//	https://go.dev/issue/27722
//	https://go.dev/issue/33993
//	https://go.dev/issue/42508
func TestPointerReceiver(t *testing.T) {
	type Values struct {
		S []CallCheck
		A [1]CallCheck
		M map[string]CallCheck
		V CallCheck
		I any
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Marshal", json.Version), func(t *testing.T) {
			var cc CallCheck
			in := Values{
				S: []CallCheck{cc},
				A: [1]CallCheck{cc},             // MarshalJSON not called on v1
				M: map[string]CallCheck{"": cc}, // MarshalJSON not called on v1
				V: cc,                           // MarshalJSON not called on v1
				I: cc,                           // MarshalJSON not called on v1
			}
			want := map[string]string{
				"v1": `{"S":["CALLED"],"A":[""],"M":{"":""},"V":"","I":""}`,
				"v2": `{"S":["CALLED"],"A":["CALLED"],"M":{"":"CALLED"},"V":"CALLED","I":"CALLED"}`,
			}[json.Version]
			got, err := json.Marshal(in)
			if err != nil {
				t.Fatalf("json.Marshal error: %v", err)
			}
			if string(got) != want {
				t.Fatalf("json.Marshal = %s, want %s", got, want)
			}
		})
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal", json.Version), func(t *testing.T) {
			in := `{"S":[""],"A":[""],"M":{"":""},"V":"","I":""}`
			called := CallCheck("CALLED") // resulting state if UnmarshalJSON is called
			want := map[string]Values{
				"v1": {
					S: []CallCheck{called},
					A: [1]CallCheck{called},
					M: map[string]CallCheck{"": called},
					V: called,
					I: "", // UnmarshalJSON not called on v1; replaced with Go string
				},
				"v2": {
					S: []CallCheck{called},
					A: [1]CallCheck{called},
					M: map[string]CallCheck{"": called},
					V: called,
					I: called,
				},
			}[json.Version]
			got := Values{
				A: [1]CallCheck{CallCheck("")},
				S: []CallCheck{CallCheck("")},
				M: map[string]CallCheck{"": CallCheck("")},
				V: CallCheck(""),
				I: CallCheck(""),
			}
			if err := json.Unmarshal([]byte(in), &got); err != nil {
				t.Fatalf("json.Unmarshal error: %v", err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("json.Unmarshal = %v, want %v", got, want)
			}
		})
	}
}

// In v1, maps are marshaled in a deterministic order.
// In v2, maps are marshaled in a non-deterministic order.
//
// The reason for the change is that v2 prioritizes performance and
// the guarantee that marshaling operates primarily in a streaming manner.
//
// The v2 API provides jsontext.Value.Canonicalize if stability is needed:
//
//	(*jsontext.Value)(&b).Canonicalize()
//
// Related issue:
//
//	https://go.dev/issue/7872
//	https://go.dev/issue/33714
func TestMapDeterminism(t *testing.T) {
	const iterations = 10
	in := map[int]int{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}

	for _, json := range jsonPackages {
		t.Run(path.Join("Marshal", json.Version), func(t *testing.T) {
			outs := make(map[string]bool)
			for range iterations {
				b, err := json.Marshal(in)
				if err != nil {
					t.Fatalf("json.Marshal error: %v", err)
				}
				outs[string(b)] = true
			}
			switch {
			case json.Version == "v1" && len(outs) != 1:
				t.Fatalf("json.Marshal encoded to %d unique forms, expected 1", len(outs))
			case json.Version == "v2" && len(outs) == 1:
				t.Logf("json.Marshal encoded to 1 unique form by chance; are you feeling lucky?")
			}
		})
	}
}

// In v1, JSON string encoding escapes special characters related to HTML.
// In v2, JSON string encoding uses a normalized representation (per RFC 8785).
//
// Users of v2 can opt into the v1 behavior by setting EscapeForHTML and EscapeForJS.
//
// Escaping HTML-specific characters in a JSON library is a layering violation.
// It presumes that JSON is always used with HTML and ignores other
// similar classes of injection attacks (e.g., SQL injection).
// Users of JSON with HTML should either manually ensure that embedded JSON is
// properly escaped or be relying on a module like "github.com/google/safehtml"
// to handle safe interoperability of JSON and HTML.
func TestEscapeHTML(t *testing.T) {
	for _, json := range jsonPackages {
		t.Run(path.Join("Marshal", json.Version), func(t *testing.T) {
			const in = `<script> console.log("Hello, world!"); </script>`
			got, err := json.Marshal(in)
			if err != nil {
				t.Fatalf("json.Marshal error: %v", err)
			}
			want := map[string]string{
				"v1": `"\u003cscript\u003e console.log(\"Hello, world!\"); \u003c/script\u003e"`,
				"v2": `"<script> console.log(\"Hello, world!\"); </script>"`,
			}[json.Version]
			if string(got) != want {
				t.Fatalf("json.Marshal = %s, want %s", got, want)
			}
		})
	}
}

// In v1, JSON serialization silently ignored invalid UTF-8 by
// replacing such bytes with the Unicode replacement character.
// In v2, JSON serialization reports an error if invalid UTF-8 is encountered.
//
// Users of v2 can opt into the v1 behavior by setting [AllowInvalidUTF8].
//
// Silently allowing invalid UTF-8 causes data corruption that can be difficult
// to detect until it is too late. Once it has been discovered, strict UTF-8
// behavior sometimes cannot be enabled since other logic may be depending
// on the current behavior due to Hyrum's Law.
//
// Tim Bray, the author of RFC 8259 recommends that implementations should
// go beyond RFC 8259 and instead target compliance with RFC 7493,
// which makes strict decisions about behavior left undefined in RFC 8259.
// In particular, RFC 7493 rejects the presence of invalid UTF-8.
// See https://www.tbray.org/ongoing/When/201x/2017/12/14/RFC-8259-STD-90
func TestInvalidUTF8(t *testing.T) {
	for _, json := range jsonPackages {
		t.Run(path.Join("Marshal", json.Version), func(t *testing.T) {
			got, err := json.Marshal("\xff")
			switch {
			case json.Version == "v1" && err != nil:
				t.Fatalf("json.Marshal error: %v", err)
			case json.Version == "v1" && string(got) != "\"\ufffd\"":
				t.Fatalf(`json.Marshal = %s, want %q`, got, "\ufffd")
			case json.Version == "v2" && err == nil:
				t.Fatal("json.Marshal error is nil, want non-nil")
			}
		})
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal", json.Version), func(t *testing.T) {
			const in = "\"\xff\""
			var got string
			err := json.Unmarshal([]byte(in), &got)
			switch {
			case json.Version == "v1" && err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			case json.Version == "v1" && got != "\ufffd":
				t.Fatalf(`json.Unmarshal = %q, want "\ufffd"`, got)
			case json.Version == "v2" && err == nil:
				t.Fatal("json.Unmarshal error is nil, want non-nil")
			}
		})
	}
}

// In v1, duplicate JSON object names are permitted by default where
// they follow the inconsistent and difficult-to-explain merge semantics of v1.
// In v2, duplicate JSON object names are rejected by default where
// they follow the merge semantics of v2 based on RFC 7396.
//
// Users of v2 can opt into the v1 behavior by setting [AllowDuplicateNames].
//
// Per RFC 8259, the handling of duplicate names is left as undefined behavior.
// Rejecting such inputs is within the realm of valid behavior.
// Tim Bray, the author of RFC 8259 recommends that implementations should
// go beyond RFC 8259 and instead target compliance with RFC 7493,
// which makes strict decisions about behavior left undefined in RFC 8259.
// In particular, RFC 7493 rejects the presence of duplicate object names.
// See https://www.tbray.org/ongoing/When/201x/2017/12/14/RFC-8259-STD-90
//
// The lack of duplicate name rejection has correctness implications where
// roundtrip unmarshal/marshal do not result in semantically equivalent JSON.
// This is surprising behavior for users when they accidentally
// send JSON objects with duplicate names.
//
// The lack of duplicate name rejection may have security implications since it
// becomes difficult for a security tool to validate the semantic meaning of a
// JSON object since meaning is undefined in the presence of duplicate names.
// See https://labs.bishopfox.com/tech-blog/an-exploration-of-json-interoperability-vulnerabilities
//
// Related issue:
//
//	https://go.dev/issue/48298
func TestDuplicateNames(t *testing.T) {
	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal", json.Version), func(t *testing.T) {
			const in = `{"Name":1,"Name":2}`
			var got struct{ Name int }
			err := json.Unmarshal([]byte(in), &got)
			switch {
			case json.Version == "v1" && err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			case json.Version == "v1" && got != struct{ Name int }{2}:
				t.Fatalf(`json.Unmarshal = %v, want {2}`, got)
			case json.Version == "v2" && err == nil:
				t.Fatal("json.Unmarshal error is nil, want non-nil")
			}
		})
	}
}

// In v1, unmarshaling a JSON null into a non-empty value was inconsistent
// in that sometimes it would be ignored and other times clear the value.
// In v2, unmarshaling a JSON null into a non-empty value would consistently
// always clear the value regardless of the value's type.
//
// The purpose of this change is to have consistent behavior with how JSON nulls
// are handled during Unmarshal. This semantic detail has no effect
// when Unmarshaling into a empty value.
//
// Related issues:
//
//	https://go.dev/issue/22177
//	https://go.dev/issue/33835
func TestMergeNull(t *testing.T) {
	type Types struct {
		Bool      bool
		String    string
		Bytes     []byte
		Int       int
		Map       map[string]string
		Struct    struct{ Field string }
		Slice     []string
		Array     [1]string
		Pointer   *string
		Interface any
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal", json.Version), func(t *testing.T) {
			// Start with a non-empty value where all fields are populated.
			in := Types{
				Bool:      true,
				String:    "old",
				Bytes:     []byte("old"),
				Int:       1234,
				Map:       map[string]string{"old": "old"},
				Struct:    struct{ Field string }{"old"},
				Slice:     []string{"old"},
				Array:     [1]string{"old"},
				Pointer:   new(string),
				Interface: "old",
			}

			// Unmarshal a JSON null into every field.
			if err := json.Unmarshal([]byte(`{
				"Bool":      null,
				"String":    null,
				"Bytes":     null,
				"Int":       null,
				"Map":       null,
				"Struct":    null,
				"Slice":     null,
				"Array":     null,
				"Pointer":   null,
				"Interface": null
			}`), &in); err != nil {
				t.Fatalf("json.Unmarshal error: %v", err)
			}

			want := map[string]Types{
				"v1": {
					Bool:   true,
					String: "old",
					Int:    1234,
					Struct: struct{ Field string }{"old"},
					Array:  [1]string{"old"},
				},
				"v2": {}, // all fields are zeroed
			}[json.Version]
			if !reflect.DeepEqual(in, want) {
				t.Fatalf("json.Unmarshal = %+v, want %+v", in, want)
			}
		})
	}
}

// In v1, merge semantics are inconsistent and difficult to explain.
// In v2, merge semantics replaces the destination value for anything
// other than a JSON object, and recursively merges JSON objects.
//
// Merge semantics in v1 are inconsistent and difficult to explain
// largely because the behavior came about organically, rather than
// having a principled approach to how the semantics should operate.
// In v2, merging follows behavior based on RFC 7396.
//
// Related issues:
//
//	https://go.dev/issue/21092
//	https://go.dev/issue/26946
//	https://go.dev/issue/27172
//	https://go.dev/issue/30701
//	https://go.dev/issue/31924
//	https://go.dev/issue/43664
func TestMergeComposite(t *testing.T) {
	type Tuple struct{ Old, New bool }
	type Composites struct {
		Slice            []Tuple
		Array            [1]Tuple
		Map              map[string]Tuple
		MapPointer       map[string]*Tuple
		Struct           struct{ Tuple Tuple }
		StructPointer    *struct{ Tuple Tuple }
		Interface        any
		InterfacePointer any
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal", json.Version), func(t *testing.T) {
			// Start with a non-empty value where all fields are populated.
			in := Composites{
				Slice:            []Tuple{{Old: true}, {Old: true}}[:1],
				Array:            [1]Tuple{{Old: true}},
				Map:              map[string]Tuple{"Tuple": {Old: true}},
				MapPointer:       map[string]*Tuple{"Tuple": {Old: true}},
				Struct:           struct{ Tuple Tuple }{Tuple{Old: true}},
				StructPointer:    &struct{ Tuple Tuple }{Tuple{Old: true}},
				Interface:        Tuple{Old: true},
				InterfacePointer: &Tuple{Old: true},
			}

			// Unmarshal into every pre-populated field.
			if err := json.Unmarshal([]byte(`{
				"Slice":            [{"New":true}, {"New":true}],
				"Array":            [{"New":true}],
				"Map":              {"Tuple": {"New":true}},
				"MapPointer":       {"Tuple": {"New":true}},
				"Struct":           {"Tuple": {"New":true}},
				"StructPointer":    {"Tuple": {"New":true}},
				"Interface":        {"New":true},
				"InterfacePointer": {"New":true}
			}`), &in); err != nil {
				t.Fatalf("json.Unmarshal error: %v", err)
			}

			merged := Tuple{Old: true, New: true}
			replaced := Tuple{Old: false, New: true}
			want := map[string]Composites{
				"v1": {
					Slice:            []Tuple{merged, merged},               // merged
					Array:            [1]Tuple{merged},                      // merged
					Map:              map[string]Tuple{"Tuple": replaced},   // replaced
					MapPointer:       map[string]*Tuple{"Tuple": &replaced}, // replaced
					Struct:           struct{ Tuple Tuple }{merged},         // merged (same as v2)
					StructPointer:    &struct{ Tuple Tuple }{merged},        // merged (same as v2)
					Interface:        map[string]any{"New": true},           // replaced
					InterfacePointer: &merged,                               // merged (same as v2)
				},
				"v2": {
					Slice:            []Tuple{replaced, replaced},         // replaced
					Array:            [1]Tuple{replaced},                  // replaced
					Map:              map[string]Tuple{"Tuple": merged},   // merged
					MapPointer:       map[string]*Tuple{"Tuple": &merged}, // merged
					Struct:           struct{ Tuple Tuple }{merged},       // merged (same as v1)
					StructPointer:    &struct{ Tuple Tuple }{merged},      // merged (same as v1)
					Interface:        merged,                              // merged
					InterfacePointer: &merged,                             // merged (same as v1)
				},
			}[json.Version]
			if !reflect.DeepEqual(in, want) {
				t.Fatalf("json.Unmarshal = %+v, want %+v", in, want)
			}
		})
	}
}

// In v1, there was no special support for time.Duration,
// which resulted in that type simply being treated as a signed integer.
// In v2, there is now first-class support for time.Duration, where the type is
// formatted and parsed using time.Duration.String and time.ParseDuration.
//
// Users of v2 can opt into the v1 behavior by setting
// the "format:nano" option in the `json` struct field tag:
//
//	struct {
//		Duration time.Duration `json:",format:nano"`
//	}
//
// Related issue:
//
//	https://go.dev/issue/10275
func TestTimeDurations(t *testing.T) {
	t.SkipNow() // TODO(https://go.dev/issue/71631): The default representation of time.Duration is still undecided.
	for _, json := range jsonPackages {
		t.Run(path.Join("Marshal", json.Version), func(t *testing.T) {
			got, err := json.Marshal(time.Minute)
			switch {
			case err != nil:
				t.Fatalf("json.Marshal error: %v", err)
			case json.Version == "v1" && string(got) != "60000000000":
				t.Fatalf("json.Marshal = %s, want 60000000000", got)
			case json.Version == "v2" && string(got) != `"1m0s"`:
				t.Fatalf(`json.Marshal = %s, want "1m0s"`, got)
			}
		})
	}

	for _, json := range jsonPackages {
		t.Run(path.Join("Unmarshal", json.Version), func(t *testing.T) {
			in := map[string]string{
				"v1": "60000000000",
				"v2": `"1m0s"`,
			}[json.Version]
			var got time.Duration
			err := json.Unmarshal([]byte(in), &got)
			switch {
			case err != nil:
				t.Fatalf("json.Unmarshal error: %v", err)
			case got != time.Minute:
				t.Fatalf("json.Unmarshal = %v, want 1m0s", got)
			}
		})
	}
}

// In v1, non-empty structs without any JSON serializable fields are permitted.
// In v2, non-empty structs without any JSON serializable fields are rejected.
//
// The purpose of this change is to avoid a common pitfall for new users
// where they expect JSON serialization to handle unexported fields.
// However, this does not work since Go reflection does not
// provide the package the ability to mutate such fields.
// Rejecting unserializable structs in v2 is intended to be a clear signal
// that the type is not supposed to be serialized.
func TestEmptyStructs(t *testing.T) {
	never := func(string) bool { return false }
	onlyV2 := func(v string) bool { return v == "v2" }
	values := []struct {
		in        any
		wantError func(string) bool
	}{
		// It is okay to marshal a truly empty struct in v1 and v2.
		{in: addr(struct{}{}), wantError: never},
		// In v1, a non-empty struct without exported fields
		// is equivalent to an empty struct, but is rejected in v2.
		// Note that errors.errorString type has only unexported fields.
		{in: errors.New("error"), wantError: onlyV2},
		// A mix of exported and unexported fields is permitted.
		{in: addr(struct{ Exported, unexported int }{}), wantError: never},
	}

	for _, json := range jsonPackages {
		t.Run("Marshal", func(t *testing.T) {
			for _, value := range values {
				wantError := value.wantError(json.Version)
				_, err := json.Marshal(value.in)
				switch {
				case (err == nil) && wantError:
					t.Fatalf("json.Marshal error is nil, want non-nil")
				case (err != nil) && !wantError:
					t.Fatalf("json.Marshal error: %v", err)
				}
			}
		})
	}

	for _, json := range jsonPackages {
		t.Run("Unmarshal", func(t *testing.T) {
			for _, value := range values {
				wantError := value.wantError(json.Version)
				out := reflect.New(reflect.TypeOf(value.in).Elem()).Interface()
				err := json.Unmarshal([]byte("{}"), out)
				switch {
				case (err == nil) && wantError:
					t.Fatalf("json.Unmarshal error is nil, want non-nil")
				case (err != nil) && !wantError:
					t.Fatalf("json.Unmarshal error: %v", err)
				}
			}
		})
	}
}
