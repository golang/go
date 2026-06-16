// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"encoding"
	"errors"
	"reflect"
	"testing"

	"encoding/json/internal/jsontest"
	"encoding/json/jsontext"
)

type unexported struct{}

func TestMakeStructFields(t *testing.T) {
	type Embed struct {
		Foo string
	}
	type Recursive struct {
		A          string
		*Recursive `json:",embed"`
		B          string
	}
	type MapStringAny map[string]any
	tests := []struct {
		name    jsontest.CaseName
		in      any
		want    structFields
		wantErr error
	}{{
		name: jsontest.Name("Names"),
		in: struct {
			F1 string
			F2 string `json:"-"`
			F3 string `json:"json_name"`
			f3 string
			F5 string `json:"json_name_nocase,case:ignore"`
		}{},
		want: structFields{
			flattened: []structField{
				{id: 0, index: []int{0}, typ: stringType, fieldOptions: fieldOptions{name: "F1", quotedName: `"F1"`}},
				{id: 1, index: []int{2}, typ: stringType, fieldOptions: fieldOptions{name: "json_name", quotedName: `"json_name"`, hasName: true}},
				{id: 2, index: []int{4}, typ: stringType, fieldOptions: fieldOptions{name: "json_name_nocase", quotedName: `"json_name_nocase"`, hasName: true, casing: caseIgnore}},
			},
		},
	}, {
		name: jsontest.Name("BreadthFirstSearch"),
		in: struct {
			L1A string
			L1B struct {
				L2A string
				L2B struct {
					L3A string
				} `json:",embed"`
				L2C string
			} `json:",embed"`
			L1C string
			L1D struct {
				L2D string
				L2E struct {
					L3B string
				} `json:",embed"`
				L2F string
			} `json:",embed"`
			L1E string
		}{},
		want: structFields{
			flattened: []structField{
				{id: 0, index: []int{0}, typ: stringType, fieldOptions: fieldOptions{name: "L1A", quotedName: `"L1A"`}},
				{id: 3, index: []int{1, 0}, typ: stringType, fieldOptions: fieldOptions{name: "L2A", quotedName: `"L2A"`}},
				{id: 7, index: []int{1, 1, 0}, typ: stringType, fieldOptions: fieldOptions{name: "L3A", quotedName: `"L3A"`}},
				{id: 4, index: []int{1, 2}, typ: stringType, fieldOptions: fieldOptions{name: "L2C", quotedName: `"L2C"`}},
				{id: 1, index: []int{2}, typ: stringType, fieldOptions: fieldOptions{name: "L1C", quotedName: `"L1C"`}},
				{id: 5, index: []int{3, 0}, typ: stringType, fieldOptions: fieldOptions{name: "L2D", quotedName: `"L2D"`}},
				{id: 8, index: []int{3, 1, 0}, typ: stringType, fieldOptions: fieldOptions{name: "L3B", quotedName: `"L3B"`}},
				{id: 6, index: []int{3, 2}, typ: stringType, fieldOptions: fieldOptions{name: "L2F", quotedName: `"L2F"`}},
				{id: 2, index: []int{4}, typ: stringType, fieldOptions: fieldOptions{name: "L1E", quotedName: `"L1E"`}},
			},
		},
	}, {
		name: jsontest.Name("NameResolution"),
		in: struct {
			X1 struct {
				X struct {
					A string // loses in precedence to A
					B string // cancels out with X2.X.B
					D string // loses in precedence to D
				} `json:",embed"`
			} `json:",embed"`
			X2 struct {
				X struct {
					B string // cancels out with X1.X.B
					C string
					D string // loses in precedence to D
				} `json:",embed"`
			} `json:",embed"`
			A string // takes precedence over X1.X.A
			D string // takes precedence over X1.X.D and X2.X.D
		}{},
		want: structFields{
			flattened: []structField{
				{id: 2, index: []int{1, 0, 1}, typ: stringType, fieldOptions: fieldOptions{name: "C", quotedName: `"C"`}},
				{id: 0, index: []int{2}, typ: stringType, fieldOptions: fieldOptions{name: "A", quotedName: `"A"`}},
				{id: 1, index: []int{3}, typ: stringType, fieldOptions: fieldOptions{name: "D", quotedName: `"D"`}},
			},
		},
	}, {
		name: jsontest.Name("NameResolution/ExplicitNameUniquePrecedence"),
		in: struct {
			X1 struct {
				A string // loses in precedence to X2.A
			} `json:",embed"`
			X2 struct {
				A string `json:"A"`
			} `json:",embed"`
			X3 struct {
				A string // loses in precedence to X2.A
			} `json:",embed"`
		}{},
		want: structFields{
			flattened: []structField{
				{id: 0, index: []int{1, 0}, typ: stringType, fieldOptions: fieldOptions{hasName: true, name: "A", quotedName: `"A"`}},
			},
		},
	}, {
		name: jsontest.Name("NameResolution/ExplicitNameCancelsOut"),
		in: struct {
			X1 struct {
				A string // loses in precedence to X2.A or X3.A
			} `json:",embed"`
			X2 struct {
				A string `json:"A"` // cancels out with X3.A
			} `json:",embed"`
			X3 struct {
				A string `json:"A"` // cancels out with X2.A
			} `json:",embed"`
		}{},
		want: structFields{flattened: []structField{}},
	}, {
		name: jsontest.Name("Embed/Implicit"),
		in: struct {
			Embed
		}{},
		want: structFields{
			flattened: []structField{
				{id: 0, index: []int{0, 0}, typ: stringType, fieldOptions: fieldOptions{name: "Foo", quotedName: `"Foo"`}},
			},
		},
	}, {
		name: jsontest.Name("Embed/Explicit"),
		in: struct {
			Embed `json:",embed"`
		}{},
		want: structFields{
			flattened: []structField{
				{id: 0, index: []int{0, 0}, typ: stringType, fieldOptions: fieldOptions{name: "Foo", quotedName: `"Foo"`}},
			},
		},
	}, {
		name: jsontest.Name("Recursive"),
		in: struct {
			A         string
			Recursive `json:",embed"`
			C         string
		}{},
		want: structFields{
			flattened: []structField{
				{id: 0, index: []int{0}, typ: stringType, fieldOptions: fieldOptions{name: "A", quotedName: `"A"`}},
				{id: 2, index: []int{1, 2}, typ: stringType, fieldOptions: fieldOptions{name: "B", quotedName: `"B"`}},
				{id: 1, index: []int{2}, typ: stringType, fieldOptions: fieldOptions{name: "C", quotedName: `"C"`}},
			},
		},
	}, {
		name: jsontest.Name("EmbeddedFallback/Cancelation"),
		in: struct {
			X1 struct {
				X jsontext.Value `json:",embed"`
			} `json:",embed"`
			X2 struct {
				X map[string]any `json:",embed"`
			} `json:",embed"`
		}{},
		want: structFields{},
	}, {
		name: jsontest.Name("EmbeddedFallback/Precedence"),
		in: struct {
			X1 struct {
				X jsontext.Value `json:",embed"`
			} `json:",embed"`
			X2 struct {
				X map[string]any `json:",embed"`
			} `json:",embed"`
			X map[string]jsontext.Value `json:",embed"`
		}{},
		want: structFields{
			embeddedFallback: &structField{id: 0, index: []int{2}, typ: T[map[string]jsontext.Value](), fieldOptions: fieldOptions{name: "X", quotedName: `"X"`, embed: true}},
		},
	}, {
		name: jsontest.Name("EmbeddedFallback/InvalidImplicit"),
		in: struct {
			MapStringAny
		}{},
		want: structFields{
			flattened: []structField{
				{id: 0, index: []int{0}, typ: reflect.TypeOf(MapStringAny(nil)), fieldOptions: fieldOptions{name: "MapStringAny", quotedName: `"MapStringAny"`}},
			},
		},
		wantErr: errors.New("embedded Go struct field MapStringAny of non-struct type must be explicitly given a JSON name"),
	}, {
		name: jsontest.Name("DuplicateName"),
		in: struct {
			A string `json:"same"`
			B string `json:"same"`
		}{},
		want:    structFields{flattened: []structField{}},
		wantErr: errors.New(`Go struct fields A and B conflict over JSON object name "same"`),
	}, {
		name: jsontest.Name("EmbedWithOptions"),
		in: struct {
			A struct{} `json:",embed,omitempty"`
		}{},
		wantErr: errors.New("Go struct field A cannot have any options other than `embed` specified"),
	}, {
		name: jsontest.Name("UnknownWithOptions"),
		in: struct {
			A map[string]any `json:",embed,omitempty"`
		}{},
		want: structFields{embeddedFallback: &structField{
			index: []int{0},
			typ:   reflect.TypeFor[map[string]any](),
			fieldOptions: fieldOptions{
				name:       "A",
				quotedName: `"A"`,
				embed:      true,
			},
		}},
		wantErr: errors.New("Go struct field A cannot have any options other than `embed` specified"),
	}, {
		name: jsontest.Name("EmbedTextMarshaler"),
		in: struct {
			A struct{ encoding.TextMarshaler } `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index: []int{0, 0},
			typ:   reflect.TypeFor[encoding.TextMarshaler](),
			fieldOptions: fieldOptions{
				name:       "TextMarshaler",
				quotedName: `"TextMarshaler"`,
			},
		}}},
		wantErr: errors.New(`embedded Go struct field A of type struct { encoding.TextMarshaler } must not implement marshal or unmarshal methods`),
	}, {
		name: jsontest.Name("EmbedTextAppender"),
		in: struct {
			A struct{ encoding.TextAppender } `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index: []int{0, 0},
			typ:   reflect.TypeFor[encoding.TextAppender](),
			fieldOptions: fieldOptions{
				name:       "TextAppender",
				quotedName: `"TextAppender"`,
			},
		}}},
		wantErr: errors.New(`embedded Go struct field A of type struct { encoding.TextAppender } must not implement marshal or unmarshal methods`),
	}, {
		name: jsontest.Name("EmbedJSONMarshaler"),
		in: struct {
			A struct{ Marshaler } `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index: []int{0, 0},
			typ:   reflect.TypeFor[Marshaler](),
			fieldOptions: fieldOptions{
				name:       "Marshaler",
				quotedName: `"Marshaler"`,
			},
		}}},
		wantErr: errors.New(`embedded Go struct field A of type struct { json.Marshaler } must not implement marshal or unmarshal methods`),
	}, {
		name: jsontest.Name("EmbedJSONMarshalerTo"),
		in: struct {
			A struct{ MarshalerTo } `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index: []int{0, 0},
			typ:   reflect.TypeFor[MarshalerTo](),
			fieldOptions: fieldOptions{
				name:       "MarshalerTo",
				quotedName: `"MarshalerTo"`,
			},
		}}},
		wantErr: errors.New(`embedded Go struct field A of type struct { json.MarshalerTo } must not implement marshal or unmarshal methods`),
	}, {
		name: jsontest.Name("EmbedTextUnmarshaler"),
		in: struct {
			A *struct{ encoding.TextUnmarshaler } `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index: []int{0, 0},
			typ:   reflect.TypeFor[encoding.TextUnmarshaler](),
			fieldOptions: fieldOptions{
				name:       "TextUnmarshaler",
				quotedName: `"TextUnmarshaler"`,
			},
		}}},
		wantErr: errors.New(`embedded Go struct field A of type struct { encoding.TextUnmarshaler } must not implement marshal or unmarshal methods`),
	}, {
		name: jsontest.Name("EmbedJSONUnmarshaler"),
		in: struct {
			A *struct{ Unmarshaler } `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index: []int{0, 0},
			typ:   reflect.TypeFor[Unmarshaler](),
			fieldOptions: fieldOptions{
				name:       "Unmarshaler",
				quotedName: `"Unmarshaler"`,
			},
		}}},
		wantErr: errors.New(`embedded Go struct field A of type struct { json.Unmarshaler } must not implement marshal or unmarshal methods`),
	}, {
		name: jsontest.Name("EmbedJSONUnmarshalerFrom"),
		in: struct {
			A struct{ UnmarshalerFrom } `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index: []int{0, 0},
			typ:   reflect.TypeFor[UnmarshalerFrom](),
			fieldOptions: fieldOptions{
				name:       "UnmarshalerFrom",
				quotedName: `"UnmarshalerFrom"`,
			},
		}}},
		wantErr: errors.New(`embedded Go struct field A of type struct { json.UnmarshalerFrom } must not implement marshal or unmarshal methods`),
	}, {
		name: jsontest.Name("EmbedUnsupported/MapIntKey"),
		in: struct {
			A map[int]any `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index:        []int{0},
			typ:          reflect.TypeFor[map[int]any](),
			fieldOptions: fieldOptions{name: "A", quotedName: `"A"`, embed: true},
		}}},
		wantErr: errors.New(`embedded Go struct field A of type map[int]interface {} must be a Go struct, Go map of string key, or jsontext.Value`),
	}, {
		name: jsontest.Name("EmbedUnsupported/MapTextMarshalerStringKey"),
		in: struct {
			A map[nocaseString]any `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index:        []int{0},
			typ:          reflect.TypeFor[map[nocaseString]any](),
			fieldOptions: fieldOptions{name: "A", quotedName: `"A"`, embed: true},
		}}},
		wantErr: errors.New(`embedded map field A of type map[json.nocaseString]interface {} must have a string key that does not implement marshal or unmarshal methods`),
	}, {
		name: jsontest.Name("EmbedUnsupported/MapMarshalerStringKey"),
		in: struct {
			A map[stringMarshalEmpty]any `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index:        []int{0},
			typ:          reflect.TypeFor[map[stringMarshalEmpty]any](),
			fieldOptions: fieldOptions{name: "A", quotedName: `"A"`, embed: true},
		}}},
		wantErr: errors.New(`embedded map field A of type map[json.stringMarshalEmpty]interface {} must have a string key that does not implement marshal or unmarshal methods`),
	}, {
		name: jsontest.Name("EmbedUnsupported/DoublePointer"),
		in: struct {
			A **struct{} `json:",embed"`
		}{},
		want: structFields{flattened: []structField{{
			index:        []int{0},
			typ:          reflect.TypeFor[**struct{}](),
			fieldOptions: fieldOptions{name: "A", quotedName: `"A"`, embed: true},
		}}},
		wantErr: errors.New(`embedded Go struct field A of type *struct {} must be a Go struct, Go map of string key, or jsontext.Value`),
	}, {
		name: jsontest.Name("DuplicateEmbed"),
		in: struct {
			A map[string]any `json:",embed"`
			B jsontext.Value `json:",embed"`
		}{},
		wantErr: errors.New(`embedded Go struct fields A and B cannot both be a Go map or jsontext.Value`),
	}, {
		name: jsontest.Name("DuplicateEmbedEmbed"),
		in: struct {
			A MapStringAny   `json:",embed"`
			B jsontext.Value `json:",embed"`
		}{},
		wantErr: errors.New(`embedded Go struct fields A and B cannot both be a Go map or jsontext.Value`),
	}}

	for _, tt := range tests {
		t.Run(tt.name.Name, func(t *testing.T) {
			got, err := makeStructFields(reflect.TypeOf(tt.in))

			// Sanity check that pointers are consistent.
			pointers := make(map[*structField]bool)
			for i := range got.flattened {
				pointers[&got.flattened[i]] = true
			}
			for _, f := range got.byActualName {
				if !pointers[f] {
					t.Errorf("%s: byActualName pointer not in flattened", tt.name.Where)
				}
			}
			for _, fs := range got.byFoldedName {
				for _, f := range fs {
					if !pointers[f] {
						t.Errorf("%s: byFoldedName pointer not in flattened", tt.name.Where)
					}
				}
			}

			// Zero out fields that are incomparable.
			for i := range got.flattened {
				got.flattened[i].fncs = nil
				got.flattened[i].isEmpty = nil
			}
			if got.embeddedFallback != nil {
				got.embeddedFallback.fncs = nil
				got.embeddedFallback.isEmpty = nil
			}

			// Reproduce maps in want.
			tt.want.byActualName = make(map[string]*structField)
			for i := range tt.want.flattened {
				f := &tt.want.flattened[i]
				tt.want.byActualName[f.name] = f
			}
			tt.want.byFoldedName = make(map[string][]*structField)
			for i, f := range tt.want.flattened {
				foldedName := string(foldName([]byte(f.name)))
				tt.want.byFoldedName[foldedName] = append(tt.want.byFoldedName[foldedName], &tt.want.flattened[i])
			}

			// Only compare underlying error to simplify test logic.
			var gotErr error
			if err != nil {
				gotErr = err.Err
			}

			tt.want.reindex()
			if !reflect.DeepEqual(got, tt.want) || !reflect.DeepEqual(gotErr, tt.wantErr) {
				t.Errorf("%s: makeStructFields(%T):\n\tgot  (%v, %v)\n\twant (%v, %v)", tt.name.Where, tt.in, got, gotErr, tt.want, tt.wantErr)
			}
		})
	}
}

func TestParseTagOptions(t *testing.T) {
	tests := []struct {
		name        jsontest.CaseName
		in          any // must be a struct with a single field
		wantOpts    fieldOptions
		wantIgnored bool
		wantErr     error
	}{{
		name: jsontest.Name("GoName"),
		in: struct {
			FieldName int
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`},
	}, {
		name: jsontest.Name("GoNameWithOptions"),
		in: struct {
			FieldName int `json:",embed"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, embed: true},
	}, {
		name: jsontest.Name("Empty"),
		in: struct {
			V int `json:""`
		}{},
		wantOpts: fieldOptions{name: "V", quotedName: `"V"`},
	}, {
		name: jsontest.Name("Unexported"),
		in: struct {
			v int `json:"Hello"`
		}{},
		wantIgnored: true,
		wantErr:     errors.New("unexported Go struct field v cannot have non-ignored `json:\"Hello\"` tag"),
	}, {
		name: jsontest.Name("UnexportedEmpty"),
		in: struct {
			v int `json:""`
		}{},
		wantIgnored: true,
		wantErr:     errors.New("unexported Go struct field v cannot have non-ignored `json:\"\"` tag"),
	}, {
		name: jsontest.Name("EmbedUnexported"),
		in: struct {
			unexported
		}{},
		wantOpts: fieldOptions{name: "unexported", quotedName: `"unexported"`},
	}, {
		name: jsontest.Name("Ignored"),
		in: struct {
			V int `json:"-"`
		}{},
		wantIgnored: true,
	}, {
		name: jsontest.Name("IgnoredEmbedUnexported"),
		in: struct {
			unexported `json:"-"`
		}{},
		wantIgnored: true,
	}, {
		name: jsontest.Name("DashComma"),
		in: struct {
			V int `json:"-,"`
		}{},
		wantOpts: fieldOptions{hasName: true, name: "-", quotedName: `"-"`},
		wantErr:  errors.New("Go struct field V has malformed `json` tag: invalid trailing ',' character"),
	}, {
		name: jsontest.Name("LatinPunctuationName"),
		in: struct {
			V int `json:"$%-/"`
		}{},
		wantOpts: fieldOptions{hasName: true, name: "$%-/", quotedName: `"$%-/"`},
	}, {
		name: jsontest.Name("LatinDigitsName"),
		in: struct {
			V int `json:"0123456789"`
		}{},
		wantOpts: fieldOptions{hasName: true, name: "0123456789", quotedName: `"0123456789"`},
	}, {
		name: jsontest.Name("LatinUppercaseName"),
		in: struct {
			V int `json:"ABCDEFGHIJKLMOPQRSTUVWXYZ"`
		}{},
		wantOpts: fieldOptions{hasName: true, name: "ABCDEFGHIJKLMOPQRSTUVWXYZ", quotedName: `"ABCDEFGHIJKLMOPQRSTUVWXYZ"`},
	}, {
		name: jsontest.Name("LatinLowercaseName"),
		in: struct {
			V int `json:"abcdefghijklmnopqrstuvwxyz_"`
		}{},
		wantOpts: fieldOptions{hasName: true, name: "abcdefghijklmnopqrstuvwxyz_", quotedName: `"abcdefghijklmnopqrstuvwxyz_"`},
	}, {
		name: jsontest.Name("GreekName"),
		in: struct {
			V string `json:"Ελλάδα"`
		}{},
		wantOpts: fieldOptions{hasName: true, name: "Ελλάδα", quotedName: `"Ελλάδα"`},
	}, {
		name: jsontest.Name("ChineseName"),
		in: struct {
			V string `json:"世界"`
		}{},
		wantOpts: fieldOptions{hasName: true, name: "世界", quotedName: `"世界"`},
	}, {
		name: jsontest.Name("PercentSlashName"),
		in: struct {
			V int `json:"text/html%"`
		}{},
		wantOpts: fieldOptions{hasName: true, name: "text/html%", quotedName: `"text/html%"`},
	}, {
		name: jsontest.Name("PunctuationName"),
		in: struct {
			V string `json:"!#$%&()*+-./:;<=>?@[]^_{|}~ "`
		}{},
		wantOpts: fieldOptions{hasName: true, name: "!#$%&()*+-./:;<=>?@[]^_{|}~ ", quotedName: `"!#$%&()*+-./:;<=>?@[]^_{|}~ "`, nameNeedEscape: true},
	}, {
		name: jsontest.Name("SingleComma"),
		in: struct {
			V int `json:","`
		}{},
		wantOpts: fieldOptions{name: "V", quotedName: `"V"`},
		wantErr:  errors.New("Go struct field V has malformed `json` tag: invalid trailing ',' character"),
	}, {
		name: jsontest.Name("SuperfluousCommas"),
		in: struct {
			V int `json:",,,,\"\",,embed,,,,,"`
		}{},
		wantOpts: fieldOptions{name: "V", quotedName: `"V"`, embed: true},
		wantErr:  errors.New("Go struct field V has malformed `json` tag: invalid character ',' at start of option (expecting Unicode letter)"),
	}, {
		name: jsontest.Name("CaseAloneOption"),
		in: struct {
			FieldName int `json:",case"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`},
		wantErr:  errors.New("Go struct field FieldName is missing value for `case` tag option; specify `case:ignore` or `case:strict` instead"),
	}, {
		name: jsontest.Name("CaseIgnoreOption"),
		in: struct {
			FieldName int `json:",case:ignore"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, casing: caseIgnore},
	}, {
		name: jsontest.Name("CaseStrictOption"),
		in: struct {
			FieldName int `json:",case:strict"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, casing: caseStrict},
	}, {
		name: jsontest.Name("CaseUnknownOption"),
		in: struct {
			FieldName int `json:",case:unknown"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`},
		wantErr:  errors.New("Go struct field FieldName has unknown `case:unknown` tag value"),
	}, {
		name: jsontest.Name("CaseQuotedOption"),
		in: struct {
			FieldName int `json:",case:'ignore'"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`},
		wantErr:  errors.New("Go struct field FieldName has malformed value for `case` tag option: invalid character '\\'' at start of option (expecting Unicode letter)"),
	}, {
		name: jsontest.Name("BothCaseOptions"),
		in: struct {
			FieldName int `json:",case:ignore,case:strict"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, casing: caseIgnore | caseStrict},
		wantErr:  errors.New("Go struct field FieldName cannot have both `case:ignore` and `case:strict` tag options"),
	}, {
		name: jsontest.Name("EmbedOption"),
		in: struct {
			FieldName int `json:",embed"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, embed: true},
	}, {
		name: jsontest.Name("OmitZeroOption"),
		in: struct {
			FieldName int `json:",omitzero"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, omitzero: true},
	}, {
		name: jsontest.Name("OmitEmptyOption"),
		in: struct {
			FieldName int `json:",omitempty"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, omitempty: true},
	}, {
		name: jsontest.Name("StringOption"),
		in: struct {
			FieldName int `json:",string"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, string: true},
	}, {
		name: jsontest.Name("FormatOptionEqual"),
		in: struct {
			FieldName int `json:",format=fizzbuzz"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`},
		wantErr:  errors.New("Go struct field FieldName is missing value for `format` tag option"),
	}, {
		name: jsontest.Name("FormatOptionColon"),
		in: struct {
			FieldName int `json:",format:fizzbuzz"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, format: "fizzbuzz"},
	}, {
		name: jsontest.Name("FormatOptionQuoted"),
		in: struct {
			FieldName int `json:",format:'2006-01-02'"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, format: "2006-01-02"},
	}, {
		name: jsontest.Name("FormatOptionInvalid"),
		in: struct {
			FieldName int `json:",format:'2006-01-02"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`},
		wantErr:  errors.New("Go struct field FieldName has malformed value for `format` tag option: single-quoted string not terminated: '2006-01-0..."),
	}, {
		name: jsontest.Name("FormatOptionNotLast"),
		in: struct {
			FieldName int `json:",format:alpha,ordered"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, format: "alpha"},
		wantErr:  errors.New("Go struct field FieldName has `format` tag option that was not specified last"),
	}, {
		name: jsontest.Name("AllOptions"),
		in: struct {
			FieldName int `json:",case:ignore,embed,omitzero,omitempty,string,format:format"`
		}{},
		wantOpts: fieldOptions{
			name:       "FieldName",
			quotedName: `"FieldName"`,
			casing:     caseIgnore,
			embed:      true,
			omitzero:   true,
			omitempty:  true,
			string:     true,
			format:     "format",
		},
	}, {
		name: jsontest.Name("AllOptionsCaseSensitive"),
		in: struct {
			FieldName int `json:",CASE:IGNORE,INLINE,UNKNOWN,OMITZERO,OMITEMPTY,STRING,FORMAT:FORMAT"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`},
		wantErr:  errors.New("Go struct field FieldName has invalid appearance of `CASE` tag option; specify `case` instead"),
	}, {
		name: jsontest.Name("AllOptionsSpaceSensitive"),
		in: struct {
			FieldName int `json:", case:ignore , embed , omitzero , omitempty , string , format:format "`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`},
		wantErr:  errors.New("Go struct field FieldName has malformed `json` tag: invalid character ' ' at start of option (expecting Unicode letter)"),
	}, {
		name: jsontest.Name("UnknownTagOption"),
		in: struct {
			FieldName int `json:",embed,whoknows,string"`
		}{},
		wantOpts: fieldOptions{name: "FieldName", quotedName: `"FieldName"`, embed: true, string: true},
	}, {
		name: jsontest.Name("MisnamedTag"),
		in: struct {
			V int `jsom:"Misnamed"`
		}{},
		wantOpts: fieldOptions{name: "V", quotedName: `"V"`},
	}}

	for _, tt := range tests {
		t.Run(tt.name.Name, func(t *testing.T) {
			fs := reflect.TypeOf(tt.in).Field(0)
			gotOpts, gotIgnored, gotErr := parseFieldOptions(fs)
			if !reflect.DeepEqual(gotOpts, tt.wantOpts) || gotIgnored != tt.wantIgnored || !reflect.DeepEqual(gotErr, tt.wantErr) {
				t.Errorf("%s: parseFieldOptions(%T) = (\n\t%v,\n\t%v,\n\t%v\n), want (\n\t%v,\n\t%v,\n\t%v\n)", tt.name.Where, tt.in, gotOpts, gotIgnored, gotErr, tt.wantOpts, tt.wantIgnored, tt.wantErr)
			}
		})
	}
}
