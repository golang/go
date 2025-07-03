// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"bytes"
	"encoding"
	"encoding/base32"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/netip"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"encoding/json/internal"
	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
	"encoding/json/internal/jsontest"
	"encoding/json/internal/jsonwire"
	"encoding/json/jsontext"
)

func newNonStringNameError(offset int64, pointer jsontext.Pointer) error {
	return &jsontext.SyntacticError{ByteOffset: offset, JSONPointer: pointer, Err: jsontext.ErrNonStringName}
}

func newInvalidCharacterError(prefix, where string, offset int64, pointer jsontext.Pointer) error {
	return &jsontext.SyntacticError{ByteOffset: offset, JSONPointer: pointer, Err: jsonwire.NewInvalidCharacterError(prefix, where)}
}

func newInvalidUTF8Error(offset int64, pointer jsontext.Pointer) error {
	return &jsontext.SyntacticError{ByteOffset: offset, JSONPointer: pointer, Err: jsonwire.ErrInvalidUTF8}
}

func newParseTimeError(layout, value, layoutElem, valueElem, message string) error {
	return &time.ParseError{Layout: layout, Value: value, LayoutElem: layoutElem, ValueElem: valueElem, Message: message}
}

func EM(err error) *SemanticError {
	return &SemanticError{action: "marshal", Err: err}
}

func EU(err error) *SemanticError {
	return &SemanticError{action: "unmarshal", Err: err}
}

func (e *SemanticError) withVal(val string) *SemanticError {
	e.JSONValue = jsontext.Value(val)
	return e
}

func (e *SemanticError) withPos(prefix string, pointer jsontext.Pointer) *SemanticError {
	e.ByteOffset = int64(len(prefix))
	e.JSONPointer = pointer
	return e
}

func (e *SemanticError) withType(k jsontext.Kind, t reflect.Type) *SemanticError {
	e.JSONKind = k
	e.GoType = t
	return e
}

var (
	errInvalidFormatFlag = errors.New(`invalid format flag "invalid"`)
	errSomeError         = errors.New("some error")
	errMustNotCall       = errors.New("must not call")
)

func T[T any]() reflect.Type { return reflect.TypeFor[T]() }

type (
	jsonObject = map[string]any
	jsonArray  = []any

	namedAny     any
	namedBool    bool
	namedString  string
	NamedString  string
	namedBytes   []byte
	namedInt64   int64
	namedUint64  uint64
	namedFloat64 float64
	namedByte    byte
	netipAddr    = netip.Addr

	recursiveMap     map[string]recursiveMap
	recursiveSlice   []recursiveSlice
	recursivePointer struct{ P *recursivePointer }

	structEmpty       struct{}
	structConflicting struct {
		A string `json:"conflict"`
		B string `json:"conflict"`
	}
	structNoneExported struct {
		unexported string
	}
	structUnexportedIgnored struct {
		ignored string `json:"-"`
	}
	structMalformedTag struct {
		Malformed string `json:"\""`
	}
	structUnexportedTag struct {
		unexported string `json:"name"`
	}
	structExportedEmbedded struct {
		NamedString
	}
	structExportedEmbeddedTag struct {
		NamedString `json:"name"`
	}
	structUnexportedEmbedded struct {
		namedString
	}
	structUnexportedEmbeddedTag struct {
		namedString `json:"name"`
	}
	structUnexportedEmbeddedMethodTag struct {
		// netipAddr cannot be marshaled since the MarshalText method
		// cannot be called on an unexported field.
		netipAddr `json:"name"`

		// Bogus MarshalText and AppendText methods are declared on
		// structUnexportedEmbeddedMethodTag to prevent it from
		// implementing those method interfaces.
	}
	structUnexportedEmbeddedStruct struct {
		structOmitZeroAll
		FizzBuzz int
		structNestedAddr
	}
	structUnexportedEmbeddedStructPointer struct {
		*structOmitZeroAll
		FizzBuzz int
		*structNestedAddr
	}
	structNestedAddr struct {
		Addr netip.Addr
	}
	structIgnoredUnexportedEmbedded struct {
		namedString `json:"-"`
	}
	structWeirdNames struct {
		Empty string `json:"''"`
		Comma string `json:"','"`
		Quote string `json:"'\"'"`
	}
	structNoCase struct {
		Aaa  string `json:",case:strict"`
		AA_A string
		AaA  string `json:",case:ignore"`
		AAa  string `json:",case:ignore"`
		AAA  string
	}
	structScalars struct {
		unexported bool
		Ignored    bool `json:"-"`

		Bool   bool
		String string
		Bytes  []byte
		Int    int64
		Uint   uint64
		Float  float64
	}
	structSlices struct {
		unexported bool
		Ignored    bool `json:"-"`

		SliceBool   []bool
		SliceString []string
		SliceBytes  [][]byte
		SliceInt    []int64
		SliceUint   []uint64
		SliceFloat  []float64
	}
	structMaps struct {
		unexported bool
		Ignored    bool `json:"-"`

		MapBool   map[string]bool
		MapString map[string]string
		MapBytes  map[string][]byte
		MapInt    map[string]int64
		MapUint   map[string]uint64
		MapFloat  map[string]float64
	}
	structAll struct {
		Bool          bool
		String        string
		Bytes         []byte
		Int           int64
		Uint          uint64
		Float         float64
		Map           map[string]string
		StructScalars structScalars
		StructMaps    structMaps
		StructSlices  structSlices
		Slice         []string
		Array         [1]string
		Pointer       *structAll
		Interface     any
	}
	structStringifiedAll struct {
		Bool          bool                  `json:",string"`
		String        string                `json:",string"`
		Bytes         []byte                `json:",string"`
		Int           int64                 `json:",string"`
		Uint          uint64                `json:",string"`
		Float         float64               `json:",string"`
		Map           map[string]string     `json:",string"`
		StructScalars structScalars         `json:",string"`
		StructMaps    structMaps            `json:",string"`
		StructSlices  structSlices          `json:",string"`
		Slice         []string              `json:",string"`
		Array         [1]string             `json:",string"`
		Pointer       *structStringifiedAll `json:",string"`
		Interface     any                   `json:",string"`
	}
	structOmitZeroAll struct {
		Bool          bool               `json:",omitzero"`
		String        string             `json:",omitzero"`
		Bytes         []byte             `json:",omitzero"`
		Int           int64              `json:",omitzero"`
		Uint          uint64             `json:",omitzero"`
		Float         float64            `json:",omitzero"`
		Map           map[string]string  `json:",omitzero"`
		StructScalars structScalars      `json:",omitzero"`
		StructMaps    structMaps         `json:",omitzero"`
		StructSlices  structSlices       `json:",omitzero"`
		Slice         []string           `json:",omitzero"`
		Array         [1]string          `json:",omitzero"`
		Pointer       *structOmitZeroAll `json:",omitzero"`
		Interface     any                `json:",omitzero"`
	}
	structOmitZeroMethodAll struct {
		ValueAlwaysZero                 valueAlwaysZero     `json:",omitzero"`
		ValueNeverZero                  valueNeverZero      `json:",omitzero"`
		PointerAlwaysZero               pointerAlwaysZero   `json:",omitzero"`
		PointerNeverZero                pointerNeverZero    `json:",omitzero"`
		PointerValueAlwaysZero          *valueAlwaysZero    `json:",omitzero"`
		PointerValueNeverZero           *valueNeverZero     `json:",omitzero"`
		PointerPointerAlwaysZero        *pointerAlwaysZero  `json:",omitzero"`
		PointerPointerNeverZero         *pointerNeverZero   `json:",omitzero"`
		PointerPointerValueAlwaysZero   **valueAlwaysZero   `json:",omitzero"`
		PointerPointerValueNeverZero    **valueNeverZero    `json:",omitzero"`
		PointerPointerPointerAlwaysZero **pointerAlwaysZero `json:",omitzero"`
		PointerPointerPointerNeverZero  **pointerNeverZero  `json:",omitzero"`
	}
	structOmitZeroMethodInterfaceAll struct {
		ValueAlwaysZero          isZeroer `json:",omitzero"`
		ValueNeverZero           isZeroer `json:",omitzero"`
		PointerValueAlwaysZero   isZeroer `json:",omitzero"`
		PointerValueNeverZero    isZeroer `json:",omitzero"`
		PointerPointerAlwaysZero isZeroer `json:",omitzero"`
		PointerPointerNeverZero  isZeroer `json:",omitzero"`
	}
	structOmitEmptyAll struct {
		Bool                  bool                    `json:",omitempty"`
		PointerBool           *bool                   `json:",omitempty"`
		String                string                  `json:",omitempty"`
		StringEmpty           stringMarshalEmpty      `json:",omitempty"`
		StringNonEmpty        stringMarshalNonEmpty   `json:",omitempty"`
		PointerString         *string                 `json:",omitempty"`
		PointerStringEmpty    *stringMarshalEmpty     `json:",omitempty"`
		PointerStringNonEmpty *stringMarshalNonEmpty  `json:",omitempty"`
		Bytes                 []byte                  `json:",omitempty"`
		BytesEmpty            bytesMarshalEmpty       `json:",omitempty"`
		BytesNonEmpty         bytesMarshalNonEmpty    `json:",omitempty"`
		PointerBytes          *[]byte                 `json:",omitempty"`
		PointerBytesEmpty     *bytesMarshalEmpty      `json:",omitempty"`
		PointerBytesNonEmpty  *bytesMarshalNonEmpty   `json:",omitempty"`
		Float                 float64                 `json:",omitempty"`
		PointerFloat          *float64                `json:",omitempty"`
		Map                   map[string]string       `json:",omitempty"`
		MapEmpty              mapMarshalEmpty         `json:",omitempty"`
		MapNonEmpty           mapMarshalNonEmpty      `json:",omitempty"`
		PointerMap            *map[string]string      `json:",omitempty"`
		PointerMapEmpty       *mapMarshalEmpty        `json:",omitempty"`
		PointerMapNonEmpty    *mapMarshalNonEmpty     `json:",omitempty"`
		Slice                 []string                `json:",omitempty"`
		SliceEmpty            sliceMarshalEmpty       `json:",omitempty"`
		SliceNonEmpty         sliceMarshalNonEmpty    `json:",omitempty"`
		PointerSlice          *[]string               `json:",omitempty"`
		PointerSliceEmpty     *sliceMarshalEmpty      `json:",omitempty"`
		PointerSliceNonEmpty  *sliceMarshalNonEmpty   `json:",omitempty"`
		Pointer               *structOmitZeroEmptyAll `json:",omitempty"`
		Interface             any                     `json:",omitempty"`
	}
	structOmitZeroEmptyAll struct {
		Bool      bool                    `json:",omitzero,omitempty"`
		String    string                  `json:",omitzero,omitempty"`
		Bytes     []byte                  `json:",omitzero,omitempty"`
		Int       int64                   `json:",omitzero,omitempty"`
		Uint      uint64                  `json:",omitzero,omitempty"`
		Float     float64                 `json:",omitzero,omitempty"`
		Map       map[string]string       `json:",omitzero,omitempty"`
		Slice     []string                `json:",omitzero,omitempty"`
		Array     [1]string               `json:",omitzero,omitempty"`
		Pointer   *structOmitZeroEmptyAll `json:",omitzero,omitempty"`
		Interface any                     `json:",omitzero,omitempty"`
	}
	structFormatBytes struct {
		Base16    []byte `json:",format:base16"`
		Base32    []byte `json:",format:base32"`
		Base32Hex []byte `json:",format:base32hex"`
		Base64    []byte `json:",format:base64"`
		Base64URL []byte `json:",format:base64url"`
		Array     []byte `json:",format:array"`
	}
	structFormatArrayBytes struct {
		Base16    [4]byte `json:",format:base16"`
		Base32    [4]byte `json:",format:base32"`
		Base32Hex [4]byte `json:",format:base32hex"`
		Base64    [4]byte `json:",format:base64"`
		Base64URL [4]byte `json:",format:base64url"`
		Array     [4]byte `json:",format:array"`
		Default   [4]byte
	}
	structFormatFloats struct {
		NonFinite        float64  `json:",format:nonfinite"`
		PointerNonFinite *float64 `json:",format:nonfinite"`
	}
	structFormatMaps struct {
		EmitNull           map[string]string  `json:",format:emitnull"`
		PointerEmitNull    *map[string]string `json:",format:emitnull"`
		EmitEmpty          map[string]string  `json:",format:emitempty"`
		PointerEmitEmpty   *map[string]string `json:",format:emitempty"`
		EmitDefault        map[string]string
		PointerEmitDefault *map[string]string
	}
	structFormatSlices struct {
		EmitNull           []string  `json:",format:emitnull"`
		PointerEmitNull    *[]string `json:",format:emitnull"`
		EmitEmpty          []string  `json:",format:emitempty"`
		PointerEmitEmpty   *[]string `json:",format:emitempty"`
		EmitDefault        []string
		PointerEmitDefault *[]string
	}
	structFormatInvalid struct {
		Bool      bool              `json:",omitzero,format:invalid"`
		String    string            `json:",omitzero,format:invalid"`
		Bytes     []byte            `json:",omitzero,format:invalid"`
		Int       int64             `json:",omitzero,format:invalid"`
		Uint      uint64            `json:",omitzero,format:invalid"`
		Float     float64           `json:",omitzero,format:invalid"`
		Map       map[string]string `json:",omitzero,format:invalid"`
		Struct    structAll         `json:",omitzero,format:invalid"`
		Slice     []string          `json:",omitzero,format:invalid"`
		Array     [1]string         `json:",omitzero,format:invalid"`
		Interface any               `json:",omitzero,format:invalid"`
	}
	structDurationFormat struct {
		D1  time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
		D2  time.Duration `json:",format:units"`
		D3  time.Duration `json:",format:sec"`
		D4  time.Duration `json:",string,format:sec"`
		D5  time.Duration `json:",format:milli"`
		D6  time.Duration `json:",string,format:milli"`
		D7  time.Duration `json:",format:micro"`
		D8  time.Duration `json:",string,format:micro"`
		D9  time.Duration `json:",format:nano"`
		D10 time.Duration `json:",string,format:nano"`
		D11 time.Duration `json:",format:iso8601"`
	}
	structTimeFormat struct {
		T1  time.Time
		T2  time.Time `json:",format:ANSIC"`
		T3  time.Time `json:",format:UnixDate"`
		T4  time.Time `json:",format:RubyDate"`
		T5  time.Time `json:",format:RFC822"`
		T6  time.Time `json:",format:RFC822Z"`
		T7  time.Time `json:",format:RFC850"`
		T8  time.Time `json:",format:RFC1123"`
		T9  time.Time `json:",format:RFC1123Z"`
		T10 time.Time `json:",format:RFC3339"`
		T11 time.Time `json:",format:RFC3339Nano"`
		T12 time.Time `json:",format:Kitchen"`
		T13 time.Time `json:",format:Stamp"`
		T14 time.Time `json:",format:StampMilli"`
		T15 time.Time `json:",format:StampMicro"`
		T16 time.Time `json:",format:StampNano"`
		T17 time.Time `json:",format:DateTime"`
		T18 time.Time `json:",format:DateOnly"`
		T19 time.Time `json:",format:TimeOnly"`
		T20 time.Time `json:",format:'2006-01-02'"`
		T21 time.Time `json:",format:'\"weird\"2006'"`
		T22 time.Time `json:",format:unix"`
		T23 time.Time `json:",string,format:unix"`
		T24 time.Time `json:",format:unixmilli"`
		T25 time.Time `json:",string,format:unixmilli"`
		T26 time.Time `json:",format:unixmicro"`
		T27 time.Time `json:",string,format:unixmicro"`
		T28 time.Time `json:",format:unixnano"`
		T29 time.Time `json:",string,format:unixnano"`
	}
	structInlined struct {
		X             structInlinedL1 `json:",inline"`
		*StructEmbed2                 // implicit inline
	}
	structInlinedL1 struct {
		X            *structInlinedL2 `json:",inline"`
		StructEmbed1 `json:",inline"`
	}
	structInlinedL2        struct{ A, B, C string }
	StructEmbed1           struct{ C, D, E string }
	StructEmbed2           struct{ E, F, G string }
	structUnknownTextValue struct {
		A int            `json:",omitzero"`
		X jsontext.Value `json:",unknown"`
		B int            `json:",omitzero"`
	}
	structInlineTextValue struct {
		A int            `json:",omitzero"`
		X jsontext.Value `json:",inline"`
		B int            `json:",omitzero"`
	}
	structInlinePointerTextValue struct {
		A int             `json:",omitzero"`
		X *jsontext.Value `json:",inline"`
		B int             `json:",omitzero"`
	}
	structInlinePointerInlineTextValue struct {
		X *struct {
			A int
			X jsontext.Value `json:",inline"`
		} `json:",inline"`
	}
	structInlineInlinePointerTextValue struct {
		X struct {
			X *jsontext.Value `json:",inline"`
		} `json:",inline"`
	}
	structInlineMapStringAny struct {
		A int        `json:",omitzero"`
		X jsonObject `json:",inline"`
		B int        `json:",omitzero"`
	}
	structInlinePointerMapStringAny struct {
		A int         `json:",omitzero"`
		X *jsonObject `json:",inline"`
		B int         `json:",omitzero"`
	}
	structInlinePointerInlineMapStringAny struct {
		X *struct {
			A int
			X jsonObject `json:",inline"`
		} `json:",inline"`
	}
	structInlineInlinePointerMapStringAny struct {
		X struct {
			X *jsonObject `json:",inline"`
		} `json:",inline"`
	}
	structInlineMapStringInt struct {
		X map[string]int `json:",inline"`
	}
	structInlineMapNamedStringInt struct {
		X map[namedString]int `json:",inline"`
	}
	structInlineMapNamedStringAny struct {
		A int                 `json:",omitzero"`
		X map[namedString]any `json:",inline"`
		B int                 `json:",omitzero"`
	}
	structNoCaseInlineTextValue struct {
		AAA  string         `json:",omitempty,case:strict"`
		AA_b string         `json:",omitempty"`
		AaA  string         `json:",omitempty,case:ignore"`
		AAa  string         `json:",omitempty,case:ignore"`
		Aaa  string         `json:",omitempty"`
		X    jsontext.Value `json:",inline"`
	}
	structNoCaseInlineMapStringAny struct {
		AAA string     `json:",omitempty"`
		AaA string     `json:",omitempty,case:ignore"`
		AAa string     `json:",omitempty,case:ignore"`
		Aaa string     `json:",omitempty"`
		X   jsonObject `json:",inline"`
	}

	allMethods struct {
		method string // the method that was called
		value  []byte // the raw value to provide or store
	}
	allMethodsExceptJSONv2 struct {
		allMethods
		MarshalJSONTo     struct{} // cancel out MarshalJSONTo method with collision
		UnmarshalJSONFrom struct{} // cancel out UnmarshalJSONFrom method with collision
	}
	allMethodsExceptJSONv1 struct {
		allMethods
		MarshalJSON   struct{} // cancel out MarshalJSON method with collision
		UnmarshalJSON struct{} // cancel out UnmarshalJSON method with collision
	}
	allMethodsExceptText struct {
		allMethods
		MarshalText   struct{} // cancel out MarshalText method with collision
		UnmarshalText struct{} // cancel out UnmarshalText method with collision
	}
	onlyMethodJSONv2 struct {
		allMethods
		MarshalJSON   struct{} // cancel out MarshalJSON method with collision
		UnmarshalJSON struct{} // cancel out UnmarshalJSON method with collision
		MarshalText   struct{} // cancel out MarshalText method with collision
		UnmarshalText struct{} // cancel out UnmarshalText method with collision
	}
	onlyMethodJSONv1 struct {
		allMethods
		MarshalJSONTo     struct{} // cancel out MarshalJSONTo method with collision
		UnmarshalJSONFrom struct{} // cancel out UnmarshalJSONFrom method with collision
		MarshalText       struct{} // cancel out MarshalText method with collision
		UnmarshalText     struct{} // cancel out UnmarshalText method with collision
	}
	onlyMethodText struct {
		allMethods
		MarshalJSONTo     struct{} // cancel out MarshalJSONTo method with collision
		UnmarshalJSONFrom struct{} // cancel out UnmarshalJSONFrom method with collision
		MarshalJSON       struct{} // cancel out MarshalJSON method with collision
		UnmarshalJSON     struct{} // cancel out UnmarshalJSON method with collision
	}

	structMethodJSONv2 struct{ value string }
	structMethodJSONv1 struct{ value string }
	structMethodText   struct{ value string }

	marshalJSONv2Func   func(*jsontext.Encoder) error
	marshalJSONv1Func   func() ([]byte, error)
	appendTextFunc      func([]byte) ([]byte, error)
	marshalTextFunc     func() ([]byte, error)
	unmarshalJSONv2Func func(*jsontext.Decoder) error
	unmarshalJSONv1Func func([]byte) error
	unmarshalTextFunc   func([]byte) error

	nocaseString string

	stringMarshalEmpty    string
	stringMarshalNonEmpty string
	bytesMarshalEmpty     []byte
	bytesMarshalNonEmpty  []byte
	mapMarshalEmpty       map[string]string
	mapMarshalNonEmpty    map[string]string
	sliceMarshalEmpty     []string
	sliceMarshalNonEmpty  []string

	valueAlwaysZero   string
	valueNeverZero    string
	pointerAlwaysZero string
	pointerNeverZero  string

	valueStringer   struct{}
	pointerStringer struct{}

	cyclicA struct {
		B1 cyclicB `json:",inline"`
		B2 cyclicB `json:",inline"`
	}
	cyclicB struct {
		F int
		A *cyclicA `json:",inline"`
	}
)

func (structUnexportedEmbeddedMethodTag) MarshalText() {}
func (structUnexportedEmbeddedMethodTag) AppendText()  {}

func (p *allMethods) MarshalJSONTo(enc *jsontext.Encoder) error {
	if got, want := "MarshalJSONTo", p.method; got != want {
		return fmt.Errorf("called wrong method: got %v, want %v", got, want)
	}
	return enc.WriteValue(p.value)
}
func (p *allMethods) MarshalJSON() ([]byte, error) {
	if got, want := "MarshalJSON", p.method; got != want {
		return nil, fmt.Errorf("called wrong method: got %v, want %v", got, want)
	}
	return p.value, nil
}
func (p *allMethods) MarshalText() ([]byte, error) {
	if got, want := "MarshalText", p.method; got != want {
		return nil, fmt.Errorf("called wrong method: got %v, want %v", got, want)
	}
	return p.value, nil
}

func (p *allMethods) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	p.method = "UnmarshalJSONFrom"
	val, err := dec.ReadValue()
	p.value = val
	return err
}
func (p *allMethods) UnmarshalJSON(val []byte) error {
	p.method = "UnmarshalJSON"
	p.value = val
	return nil
}
func (p *allMethods) UnmarshalText(val []byte) error {
	p.method = "UnmarshalText"
	p.value = val
	return nil
}

func (s structMethodJSONv2) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteToken(jsontext.String(s.value))
}
func (s *structMethodJSONv2) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	tok, err := dec.ReadToken()
	if err != nil {
		return err
	}
	if k := tok.Kind(); k != '"' {
		return EU(nil).withType(k, T[structMethodJSONv2]())
	}
	s.value = tok.String()
	return nil
}

func (s structMethodJSONv1) MarshalJSON() ([]byte, error) {
	return jsontext.AppendQuote(nil, s.value)
}
func (s *structMethodJSONv1) UnmarshalJSON(b []byte) error {
	if k := jsontext.Value(b).Kind(); k != '"' {
		return EU(nil).withType(k, T[structMethodJSONv1]())
	}
	b, _ = jsontext.AppendUnquote(nil, b)
	s.value = string(b)
	return nil
}

func (s structMethodText) MarshalText() ([]byte, error) {
	return []byte(s.value), nil
}
func (s *structMethodText) UnmarshalText(b []byte) error {
	s.value = string(b)
	return nil
}

func (f marshalJSONv2Func) MarshalJSONTo(enc *jsontext.Encoder) error {
	return f(enc)
}
func (f marshalJSONv1Func) MarshalJSON() ([]byte, error) {
	return f()
}
func (f appendTextFunc) AppendText(b []byte) ([]byte, error) {
	return f(b)
}
func (f marshalTextFunc) MarshalText() ([]byte, error) {
	return f()
}
func (f unmarshalJSONv2Func) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	return f(dec)
}
func (f unmarshalJSONv1Func) UnmarshalJSON(b []byte) error {
	return f(b)
}
func (f unmarshalTextFunc) UnmarshalText(b []byte) error {
	return f(b)
}

func (k nocaseString) MarshalText() ([]byte, error) {
	return []byte(strings.ToLower(string(k))), nil
}
func (k *nocaseString) UnmarshalText(b []byte) error {
	*k = nocaseString(strings.ToLower(string(b)))
	return nil
}

func (stringMarshalEmpty) MarshalJSON() ([]byte, error)    { return []byte(`""`), nil }
func (stringMarshalNonEmpty) MarshalJSON() ([]byte, error) { return []byte(`"value"`), nil }
func (bytesMarshalEmpty) MarshalJSON() ([]byte, error)     { return []byte(`[]`), nil }
func (bytesMarshalNonEmpty) MarshalJSON() ([]byte, error)  { return []byte(`["value"]`), nil }
func (mapMarshalEmpty) MarshalJSON() ([]byte, error)       { return []byte(`{}`), nil }
func (mapMarshalNonEmpty) MarshalJSON() ([]byte, error)    { return []byte(`{"key":"value"}`), nil }
func (sliceMarshalEmpty) MarshalJSON() ([]byte, error)     { return []byte(`[]`), nil }
func (sliceMarshalNonEmpty) MarshalJSON() ([]byte, error)  { return []byte(`["value"]`), nil }

func (valueAlwaysZero) IsZero() bool    { return true }
func (valueNeverZero) IsZero() bool     { return false }
func (*pointerAlwaysZero) IsZero() bool { return true }
func (*pointerNeverZero) IsZero() bool  { return false }

func (valueStringer) String() string    { return "" }
func (*pointerStringer) String() string { return "" }

func addr[T any](v T) *T {
	return &v
}

func mustParseTime(layout, value string) time.Time {
	t, err := time.Parse(layout, value)
	if err != nil {
		panic(err)
	}
	return t
}

var invalidFormatOption = &jsonopts.Struct{
	ArshalValues: jsonopts.ArshalValues{FormatDepth: 1000, Format: "invalid"},
}

func TestMarshal(t *testing.T) {
	tests := []struct {
		name    jsontest.CaseName
		opts    []Options
		in      any
		want    string
		wantErr error

		canonicalize bool // canonicalize the output before comparing?
		useWriter    bool // call MarshalWrite instead of Marshal
	}{{
		name: jsontest.Name("Nil"),
		in:   nil,
		want: `null`,
	}, {
		name: jsontest.Name("Bools"),
		in:   []bool{false, true},
		want: `[false,true]`,
	}, {
		name: jsontest.Name("Bools/Named"),
		in:   []namedBool{false, true},
		want: `[false,true]`,
	}, {
		name: jsontest.Name("Bools/NotStringified"),
		opts: []Options{StringifyNumbers(true)},
		in:   []bool{false, true},
		want: `[false,true]`,
	}, {
		name: jsontest.Name("Bools/StringifiedBool"),
		opts: []Options{jsonflags.StringifyBoolsAndStrings | 1},
		in:   []bool{false, true},
		want: `["false","true"]`,
	}, {
		name: jsontest.Name("Bools/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   true,
		want: `true`,
	}, {
		name: jsontest.Name("Strings"),
		in:   []string{"", "hello", "世界"},
		want: `["","hello","世界"]`,
	}, {
		name: jsontest.Name("Strings/Named"),
		in:   []namedString{"", "hello", "世界"},
		want: `["","hello","世界"]`,
	}, {
		name: jsontest.Name("Strings/StringifiedBool"),
		opts: []Options{jsonflags.StringifyBoolsAndStrings | 1},
		in:   []string{"", "hello", "世界"},
		want: `["\"\"","\"hello\"","\"世界\""]`,
	}, {
		name: jsontest.Name("Strings/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   "string",
		want: `"string"`,
	}, {
		name: jsontest.Name("Bytes"),
		in:   [][]byte{nil, {}, {1}, {1, 2}, {1, 2, 3}},
		want: `["","","AQ==","AQI=","AQID"]`,
	}, {
		name: jsontest.Name("Bytes/FormatNilSliceAsNull"),
		opts: []Options{FormatNilSliceAsNull(true)},
		in:   [][]byte{nil, {}},
		want: `[null,""]`,
	}, {
		name: jsontest.Name("Bytes/Large"),
		in:   []byte("the quick brown fox jumped over the lazy dog and ate the homework that I spent so much time on."),
		want: `"dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wZWQgb3ZlciB0aGUgbGF6eSBkb2cgYW5kIGF0ZSB0aGUgaG9tZXdvcmsgdGhhdCBJIHNwZW50IHNvIG11Y2ggdGltZSBvbi4="`,
	}, {
		name: jsontest.Name("Bytes/Named"),
		in:   []namedBytes{nil, {}, {1}, {1, 2}, {1, 2, 3}},
		want: `["","","AQ==","AQI=","AQID"]`,
	}, {
		name: jsontest.Name("Bytes/NotStringified"),
		opts: []Options{StringifyNumbers(true)},
		in:   [][]byte{nil, {}, {1}, {1, 2}, {1, 2, 3}},
		want: `["","","AQ==","AQI=","AQID"]`,
	}, {
		// NOTE: []namedByte is not assignable to []byte,
		// so the following should be treated as a slice of uints.
		name: jsontest.Name("Bytes/Invariant"),
		in:   [][]namedByte{nil, {}, {1}, {1, 2}, {1, 2, 3}},
		want: `[[],[],[1],[1,2],[1,2,3]]`,
	}, {
		// NOTE: This differs in behavior from v1,
		// but keeps the representation of slices and arrays more consistent.
		name: jsontest.Name("Bytes/ByteArray"),
		in:   [5]byte{'h', 'e', 'l', 'l', 'o'},
		want: `"aGVsbG8="`,
	}, {
		// NOTE: []namedByte is not assignable to []byte,
		// so the following should be treated as an array of uints.
		name: jsontest.Name("Bytes/NamedByteArray"),
		in:   [5]namedByte{'h', 'e', 'l', 'l', 'o'},
		want: `[104,101,108,108,111]`,
	}, {
		name: jsontest.Name("Bytes/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   []byte("hello"),
		want: `"aGVsbG8="`,
	}, {
		name: jsontest.Name("Ints"),
		in: []any{
			int(0), int8(math.MinInt8), int16(math.MinInt16), int32(math.MinInt32), int64(math.MinInt64), namedInt64(-6464),
		},
		want: `[0,-128,-32768,-2147483648,-9223372036854775808,-6464]`,
	}, {
		name: jsontest.Name("Ints/Stringified"),
		opts: []Options{StringifyNumbers(true)},
		in: []any{
			int(0), int8(math.MinInt8), int16(math.MinInt16), int32(math.MinInt32), int64(math.MinInt64), namedInt64(-6464),
		},
		want: `["0","-128","-32768","-2147483648","-9223372036854775808","-6464"]`,
	}, {
		name: jsontest.Name("Ints/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   int(0),
		want: `0`,
	}, {
		name: jsontest.Name("Uints"),
		in: []any{
			uint(0), uint8(math.MaxUint8), uint16(math.MaxUint16), uint32(math.MaxUint32), uint64(math.MaxUint64), namedUint64(6464), uintptr(1234),
		},
		want: `[0,255,65535,4294967295,18446744073709551615,6464,1234]`,
	}, {
		name: jsontest.Name("Uints/Stringified"),
		opts: []Options{StringifyNumbers(true)},
		in: []any{
			uint(0), uint8(math.MaxUint8), uint16(math.MaxUint16), uint32(math.MaxUint32), uint64(math.MaxUint64), namedUint64(6464),
		},
		want: `["0","255","65535","4294967295","18446744073709551615","6464"]`,
	}, {
		name: jsontest.Name("Uints/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   uint(0),
		want: `0`,
	}, {
		name: jsontest.Name("Floats"),
		in: []any{
			float32(math.MaxFloat32), float64(math.MaxFloat64), namedFloat64(64.64),
		},
		want: `[3.4028235e+38,1.7976931348623157e+308,64.64]`,
	}, {
		name: jsontest.Name("Floats/Stringified"),
		opts: []Options{StringifyNumbers(true)},
		in: []any{
			float32(math.MaxFloat32), float64(math.MaxFloat64), namedFloat64(64.64),
		},
		want: `["3.4028235e+38","1.7976931348623157e+308","64.64"]`,
	}, {
		name:    jsontest.Name("Floats/Invalid/NaN"),
		opts:    []Options{StringifyNumbers(true)},
		in:      math.NaN(),
		wantErr: EM(fmt.Errorf("unsupported value: %v", math.NaN())).withType(0, float64Type),
	}, {
		name:    jsontest.Name("Floats/Invalid/PositiveInfinity"),
		in:      math.Inf(+1),
		wantErr: EM(fmt.Errorf("unsupported value: %v", math.Inf(+1))).withType(0, float64Type),
	}, {
		name:    jsontest.Name("Floats/Invalid/NegativeInfinity"),
		in:      math.Inf(-1),
		wantErr: EM(fmt.Errorf("unsupported value: %v", math.Inf(-1))).withType(0, float64Type),
	}, {
		name: jsontest.Name("Floats/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   float64(0),
		want: `0`,
	}, {
		name:    jsontest.Name("Maps/InvalidKey/Bool"),
		in:      map[bool]string{false: "value"},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(`{`), "")).withPos(`{`, "").withType(0, boolType),
	}, {
		name:    jsontest.Name("Maps/InvalidKey/NamedBool"),
		in:      map[namedBool]string{false: "value"},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(`{`), "")).withPos(`{`, "").withType(0, T[namedBool]()),
	}, {
		name:    jsontest.Name("Maps/InvalidKey/Array"),
		in:      map[[1]string]string{{"key"}: "value"},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(`{`), "")).withPos(`{`, "").withType(0, T[[1]string]()),
	}, {
		name:    jsontest.Name("Maps/InvalidKey/Channel"),
		in:      map[chan string]string{make(chan string): "value"},
		want:    `{`,
		wantErr: EM(nil).withPos(`{`, "").withType(0, T[chan string]()),
	}, {
		name:         jsontest.Name("Maps/ValidKey/Int"),
		in:           map[int64]string{math.MinInt64: "MinInt64", 0: "Zero", math.MaxInt64: "MaxInt64"},
		canonicalize: true,
		want:         `{"-9223372036854775808":"MinInt64","0":"Zero","9223372036854775807":"MaxInt64"}`,
	}, {
		name:         jsontest.Name("Maps/ValidKey/PointerInt"),
		in:           map[*int64]string{addr(int64(math.MinInt64)): "MinInt64", addr(int64(0)): "Zero", addr(int64(math.MaxInt64)): "MaxInt64"},
		canonicalize: true,
		want:         `{"-9223372036854775808":"MinInt64","0":"Zero","9223372036854775807":"MaxInt64"}`,
	}, {
		name:         jsontest.Name("Maps/DuplicateName/PointerInt"),
		in:           map[*int64]string{addr(int64(0)): "0", addr(int64(0)): "0"},
		canonicalize: true,
		want:         `{"0":"0"`,
		wantErr:      newDuplicateNameError("", []byte(`"0"`), len64(`{"0":"0",`)),
	}, {
		name:         jsontest.Name("Maps/ValidKey/NamedInt"),
		in:           map[namedInt64]string{math.MinInt64: "MinInt64", 0: "Zero", math.MaxInt64: "MaxInt64"},
		canonicalize: true,
		want:         `{"-9223372036854775808":"MinInt64","0":"Zero","9223372036854775807":"MaxInt64"}`,
	}, {
		name:         jsontest.Name("Maps/ValidKey/Uint"),
		in:           map[uint64]string{0: "Zero", math.MaxUint64: "MaxUint64"},
		canonicalize: true,
		want:         `{"0":"Zero","18446744073709551615":"MaxUint64"}`,
	}, {
		name:         jsontest.Name("Maps/ValidKey/NamedUint"),
		in:           map[namedUint64]string{0: "Zero", math.MaxUint64: "MaxUint64"},
		canonicalize: true,
		want:         `{"0":"Zero","18446744073709551615":"MaxUint64"}`,
	}, {
		name: jsontest.Name("Maps/ValidKey/Float"),
		in:   map[float64]string{3.14159: "value"},
		want: `{"3.14159":"value"}`,
	}, {
		name:    jsontest.Name("Maps/InvalidKey/Float/NaN"),
		in:      map[float64]string{math.NaN(): "NaN", math.NaN(): "NaN"},
		want:    `{`,
		wantErr: EM(errors.New("unsupported value: NaN")).withPos(`{`, "").withType(0, float64Type),
	}, {
		name: jsontest.Name("Maps/ValidKey/Interface"),
		in: map[any]any{
			"key":               "key",
			namedInt64(-64):     int32(-32),
			namedUint64(+64):    uint32(+32),
			namedFloat64(64.64): float32(32.32),
		},
		canonicalize: true,
		want:         `{"-64":-32,"64":32,"64.64":32.32,"key":"key"}`,
	}, {
		name: jsontest.Name("Maps/DuplicateName/String/AllowInvalidUTF8+AllowDuplicateNames"),
		opts: []Options{jsontext.AllowInvalidUTF8(true), jsontext.AllowDuplicateNames(true)},
		in:   map[string]string{"\x80": "", "\x81": ""},
		want: `{"�":"","�":""}`,
	}, {
		name:    jsontest.Name("Maps/DuplicateName/String/AllowInvalidUTF8"),
		opts:    []Options{jsontext.AllowInvalidUTF8(true)},
		in:      map[string]string{"\x80": "", "\x81": ""},
		want:    `{"�":""`,
		wantErr: newDuplicateNameError("", []byte(`"�"`), len64(`{"�":"",`)),
	}, {
		name: jsontest.Name("Maps/DuplicateName/NoCaseString/AllowDuplicateNames"),
		opts: []Options{jsontext.AllowDuplicateNames(true)},
		in:   map[nocaseString]string{"hello": "", "HELLO": ""},
		want: `{"hello":"","hello":""}`,
	}, {
		name:    jsontest.Name("Maps/DuplicateName/NoCaseString"),
		in:      map[nocaseString]string{"hello": "", "HELLO": ""},
		want:    `{"hello":""`,
		wantErr: EM(newDuplicateNameError("", []byte(`"hello"`), len64(`{"hello":"",`))).withPos(`{"hello":"",`, "").withType(0, T[nocaseString]()),
	}, {
		name: jsontest.Name("Maps/DuplicateName/NaNs/Deterministic+AllowDuplicateNames"),
		opts: []Options{
			WithMarshalers(
				MarshalFunc(func(v float64) ([]byte, error) { return []byte(`"NaN"`), nil }),
			),
			Deterministic(true),
			jsontext.AllowDuplicateNames(true),
		},
		in:   map[float64]string{math.NaN(): "NaN", math.NaN(): "NaN"},
		want: `{"NaN":"NaN","NaN":"NaN"}`,
	}, {
		name: jsontest.Name("Maps/InvalidValue/Channel"),
		in: map[string]chan string{
			"key": nil,
		},
		want:    `{"key"`,
		wantErr: EM(nil).withPos(`{"key":`, "/key").withType(0, T[chan string]()),
	}, {
		name: jsontest.Name("Maps/String/Deterministic"),
		opts: []Options{Deterministic(true)},
		in:   map[string]int{"a": 0, "b": 1, "c": 2},
		want: `{"a":0,"b":1,"c":2}`,
	}, {
		name: jsontest.Name("Maps/String/Deterministic+AllowInvalidUTF8+RejectDuplicateNames"),
		opts: []Options{
			Deterministic(true),
			jsontext.AllowInvalidUTF8(true),
			jsontext.AllowDuplicateNames(false),
		},
		in:      map[string]int{"\xff": 0, "\xfe": 1},
		want:    `{"�":1`,
		wantErr: newDuplicateNameError("", []byte(`"�"`), len64(`{"�":1,`)),
	}, {
		name: jsontest.Name("Maps/String/Deterministic+AllowInvalidUTF8+AllowDuplicateNames"),
		opts: []Options{
			Deterministic(true),
			jsontext.AllowInvalidUTF8(true),
			jsontext.AllowDuplicateNames(true),
		},
		in:   map[string]int{"\xff": 0, "\xfe": 1},
		want: `{"�":1,"�":0}`,
	}, {
		name: jsontest.Name("Maps/String/Deterministic+MarshalFuncs"),
		opts: []Options{
			Deterministic(true),
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v string) error {
				if p := enc.StackPointer(); p != "/X" {
					return fmt.Errorf("invalid stack pointer: got %s, want /X", p)
				}
				switch v {
				case "a":
					return enc.WriteToken(jsontext.String("b"))
				case "b":
					return enc.WriteToken(jsontext.String("a"))
				default:
					return fmt.Errorf("invalid value: %q", v)
				}
			})),
		},
		in:   map[namedString]map[string]int{"X": {"a": -1, "b": 1}},
		want: `{"X":{"a":1,"b":-1}}`,
	}, {
		name: jsontest.Name("Maps/String/Deterministic+MarshalFuncs+RejectDuplicateNames"),
		opts: []Options{
			Deterministic(true),
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v string) error {
				if p := enc.StackPointer(); p != "/X" {
					return fmt.Errorf("invalid stack pointer: got %s, want /X", p)
				}
				switch v {
				case "a", "b":
					return enc.WriteToken(jsontext.String("x"))
				default:
					return fmt.Errorf("invalid value: %q", v)
				}
			})),
			jsontext.AllowDuplicateNames(false),
		},
		in:      map[namedString]map[string]int{"X": {"a": 1, "b": 1}},
		want:    `{"X":{"x":1`,
		wantErr: newDuplicateNameError("/X/x", nil, len64(`{"X":{"x":1,`)),
	}, {
		name: jsontest.Name("Maps/String/Deterministic+MarshalFuncs+AllowDuplicateNames"),
		opts: []Options{
			Deterministic(true),
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v string) error {
				if p := enc.StackPointer(); p != "/X" {
					return fmt.Errorf("invalid stack pointer: got %s, want /0", p)
				}
				switch v {
				case "a", "b":
					return enc.WriteToken(jsontext.String("x"))
				default:
					return fmt.Errorf("invalid value: %q", v)
				}
			})),
			jsontext.AllowDuplicateNames(true),
		},
		in: map[namedString]map[string]int{"X": {"a": 1, "b": 1}},
		// NOTE: Since the names are identical, the exact values may be
		// non-deterministic since sort cannot distinguish between members.
		want: `{"X":{"x":1,"x":1}}`,
	}, {
		name: jsontest.Name("Maps/RecursiveMap"),
		in: recursiveMap{
			"fizz": {
				"foo": {},
				"bar": nil,
			},
			"buzz": nil,
		},
		canonicalize: true,
		want:         `{"buzz":{},"fizz":{"bar":{},"foo":{}}}`,
	}, {
		name: jsontest.Name("Maps/CyclicMap"),
		in: func() recursiveMap {
			m := recursiveMap{"k": nil}
			m["k"] = m
			return m
		}(),
		want:    strings.Repeat(`{"k":`, startDetectingCyclesAfter) + `{"k"`,
		wantErr: EM(internal.ErrCycle).withPos(strings.Repeat(`{"k":`, startDetectingCyclesAfter+1), jsontext.Pointer(strings.Repeat("/k", startDetectingCyclesAfter+1))).withType(0, T[recursiveMap]()),
	}, {
		name: jsontest.Name("Maps/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   map[string]string{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/Empty"),
		in:   structEmpty{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/UnexportedIgnored"),
		in:   structUnexportedIgnored{ignored: "ignored"},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/IgnoredUnexportedEmbedded"),
		in:   structIgnoredUnexportedEmbedded{namedString: "ignored"},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/WeirdNames"),
		in:   structWeirdNames{Empty: "empty", Comma: "comma", Quote: "quote"},
		want: `{"":"empty",",":"comma","\"":"quote"}`,
	}, {
		name: jsontest.Name("Structs/EscapedNames"),
		opts: []Options{jsontext.EscapeForHTML(true), jsontext.EscapeForJS(true)},
		in: struct {
			S string "json:\"'abc<>&\u2028\u2029xyz'\""
			M any
			I structInlineTextValue
		}{
			S: "abc<>&\u2028\u2029xyz",
			M: map[string]string{"abc<>&\u2028\u2029xyz": "abc<>&\u2028\u2029xyz"},
			I: structInlineTextValue{X: jsontext.Value(`{"abc<>&` + "\u2028\u2029" + `xyz":"abc<>&` + "\u2028\u2029" + `xyz"}`)},
		},
		want: `{"abc\u003c\u003e\u0026\u2028\u2029xyz":"abc\u003c\u003e\u0026\u2028\u2029xyz","M":{"abc\u003c\u003e\u0026\u2028\u2029xyz":"abc\u003c\u003e\u0026\u2028\u2029xyz"},"I":{"abc\u003c\u003e\u0026\u2028\u2029xyz":"abc\u003c\u003e\u0026\u2028\u2029xyz"}}`,
	}, {
		name: jsontest.Name("Structs/NoCase"),
		in:   structNoCase{AaA: "AaA", AAa: "AAa", Aaa: "Aaa", AAA: "AAA", AA_A: "AA_A"},
		want: `{"Aaa":"Aaa","AA_A":"AA_A","AaA":"AaA","AAa":"AAa","AAA":"AAA"}`,
	}, {
		name: jsontest.Name("Structs/NoCase/MatchCaseInsensitiveNames"),
		opts: []Options{MatchCaseInsensitiveNames(true)},
		in:   structNoCase{AaA: "AaA", AAa: "AAa", Aaa: "Aaa", AAA: "AAA", AA_A: "AA_A"},
		want: `{"Aaa":"Aaa","AA_A":"AA_A","AaA":"AaA","AAa":"AAa","AAA":"AAA"}`,
	}, {
		name: jsontest.Name("Structs/NoCase/MatchCaseInsensitiveNames+MatchCaseSensitiveDelimiter"),
		opts: []Options{MatchCaseInsensitiveNames(true), jsonflags.MatchCaseSensitiveDelimiter | 1},
		in:   structNoCase{AaA: "AaA", AAa: "AAa", Aaa: "Aaa", AAA: "AAA", AA_A: "AA_A"},
		want: `{"Aaa":"Aaa","AA_A":"AA_A","AaA":"AaA","AAa":"AAa","AAA":"AAA"}`,
	}, {
		name: jsontest.Name("Structs/Normal"),
		opts: []Options{jsontext.Multiline(true)},
		in: structAll{
			Bool:   true,
			String: "hello",
			Bytes:  []byte{1, 2, 3},
			Int:    -64,
			Uint:   +64,
			Float:  3.14159,
			Map:    map[string]string{"key": "value"},
			StructScalars: structScalars{
				Bool:   true,
				String: "hello",
				Bytes:  []byte{1, 2, 3},
				Int:    -64,
				Uint:   +64,
				Float:  3.14159,
			},
			StructMaps: structMaps{
				MapBool:   map[string]bool{"": true},
				MapString: map[string]string{"": "hello"},
				MapBytes:  map[string][]byte{"": {1, 2, 3}},
				MapInt:    map[string]int64{"": -64},
				MapUint:   map[string]uint64{"": +64},
				MapFloat:  map[string]float64{"": 3.14159},
			},
			StructSlices: structSlices{
				SliceBool:   []bool{true},
				SliceString: []string{"hello"},
				SliceBytes:  [][]byte{{1, 2, 3}},
				SliceInt:    []int64{-64},
				SliceUint:   []uint64{+64},
				SliceFloat:  []float64{3.14159},
			},
			Slice:     []string{"fizz", "buzz"},
			Array:     [1]string{"goodbye"},
			Pointer:   new(structAll),
			Interface: (*structAll)(nil),
		},
		want: `{
	"Bool": true,
	"String": "hello",
	"Bytes": "AQID",
	"Int": -64,
	"Uint": 64,
	"Float": 3.14159,
	"Map": {
		"key": "value"
	},
	"StructScalars": {
		"Bool": true,
		"String": "hello",
		"Bytes": "AQID",
		"Int": -64,
		"Uint": 64,
		"Float": 3.14159
	},
	"StructMaps": {
		"MapBool": {
			"": true
		},
		"MapString": {
			"": "hello"
		},
		"MapBytes": {
			"": "AQID"
		},
		"MapInt": {
			"": -64
		},
		"MapUint": {
			"": 64
		},
		"MapFloat": {
			"": 3.14159
		}
	},
	"StructSlices": {
		"SliceBool": [
			true
		],
		"SliceString": [
			"hello"
		],
		"SliceBytes": [
			"AQID"
		],
		"SliceInt": [
			-64
		],
		"SliceUint": [
			64
		],
		"SliceFloat": [
			3.14159
		]
	},
	"Slice": [
		"fizz",
		"buzz"
	],
	"Array": [
		"goodbye"
	],
	"Pointer": {
		"Bool": false,
		"String": "",
		"Bytes": "",
		"Int": 0,
		"Uint": 0,
		"Float": 0,
		"Map": {},
		"StructScalars": {
			"Bool": false,
			"String": "",
			"Bytes": "",
			"Int": 0,
			"Uint": 0,
			"Float": 0
		},
		"StructMaps": {
			"MapBool": {},
			"MapString": {},
			"MapBytes": {},
			"MapInt": {},
			"MapUint": {},
			"MapFloat": {}
		},
		"StructSlices": {
			"SliceBool": [],
			"SliceString": [],
			"SliceBytes": [],
			"SliceInt": [],
			"SliceUint": [],
			"SliceFloat": []
		},
		"Slice": [],
		"Array": [
			""
		],
		"Pointer": null,
		"Interface": null
	},
	"Interface": null
}`,
	}, {
		name: jsontest.Name("Structs/SpaceAfterColonAndComma"),
		opts: []Options{jsontext.SpaceAfterColon(true), jsontext.SpaceAfterComma(true)},
		in:   structOmitZeroAll{Int: 1, Uint: 1},
		want: `{"Int": 1, "Uint": 1}`,
	}, {
		name: jsontest.Name("Structs/SpaceAfterColon"),
		opts: []Options{jsontext.SpaceAfterColon(true)},
		in:   structOmitZeroAll{Int: 1, Uint: 1},
		want: `{"Int": 1,"Uint": 1}`,
	}, {
		name: jsontest.Name("Structs/SpaceAfterComma"),
		opts: []Options{jsontext.SpaceAfterComma(true)},
		in:   structOmitZeroAll{Int: 1, Uint: 1, Slice: []string{"a", "b"}},
		want: `{"Int":1, "Uint":1, "Slice":["a", "b"]}`,
	}, {
		name: jsontest.Name("Structs/Stringified"),
		opts: []Options{jsontext.Multiline(true)},
		in: structStringifiedAll{
			Bool:   true,
			String: "hello",
			Bytes:  []byte{1, 2, 3},
			Int:    -64,     // should be stringified
			Uint:   +64,     // should be stringified
			Float:  3.14159, // should be stringified
			Map:    map[string]string{"key": "value"},
			StructScalars: structScalars{
				Bool:   true,
				String: "hello",
				Bytes:  []byte{1, 2, 3},
				Int:    -64,     // should be stringified
				Uint:   +64,     // should be stringified
				Float:  3.14159, // should be stringified
			},
			StructMaps: structMaps{
				MapBool:   map[string]bool{"": true},
				MapString: map[string]string{"": "hello"},
				MapBytes:  map[string][]byte{"": {1, 2, 3}},
				MapInt:    map[string]int64{"": -64},       // should be stringified
				MapUint:   map[string]uint64{"": +64},      // should be stringified
				MapFloat:  map[string]float64{"": 3.14159}, // should be stringified
			},
			StructSlices: structSlices{
				SliceBool:   []bool{true},
				SliceString: []string{"hello"},
				SliceBytes:  [][]byte{{1, 2, 3}},
				SliceInt:    []int64{-64},       // should be stringified
				SliceUint:   []uint64{+64},      // should be stringified
				SliceFloat:  []float64{3.14159}, // should be stringified
			},
			Slice:     []string{"fizz", "buzz"},
			Array:     [1]string{"goodbye"},
			Pointer:   new(structStringifiedAll), // should be stringified
			Interface: (*structStringifiedAll)(nil),
		},
		want: `{
	"Bool": true,
	"String": "hello",
	"Bytes": "AQID",
	"Int": "-64",
	"Uint": "64",
	"Float": "3.14159",
	"Map": {
		"key": "value"
	},
	"StructScalars": {
		"Bool": true,
		"String": "hello",
		"Bytes": "AQID",
		"Int": "-64",
		"Uint": "64",
		"Float": "3.14159"
	},
	"StructMaps": {
		"MapBool": {
			"": true
		},
		"MapString": {
			"": "hello"
		},
		"MapBytes": {
			"": "AQID"
		},
		"MapInt": {
			"": "-64"
		},
		"MapUint": {
			"": "64"
		},
		"MapFloat": {
			"": "3.14159"
		}
	},
	"StructSlices": {
		"SliceBool": [
			true
		],
		"SliceString": [
			"hello"
		],
		"SliceBytes": [
			"AQID"
		],
		"SliceInt": [
			"-64"
		],
		"SliceUint": [
			"64"
		],
		"SliceFloat": [
			"3.14159"
		]
	},
	"Slice": [
		"fizz",
		"buzz"
	],
	"Array": [
		"goodbye"
	],
	"Pointer": {
		"Bool": false,
		"String": "",
		"Bytes": "",
		"Int": "0",
		"Uint": "0",
		"Float": "0",
		"Map": {},
		"StructScalars": {
			"Bool": false,
			"String": "",
			"Bytes": "",
			"Int": "0",
			"Uint": "0",
			"Float": "0"
		},
		"StructMaps": {
			"MapBool": {},
			"MapString": {},
			"MapBytes": {},
			"MapInt": {},
			"MapUint": {},
			"MapFloat": {}
		},
		"StructSlices": {
			"SliceBool": [],
			"SliceString": [],
			"SliceBytes": [],
			"SliceInt": [],
			"SliceUint": [],
			"SliceFloat": []
		},
		"Slice": [],
		"Array": [
			""
		],
		"Pointer": null,
		"Interface": null
	},
	"Interface": null
}`,
	}, {
		name: jsontest.Name("Structs/LegacyStringified"),
		opts: []Options{jsontext.Multiline(true), jsonflags.StringifyWithLegacySemantics | 1},
		in: structStringifiedAll{
			Bool:   true,    // should be stringified
			String: "hello", // should be stringified
			Bytes:  []byte{1, 2, 3},
			Int:    -64,     // should be stringified
			Uint:   +64,     // should be stringified
			Float:  3.14159, // should be stringified
			Map:    map[string]string{"key": "value"},
			StructScalars: structScalars{
				Bool:   true,
				String: "hello",
				Bytes:  []byte{1, 2, 3},
				Int:    -64,
				Uint:   +64,
				Float:  3.14159,
			},
			StructMaps: structMaps{
				MapBool:   map[string]bool{"": true},
				MapString: map[string]string{"": "hello"},
				MapBytes:  map[string][]byte{"": {1, 2, 3}},
				MapInt:    map[string]int64{"": -64},
				MapUint:   map[string]uint64{"": +64},
				MapFloat:  map[string]float64{"": 3.14159},
			},
			StructSlices: structSlices{
				SliceBool:   []bool{true},
				SliceString: []string{"hello"},
				SliceBytes:  [][]byte{{1, 2, 3}},
				SliceInt:    []int64{-64},
				SliceUint:   []uint64{+64},
				SliceFloat:  []float64{3.14159},
			},
			Slice:     []string{"fizz", "buzz"},
			Array:     [1]string{"goodbye"},
			Pointer:   new(structStringifiedAll), // should be stringified
			Interface: (*structStringifiedAll)(nil),
		},
		want: `{
	"Bool": "true",
	"String": "\"hello\"",
	"Bytes": "AQID",
	"Int": "-64",
	"Uint": "64",
	"Float": "3.14159",
	"Map": {
		"key": "value"
	},
	"StructScalars": {
		"Bool": true,
		"String": "hello",
		"Bytes": "AQID",
		"Int": -64,
		"Uint": 64,
		"Float": 3.14159
	},
	"StructMaps": {
		"MapBool": {
			"": true
		},
		"MapString": {
			"": "hello"
		},
		"MapBytes": {
			"": "AQID"
		},
		"MapInt": {
			"": -64
		},
		"MapUint": {
			"": 64
		},
		"MapFloat": {
			"": 3.14159
		}
	},
	"StructSlices": {
		"SliceBool": [
			true
		],
		"SliceString": [
			"hello"
		],
		"SliceBytes": [
			"AQID"
		],
		"SliceInt": [
			-64
		],
		"SliceUint": [
			64
		],
		"SliceFloat": [
			3.14159
		]
	},
	"Slice": [
		"fizz",
		"buzz"
	],
	"Array": [
		"goodbye"
	],
	"Pointer": {
		"Bool": "false",
		"String": "\"\"",
		"Bytes": "",
		"Int": "0",
		"Uint": "0",
		"Float": "0",
		"Map": {},
		"StructScalars": {
			"Bool": false,
			"String": "",
			"Bytes": "",
			"Int": 0,
			"Uint": 0,
			"Float": 0
		},
		"StructMaps": {
			"MapBool": {},
			"MapString": {},
			"MapBytes": {},
			"MapInt": {},
			"MapUint": {},
			"MapFloat": {}
		},
		"StructSlices": {
			"SliceBool": [],
			"SliceString": [],
			"SliceBytes": [],
			"SliceInt": [],
			"SliceUint": [],
			"SliceFloat": []
		},
		"Slice": [],
		"Array": [
			""
		],
		"Pointer": null,
		"Interface": null
	},
	"Interface": null
}`,
	}, {
		name: jsontest.Name("Structs/OmitZero/Zero"),
		in:   structOmitZeroAll{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/OmitZeroOption/Zero"),
		opts: []Options{OmitZeroStructFields(true)},
		in:   structAll{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/OmitZero/NonZero"),
		opts: []Options{jsontext.Multiline(true)},
		in: structOmitZeroAll{
			Bool:          true,                                   // not omitted since true is non-zero
			String:        " ",                                    // not omitted since non-empty string is non-zero
			Bytes:         []byte{},                               // not omitted since allocated slice is non-zero
			Int:           1,                                      // not omitted since 1 is non-zero
			Uint:          1,                                      // not omitted since 1 is non-zero
			Float:         math.SmallestNonzeroFloat64,            // not omitted since still slightly above zero
			Map:           map[string]string{},                    // not omitted since allocated map is non-zero
			StructScalars: structScalars{unexported: true},        // not omitted since unexported is non-zero
			StructSlices:  structSlices{Ignored: true},            // not omitted since Ignored is non-zero
			StructMaps:    structMaps{MapBool: map[string]bool{}}, // not omitted since MapBool is non-zero
			Slice:         []string{},                             // not omitted since allocated slice is non-zero
			Array:         [1]string{" "},                         // not omitted since single array element is non-zero
			Pointer:       new(structOmitZeroAll),                 // not omitted since pointer is non-zero (even if all fields of the struct value are zero)
			Interface:     (*structOmitZeroAll)(nil),              // not omitted since interface value is non-zero (even if interface value is a nil pointer)
		},
		want: `{
	"Bool": true,
	"String": " ",
	"Bytes": "",
	"Int": 1,
	"Uint": 1,
	"Float": 5e-324,
	"Map": {},
	"StructScalars": {
		"Bool": false,
		"String": "",
		"Bytes": "",
		"Int": 0,
		"Uint": 0,
		"Float": 0
	},
	"StructMaps": {
		"MapBool": {},
		"MapString": {},
		"MapBytes": {},
		"MapInt": {},
		"MapUint": {},
		"MapFloat": {}
	},
	"StructSlices": {
		"SliceBool": [],
		"SliceString": [],
		"SliceBytes": [],
		"SliceInt": [],
		"SliceUint": [],
		"SliceFloat": []
	},
	"Slice": [],
	"Array": [
		" "
	],
	"Pointer": {},
	"Interface": null
}`,
	}, {
		name: jsontest.Name("Structs/OmitZeroOption/NonZero"),
		opts: []Options{OmitZeroStructFields(true), jsontext.Multiline(true)},
		in: structAll{
			Bool:          true,
			String:        " ",
			Bytes:         []byte{},
			Int:           1,
			Uint:          1,
			Float:         math.SmallestNonzeroFloat64,
			Map:           map[string]string{},
			StructScalars: structScalars{unexported: true},
			StructSlices:  structSlices{Ignored: true},
			StructMaps:    structMaps{MapBool: map[string]bool{}},
			Slice:         []string{},
			Array:         [1]string{" "},
			Pointer:       new(structAll),
			Interface:     (*structAll)(nil),
		},
		want: `{
	"Bool": true,
	"String": " ",
	"Bytes": "",
	"Int": 1,
	"Uint": 1,
	"Float": 5e-324,
	"Map": {},
	"StructScalars": {},
	"StructMaps": {
		"MapBool": {}
	},
	"StructSlices": {},
	"Slice": [],
	"Array": [
		" "
	],
	"Pointer": {},
	"Interface": null
}`,
	}, {
		name: jsontest.Name("Structs/OmitZeroMethod/Zero"),
		in:   structOmitZeroMethodAll{},
		want: `{"ValueNeverZero":"","PointerNeverZero":""}`,
	}, {
		name: jsontest.Name("Structs/OmitZeroMethod/NonZero"),
		opts: []Options{jsontext.Multiline(true)},
		in: structOmitZeroMethodAll{
			ValueAlwaysZero:                 valueAlwaysZero("nonzero"),
			ValueNeverZero:                  valueNeverZero("nonzero"),
			PointerAlwaysZero:               pointerAlwaysZero("nonzero"),
			PointerNeverZero:                pointerNeverZero("nonzero"),
			PointerValueAlwaysZero:          addr(valueAlwaysZero("nonzero")),
			PointerValueNeverZero:           addr(valueNeverZero("nonzero")),
			PointerPointerAlwaysZero:        addr(pointerAlwaysZero("nonzero")),
			PointerPointerNeverZero:         addr(pointerNeverZero("nonzero")),
			PointerPointerValueAlwaysZero:   addr(addr(valueAlwaysZero("nonzero"))), // marshaled since **valueAlwaysZero does not implement IsZero
			PointerPointerValueNeverZero:    addr(addr(valueNeverZero("nonzero"))),
			PointerPointerPointerAlwaysZero: addr(addr(pointerAlwaysZero("nonzero"))), // marshaled since **pointerAlwaysZero does not implement IsZero
			PointerPointerPointerNeverZero:  addr(addr(pointerNeverZero("nonzero"))),
		},
		want: `{
	"ValueNeverZero": "nonzero",
	"PointerNeverZero": "nonzero",
	"PointerValueNeverZero": "nonzero",
	"PointerPointerNeverZero": "nonzero",
	"PointerPointerValueAlwaysZero": "nonzero",
	"PointerPointerValueNeverZero": "nonzero",
	"PointerPointerPointerAlwaysZero": "nonzero",
	"PointerPointerPointerNeverZero": "nonzero"
}`,
	}, {
		name: jsontest.Name("Structs/OmitZeroMethod/Interface/Zero"),
		opts: []Options{jsontext.Multiline(true)},
		in:   structOmitZeroMethodInterfaceAll{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/OmitZeroMethod/Interface/PartialZero"),
		opts: []Options{jsontext.Multiline(true)},
		in: structOmitZeroMethodInterfaceAll{
			ValueAlwaysZero:          valueAlwaysZero(""),
			ValueNeverZero:           valueNeverZero(""),
			PointerValueAlwaysZero:   (*valueAlwaysZero)(nil),
			PointerValueNeverZero:    (*valueNeverZero)(nil), // nil pointer, so method not called
			PointerPointerAlwaysZero: (*pointerAlwaysZero)(nil),
			PointerPointerNeverZero:  (*pointerNeverZero)(nil), // nil pointer, so method not called
		},
		want: `{
	"ValueNeverZero": ""
}`,
	}, {
		name: jsontest.Name("Structs/OmitZeroMethod/Interface/NonZero"),
		opts: []Options{jsontext.Multiline(true)},
		in: structOmitZeroMethodInterfaceAll{
			ValueAlwaysZero:          valueAlwaysZero("nonzero"),
			ValueNeverZero:           valueNeverZero("nonzero"),
			PointerValueAlwaysZero:   addr(valueAlwaysZero("nonzero")),
			PointerValueNeverZero:    addr(valueNeverZero("nonzero")),
			PointerPointerAlwaysZero: addr(pointerAlwaysZero("nonzero")),
			PointerPointerNeverZero:  addr(pointerNeverZero("nonzero")),
		},
		want: `{
	"ValueNeverZero": "nonzero",
	"PointerValueNeverZero": "nonzero",
	"PointerPointerNeverZero": "nonzero"
}`,
	}, {
		name: jsontest.Name("Structs/OmitEmpty/Zero"),
		opts: []Options{jsontext.Multiline(true)},
		in:   structOmitEmptyAll{},
		want: `{
	"Bool": false,
	"StringNonEmpty": "value",
	"BytesNonEmpty": [
		"value"
	],
	"Float": 0,
	"MapNonEmpty": {
		"key": "value"
	},
	"SliceNonEmpty": [
		"value"
	]
}`,
	}, {
		name: jsontest.Name("Structs/OmitEmpty/EmptyNonZero"),
		opts: []Options{jsontext.Multiline(true)},
		in: structOmitEmptyAll{
			String:                string(""),
			StringEmpty:           stringMarshalEmpty(""),
			StringNonEmpty:        stringMarshalNonEmpty(""),
			PointerString:         addr(string("")),
			PointerStringEmpty:    addr(stringMarshalEmpty("")),
			PointerStringNonEmpty: addr(stringMarshalNonEmpty("")),
			Bytes:                 []byte(""),
			BytesEmpty:            bytesMarshalEmpty([]byte("")),
			BytesNonEmpty:         bytesMarshalNonEmpty([]byte("")),
			PointerBytes:          addr([]byte("")),
			PointerBytesEmpty:     addr(bytesMarshalEmpty([]byte(""))),
			PointerBytesNonEmpty:  addr(bytesMarshalNonEmpty([]byte(""))),
			Map:                   map[string]string{},
			MapEmpty:              mapMarshalEmpty{},
			MapNonEmpty:           mapMarshalNonEmpty{},
			PointerMap:            addr(map[string]string{}),
			PointerMapEmpty:       addr(mapMarshalEmpty{}),
			PointerMapNonEmpty:    addr(mapMarshalNonEmpty{}),
			Slice:                 []string{},
			SliceEmpty:            sliceMarshalEmpty{},
			SliceNonEmpty:         sliceMarshalNonEmpty{},
			PointerSlice:          addr([]string{}),
			PointerSliceEmpty:     addr(sliceMarshalEmpty{}),
			PointerSliceNonEmpty:  addr(sliceMarshalNonEmpty{}),
			Pointer:               &structOmitZeroEmptyAll{},
			Interface:             []string{},
		},
		want: `{
	"Bool": false,
	"StringNonEmpty": "value",
	"PointerStringNonEmpty": "value",
	"BytesNonEmpty": [
		"value"
	],
	"PointerBytesNonEmpty": [
		"value"
	],
	"Float": 0,
	"MapNonEmpty": {
		"key": "value"
	},
	"PointerMapNonEmpty": {
		"key": "value"
	},
	"SliceNonEmpty": [
		"value"
	],
	"PointerSliceNonEmpty": [
		"value"
	]
}`,
	}, {
		name: jsontest.Name("Structs/OmitEmpty/NonEmpty"),
		opts: []Options{jsontext.Multiline(true)},
		in: structOmitEmptyAll{
			Bool:                  true,
			PointerBool:           addr(true),
			String:                string("value"),
			StringEmpty:           stringMarshalEmpty("value"),
			StringNonEmpty:        stringMarshalNonEmpty("value"),
			PointerString:         addr(string("value")),
			PointerStringEmpty:    addr(stringMarshalEmpty("value")),
			PointerStringNonEmpty: addr(stringMarshalNonEmpty("value")),
			Bytes:                 []byte("value"),
			BytesEmpty:            bytesMarshalEmpty([]byte("value")),
			BytesNonEmpty:         bytesMarshalNonEmpty([]byte("value")),
			PointerBytes:          addr([]byte("value")),
			PointerBytesEmpty:     addr(bytesMarshalEmpty([]byte("value"))),
			PointerBytesNonEmpty:  addr(bytesMarshalNonEmpty([]byte("value"))),
			Float:                 math.Copysign(0, -1),
			PointerFloat:          addr(math.Copysign(0, -1)),
			Map:                   map[string]string{"": ""},
			MapEmpty:              mapMarshalEmpty{"key": "value"},
			MapNonEmpty:           mapMarshalNonEmpty{"key": "value"},
			PointerMap:            addr(map[string]string{"": ""}),
			PointerMapEmpty:       addr(mapMarshalEmpty{"key": "value"}),
			PointerMapNonEmpty:    addr(mapMarshalNonEmpty{"key": "value"}),
			Slice:                 []string{""},
			SliceEmpty:            sliceMarshalEmpty{"value"},
			SliceNonEmpty:         sliceMarshalNonEmpty{"value"},
			PointerSlice:          addr([]string{""}),
			PointerSliceEmpty:     addr(sliceMarshalEmpty{"value"}),
			PointerSliceNonEmpty:  addr(sliceMarshalNonEmpty{"value"}),
			Pointer:               &structOmitZeroEmptyAll{Float: math.SmallestNonzeroFloat64},
			Interface:             []string{""},
		},
		want: `{
	"Bool": true,
	"PointerBool": true,
	"String": "value",
	"StringNonEmpty": "value",
	"PointerString": "value",
	"PointerStringNonEmpty": "value",
	"Bytes": "dmFsdWU=",
	"BytesNonEmpty": [
		"value"
	],
	"PointerBytes": "dmFsdWU=",
	"PointerBytesNonEmpty": [
		"value"
	],
	"Float": -0,
	"PointerFloat": -0,
	"Map": {
		"": ""
	},
	"MapNonEmpty": {
		"key": "value"
	},
	"PointerMap": {
		"": ""
	},
	"PointerMapNonEmpty": {
		"key": "value"
	},
	"Slice": [
		""
	],
	"SliceNonEmpty": [
		"value"
	],
	"PointerSlice": [
		""
	],
	"PointerSliceNonEmpty": [
		"value"
	],
	"Pointer": {
		"Float": 5e-324
	},
	"Interface": [
		""
	]
}`,
	}, {
		name: jsontest.Name("Structs/OmitEmpty/Legacy/Zero"),
		opts: []Options{jsonflags.OmitEmptyWithLegacyDefinition | 1},
		in:   structOmitEmptyAll{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/OmitEmpty/Legacy/NonEmpty"),
		opts: []Options{jsontext.Multiline(true), jsonflags.OmitEmptyWithLegacyDefinition | 1},
		in: structOmitEmptyAll{
			Bool:                  true,
			PointerBool:           addr(true),
			String:                string("value"),
			StringEmpty:           stringMarshalEmpty("value"),
			StringNonEmpty:        stringMarshalNonEmpty("value"),
			PointerString:         addr(string("value")),
			PointerStringEmpty:    addr(stringMarshalEmpty("value")),
			PointerStringNonEmpty: addr(stringMarshalNonEmpty("value")),
			Bytes:                 []byte("value"),
			BytesEmpty:            bytesMarshalEmpty([]byte("value")),
			BytesNonEmpty:         bytesMarshalNonEmpty([]byte("value")),
			PointerBytes:          addr([]byte("value")),
			PointerBytesEmpty:     addr(bytesMarshalEmpty([]byte("value"))),
			PointerBytesNonEmpty:  addr(bytesMarshalNonEmpty([]byte("value"))),
			Float:                 math.Copysign(0, -1),
			PointerFloat:          addr(math.Copysign(0, -1)),
			Map:                   map[string]string{"": ""},
			MapEmpty:              mapMarshalEmpty{"key": "value"},
			MapNonEmpty:           mapMarshalNonEmpty{"key": "value"},
			PointerMap:            addr(map[string]string{"": ""}),
			PointerMapEmpty:       addr(mapMarshalEmpty{"key": "value"}),
			PointerMapNonEmpty:    addr(mapMarshalNonEmpty{"key": "value"}),
			Slice:                 []string{""},
			SliceEmpty:            sliceMarshalEmpty{"value"},
			SliceNonEmpty:         sliceMarshalNonEmpty{"value"},
			PointerSlice:          addr([]string{""}),
			PointerSliceEmpty:     addr(sliceMarshalEmpty{"value"}),
			PointerSliceNonEmpty:  addr(sliceMarshalNonEmpty{"value"}),
			Pointer:               &structOmitZeroEmptyAll{Float: math.Copysign(0, -1)},
			Interface:             []string{""},
		},
		want: `{
	"Bool": true,
	"PointerBool": true,
	"String": "value",
	"StringEmpty": "",
	"StringNonEmpty": "value",
	"PointerString": "value",
	"PointerStringEmpty": "",
	"PointerStringNonEmpty": "value",
	"Bytes": "dmFsdWU=",
	"BytesEmpty": [],
	"BytesNonEmpty": [
		"value"
	],
	"PointerBytes": "dmFsdWU=",
	"PointerBytesEmpty": [],
	"PointerBytesNonEmpty": [
		"value"
	],
	"PointerFloat": -0,
	"Map": {
		"": ""
	},
	"MapEmpty": {},
	"MapNonEmpty": {
		"key": "value"
	},
	"PointerMap": {
		"": ""
	},
	"PointerMapEmpty": {},
	"PointerMapNonEmpty": {
		"key": "value"
	},
	"Slice": [
		""
	],
	"SliceEmpty": [],
	"SliceNonEmpty": [
		"value"
	],
	"PointerSlice": [
		""
	],
	"PointerSliceEmpty": [],
	"PointerSliceNonEmpty": [
		"value"
	],
	"Pointer": {},
	"Interface": [
		""
	]
}`,
	}, {
		name: jsontest.Name("Structs/OmitEmpty/NonEmptyString"),
		in: struct {
			X string `json:",omitempty"`
		}{`"`},
		want: `{"X":"\""}`,
	}, {
		name: jsontest.Name("Structs/OmitZeroEmpty/Zero"),
		in:   structOmitZeroEmptyAll{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/OmitZeroEmpty/Empty"),
		in: structOmitZeroEmptyAll{
			Bytes:     []byte{},
			Map:       map[string]string{},
			Slice:     []string{},
			Pointer:   &structOmitZeroEmptyAll{},
			Interface: []string{},
		},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/OmitEmpty/PathologicalDepth"),
		in: func() any {
			type X struct {
				X *X `json:",omitempty"`
			}
			var make func(int) *X
			make = func(n int) *X {
				if n == 0 {
					return nil
				}
				return &X{make(n - 1)}
			}
			return make(100)
		}(),
		want:      `{}`,
		useWriter: true,
	}, {
		name: jsontest.Name("Structs/OmitEmpty/PathologicalBreadth"),
		in: func() any {
			var fields []reflect.StructField
			for i := range 100 {
				fields = append(fields, reflect.StructField{
					Name: fmt.Sprintf("X%d", i),
					Type: T[stringMarshalEmpty](),
					Tag:  `json:",omitempty"`,
				})
			}
			return reflect.New(reflect.StructOf(fields)).Interface()
		}(),
		want:      `{}`,
		useWriter: true,
	}, {
		name: jsontest.Name("Structs/OmitEmpty/PathologicalTree"),
		in: func() any {
			type X struct {
				XL, XR *X `json:",omitempty"`
			}
			var make func(int) *X
			make = func(n int) *X {
				if n == 0 {
					return nil
				}
				return &X{make(n - 1), make(n - 1)}
			}
			return make(8)
		}(),
		want:      `{}`,
		useWriter: true,
	}, {
		name: jsontest.Name("Structs/OmitZeroEmpty/NonEmpty"),
		in: structOmitZeroEmptyAll{
			Bytes:     []byte("value"),
			Map:       map[string]string{"": ""},
			Slice:     []string{""},
			Pointer:   &structOmitZeroEmptyAll{Bool: true},
			Interface: []string{""},
		},
		want: `{"Bytes":"dmFsdWU=","Map":{"":""},"Slice":[""],"Pointer":{"Bool":true},"Interface":[""]}`,
	}, {
		name: jsontest.Name("Structs/Format/Bytes"),
		opts: []Options{jsontext.Multiline(true)},
		in: structFormatBytes{
			Base16:    []byte("\x01\x23\x45\x67\x89\xab\xcd\xef"),
			Base32:    []byte("\x00D2\x14\xc7BT\xb65τe:V\xd7\xc6u\xbew\xdf"),
			Base32Hex: []byte("\x00D2\x14\xc7BT\xb65τe:V\xd7\xc6u\xbew\xdf"),
			Base64:    []byte("\x00\x10\x83\x10Q\x87 \x92\x8b0ӏA\x14\x93QU\x97a\x96\x9bqן\x82\x18\xa3\x92Y\xa7\xa2\x9a\xab\xb2ۯ\xc3\x1c\xb3\xd3]\xb7㞻\xf3߿"),
			Base64URL: []byte("\x00\x10\x83\x10Q\x87 \x92\x8b0ӏA\x14\x93QU\x97a\x96\x9bqן\x82\x18\xa3\x92Y\xa7\xa2\x9a\xab\xb2ۯ\xc3\x1c\xb3\xd3]\xb7㞻\xf3߿"),
			Array:     []byte{1, 2, 3, 4},
		},
		want: `{
	"Base16": "0123456789abcdef",
	"Base32": "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567",
	"Base32Hex": "0123456789ABCDEFGHIJKLMNOPQRSTUV",
	"Base64": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
	"Base64URL": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
	"Array": [
		1,
		2,
		3,
		4
	]
}`}, {
		name: jsontest.Name("Structs/Format/ArrayBytes"),
		opts: []Options{jsontext.Multiline(true)},
		in: structFormatArrayBytes{
			Base16:    [4]byte{1, 2, 3, 4},
			Base32:    [4]byte{1, 2, 3, 4},
			Base32Hex: [4]byte{1, 2, 3, 4},
			Base64:    [4]byte{1, 2, 3, 4},
			Base64URL: [4]byte{1, 2, 3, 4},
			Array:     [4]byte{1, 2, 3, 4},
			Default:   [4]byte{1, 2, 3, 4},
		},
		want: `{
	"Base16": "01020304",
	"Base32": "AEBAGBA=",
	"Base32Hex": "0410610=",
	"Base64": "AQIDBA==",
	"Base64URL": "AQIDBA==",
	"Array": [
		1,
		2,
		3,
		4
	],
	"Default": "AQIDBA=="
}`}, {
		name: jsontest.Name("Structs/Format/ArrayBytes/Legacy"),
		opts: []Options{jsontext.Multiline(true), jsonflags.FormatBytesWithLegacySemantics | 1},
		in: structFormatArrayBytes{
			Base16:    [4]byte{1, 2, 3, 4},
			Base32:    [4]byte{1, 2, 3, 4},
			Base32Hex: [4]byte{1, 2, 3, 4},
			Base64:    [4]byte{1, 2, 3, 4},
			Base64URL: [4]byte{1, 2, 3, 4},
			Array:     [4]byte{1, 2, 3, 4},
			Default:   [4]byte{1, 2, 3, 4},
		},
		want: `{
	"Base16": "01020304",
	"Base32": "AEBAGBA=",
	"Base32Hex": "0410610=",
	"Base64": "AQIDBA==",
	"Base64URL": "AQIDBA==",
	"Array": [
		1,
		2,
		3,
		4
	],
	"Default": [
		1,
		2,
		3,
		4
	]
}`}, {
		name: jsontest.Name("Structs/Format/Bytes/Array"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(in byte) ([]byte, error) {
				if in > 3 {
					return []byte("true"), nil
				} else {
					return []byte("false"), nil
				}
			})),
		},
		in: struct {
			Array []byte `json:",format:array"`
		}{
			Array: []byte{1, 6, 2, 5, 3, 4},
		},
		want: `{"Array":[false,true,false,true,false,true]}`,
	}, {
		name: jsontest.Name("Structs/Format/Floats"),
		opts: []Options{jsontext.Multiline(true)},
		in: []structFormatFloats{
			{NonFinite: math.Pi, PointerNonFinite: addr(math.Pi)},
			{NonFinite: math.NaN(), PointerNonFinite: addr(math.NaN())},
			{NonFinite: math.Inf(-1), PointerNonFinite: addr(math.Inf(-1))},
			{NonFinite: math.Inf(+1), PointerNonFinite: addr(math.Inf(+1))},
		},
		want: `[
	{
		"NonFinite": 3.141592653589793,
		"PointerNonFinite": 3.141592653589793
	},
	{
		"NonFinite": "NaN",
		"PointerNonFinite": "NaN"
	},
	{
		"NonFinite": "-Infinity",
		"PointerNonFinite": "-Infinity"
	},
	{
		"NonFinite": "Infinity",
		"PointerNonFinite": "Infinity"
	}
]`,
	}, {
		name: jsontest.Name("Structs/Format/Maps"),
		opts: []Options{jsontext.Multiline(true)},
		in: []structFormatMaps{{
			EmitNull: map[string]string(nil), PointerEmitNull: addr(map[string]string(nil)),
			EmitEmpty: map[string]string(nil), PointerEmitEmpty: addr(map[string]string(nil)),
			EmitDefault: map[string]string(nil), PointerEmitDefault: addr(map[string]string(nil)),
		}, {
			EmitNull: map[string]string{}, PointerEmitNull: addr(map[string]string{}),
			EmitEmpty: map[string]string{}, PointerEmitEmpty: addr(map[string]string{}),
			EmitDefault: map[string]string{}, PointerEmitDefault: addr(map[string]string{}),
		}, {
			EmitNull: map[string]string{"k": "v"}, PointerEmitNull: addr(map[string]string{"k": "v"}),
			EmitEmpty: map[string]string{"k": "v"}, PointerEmitEmpty: addr(map[string]string{"k": "v"}),
			EmitDefault: map[string]string{"k": "v"}, PointerEmitDefault: addr(map[string]string{"k": "v"}),
		}},
		want: `[
	{
		"EmitNull": null,
		"PointerEmitNull": null,
		"EmitEmpty": {},
		"PointerEmitEmpty": {},
		"EmitDefault": {},
		"PointerEmitDefault": {}
	},
	{
		"EmitNull": {},
		"PointerEmitNull": {},
		"EmitEmpty": {},
		"PointerEmitEmpty": {},
		"EmitDefault": {},
		"PointerEmitDefault": {}
	},
	{
		"EmitNull": {
			"k": "v"
		},
		"PointerEmitNull": {
			"k": "v"
		},
		"EmitEmpty": {
			"k": "v"
		},
		"PointerEmitEmpty": {
			"k": "v"
		},
		"EmitDefault": {
			"k": "v"
		},
		"PointerEmitDefault": {
			"k": "v"
		}
	}
]`,
	}, {
		name: jsontest.Name("Structs/Format/Maps/FormatNilMapAsNull"),
		opts: []Options{
			FormatNilMapAsNull(true),
			jsontext.Multiline(true),
		},
		in: []structFormatMaps{{
			EmitNull: map[string]string(nil), PointerEmitNull: addr(map[string]string(nil)),
			EmitEmpty: map[string]string(nil), PointerEmitEmpty: addr(map[string]string(nil)),
			EmitDefault: map[string]string(nil), PointerEmitDefault: addr(map[string]string(nil)),
		}, {
			EmitNull: map[string]string{}, PointerEmitNull: addr(map[string]string{}),
			EmitEmpty: map[string]string{}, PointerEmitEmpty: addr(map[string]string{}),
			EmitDefault: map[string]string{}, PointerEmitDefault: addr(map[string]string{}),
		}, {
			EmitNull: map[string]string{"k": "v"}, PointerEmitNull: addr(map[string]string{"k": "v"}),
			EmitEmpty: map[string]string{"k": "v"}, PointerEmitEmpty: addr(map[string]string{"k": "v"}),
			EmitDefault: map[string]string{"k": "v"}, PointerEmitDefault: addr(map[string]string{"k": "v"}),
		}},
		want: `[
	{
		"EmitNull": null,
		"PointerEmitNull": null,
		"EmitEmpty": {},
		"PointerEmitEmpty": {},
		"EmitDefault": null,
		"PointerEmitDefault": null
	},
	{
		"EmitNull": {},
		"PointerEmitNull": {},
		"EmitEmpty": {},
		"PointerEmitEmpty": {},
		"EmitDefault": {},
		"PointerEmitDefault": {}
	},
	{
		"EmitNull": {
			"k": "v"
		},
		"PointerEmitNull": {
			"k": "v"
		},
		"EmitEmpty": {
			"k": "v"
		},
		"PointerEmitEmpty": {
			"k": "v"
		},
		"EmitDefault": {
			"k": "v"
		},
		"PointerEmitDefault": {
			"k": "v"
		}
	}
]`,
	}, {
		name: jsontest.Name("Structs/Format/Slices"),
		opts: []Options{jsontext.Multiline(true)},
		in: []structFormatSlices{{
			EmitNull: []string(nil), PointerEmitNull: addr([]string(nil)),
			EmitEmpty: []string(nil), PointerEmitEmpty: addr([]string(nil)),
			EmitDefault: []string(nil), PointerEmitDefault: addr([]string(nil)),
		}, {
			EmitNull: []string{}, PointerEmitNull: addr([]string{}),
			EmitEmpty: []string{}, PointerEmitEmpty: addr([]string{}),
			EmitDefault: []string{}, PointerEmitDefault: addr([]string{}),
		}, {
			EmitNull: []string{"v"}, PointerEmitNull: addr([]string{"v"}),
			EmitEmpty: []string{"v"}, PointerEmitEmpty: addr([]string{"v"}),
			EmitDefault: []string{"v"}, PointerEmitDefault: addr([]string{"v"}),
		}},
		want: `[
	{
		"EmitNull": null,
		"PointerEmitNull": null,
		"EmitEmpty": [],
		"PointerEmitEmpty": [],
		"EmitDefault": [],
		"PointerEmitDefault": []
	},
	{
		"EmitNull": [],
		"PointerEmitNull": [],
		"EmitEmpty": [],
		"PointerEmitEmpty": [],
		"EmitDefault": [],
		"PointerEmitDefault": []
	},
	{
		"EmitNull": [
			"v"
		],
		"PointerEmitNull": [
			"v"
		],
		"EmitEmpty": [
			"v"
		],
		"PointerEmitEmpty": [
			"v"
		],
		"EmitDefault": [
			"v"
		],
		"PointerEmitDefault": [
			"v"
		]
	}
]`,
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Bool"),
		in:      structFormatInvalid{Bool: true},
		want:    `{"Bool"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Bool":`, "/Bool").withType(0, boolType),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/String"),
		in:      structFormatInvalid{String: "string"},
		want:    `{"String"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"String":`, "/String").withType(0, stringType),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Bytes"),
		in:      structFormatInvalid{Bytes: []byte("bytes")},
		want:    `{"Bytes"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Bytes":`, "/Bytes").withType(0, bytesType),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Int"),
		in:      structFormatInvalid{Int: 1},
		want:    `{"Int"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Int":`, "/Int").withType(0, T[int64]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Uint"),
		in:      structFormatInvalid{Uint: 1},
		want:    `{"Uint"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Uint":`, "/Uint").withType(0, T[uint64]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Float"),
		in:      structFormatInvalid{Float: 1},
		want:    `{"Float"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Float":`, "/Float").withType(0, T[float64]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Map"),
		in:      structFormatInvalid{Map: map[string]string{}},
		want:    `{"Map"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Map":`, "/Map").withType(0, T[map[string]string]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Struct"),
		in:      structFormatInvalid{Struct: structAll{Bool: true}},
		want:    `{"Struct"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Struct":`, "/Struct").withType(0, T[structAll]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Slice"),
		in:      structFormatInvalid{Slice: []string{}},
		want:    `{"Slice"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Slice":`, "/Slice").withType(0, T[[]string]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Array"),
		in:      structFormatInvalid{Array: [1]string{"string"}},
		want:    `{"Array"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Array":`, "/Array").withType(0, T[[1]string]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Interface"),
		in:      structFormatInvalid{Interface: "anything"},
		want:    `{"Interface"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"Interface":`, "/Interface").withType(0, T[any]()),
	}, {
		name: jsontest.Name("Structs/Inline/Zero"),
		in:   structInlined{},
		want: `{"D":""}`,
	}, {
		name: jsontest.Name("Structs/Inline/Alloc"),
		in: structInlined{
			X: structInlinedL1{
				X:            &structInlinedL2{},
				StructEmbed1: StructEmbed1{},
			},
			StructEmbed2: &StructEmbed2{},
		},
		want: `{"A":"","B":"","D":"","E":"","F":"","G":""}`,
	}, {
		name: jsontest.Name("Structs/Inline/NonZero"),
		in: structInlined{
			X: structInlinedL1{
				X:            &structInlinedL2{A: "A1", B: "B1", C: "C1"},
				StructEmbed1: StructEmbed1{C: "C2", D: "D2", E: "E2"},
			},
			StructEmbed2: &StructEmbed2{E: "E3", F: "F3", G: "G3"},
		},
		want: `{"A":"A1","B":"B1","D":"D2","E":"E3","F":"F3","G":"G3"}`,
	}, {
		name: jsontest.Name("Structs/Inline/DualCycle"),
		in: cyclicA{
			B1: cyclicB{F: 1}, // B1.F ignored since it conflicts with B2.F
			B2: cyclicB{F: 2}, // B2.F ignored since it conflicts with B1.F
		},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/TextValue/Nil"),
		in:   structInlineTextValue{X: jsontext.Value(nil)},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/TextValue/Empty"),
		in:   structInlineTextValue{X: jsontext.Value("")},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/TextValue/NonEmptyN1"),
		in:   structInlineTextValue{X: jsontext.Value(` { "fizz" : "buzz" } `)},
		want: `{"fizz":"buzz"}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/TextValue/NonEmptyN2"),
		in:   structInlineTextValue{X: jsontext.Value(` { "fizz" : "buzz" , "foo" : "bar" } `)},
		want: `{"fizz":"buzz","foo":"bar"}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/TextValue/NonEmptyWithOthers"),
		in: structInlineTextValue{
			A: 1,
			X: jsontext.Value(` { "fizz" : "buzz" , "foo" : "bar" } `),
			B: 2,
		},
		// NOTE: Inlined fallback fields are always serialized last.
		want: `{"A":1,"B":2,"fizz":"buzz","foo":"bar"}`,
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/RejectDuplicateNames"),
		opts:    []Options{jsontext.AllowDuplicateNames(false)},
		in:      structInlineTextValue{X: jsontext.Value(` { "fizz" : "buzz" , "fizz" : "buzz" } `)},
		want:    `{"fizz":"buzz"`,
		wantErr: newDuplicateNameError("/fizz", nil, len64(`{"fizz":"buzz"`)),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/TextValue/AllowDuplicateNames"),
		opts: []Options{jsontext.AllowDuplicateNames(true)},
		in:   structInlineTextValue{X: jsontext.Value(` { "fizz" : "buzz" , "fizz" : "buzz" } `)},
		want: `{"fizz":"buzz","fizz":"buzz"}`,
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/RejectInvalidUTF8"),
		opts:    []Options{jsontext.AllowInvalidUTF8(false)},
		in:      structInlineTextValue{X: jsontext.Value(`{"` + "\xde\xad\xbe\xef" + `":"value"}`)},
		want:    `{`,
		wantErr: newInvalidUTF8Error(len64(`{"`+"\xde\xad"), ""),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/TextValue/AllowInvalidUTF8"),
		opts: []Options{jsontext.AllowInvalidUTF8(true)},
		in:   structInlineTextValue{X: jsontext.Value(`{"` + "\xde\xad\xbe\xef" + `":"value"}`)},
		want: `{"ޭ��":"value"}`,
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/InvalidWhitespace"),
		in:      structInlineTextValue{X: jsontext.Value("\n\r\t ")},
		want:    `{`,
		wantErr: EM(io.ErrUnexpectedEOF).withPos(`{`, "").withType(0, T[jsontext.Value]()),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/InvalidObject"),
		in:      structInlineTextValue{X: jsontext.Value(` true `)},
		want:    `{`,
		wantErr: EM(errRawInlinedNotObject).withPos(`{`, "").withType(0, T[jsontext.Value]()),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/InvalidObjectName"),
		in:      structInlineTextValue{X: jsontext.Value(` { true : false } `)},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(" { "), "")).withPos(`{`, "").withType(0, T[jsontext.Value]()),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/InvalidEndObject"),
		in:      structInlineTextValue{X: jsontext.Value(` { "name" : false , } `)},
		want:    `{"name":false`,
		wantErr: EM(newInvalidCharacterError(",", "at start of value", len64(` { "name" : false `), "")).withPos(`{"name":false,`, "").withType(0, T[jsontext.Value]()),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/InvalidDualObject"),
		in:      structInlineTextValue{X: jsontext.Value(`{}{}`)},
		want:    `{`,
		wantErr: EM(newInvalidCharacterError("{", "after top-level value", len64(`{}`), "")).withPos(`{`, "").withType(0, T[jsontext.Value]()),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/TextValue/Nested/Nil"),
		in:   structInlinePointerInlineTextValue{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/PointerTextValue/Nil"),
		in:   structInlinePointerTextValue{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/PointerTextValue/NonEmpty"),
		in:   structInlinePointerTextValue{X: addr(jsontext.Value(` { "fizz" : "buzz" } `))},
		want: `{"fizz":"buzz"}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/PointerTextValue/Nested/Nil"),
		in:   structInlineInlinePointerTextValue{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringAny/Nil"),
		in:   structInlineMapStringAny{X: nil},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringAny/Empty"),
		in:   structInlineMapStringAny{X: make(jsonObject)},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringAny/NonEmptyN1"),
		in:   structInlineMapStringAny{X: jsonObject{"fizz": nil}},
		want: `{"fizz":null}`,
	}, {
		name:         jsontest.Name("Structs/InlinedFallback/MapStringAny/NonEmptyN2"),
		in:           structInlineMapStringAny{X: jsonObject{"fizz": time.Time{}, "buzz": math.Pi}},
		want:         `{"buzz":3.141592653589793,"fizz":"0001-01-01T00:00:00Z"}`,
		canonicalize: true,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringAny/NonEmptyWithOthers"),
		in: structInlineMapStringAny{
			A: 1,
			X: jsonObject{"fizz": nil},
			B: 2,
		},
		// NOTE: Inlined fallback fields are always serialized last.
		want: `{"A":1,"B":2,"fizz":null}`,
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapStringAny/RejectInvalidUTF8"),
		opts:    []Options{jsontext.AllowInvalidUTF8(false)},
		in:      structInlineMapStringAny{X: jsonObject{"\xde\xad\xbe\xef": nil}},
		want:    `{`,
		wantErr: EM(jsonwire.ErrInvalidUTF8).withPos(`{`, "").withType(0, stringType),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringAny/AllowInvalidUTF8"),
		opts: []Options{jsontext.AllowInvalidUTF8(true)},
		in:   structInlineMapStringAny{X: jsonObject{"\xde\xad\xbe\xef": nil}},
		want: `{"ޭ��":null}`,
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapStringAny/InvalidValue"),
		opts:    []Options{jsontext.AllowInvalidUTF8(true)},
		in:      structInlineMapStringAny{X: jsonObject{"name": make(chan string)}},
		want:    `{"name"`,
		wantErr: EM(nil).withPos(`{"name":`, "/name").withType(0, T[chan string]()),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringAny/Nested/Nil"),
		in:   structInlinePointerInlineMapStringAny{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringAny/MarshalFunc"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v float64) ([]byte, error) {
				return []byte(fmt.Sprintf(`"%v"`, v)), nil
			})),
		},
		in:   structInlineMapStringAny{X: jsonObject{"fizz": 3.14159}},
		want: `{"fizz":"3.14159"}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/PointerMapStringAny/Nil"),
		in:   structInlinePointerMapStringAny{X: nil},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/PointerMapStringAny/NonEmpty"),
		in:   structInlinePointerMapStringAny{X: addr(jsonObject{"name": "value"})},
		want: `{"name":"value"}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/PointerMapStringAny/Nested/Nil"),
		in:   structInlineInlinePointerMapStringAny{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringInt"),
		in: structInlineMapStringInt{
			X: map[string]int{"zero": 0, "one": 1, "two": 2},
		},
		want:         `{"one":1,"two":2,"zero":0}`,
		canonicalize: true,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringInt/Deterministic"),
		opts: []Options{Deterministic(true)},
		in: structInlineMapStringInt{
			X: map[string]int{"zero": 0, "one": 1, "two": 2},
		},
		want: `{"one":1,"two":2,"zero":0}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringInt/Deterministic+AllowInvalidUTF8+RejectDuplicateNames"),
		opts: []Options{Deterministic(true), jsontext.AllowInvalidUTF8(true), jsontext.AllowDuplicateNames(false)},
		in: structInlineMapStringInt{
			X: map[string]int{"\xff": 0, "\xfe": 1},
		},
		want:    `{"�":1`,
		wantErr: newDuplicateNameError("", []byte(`"�"`), len64(`{"�":1`)),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringInt/Deterministic+AllowInvalidUTF8+AllowDuplicateNames"),
		opts: []Options{Deterministic(true), jsontext.AllowInvalidUTF8(true), jsontext.AllowDuplicateNames(true)},
		in: structInlineMapStringInt{
			X: map[string]int{"\xff": 0, "\xfe": 1},
		},
		want: `{"�":1,"�":0}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringInt/StringifiedNumbers"),
		opts: []Options{StringifyNumbers(true)},
		in: structInlineMapStringInt{
			X: map[string]int{"zero": 0, "one": 1, "two": 2},
		},
		want:         `{"one":"1","two":"2","zero":"0"}`,
		canonicalize: true,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringInt/MarshalFunc"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				// Marshalers do not affect the string key of inlined maps.
				MarshalFunc(func(v string) ([]byte, error) {
					return []byte(fmt.Sprintf(`"%q"`, strings.ToUpper(v))), nil
				}),
				MarshalFunc(func(v int) ([]byte, error) {
					return []byte(fmt.Sprintf(`"%v"`, v)), nil
				}),
			)),
		},
		in: structInlineMapStringInt{
			X: map[string]int{"zero": 0, "one": 1, "two": 2},
		},
		want:         `{"one":"1","two":"2","zero":"0"}`,
		canonicalize: true,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapNamedStringInt"),
		in: structInlineMapNamedStringInt{
			X: map[namedString]int{"zero": 0, "one": 1, "two": 2},
		},
		want:         `{"one":1,"two":2,"zero":0}`,
		canonicalize: true,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapNamedStringInt/Deterministic"),
		opts: []Options{Deterministic(true)},
		in: structInlineMapNamedStringInt{
			X: map[namedString]int{"zero": 0, "one": 1, "two": 2},
		},
		want: `{"one":1,"two":2,"zero":0}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/Nil"),
		in:   structInlineMapNamedStringAny{X: nil},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/Empty"),
		in:   structInlineMapNamedStringAny{X: make(map[namedString]any)},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/NonEmptyN1"),
		in:   structInlineMapNamedStringAny{X: map[namedString]any{"fizz": nil}},
		want: `{"fizz":null}`,
	}, {
		name:         jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/NonEmptyN2"),
		in:           structInlineMapNamedStringAny{X: map[namedString]any{"fizz": time.Time{}, "buzz": math.Pi}},
		want:         `{"buzz":3.141592653589793,"fizz":"0001-01-01T00:00:00Z"}`,
		canonicalize: true,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/NonEmptyWithOthers"),
		in: structInlineMapNamedStringAny{
			A: 1,
			X: map[namedString]any{"fizz": nil},
			B: 2,
		},
		// NOTE: Inlined fallback fields are always serialized last.
		want: `{"A":1,"B":2,"fizz":null}`,
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/RejectInvalidUTF8"),
		opts:    []Options{jsontext.AllowInvalidUTF8(false)},
		in:      structInlineMapNamedStringAny{X: map[namedString]any{"\xde\xad\xbe\xef": nil}},
		want:    `{`,
		wantErr: EM(jsonwire.ErrInvalidUTF8).withPos(`{`, "").withType(0, T[namedString]()),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/AllowInvalidUTF8"),
		opts: []Options{jsontext.AllowInvalidUTF8(true)},
		in:   structInlineMapNamedStringAny{X: map[namedString]any{"\xde\xad\xbe\xef": nil}},
		want: `{"ޭ��":null}`,
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/InvalidValue"),
		opts:    []Options{jsontext.AllowInvalidUTF8(true)},
		in:      structInlineMapNamedStringAny{X: map[namedString]any{"name": make(chan string)}},
		want:    `{"name"`,
		wantErr: EM(nil).withPos(`{"name":`, "/name").withType(0, T[chan string]()),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/MarshalFunc"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v float64) ([]byte, error) {
				return []byte(fmt.Sprintf(`"%v"`, v)), nil
			})),
		},
		in:   structInlineMapNamedStringAny{X: map[namedString]any{"fizz": 3.14159}},
		want: `{"fizz":"3.14159"}`,
	}, {
		name: jsontest.Name("Structs/InlinedFallback/DiscardUnknownMembers"),
		opts: []Options{DiscardUnknownMembers(true)},
		in: structInlineTextValue{
			A: 1,
			X: jsontext.Value(` { "fizz" : "buzz" } `),
			B: 2,
		},
		// NOTE: DiscardUnknownMembers has no effect since this is "inline".
		want: `{"A":1,"B":2,"fizz":"buzz"}`,
	}, {
		name: jsontest.Name("Structs/UnknownFallback/DiscardUnknownMembers"),
		opts: []Options{DiscardUnknownMembers(true)},
		in: structUnknownTextValue{
			A: 1,
			X: jsontext.Value(` { "fizz" : "buzz" } `),
			B: 2,
		},
		want: `{"A":1,"B":2}`,
	}, {
		name: jsontest.Name("Structs/UnknownFallback"),
		in: structUnknownTextValue{
			A: 1,
			X: jsontext.Value(` { "fizz" : "buzz" } `),
			B: 2,
		},
		want: `{"A":1,"B":2,"fizz":"buzz"}`,
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/Other"),
		in: structNoCaseInlineTextValue{
			X: jsontext.Value(`{"dupe":"","dupe":""}`),
		},
		want:    `{"dupe":""`,
		wantErr: newDuplicateNameError("", []byte(`"dupe"`), len64(`{"dupe":""`)),
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/Other/AllowDuplicateNames"),
		opts: []Options{jsontext.AllowDuplicateNames(true)},
		in: structNoCaseInlineTextValue{
			X: jsontext.Value(`{"dupe": "", "dupe": ""}`),
		},
		want: `{"dupe":"","dupe":""}`,
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/ExactDifferent"),
		in: structNoCaseInlineTextValue{
			X: jsontext.Value(`{"Aaa": "", "AaA": "", "AAa": "", "AAA": ""}`),
		},
		want: `{"Aaa":"","AaA":"","AAa":"","AAA":""}`,
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/ExactConflict"),
		in: structNoCaseInlineTextValue{
			X: jsontext.Value(`{"Aaa": "", "Aaa": ""}`),
		},
		want:    `{"Aaa":""`,
		wantErr: newDuplicateNameError("", []byte(`"Aaa"`), len64(`{"Aaa":""`)),
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/ExactConflict/AllowDuplicateNames"),
		opts: []Options{jsontext.AllowDuplicateNames(true)},
		in: structNoCaseInlineTextValue{
			X: jsontext.Value(`{"Aaa": "", "Aaa": ""}`),
		},
		want: `{"Aaa":"","Aaa":""}`,
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/NoCaseConflict"),
		in: structNoCaseInlineTextValue{
			X: jsontext.Value(`{"Aaa": "", "AaA": "", "aaa": ""}`),
		},
		want:    `{"Aaa":"","AaA":""`,
		wantErr: newDuplicateNameError("", []byte(`"aaa"`), len64(`{"Aaa":"","AaA":""`)),
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/NoCaseConflict/AllowDuplicateNames"),
		opts: []Options{jsontext.AllowDuplicateNames(true)},
		in: structNoCaseInlineTextValue{
			X: jsontext.Value(`{"Aaa": "", "AaA": "", "aaa": ""}`),
		},
		want: `{"Aaa":"","AaA":"","aaa":""}`,
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/ExactDifferentWithField"),
		in: structNoCaseInlineTextValue{
			AAA: "x",
			AaA: "x",
			X:   jsontext.Value(`{"Aaa": ""}`),
		},
		want: `{"AAA":"x","AaA":"x","Aaa":""}`,
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/ExactConflictWithField"),
		in: structNoCaseInlineTextValue{
			AAA: "x",
			AaA: "x",
			X:   jsontext.Value(`{"AAA": ""}`),
		},
		want:    `{"AAA":"x","AaA":"x"`,
		wantErr: newDuplicateNameError("", []byte(`"AAA"`), len64(`{"AAA":"x","AaA":"x"`)),
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineTextValue/NoCaseConflictWithField"),
		in: structNoCaseInlineTextValue{
			AAA: "x",
			AaA: "x",
			X:   jsontext.Value(`{"aaa": ""}`),
		},
		want:    `{"AAA":"x","AaA":"x"`,
		wantErr: newDuplicateNameError("", []byte(`"aaa"`), len64(`{"AAA":"x","AaA":"x"`)),
	}, {
		name: jsontest.Name("Structs/DuplicateName/MatchCaseInsensitiveDelimiter"),
		in: structNoCaseInlineTextValue{
			AaA: "x",
			X:   jsontext.Value(`{"aa_a": ""}`),
		},
		want:    `{"AaA":"x"`,
		wantErr: newDuplicateNameError("", []byte(`"aa_a"`), len64(`{"AaA":"x"`)),
	}, {
		name: jsontest.Name("Structs/DuplicateName/MatchCaseSensitiveDelimiter"),
		opts: []Options{jsonflags.MatchCaseSensitiveDelimiter | 1},
		in: structNoCaseInlineTextValue{
			AaA: "x",
			X:   jsontext.Value(`{"aa_a": ""}`),
		},
		want: `{"AaA":"x","aa_a":""}`,
	}, {
		name: jsontest.Name("Structs/DuplicateName/MatchCaseInsensitiveNames+MatchCaseSensitiveDelimiter"),
		opts: []Options{MatchCaseInsensitiveNames(true), jsonflags.MatchCaseSensitiveDelimiter | 1},
		in: structNoCaseInlineTextValue{
			AaA: "x",
			X:   jsontext.Value(`{"aa_a": ""}`),
		},
		want: `{"AaA":"x","aa_a":""}`,
	}, {
		name: jsontest.Name("Structs/DuplicateName/MatchCaseInsensitiveNames+MatchCaseSensitiveDelimiter"),
		opts: []Options{MatchCaseInsensitiveNames(true), jsonflags.MatchCaseSensitiveDelimiter | 1},
		in: structNoCaseInlineTextValue{
			AA_b: "x",
			X:    jsontext.Value(`{"aa_b": ""}`),
		},
		want:    `{"AA_b":"x"`,
		wantErr: newDuplicateNameError("", []byte(`"aa_b"`), len64(`{"AA_b":"x"`)),
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineMapStringAny/ExactDifferent"),
		in: structNoCaseInlineMapStringAny{
			X: jsonObject{"Aaa": "", "AaA": "", "AAa": "", "AAA": ""},
		},
		want:         `{"AAA":"","AAa":"","AaA":"","Aaa":""}`,
		canonicalize: true,
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineMapStringAny/ExactDifferentWithField"),
		in: structNoCaseInlineMapStringAny{
			AAA: "x",
			AaA: "x",
			X:   jsonObject{"Aaa": ""},
		},
		want: `{"AAA":"x","AaA":"x","Aaa":""}`,
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineMapStringAny/ExactConflictWithField"),
		in: structNoCaseInlineMapStringAny{
			AAA: "x",
			AaA: "x",
			X:   jsonObject{"AAA": ""},
		},
		want:    `{"AAA":"x","AaA":"x"`,
		wantErr: newDuplicateNameError("", []byte(`"AAA"`), len64(`{"AAA":"x","AaA":"x"`)),
	}, {
		name: jsontest.Name("Structs/DuplicateName/NoCaseInlineMapStringAny/NoCaseConflictWithField"),
		in: structNoCaseInlineMapStringAny{
			AAA: "x",
			AaA: "x",
			X:   jsonObject{"aaa": ""},
		},
		want:    `{"AAA":"x","AaA":"x"`,
		wantErr: newDuplicateNameError("", []byte(`"aaa"`), len64(`{"AAA":"x","AaA":"x"`)),
	}, {
		name:    jsontest.Name("Structs/Invalid/Conflicting"),
		in:      structConflicting{},
		want:    ``,
		wantErr: EM(errors.New("Go struct fields A and B conflict over JSON object name \"conflict\"")).withType(0, T[structConflicting]()),
	}, {
		name:    jsontest.Name("Structs/Invalid/NoneExported"),
		in:      structNoneExported{},
		want:    ``,
		wantErr: EM(errNoExportedFields).withType(0, T[structNoneExported]()),
	}, {
		name:    jsontest.Name("Structs/Invalid/MalformedTag"),
		in:      structMalformedTag{},
		want:    ``,
		wantErr: EM(errors.New("Go struct field Malformed has malformed `json` tag: invalid character '\"' at start of option (expecting Unicode letter or single quote)")).withType(0, T[structMalformedTag]()),
	}, {
		name:    jsontest.Name("Structs/Invalid/UnexportedTag"),
		in:      structUnexportedTag{},
		want:    ``,
		wantErr: EM(errors.New("unexported Go struct field unexported cannot have non-ignored `json:\"name\"` tag")).withType(0, T[structUnexportedTag]()),
	}, {
		name:    jsontest.Name("Structs/Invalid/ExportedEmbedded"),
		in:      structExportedEmbedded{"hello"},
		want:    ``,
		wantErr: EM(errors.New("embedded Go struct field NamedString of non-struct type must be explicitly given a JSON name")).withType(0, T[structExportedEmbedded]()),
	}, {
		name: jsontest.Name("Structs/Valid/ExportedEmbedded"),
		opts: []Options{jsonflags.ReportErrorsWithLegacySemantics | 1},
		in:   structExportedEmbedded{"hello"},
		want: `{"NamedString":"hello"}`,
	}, {
		name: jsontest.Name("Structs/Valid/ExportedEmbeddedTag"),
		in:   structExportedEmbeddedTag{"hello"},
		want: `{"name":"hello"}`,
	}, {
		name:    jsontest.Name("Structs/Invalid/UnexportedEmbedded"),
		in:      structUnexportedEmbedded{},
		want:    ``,
		wantErr: EM(errors.New("embedded Go struct field namedString of non-struct type must be explicitly given a JSON name")).withType(0, T[structUnexportedEmbedded]()),
	}, {
		name: jsontest.Name("Structs/Valid/UnexportedEmbedded"),
		opts: []Options{jsonflags.ReportErrorsWithLegacySemantics | 1},
		in:   structUnexportedEmbedded{},
		want: `{}`,
	}, {
		name:    jsontest.Name("Structs/Invalid/UnexportedEmbeddedTag"),
		in:      structUnexportedEmbeddedTag{},
		wantErr: EM(errors.New("Go struct field namedString is not exported")).withType(0, T[structUnexportedEmbeddedTag]()),
	}, {
		name: jsontest.Name("Structs/Valid/UnexportedEmbeddedTag"),
		opts: []Options{jsonflags.ReportErrorsWithLegacySemantics | 1},
		in:   structUnexportedEmbeddedTag{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/Invalid/UnexportedEmbeddedMethodTag"),
		opts: []Options{jsonflags.ReportErrorsWithLegacySemantics | 1},
		in:   structUnexportedEmbeddedMethodTag{},
		want: `{}`,
	}, {
		name: jsontest.Name("Structs/UnexportedEmbeddedStruct/Zero"),
		in:   structUnexportedEmbeddedStruct{},
		want: `{"FizzBuzz":0,"Addr":""}`,
	}, {
		name: jsontest.Name("Structs/UnexportedEmbeddedStruct/NonZero"),
		in:   structUnexportedEmbeddedStruct{structOmitZeroAll{Bool: true}, 5, structNestedAddr{netip.AddrFrom4([4]byte{192, 168, 0, 1})}},
		want: `{"Bool":true,"FizzBuzz":5,"Addr":"192.168.0.1"}`,
	}, {
		name: jsontest.Name("Structs/UnexportedEmbeddedStructPointer/Nil"),
		in:   structUnexportedEmbeddedStructPointer{},
		want: `{"FizzBuzz":0}`,
	}, {
		name: jsontest.Name("Structs/UnexportedEmbeddedStructPointer/Zero"),
		in:   structUnexportedEmbeddedStructPointer{&structOmitZeroAll{}, 0, &structNestedAddr{}},
		want: `{"FizzBuzz":0,"Addr":""}`,
	}, {
		name: jsontest.Name("Structs/UnexportedEmbeddedStructPointer/NonZero"),
		in:   structUnexportedEmbeddedStructPointer{&structOmitZeroAll{Bool: true}, 5, &structNestedAddr{netip.AddrFrom4([4]byte{192, 168, 0, 1})}},
		want: `{"Bool":true,"FizzBuzz":5,"Addr":"192.168.0.1"}`,
	}, {
		name: jsontest.Name("Structs/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   struct{}{},
		want: `{}`,
	}, {
		name: jsontest.Name("Slices/Interface"),
		in: []any{
			false, true,
			"hello", []byte("world"),
			int32(-32), namedInt64(-64),
			uint32(+32), namedUint64(+64),
			float32(32.32), namedFloat64(64.64),
		},
		want: `[false,true,"hello","d29ybGQ=",-32,-64,32,64,32.32,64.64]`,
	}, {
		name:    jsontest.Name("Slices/Invalid/Channel"),
		in:      [](chan string){nil},
		want:    `[`,
		wantErr: EM(nil).withPos(`[`, "/0").withType(0, T[chan string]()),
	}, {
		name: jsontest.Name("Slices/RecursiveSlice"),
		in: recursiveSlice{
			nil,
			{},
			{nil},
			{nil, {}},
		},
		want: `[[],[],[[]],[[],[]]]`,
	}, {
		name: jsontest.Name("Slices/CyclicSlice"),
		in: func() recursiveSlice {
			s := recursiveSlice{{}}
			s[0] = s
			return s
		}(),
		want:    strings.Repeat(`[`, startDetectingCyclesAfter) + `[`,
		wantErr: EM(internal.ErrCycle).withPos(strings.Repeat("[", startDetectingCyclesAfter+1), jsontext.Pointer(strings.Repeat("/0", startDetectingCyclesAfter+1))).withType(0, T[recursiveSlice]()),
	}, {
		name: jsontest.Name("Slices/NonCyclicSlice"),
		in: func() []any {
			v := []any{nil, nil}
			v[1] = v[:1]
			for i := 1000; i > 0; i-- {
				v = []any{v}
			}
			return v
		}(),
		want: strings.Repeat(`[`, startDetectingCyclesAfter) + `[null,[null]]` + strings.Repeat(`]`, startDetectingCyclesAfter),
	}, {
		name: jsontest.Name("Slices/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   []string{"hello", "goodbye"},
		want: `["hello","goodbye"]`,
	}, {
		name: jsontest.Name("Arrays/Empty"),
		in:   [0]struct{}{},
		want: `[]`,
	}, {
		name: jsontest.Name("Arrays/Bool"),
		in:   [2]bool{false, true},
		want: `[false,true]`,
	}, {
		name: jsontest.Name("Arrays/String"),
		in:   [2]string{"hello", "goodbye"},
		want: `["hello","goodbye"]`,
	}, {
		name: jsontest.Name("Arrays/Bytes"),
		in:   [2][]byte{[]byte("hello"), []byte("goodbye")},
		want: `["aGVsbG8=","Z29vZGJ5ZQ=="]`,
	}, {
		name: jsontest.Name("Arrays/Int"),
		in:   [2]int64{math.MinInt64, math.MaxInt64},
		want: `[-9223372036854775808,9223372036854775807]`,
	}, {
		name: jsontest.Name("Arrays/Uint"),
		in:   [2]uint64{0, math.MaxUint64},
		want: `[0,18446744073709551615]`,
	}, {
		name: jsontest.Name("Arrays/Float"),
		in:   [2]float64{-math.MaxFloat64, +math.MaxFloat64},
		want: `[-1.7976931348623157e+308,1.7976931348623157e+308]`,
	}, {
		name:    jsontest.Name("Arrays/Invalid/Channel"),
		in:      new([1]chan string),
		want:    `[`,
		wantErr: EM(nil).withPos(`[`, "/0").withType(0, T[chan string]()),
	}, {
		name: jsontest.Name("Arrays/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   [2]string{"hello", "goodbye"},
		want: `["hello","goodbye"]`,
	}, {
		name: jsontest.Name("Pointers/NilL0"),
		in:   (*int)(nil),
		want: `null`,
	}, {
		name: jsontest.Name("Pointers/NilL1"),
		in:   new(*int),
		want: `null`,
	}, {
		name: jsontest.Name("Pointers/Bool"),
		in:   addr(addr(bool(true))),
		want: `true`,
	}, {
		name: jsontest.Name("Pointers/String"),
		in:   addr(addr(string("string"))),
		want: `"string"`,
	}, {
		name: jsontest.Name("Pointers/Bytes"),
		in:   addr(addr([]byte("bytes"))),
		want: `"Ynl0ZXM="`,
	}, {
		name: jsontest.Name("Pointers/Int"),
		in:   addr(addr(int(-100))),
		want: `-100`,
	}, {
		name: jsontest.Name("Pointers/Uint"),
		in:   addr(addr(uint(100))),
		want: `100`,
	}, {
		name: jsontest.Name("Pointers/Float"),
		in:   addr(addr(float64(3.14159))),
		want: `3.14159`,
	}, {
		name: jsontest.Name("Pointers/CyclicPointer"),
		in: func() *recursivePointer {
			p := new(recursivePointer)
			p.P = p
			return p
		}(),
		want:    strings.Repeat(`{"P":`, startDetectingCyclesAfter) + `{"P"`,
		wantErr: EM(internal.ErrCycle).withPos(strings.Repeat(`{"P":`, startDetectingCyclesAfter+1), jsontext.Pointer(strings.Repeat("/P", startDetectingCyclesAfter+1))).withType(0, T[*recursivePointer]()),
	}, {
		name: jsontest.Name("Pointers/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   addr(addr(bool(true))),
		want: `true`,
	}, {
		name: jsontest.Name("Interfaces/Nil/Empty"),
		in:   [1]any{nil},
		want: `[null]`,
	}, {
		name: jsontest.Name("Interfaces/Nil/NonEmpty"),
		in:   [1]io.Reader{nil},
		want: `[null]`,
	}, {
		name: jsontest.Name("Interfaces/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   [1]io.Reader{nil},
		want: `[null]`,
	}, {
		name: jsontest.Name("Interfaces/Any"),
		in:   struct{ X any }{[]any{nil, false, "", 0.0, map[string]any{}, []any{}, [8]byte{}}},
		want: `{"X":[null,false,"",0,{},[],"AAAAAAAAAAA="]}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Named"),
		in:   struct{ X namedAny }{[]namedAny{nil, false, "", 0.0, map[string]namedAny{}, []namedAny{}, [8]byte{}}},
		want: `{"X":[null,false,"",0,{},[],"AAAAAAAAAAA="]}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Stringified"),
		opts: []Options{StringifyNumbers(true)},
		in:   struct{ X any }{0.0},
		want: `{"X":"0"}`,
	}, {
		name: jsontest.Name("Interfaces/Any/MarshalFunc/Any"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v any) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   struct{ X any }{[]any{nil, false, "", 0.0, map[string]any{}, []any{}}},
		want: `"called"`,
	}, {
		name: jsontest.Name("Interfaces/Any/MarshalFunc/Bool"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v bool) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   struct{ X any }{[]any{nil, false, "", 0.0, map[string]any{}, []any{}}},
		want: `{"X":[null,"called","",0,{},[]]}`,
	}, {
		name: jsontest.Name("Interfaces/Any/MarshalFunc/String"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v string) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   struct{ X any }{[]any{nil, false, "", 0.0, map[string]any{}, []any{}}},
		want: `{"X":[null,false,"called",0,{},[]]}`,
	}, {
		name: jsontest.Name("Interfaces/Any/MarshalFunc/Float64"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v float64) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   struct{ X any }{[]any{nil, false, "", 0.0, map[string]any{}, []any{}}},
		want: `{"X":[null,false,"","called",{},[]]}`,
	}, {
		name: jsontest.Name("Interfaces/Any/MarshalFunc/MapStringAny"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v map[string]any) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   struct{ X any }{[]any{nil, false, "", 0.0, map[string]any{}, []any{}}},
		want: `{"X":[null,false,"",0,"called",[]]}`,
	}, {
		name: jsontest.Name("Interfaces/Any/MarshalFunc/SliceAny"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v []any) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   struct{ X any }{[]any{nil, false, "", 0.0, map[string]any{}, []any{}}},
		want: `{"X":"called"}`,
	}, {
		name: jsontest.Name("Interfaces/Any/MarshalFunc/Bytes"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v [8]byte) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   struct{ X any }{[8]byte{}},
		want: `{"X":"called"}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Maps/Nil"),
		in:   struct{ X any }{map[string]any(nil)},
		want: `{"X":{}}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Maps/Nil/FormatNilMapAsNull"),
		opts: []Options{FormatNilMapAsNull(true)},
		in:   struct{ X any }{map[string]any(nil)},
		want: `{"X":null}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Maps/Empty"),
		in:   struct{ X any }{map[string]any{}},
		want: `{"X":{}}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Maps/Empty/Multiline"),
		opts: []Options{jsontext.Multiline(true), jsontext.WithIndent("")},
		in:   struct{ X any }{map[string]any{}},
		want: "{\n\"X\": {}\n}",
	}, {
		name: jsontest.Name("Interfaces/Any/Maps/NonEmpty"),
		in:   struct{ X any }{map[string]any{"fizz": "buzz"}},
		want: `{"X":{"fizz":"buzz"}}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Maps/Deterministic"),
		opts: []Options{Deterministic(true)},
		in:   struct{ X any }{map[string]any{"alpha": "", "bravo": ""}},
		want: `{"X":{"alpha":"","bravo":""}}`,
	}, {
		name:    jsontest.Name("Interfaces/Any/Maps/Deterministic+AllowInvalidUTF8+RejectDuplicateNames"),
		opts:    []Options{Deterministic(true), jsontext.AllowInvalidUTF8(true), jsontext.AllowDuplicateNames(false)},
		in:      struct{ X any }{map[string]any{"\xff": "", "\xfe": ""}},
		want:    `{"X":{"�":""`,
		wantErr: newDuplicateNameError("/X", []byte(`"�"`), len64(`{"X":{"�":"",`)),
	}, {
		name: jsontest.Name("Interfaces/Any/Maps/Deterministic+AllowInvalidUTF8+AllowDuplicateNames"),
		opts: []Options{Deterministic(true), jsontext.AllowInvalidUTF8(true), jsontext.AllowDuplicateNames(true)},
		in:   struct{ X any }{map[string]any{"\xff": "alpha", "\xfe": "bravo"}},
		want: `{"X":{"�":"bravo","�":"alpha"}}`,
	}, {
		name:    jsontest.Name("Interfaces/Any/Maps/RejectInvalidUTF8"),
		in:      struct{ X any }{map[string]any{"\xff": "", "\xfe": ""}},
		want:    `{"X":{`,
		wantErr: newInvalidUTF8Error(len64(`{"X":{`), "/X"),
	}, {
		name:    jsontest.Name("Interfaces/Any/Maps/AllowInvalidUTF8+RejectDuplicateNames"),
		opts:    []Options{jsontext.AllowInvalidUTF8(true)},
		in:      struct{ X any }{map[string]any{"\xff": "", "\xfe": ""}},
		want:    `{"X":{"�":""`,
		wantErr: newDuplicateNameError("/X", []byte(`"�"`), len64(`{"X":{"�":"",`)),
	}, {
		name: jsontest.Name("Interfaces/Any/Maps/AllowInvalidUTF8+AllowDuplicateNames"),
		opts: []Options{jsontext.AllowInvalidUTF8(true), jsontext.AllowDuplicateNames(true)},
		in:   struct{ X any }{map[string]any{"\xff": "", "\xfe": ""}},
		want: `{"X":{"�":"","�":""}}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Maps/Cyclic"),
		in: func() any {
			m := map[string]any{}
			m[""] = m
			return struct{ X any }{m}
		}(),
		want:    `{"X"` + strings.Repeat(`:{""`, startDetectingCyclesAfter),
		wantErr: EM(internal.ErrCycle).withPos(`{"X":`+strings.Repeat(`{"":`, startDetectingCyclesAfter), "/X"+jsontext.Pointer(strings.Repeat("/", startDetectingCyclesAfter))).withType(0, T[any]()),
	}, {
		name: jsontest.Name("Interfaces/Any/Slices/Nil"),
		in:   struct{ X any }{[]any(nil)},
		want: `{"X":[]}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Slices/Nil/FormatNilSliceAsNull"),
		opts: []Options{FormatNilSliceAsNull(true)},
		in:   struct{ X any }{[]any(nil)},
		want: `{"X":null}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Slices/Empty"),
		in:   struct{ X any }{[]any{}},
		want: `{"X":[]}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Slices/Empty/Multiline"),
		opts: []Options{jsontext.Multiline(true), jsontext.WithIndent("")},
		in:   struct{ X any }{[]any{}},
		want: "{\n\"X\": []\n}",
	}, {
		name: jsontest.Name("Interfaces/Any/Slices/NonEmpty"),
		in:   struct{ X any }{[]any{"fizz", "buzz"}},
		want: `{"X":["fizz","buzz"]}`,
	}, {
		name: jsontest.Name("Interfaces/Any/Slices/Cyclic"),
		in: func() any {
			s := make([]any, 1)
			s[0] = s
			return struct{ X any }{s}
		}(),
		want:    `{"X":` + strings.Repeat(`[`, startDetectingCyclesAfter),
		wantErr: EM(internal.ErrCycle).withPos(`{"X":`+strings.Repeat(`[`, startDetectingCyclesAfter), "/X"+jsontext.Pointer(strings.Repeat("/0", startDetectingCyclesAfter))).withType(0, T[[]any]()),
	}, {
		name: jsontest.Name("Methods/NilPointer"),
		in:   struct{ X *allMethods }{X: (*allMethods)(nil)}, // method should not be called
		want: `{"X":null}`,
	}, {
		// NOTE: Fixes https://github.com/dominikh/go-tools/issues/975.
		name: jsontest.Name("Methods/NilInterface"),
		in:   struct{ X MarshalerTo }{X: (*allMethods)(nil)}, // method should not be called
		want: `{"X":null}`,
	}, {
		name: jsontest.Name("Methods/AllMethods"),
		in:   struct{ X *allMethods }{X: &allMethods{method: "MarshalJSONTo", value: []byte(`"hello"`)}},
		want: `{"X":"hello"}`,
	}, {
		name: jsontest.Name("Methods/AllMethodsExceptJSONv2"),
		in:   struct{ X *allMethodsExceptJSONv2 }{X: &allMethodsExceptJSONv2{allMethods: allMethods{method: "MarshalJSON", value: []byte(`"hello"`)}}},
		want: `{"X":"hello"}`,
	}, {
		name: jsontest.Name("Methods/AllMethodsExceptJSONv1"),
		in:   struct{ X *allMethodsExceptJSONv1 }{X: &allMethodsExceptJSONv1{allMethods: allMethods{method: "MarshalJSONTo", value: []byte(`"hello"`)}}},
		want: `{"X":"hello"}`,
	}, {
		name: jsontest.Name("Methods/AllMethodsExceptText"),
		in:   struct{ X *allMethodsExceptText }{X: &allMethodsExceptText{allMethods: allMethods{method: "MarshalJSONTo", value: []byte(`"hello"`)}}},
		want: `{"X":"hello"}`,
	}, {
		name: jsontest.Name("Methods/OnlyMethodJSONv2"),
		in:   struct{ X *onlyMethodJSONv2 }{X: &onlyMethodJSONv2{allMethods: allMethods{method: "MarshalJSONTo", value: []byte(`"hello"`)}}},
		want: `{"X":"hello"}`,
	}, {
		name: jsontest.Name("Methods/OnlyMethodJSONv1"),
		in:   struct{ X *onlyMethodJSONv1 }{X: &onlyMethodJSONv1{allMethods: allMethods{method: "MarshalJSON", value: []byte(`"hello"`)}}},
		want: `{"X":"hello"}`,
	}, {
		name: jsontest.Name("Methods/OnlyMethodText"),
		in:   struct{ X *onlyMethodText }{X: &onlyMethodText{allMethods: allMethods{method: "MarshalText", value: []byte(`hello`)}}},
		want: `{"X":"hello"}`,
	}, {
		name: jsontest.Name("Methods/IP"),
		in:   net.IPv4(192, 168, 0, 100),
		want: `"192.168.0.100"`,
	}, {
		name: jsontest.Name("Methods/NetIP"),
		in: struct {
			Addr     netip.Addr
			AddrPort netip.AddrPort
			Prefix   netip.Prefix
		}{
			Addr:     netip.AddrFrom4([4]byte{1, 2, 3, 4}),
			AddrPort: netip.AddrPortFrom(netip.AddrFrom4([4]byte{1, 2, 3, 4}), 1234),
			Prefix:   netip.PrefixFrom(netip.AddrFrom4([4]byte{1, 2, 3, 4}), 24),
		},
		want: `{"Addr":"1.2.3.4","AddrPort":"1.2.3.4:1234","Prefix":"1.2.3.4/24"}`,
	}, {
		// NOTE: Fixes https://go.dev/issue/46516.
		name: jsontest.Name("Methods/Anonymous"),
		in:   struct{ X struct{ allMethods } }{X: struct{ allMethods }{allMethods{method: "MarshalJSONTo", value: []byte(`"hello"`)}}},
		want: `{"X":"hello"}`,
	}, {
		// NOTE: Fixes https://go.dev/issue/22967.
		name: jsontest.Name("Methods/Addressable"),
		in: struct {
			V allMethods
			M map[string]allMethods
			I any
		}{
			V: allMethods{method: "MarshalJSONTo", value: []byte(`"hello"`)},
			M: map[string]allMethods{"K": {method: "MarshalJSONTo", value: []byte(`"hello"`)}},
			I: allMethods{method: "MarshalJSONTo", value: []byte(`"hello"`)},
		},
		want: `{"V":"hello","M":{"K":"hello"},"I":"hello"}`,
	}, {
		// NOTE: Fixes https://go.dev/issue/29732.
		name:         jsontest.Name("Methods/MapKey/JSONv2"),
		in:           map[structMethodJSONv2]string{{"k1"}: "v1", {"k2"}: "v2"},
		want:         `{"k1":"v1","k2":"v2"}`,
		canonicalize: true,
	}, {
		// NOTE: Fixes https://go.dev/issue/29732.
		name:         jsontest.Name("Methods/MapKey/JSONv1"),
		in:           map[structMethodJSONv1]string{{"k1"}: "v1", {"k2"}: "v2"},
		want:         `{"k1":"v1","k2":"v2"}`,
		canonicalize: true,
	}, {
		name:         jsontest.Name("Methods/MapKey/Text"),
		in:           map[structMethodText]string{{"k1"}: "v1", {"k2"}: "v2"},
		want:         `{"k1":"v1","k2":"v2"}`,
		canonicalize: true,
	}, {
		name: jsontest.Name("Methods/Invalid/JSONv2/Error"),
		in: marshalJSONv2Func(func(*jsontext.Encoder) error {
			return errSomeError
		}),
		wantErr: EM(errSomeError).withType(0, T[marshalJSONv2Func]()),
	}, {
		name: jsontest.Name("Methods/Invalid/JSONv2/TooFew"),
		in: marshalJSONv2Func(func(*jsontext.Encoder) error {
			return nil // do nothing
		}),
		wantErr: EM(errNonSingularValue).withType(0, T[marshalJSONv2Func]()),
	}, {
		name: jsontest.Name("Methods/Invalid/JSONv2/TooMany"),
		in: marshalJSONv2Func(func(enc *jsontext.Encoder) error {
			enc.WriteToken(jsontext.Null)
			enc.WriteToken(jsontext.Null)
			return nil
		}),
		want:    `nullnull`,
		wantErr: EM(errNonSingularValue).withPos(`nullnull`, "").withType(0, T[marshalJSONv2Func]()),
	}, {
		name: jsontest.Name("Methods/Invalid/JSONv2/SkipFunc"),
		in: marshalJSONv2Func(func(enc *jsontext.Encoder) error {
			return SkipFunc
		}),
		wantErr: EM(errors.New("marshal method cannot be skipped")).withType(0, T[marshalJSONv2Func]()),
	}, {
		name: jsontest.Name("Methods/Invalid/JSONv1/Error"),
		in: marshalJSONv1Func(func() ([]byte, error) {
			return nil, errSomeError
		}),
		wantErr: EM(errSomeError).withType(0, T[marshalJSONv1Func]()),
	}, {
		name: jsontest.Name("Methods/Invalid/JSONv1/Syntax"),
		in: marshalJSONv1Func(func() ([]byte, error) {
			return []byte("invalid"), nil
		}),
		wantErr: EM(newInvalidCharacterError("i", "at start of value", 0, "")).withType(0, T[marshalJSONv1Func]()),
	}, {
		name: jsontest.Name("Methods/Invalid/JSONv1/SkipFunc"),
		in: marshalJSONv1Func(func() ([]byte, error) {
			return nil, SkipFunc
		}),
		wantErr: EM(errors.New("marshal method cannot be skipped")).withType(0, T[marshalJSONv1Func]()),
	}, {
		name: jsontest.Name("Methods/AppendText"),
		in:   appendTextFunc(func(b []byte) ([]byte, error) { return append(b, "hello"...), nil }),
		want: `"hello"`,
	}, {
		name:    jsontest.Name("Methods/AppendText/Error"),
		in:      appendTextFunc(func(b []byte) ([]byte, error) { return append(b, "hello"...), errSomeError }),
		wantErr: EM(errSomeError).withType(0, T[appendTextFunc]()),
	}, {
		name: jsontest.Name("Methods/AppendText/NeedEscape"),
		in:   appendTextFunc(func(b []byte) ([]byte, error) { return append(b, `"`...), nil }),
		want: `"\""`,
	}, {
		name:    jsontest.Name("Methods/AppendText/RejectInvalidUTF8"),
		in:      appendTextFunc(func(b []byte) ([]byte, error) { return append(b, "\xde\xad\xbe\xef"...), nil }),
		wantErr: EM(newInvalidUTF8Error(0, "")).withType(0, T[appendTextFunc]()),
	}, {
		name: jsontest.Name("Methods/AppendText/AllowInvalidUTF8"),
		opts: []Options{jsontext.AllowInvalidUTF8(true)},
		in:   appendTextFunc(func(b []byte) ([]byte, error) { return append(b, "\xde\xad\xbe\xef"...), nil }),
		want: "\"\xde\xad\ufffd\ufffd\"",
	}, {
		name: jsontest.Name("Methods/Invalid/Text/Error"),
		in: marshalTextFunc(func() ([]byte, error) {
			return nil, errSomeError
		}),
		wantErr: EM(errSomeError).withType(0, T[marshalTextFunc]()),
	}, {
		name: jsontest.Name("Methods/Text/RejectInvalidUTF8"),
		in: marshalTextFunc(func() ([]byte, error) {
			return []byte("\xde\xad\xbe\xef"), nil
		}),
		wantErr: EM(newInvalidUTF8Error(0, "")).withType(0, T[marshalTextFunc]()),
	}, {
		name: jsontest.Name("Methods/Text/AllowInvalidUTF8"),
		opts: []Options{jsontext.AllowInvalidUTF8(true)},
		in: marshalTextFunc(func() ([]byte, error) {
			return []byte("\xde\xad\xbe\xef"), nil
		}),
		want: "\"\xde\xad\ufffd\ufffd\"",
	}, {
		name: jsontest.Name("Methods/Invalid/Text/SkipFunc"),
		in: marshalTextFunc(func() ([]byte, error) {
			return nil, SkipFunc
		}),
		wantErr: EM(wrapSkipFunc(SkipFunc, "marshal method")).withType(0, T[marshalTextFunc]()),
	}, {
		name: jsontest.Name("Methods/Invalid/MapKey/JSONv2/Syntax"),
		in: map[any]string{
			addr(marshalJSONv2Func(func(enc *jsontext.Encoder) error {
				return enc.WriteToken(jsontext.Null)
			})): "invalid",
		},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(`{`), "")).withPos(`{`, "").withType(0, T[marshalJSONv2Func]()),
	}, {
		name: jsontest.Name("Methods/Invalid/MapKey/JSONv1/Syntax"),
		in: map[any]string{
			addr(marshalJSONv1Func(func() ([]byte, error) {
				return []byte(`null`), nil
			})): "invalid",
		},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(`{`), "")).withPos(`{`, "").withType(0, T[marshalJSONv1Func]()),
	}, {
		name: jsontest.Name("Functions/Bool/V1"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(bool) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   true,
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/Bool/Empty"),
		opts: []Options{WithMarshalers(nil)},
		in:   true,
		want: `true`,
	}, {
		name: jsontest.Name("Functions/NamedBool/V1/NoMatch"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(namedBool) ([]byte, error) {
				return nil, errMustNotCall
			})),
		},
		in:   true,
		want: `true`,
	}, {
		name: jsontest.Name("Functions/NamedBool/V1/Match"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(namedBool) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   namedBool(true),
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/PointerBool/V1/Match"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v *bool) ([]byte, error) {
				_ = *v // must be a non-nil pointer
				return []byte(`"called"`), nil
			})),
		},
		in:   true,
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/Bool/V2"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
				return enc.WriteToken(jsontext.String("called"))
			})),
		},
		in:   true,
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/NamedBool/V2/NoMatch"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v namedBool) error {
				return errMustNotCall
			})),
		},
		in:   true,
		want: `true`,
	}, {
		name: jsontest.Name("Functions/NamedBool/V2/Match"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v namedBool) error {
				return enc.WriteToken(jsontext.String("called"))
			})),
		},
		in:   namedBool(true),
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/PointerBool/V2/Match"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v *bool) error {
				_ = *v // must be a non-nil pointer
				return enc.WriteToken(jsontext.String("called"))
			})),
		},
		in:   true,
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/Bool/Empty1/NoMatch"),
		opts: []Options{
			WithMarshalers(new(Marshalers)),
		},
		in:   true,
		want: `true`,
	}, {
		name: jsontest.Name("Functions/Bool/Empty2/NoMatch"),
		opts: []Options{
			WithMarshalers(JoinMarshalers()),
		},
		in:   true,
		want: `true`,
	}, {
		name: jsontest.Name("Functions/Bool/V1/DirectError"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(bool) ([]byte, error) {
				return nil, errSomeError
			})),
		},
		in:      true,
		wantErr: EM(errSomeError).withType(0, T[bool]()),
	}, {
		name: jsontest.Name("Functions/Bool/V1/SkipError"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(bool) ([]byte, error) {
				return nil, SkipFunc
			})),
		},
		in:      true,
		wantErr: EM(wrapSkipFunc(SkipFunc, "marshal function of type func(T) ([]byte, error)")).withType(0, T[bool]()),
	}, {
		name: jsontest.Name("Functions/Bool/V1/InvalidValue"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(bool) ([]byte, error) {
				return []byte("invalid"), nil
			})),
		},
		in:      true,
		wantErr: EM(newInvalidCharacterError("i", "at start of value", 0, "")).withType(0, T[bool]()),
	}, {
		name: jsontest.Name("Functions/Bool/V2/DirectError"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
				return errSomeError
			})),
		},
		in:      true,
		wantErr: EM(errSomeError).withType(0, T[bool]()),
	}, {
		name: jsontest.Name("Functions/Bool/V2/TooFew"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
				return nil
			})),
		},
		in:      true,
		wantErr: EM(errNonSingularValue).withType(0, T[bool]()),
	}, {
		name: jsontest.Name("Functions/Bool/V2/TooMany"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
				enc.WriteValue([]byte(`"hello"`))
				enc.WriteValue([]byte(`"world"`))
				return nil
			})),
		},
		in:      true,
		want:    `"hello""world"`,
		wantErr: EM(errNonSingularValue).withPos(`"hello""world"`, "").withType(0, T[bool]()),
	}, {
		name: jsontest.Name("Functions/Bool/V2/Skipped"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
				return SkipFunc
			})),
		},
		in:   true,
		want: `true`,
	}, {
		name: jsontest.Name("Functions/Bool/V2/ProcessBeforeSkip"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
				enc.WriteValue([]byte(`"hello"`))
				return SkipFunc
			})),
		},
		in:      true,
		want:    `"hello"`,
		wantErr: EM(errSkipMutation).withPos(`"hello"`, "").withType(0, T[bool]()),
	}, {
		name: jsontest.Name("Functions/Bool/V2/WrappedSkipError"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
				return fmt.Errorf("wrap: %w", SkipFunc)
			})),
		},
		in:      true,
		wantErr: EM(fmt.Errorf("wrap: %w", SkipFunc)).withType(0, T[bool]()),
	}, {
		name: jsontest.Name("Functions/Map/Key/NoCaseString/V1"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v nocaseString) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   map[nocaseString]string{"hello": "world"},
		want: `{"called":"world"}`,
	}, {
		name: jsontest.Name("Functions/Map/Key/PointerNoCaseString/V1"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v *nocaseString) ([]byte, error) {
				_ = *v // must be a non-nil pointer
				return []byte(`"called"`), nil
			})),
		},
		in:   map[nocaseString]string{"hello": "world"},
		want: `{"called":"world"}`,
	}, {
		name: jsontest.Name("Functions/Map/Key/TextMarshaler/V1"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v encoding.TextMarshaler) ([]byte, error) {
				_ = *v.(*nocaseString) // must be a non-nil *nocaseString
				return []byte(`"called"`), nil
			})),
		},
		in:   map[nocaseString]string{"hello": "world"},
		want: `{"called":"world"}`,
	}, {
		name: jsontest.Name("Functions/Map/Key/NoCaseString/V1/InvalidValue"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v nocaseString) ([]byte, error) {
				return []byte(`null`), nil
			})),
		},
		in:      map[nocaseString]string{"hello": "world"},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(`{`), "")).withPos(`{`, "").withType(0, T[nocaseString]()),
	}, {
		name: jsontest.Name("Functions/Map/Key/NoCaseString/V2/InvalidKind"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v nocaseString) ([]byte, error) {
				return []byte(`null`), nil
			})),
		},
		in:      map[nocaseString]string{"hello": "world"},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(`{`), "")).withPos(`{`, "").withType(0, T[nocaseString]()),
	}, {
		name: jsontest.Name("Functions/Map/Key/String/V1/DuplicateName"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v string) ([]byte, error) {
				return []byte(`"name"`), nil
			})),
		},
		in:   map[string]string{"name1": "value", "name2": "value"},
		want: `{"name":"name"`,
		wantErr: EM(newDuplicateNameError("", []byte(`"name"`), len64(`{"name":"name",`))).
			withPos(`{"name":"name",`, "").withType(0, T[string]()),
	}, {
		name: jsontest.Name("Functions/Map/Key/NoCaseString/V2"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v nocaseString) error {
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   map[nocaseString]string{"hello": "world"},
		want: `{"called":"world"}`,
	}, {
		name: jsontest.Name("Functions/Map/Key/PointerNoCaseString/V2"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v *nocaseString) error {
				_ = *v // must be a non-nil pointer
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   map[nocaseString]string{"hello": "world"},
		want: `{"called":"world"}`,
	}, {
		name: jsontest.Name("Functions/Map/Key/TextMarshaler/V2"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v encoding.TextMarshaler) error {
				_ = *v.(*nocaseString) // must be a non-nil *nocaseString
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   map[nocaseString]string{"hello": "world"},
		want: `{"called":"world"}`,
	}, {
		name: jsontest.Name("Functions/Map/Key/NoCaseString/V2/InvalidToken"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v nocaseString) error {
				return enc.WriteToken(jsontext.Null)
			})),
		},
		in:      map[nocaseString]string{"hello": "world"},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(`{`), "")).withPos(`{`, "").withType(0, T[nocaseString]()),
	}, {
		name: jsontest.Name("Functions/Map/Key/NoCaseString/V2/InvalidValue"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v nocaseString) error {
				return enc.WriteValue([]byte(`null`))
			})),
		},
		in:      map[nocaseString]string{"hello": "world"},
		want:    `{`,
		wantErr: EM(newNonStringNameError(len64(`{`), "")).withPos(`{`, "").withType(0, T[nocaseString]()),
	}, {
		name: jsontest.Name("Functions/Map/Value/NoCaseString/V1"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v nocaseString) ([]byte, error) {
				return []byte(`"called"`), nil
			})),
		},
		in:   map[string]nocaseString{"hello": "world"},
		want: `{"hello":"called"}`,
	}, {
		name: jsontest.Name("Functions/Map/Value/PointerNoCaseString/V1"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v *nocaseString) ([]byte, error) {
				_ = *v // must be a non-nil pointer
				return []byte(`"called"`), nil
			})),
		},
		in:   map[string]nocaseString{"hello": "world"},
		want: `{"hello":"called"}`,
	}, {
		name: jsontest.Name("Functions/Map/Value/TextMarshaler/V1"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v encoding.TextMarshaler) ([]byte, error) {
				_ = *v.(*nocaseString) // must be a non-nil *nocaseString
				return []byte(`"called"`), nil
			})),
		},
		in:   map[string]nocaseString{"hello": "world"},
		want: `{"hello":"called"}`,
	}, {
		name: jsontest.Name("Functions/Map/Value/NoCaseString/V2"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v nocaseString) error {
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   map[string]nocaseString{"hello": "world"},
		want: `{"hello":"called"}`,
	}, {
		name: jsontest.Name("Functions/Map/Value/PointerNoCaseString/V2"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v *nocaseString) error {
				_ = *v // must be a non-nil pointer
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   map[string]nocaseString{"hello": "world"},
		want: `{"hello":"called"}`,
	}, {
		name: jsontest.Name("Functions/Map/Value/TextMarshaler/V2"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v encoding.TextMarshaler) error {
				_ = *v.(*nocaseString) // must be a non-nil *nocaseString
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   map[string]nocaseString{"hello": "world"},
		want: `{"hello":"called"}`,
	}, {
		name: jsontest.Name("Funtions/Struct/Fields"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				MarshalFunc(func(v bool) ([]byte, error) {
					return []byte(`"called1"`), nil
				}),
				MarshalFunc(func(v *string) ([]byte, error) {
					return []byte(`"called2"`), nil
				}),
				MarshalToFunc(func(enc *jsontext.Encoder, v []byte) error {
					return enc.WriteValue([]byte(`"called3"`))
				}),
				MarshalToFunc(func(enc *jsontext.Encoder, v *int64) error {
					return enc.WriteValue([]byte(`"called4"`))
				}),
			)),
		},
		in:   structScalars{},
		want: `{"Bool":"called1","String":"called2","Bytes":"called3","Int":"called4","Uint":0,"Float":0}`,
	}, {
		name: jsontest.Name("Functions/Struct/OmitEmpty"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				MarshalFunc(func(v bool) ([]byte, error) {
					return []byte(`null`), nil
				}),
				MarshalFunc(func(v string) ([]byte, error) {
					return []byte(`"called1"`), nil
				}),
				MarshalFunc(func(v *stringMarshalNonEmpty) ([]byte, error) {
					return []byte(`""`), nil
				}),
				MarshalToFunc(func(enc *jsontext.Encoder, v bytesMarshalNonEmpty) error {
					return enc.WriteValue([]byte(`{}`))
				}),
				MarshalToFunc(func(enc *jsontext.Encoder, v *float64) error {
					return enc.WriteValue([]byte(`[]`))
				}),
				MarshalFunc(func(v mapMarshalNonEmpty) ([]byte, error) {
					return []byte(`"called2"`), nil
				}),
				MarshalFunc(func(v []string) ([]byte, error) {
					return []byte(`"called3"`), nil
				}),
				MarshalToFunc(func(enc *jsontext.Encoder, v *sliceMarshalNonEmpty) error {
					return enc.WriteValue([]byte(`"called4"`))
				}),
			)),
		},
		in:   structOmitEmptyAll{},
		want: `{"String":"called1","MapNonEmpty":"called2","Slice":"called3","SliceNonEmpty":"called4"}`,
	}, {
		name: jsontest.Name("Functions/Struct/OmitZero"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				MarshalFunc(func(v bool) ([]byte, error) {
					panic("should not be called")
				}),
				MarshalFunc(func(v *string) ([]byte, error) {
					panic("should not be called")
				}),
				MarshalToFunc(func(enc *jsontext.Encoder, v []byte) error {
					panic("should not be called")
				}),
				MarshalToFunc(func(enc *jsontext.Encoder, v *int64) error {
					panic("should not be called")
				}),
			)),
		},
		in:   structOmitZeroAll{},
		want: `{}`,
	}, {
		name: jsontest.Name("Functions/Struct/Inlined"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				MarshalFunc(func(v structInlinedL1) ([]byte, error) {
					panic("should not be called")
				}),
				MarshalToFunc(func(enc *jsontext.Encoder, v *StructEmbed2) error {
					panic("should not be called")
				}),
			)),
		},
		in:   structInlined{},
		want: `{"D":""}`,
	}, {
		name: jsontest.Name("Functions/Slice/Elem"),
		opts: []Options{
			WithMarshalers(MarshalFunc(func(v bool) ([]byte, error) {
				return []byte(`"` + strconv.FormatBool(v) + `"`), nil
			})),
		},
		in:   []bool{true, false},
		want: `["true","false"]`,
	}, {
		name: jsontest.Name("Functions/Array/Elem"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v *bool) error {
				return enc.WriteValue([]byte(`"` + strconv.FormatBool(*v) + `"`))
			})),
		},
		in:   [2]bool{true, false},
		want: `["true","false"]`,
	}, {
		name: jsontest.Name("Functions/Pointer/Nil"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v *bool) error {
				panic("should not be called")
			})),
		},
		in:   struct{ X *bool }{nil},
		want: `{"X":null}`,
	}, {
		name: jsontest.Name("Functions/Pointer/NonNil"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v *bool) error {
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   struct{ X *bool }{addr(false)},
		want: `{"X":"called"}`,
	}, {
		name: jsontest.Name("Functions/Interface/Nil"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v fmt.Stringer) error {
				panic("should not be called")
			})),
		},
		in:   struct{ X fmt.Stringer }{nil},
		want: `{"X":null}`,
	}, {
		name: jsontest.Name("Functions/Interface/NonNil/MatchInterface"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v fmt.Stringer) error {
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   struct{ X fmt.Stringer }{valueStringer{}},
		want: `{"X":"called"}`,
	}, {
		name: jsontest.Name("Functions/Interface/NonNil/MatchConcrete"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v valueStringer) error {
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   struct{ X fmt.Stringer }{valueStringer{}},
		want: `{"X":"called"}`,
	}, {
		name: jsontest.Name("Functions/Interface/NonNil/MatchPointer"),
		opts: []Options{
			WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, v *valueStringer) error {
				return enc.WriteValue([]byte(`"called"`))
			})),
		},
		in:   struct{ X fmt.Stringer }{valueStringer{}},
		want: `{"X":"called"}`,
	}, {
		name: jsontest.Name("Functions/Interface/Any"),
		in: []any{
			nil,                           // nil
			valueStringer{},               // T
			(*valueStringer)(nil),         // *T
			addr(valueStringer{}),         // *T
			(**valueStringer)(nil),        // **T
			addr((*valueStringer)(nil)),   // **T
			addr(addr(valueStringer{})),   // **T
			pointerStringer{},             // T
			(*pointerStringer)(nil),       // *T
			addr(pointerStringer{}),       // *T
			(**pointerStringer)(nil),      // **T
			addr((*pointerStringer)(nil)), // **T
			addr(addr(pointerStringer{})), // **T
			"LAST",
		},
		want: `[null,{},null,{},null,null,{},{},null,{},null,null,{},"LAST"]`,
		opts: []Options{
			WithMarshalers(func() *Marshalers {
				type P struct {
					D int
					N int64
				}
				type PV struct {
					P P
					V any
				}

				var lastChecks []func() error
				checkLast := func() error {
					for _, fn := range lastChecks {
						if err := fn(); err != nil {
							return err
						}
					}
					return SkipFunc
				}
				makeValueChecker := func(name string, want []PV) func(e *jsontext.Encoder, v any) error {
					checkNext := func(e *jsontext.Encoder, v any) error {
						xe := export.Encoder(e)
						p := P{len(xe.Tokens.Stack), xe.Tokens.Last.Length()}
						rv := reflect.ValueOf(v)
						pv := PV{p, v}
						switch {
						case len(want) == 0:
							return fmt.Errorf("%s: %v: got more values than expected", name, p)
						case !rv.IsValid() || rv.Kind() != reflect.Pointer || rv.IsNil():
							return fmt.Errorf("%s: %v: got %#v, want non-nil pointer type", name, p, v)
						case !reflect.DeepEqual(pv, want[0]):
							return fmt.Errorf("%s:\n\tgot  %#v\n\twant %#v", name, pv, want[0])
						default:
							want = want[1:]
							return SkipFunc
						}
					}
					lastChecks = append(lastChecks, func() error {
						if len(want) > 0 {
							return fmt.Errorf("%s: did not get enough values, want %d more", name, len(want))
						}
						return nil
					})
					return checkNext
				}
				makePositionChecker := func(name string, want []P) func(e *jsontext.Encoder, v any) error {
					checkNext := func(e *jsontext.Encoder, v any) error {
						xe := export.Encoder(e)
						p := P{len(xe.Tokens.Stack), xe.Tokens.Last.Length()}
						switch {
						case len(want) == 0:
							return fmt.Errorf("%s: %v: got more values than wanted", name, p)
						case p != want[0]:
							return fmt.Errorf("%s: got %v, want %v", name, p, want[0])
						default:
							want = want[1:]
							return SkipFunc
						}
					}
					lastChecks = append(lastChecks, func() error {
						if len(want) > 0 {
							return fmt.Errorf("%s: did not get enough values, want %d more", name, len(want))
						}
						return nil
					})
					return checkNext
				}

				wantAny := []PV{
					{P{0, 0}, addr([]any{
						nil,
						valueStringer{},
						(*valueStringer)(nil),
						addr(valueStringer{}),
						(**valueStringer)(nil),
						addr((*valueStringer)(nil)),
						addr(addr(valueStringer{})),
						pointerStringer{},
						(*pointerStringer)(nil),
						addr(pointerStringer{}),
						(**pointerStringer)(nil),
						addr((*pointerStringer)(nil)),
						addr(addr(pointerStringer{})),
						"LAST",
					})},
					{P{1, 0}, addr(any(nil))},
					{P{1, 1}, addr(any(valueStringer{}))},
					{P{1, 1}, addr(valueStringer{})},
					{P{1, 2}, addr(any((*valueStringer)(nil)))},
					{P{1, 2}, addr((*valueStringer)(nil))},
					{P{1, 3}, addr(any(addr(valueStringer{})))},
					{P{1, 3}, addr(addr(valueStringer{}))},
					{P{1, 3}, addr(valueStringer{})},
					{P{1, 4}, addr(any((**valueStringer)(nil)))},
					{P{1, 4}, addr((**valueStringer)(nil))},
					{P{1, 5}, addr(any(addr((*valueStringer)(nil))))},
					{P{1, 5}, addr(addr((*valueStringer)(nil)))},
					{P{1, 5}, addr((*valueStringer)(nil))},
					{P{1, 6}, addr(any(addr(addr(valueStringer{}))))},
					{P{1, 6}, addr(addr(addr(valueStringer{})))},
					{P{1, 6}, addr(addr(valueStringer{}))},
					{P{1, 6}, addr(valueStringer{})},
					{P{1, 7}, addr(any(pointerStringer{}))},
					{P{1, 7}, addr(pointerStringer{})},
					{P{1, 8}, addr(any((*pointerStringer)(nil)))},
					{P{1, 8}, addr((*pointerStringer)(nil))},
					{P{1, 9}, addr(any(addr(pointerStringer{})))},
					{P{1, 9}, addr(addr(pointerStringer{}))},
					{P{1, 9}, addr(pointerStringer{})},
					{P{1, 10}, addr(any((**pointerStringer)(nil)))},
					{P{1, 10}, addr((**pointerStringer)(nil))},
					{P{1, 11}, addr(any(addr((*pointerStringer)(nil))))},
					{P{1, 11}, addr(addr((*pointerStringer)(nil)))},
					{P{1, 11}, addr((*pointerStringer)(nil))},
					{P{1, 12}, addr(any(addr(addr(pointerStringer{}))))},
					{P{1, 12}, addr(addr(addr(pointerStringer{})))},
					{P{1, 12}, addr(addr(pointerStringer{}))},
					{P{1, 12}, addr(pointerStringer{})},
					{P{1, 13}, addr(any("LAST"))},
					{P{1, 13}, addr("LAST")},
				}
				checkAny := makeValueChecker("any", wantAny)
				anyMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v any) error {
					return checkAny(enc, v)
				})

				var wantPointerAny []PV
				for _, v := range wantAny {
					if _, ok := v.V.(*any); ok {
						wantPointerAny = append(wantPointerAny, v)
					}
				}
				checkPointerAny := makeValueChecker("*any", wantPointerAny)
				pointerAnyMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v *any) error {
					return checkPointerAny(enc, v)
				})

				checkNamedAny := makeValueChecker("namedAny", wantAny)
				namedAnyMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v namedAny) error {
					return checkNamedAny(enc, v)
				})

				checkPointerNamedAny := makeValueChecker("*namedAny", nil)
				pointerNamedAnyMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v *namedAny) error {
					return checkPointerNamedAny(enc, v)
				})

				type stringer = fmt.Stringer
				var wantStringer []PV
				for _, v := range wantAny {
					if _, ok := v.V.(stringer); ok {
						wantStringer = append(wantStringer, v)
					}
				}
				checkStringer := makeValueChecker("stringer", wantStringer)
				stringerMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v stringer) error {
					return checkStringer(enc, v)
				})

				checkPointerStringer := makeValueChecker("*stringer", nil)
				pointerStringerMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v *stringer) error {
					return checkPointerStringer(enc, v)
				})

				wantValueStringer := []P{{1, 1}, {1, 3}, {1, 6}}
				checkValueValueStringer := makePositionChecker("valueStringer", wantValueStringer)
				valueValueStringerMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v valueStringer) error {
					return checkValueValueStringer(enc, v)
				})

				checkPointerValueStringer := makePositionChecker("*valueStringer", wantValueStringer)
				pointerValueStringerMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v *valueStringer) error {
					return checkPointerValueStringer(enc, v)
				})

				wantPointerStringer := []P{{1, 7}, {1, 9}, {1, 12}}
				checkValuePointerStringer := makePositionChecker("pointerStringer", wantPointerStringer)
				valuePointerStringerMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v pointerStringer) error {
					return checkValuePointerStringer(enc, v)
				})

				checkPointerPointerStringer := makePositionChecker("*pointerStringer", wantPointerStringer)
				pointerPointerStringerMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v *pointerStringer) error {
					return checkPointerPointerStringer(enc, v)
				})

				lastMarshaler := MarshalToFunc(func(enc *jsontext.Encoder, v string) error {
					return checkLast()
				})

				return JoinMarshalers(
					anyMarshaler,
					pointerAnyMarshaler,
					namedAnyMarshaler,
					pointerNamedAnyMarshaler, // never called
					stringerMarshaler,
					pointerStringerMarshaler, // never called
					valueValueStringerMarshaler,
					pointerValueStringerMarshaler,
					valuePointerStringerMarshaler,
					pointerPointerStringerMarshaler,
					lastMarshaler,
				)
			}()),
		},
	}, {
		name: jsontest.Name("Functions/Precedence/V1First"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				MarshalFunc(func(bool) ([]byte, error) {
					return []byte(`"called"`), nil
				}),
				MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
					panic("should not be called")
				}),
			)),
		},
		in:   true,
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/Precedence/V2First"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
					return enc.WriteToken(jsontext.String("called"))
				}),
				MarshalFunc(func(bool) ([]byte, error) {
					panic("should not be called")
				}),
			)),
		},
		in:   true,
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/Precedence/V2Skipped"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				MarshalToFunc(func(enc *jsontext.Encoder, v bool) error {
					return SkipFunc
				}),
				MarshalFunc(func(bool) ([]byte, error) {
					return []byte(`"called"`), nil
				}),
			)),
		},
		in:   true,
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/Precedence/NestedFirst"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				JoinMarshalers(
					MarshalFunc(func(bool) ([]byte, error) {
						return []byte(`"called"`), nil
					}),
				),
				MarshalFunc(func(bool) ([]byte, error) {
					panic("should not be called")
				}),
			)),
		},
		in:   true,
		want: `"called"`,
	}, {
		name: jsontest.Name("Functions/Precedence/NestedLast"),
		opts: []Options{
			WithMarshalers(JoinMarshalers(
				MarshalFunc(func(bool) ([]byte, error) {
					return []byte(`"called"`), nil
				}),
				JoinMarshalers(
					MarshalFunc(func(bool) ([]byte, error) {
						panic("should not be called")
					}),
				),
			)),
		},
		in:   true,
		want: `"called"`,
	}, {
		name: jsontest.Name("Duration/Zero"),
		in: struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}{0, 0},
		want: `{"D1":"0s","D2":0}`,
	}, {
		name: jsontest.Name("Duration/Positive"),
		in: struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}{
			123456789123456789,
			123456789123456789,
		},
		want: `{"D1":"34293h33m9.123456789s","D2":123456789123456789}`,
	}, {
		name: jsontest.Name("Duration/Negative"),
		in: struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}{
			-123456789123456789,
			-123456789123456789,
		},
		want: `{"D1":"-34293h33m9.123456789s","D2":-123456789123456789}`,
	}, {
		name: jsontest.Name("Duration/Nanos/String"),
		in: struct {
			D1 time.Duration `json:",string,format:nano"`
			D2 time.Duration `json:",string,format:nano"`
			D3 time.Duration `json:",string,format:nano"`
		}{
			math.MinInt64,
			0,
			math.MaxInt64,
		},
		want: `{"D1":"-9223372036854775808","D2":"0","D3":"9223372036854775807"}`,
	}, {
		name: jsontest.Name("Duration/Format/Invalid"),
		in: struct {
			D time.Duration `json:",format:invalid"`
		}{},
		want:    `{"D"`,
		wantErr: EM(errInvalidFormatFlag).withPos(`{"D":`, "/D").withType(0, T[time.Duration]()),
	}, {
		/* TODO(https://go.dev/issue/71631): Re-enable this test case.
		name: jsontest.Name("Duration/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   time.Duration(0),
		want: `"0s"`,
		}, { */
		name: jsontest.Name("Duration/Format"),
		opts: []Options{jsontext.Multiline(true)},
		in: structDurationFormat{
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
		},
		want: `{
	"D1": "12h34m56.078090012s",
	"D2": "12h34m56.078090012s",
	"D3": 45296.078090012,
	"D4": "45296.078090012",
	"D5": 45296078.090012,
	"D6": "45296078.090012",
	"D7": 45296078090.012,
	"D8": "45296078090.012",
	"D9": 45296078090012,
	"D10": "45296078090012",
	"D11": "PT12H34M56.078090012S"
}`,
	}, {
		/* TODO(https://go.dev/issue/71631): Re-enable this test case.
		name: jsontest.Name("Duration/Format/Legacy"),
		opts: []Options{jsonflags.FormatTimeWithLegacySemantics | 1},
		in: structDurationFormat{
			D1: 12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			D2: 12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
		},
		want: `{"D1":45296078090012,"D2":"12h34m56.078090012s","D3":0,"D4":"0","D5":0,"D6":"0","D7":0,"D8":"0","D9":0,"D10":"0","D11":"PT0S"}`,
		}, { */
		/* TODO(https://go.dev/issue/71631): Re-enable this test case.
		name: jsontest.Name("Duration/MapKey"),
		in:   map[time.Duration]string{time.Second: ""},
		want: `{"1s":""}`,
		}, { */
		name: jsontest.Name("Duration/MapKey/Legacy"),
		opts: []Options{jsonflags.FormatTimeWithLegacySemantics | 1},
		in:   map[time.Duration]string{time.Second: ""},
		want: `{"1000000000":""}`,
	}, {
		name: jsontest.Name("Time/Zero"),
		in: struct {
			T1 time.Time
			T2 time.Time `json:",format:RFC822"`
			T3 time.Time `json:",format:'2006-01-02'"`
			T4 time.Time `json:",omitzero"`
			T5 time.Time `json:",omitempty"`
		}{
			time.Time{},
			time.Time{},
			time.Time{},
			// This is zero according to time.Time.IsZero,
			// but non-zero according to reflect.Value.IsZero.
			time.Date(1, 1, 1, 0, 0, 0, 0, time.FixedZone("UTC", 0)),
			time.Time{},
		},
		want: `{"T1":"0001-01-01T00:00:00Z","T2":"01 Jan 01 00:00 UTC","T3":"0001-01-01","T5":"0001-01-01T00:00:00Z"}`,
	}, {
		name: jsontest.Name("Time/Format"),
		opts: []Options{jsontext.Multiline(true)},
		in: structTimeFormat{
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
			time.Date(1234, 1, 2, 3, 4, 5, 6, time.UTC),
		},
		want: `{
	"T1": "1234-01-02T03:04:05.000000006Z",
	"T2": "Mon Jan  2 03:04:05 1234",
	"T3": "Mon Jan  2 03:04:05 UTC 1234",
	"T4": "Mon Jan 02 03:04:05 +0000 1234",
	"T5": "02 Jan 34 03:04 UTC",
	"T6": "02 Jan 34 03:04 +0000",
	"T7": "Monday, 02-Jan-34 03:04:05 UTC",
	"T8": "Mon, 02 Jan 1234 03:04:05 UTC",
	"T9": "Mon, 02 Jan 1234 03:04:05 +0000",
	"T10": "1234-01-02T03:04:05Z",
	"T11": "1234-01-02T03:04:05.000000006Z",
	"T12": "3:04AM",
	"T13": "Jan  2 03:04:05",
	"T14": "Jan  2 03:04:05.000",
	"T15": "Jan  2 03:04:05.000000",
	"T16": "Jan  2 03:04:05.000000006",
	"T17": "1234-01-02 03:04:05",
	"T18": "1234-01-02",
	"T19": "03:04:05",
	"T20": "1234-01-02",
	"T21": "\"weird\"1234",
	"T22": -23225777754.999999994,
	"T23": "-23225777754.999999994",
	"T24": -23225777754999.999994,
	"T25": "-23225777754999.999994",
	"T26": -23225777754999999.994,
	"T27": "-23225777754999999.994",
	"T28": -23225777754999999994,
	"T29": "-23225777754999999994"
}`,
	}, {
		name: jsontest.Name("Time/Format/Invalid"),
		in: struct {
			T time.Time `json:",format:UndefinedConstant"`
		}{},
		want:    `{"T"`,
		wantErr: EM(errors.New(`invalid format flag "UndefinedConstant"`)).withPos(`{"T":`, "/T").withType(0, timeTimeType),
	}, {
		name: jsontest.Name("Time/Format/YearOverflow"),
		in: struct {
			T1 time.Time
			T2 time.Time
		}{
			time.Date(10000, 1, 1, 0, 0, 0, 0, time.UTC).Add(-time.Second),
			time.Date(10000, 1, 1, 0, 0, 0, 0, time.UTC),
		},
		want:    `{"T1":"9999-12-31T23:59:59Z","T2"`,
		wantErr: EM(errors.New(`year outside of range [0,9999]`)).withPos(`{"T1":"9999-12-31T23:59:59Z","T2":`, "/T2").withType(0, timeTimeType),
	}, {
		name: jsontest.Name("Time/Format/YearUnderflow"),
		in: struct {
			T1 time.Time
			T2 time.Time
		}{
			time.Date(0, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(0, 1, 1, 0, 0, 0, 0, time.UTC).Add(-time.Second),
		},
		want:    `{"T1":"0000-01-01T00:00:00Z","T2"`,
		wantErr: EM(errors.New(`year outside of range [0,9999]`)).withPos(`{"T1":"0000-01-01T00:00:00Z","T2":`, "/T2").withType(0, timeTimeType),
	}, {
		name:    jsontest.Name("Time/Format/YearUnderflow"),
		in:      struct{ T time.Time }{time.Date(-998, 1, 1, 0, 0, 0, 0, time.UTC).Add(-time.Second)},
		want:    `{"T"`,
		wantErr: EM(errors.New(`year outside of range [0,9999]`)).withPos(`{"T":`, "/T").withType(0, timeTimeType),
	}, {
		name: jsontest.Name("Time/Format/ZoneExact"),
		in:   struct{ T time.Time }{time.Date(2020, 1, 1, 0, 0, 0, 0, time.FixedZone("", 23*60*60+59*60))},
		want: `{"T":"2020-01-01T00:00:00+23:59"}`,
	}, {
		name:    jsontest.Name("Time/Format/ZoneHourOverflow"),
		in:      struct{ T time.Time }{time.Date(2020, 1, 1, 0, 0, 0, 0, time.FixedZone("", 24*60*60))},
		want:    `{"T"`,
		wantErr: EM(errors.New(`timezone hour outside of range [0,23]`)).withPos(`{"T":`, "/T").withType(0, timeTimeType),
	}, {
		name:    jsontest.Name("Time/Format/ZoneHourOverflow"),
		in:      struct{ T time.Time }{time.Date(2020, 1, 1, 0, 0, 0, 0, time.FixedZone("", 123*60*60))},
		want:    `{"T"`,
		wantErr: EM(errors.New(`timezone hour outside of range [0,23]`)).withPos(`{"T":`, "/T").withType(0, timeTimeType),
	}, {
		name: jsontest.Name("Time/IgnoreInvalidFormat"),
		opts: []Options{invalidFormatOption},
		in:   time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC),
		want: `"2000-01-01T00:00:00Z"`,
	}}

	for _, tt := range tests {
		t.Run(tt.name.Name, func(t *testing.T) {
			var got []byte
			var gotErr error
			if tt.useWriter {
				bb := new(struct{ bytes.Buffer }) // avoid optimizations with bytes.Buffer
				gotErr = MarshalWrite(bb, tt.in, tt.opts...)
				got = bb.Bytes()
			} else {
				got, gotErr = Marshal(tt.in, tt.opts...)
			}
			if tt.canonicalize {
				(*jsontext.Value)(&got).Canonicalize()
			}
			if string(got) != tt.want {
				t.Errorf("%s: Marshal output mismatch:\ngot  %s\nwant %s", tt.name.Where, got, tt.want)
			}
			if !reflect.DeepEqual(gotErr, tt.wantErr) {
				t.Errorf("%s: Marshal error mismatch:\ngot  %v\nwant %v", tt.name.Where, gotErr, tt.wantErr)
			}
		})
	}
}

func TestUnmarshal(t *testing.T) {
	tests := []struct {
		name    jsontest.CaseName
		opts    []Options
		inBuf   string
		inVal   any
		want    any
		wantErr error
	}{{
		name:    jsontest.Name("Nil"),
		inBuf:   `null`,
		wantErr: EU(internal.ErrNonNilReference),
	}, {
		name:    jsontest.Name("NilPointer"),
		inBuf:   `null`,
		inVal:   (*string)(nil),
		want:    (*string)(nil),
		wantErr: EU(internal.ErrNonNilReference).withType(0, T[*string]()),
	}, {
		name:    jsontest.Name("NonPointer"),
		inBuf:   `null`,
		inVal:   "unchanged",
		want:    "unchanged",
		wantErr: EU(internal.ErrNonNilReference).withType(0, T[string]()),
	}, {
		name:    jsontest.Name("Bools/TrailingJunk"),
		inBuf:   `falsetrue`,
		inVal:   addr(true),
		want:    addr(false),
		wantErr: newInvalidCharacterError("t", "after top-level value", len64(`false`), ""),
	}, {
		name:  jsontest.Name("Bools/Null"),
		inBuf: `null`,
		inVal: addr(true),
		want:  addr(false),
	}, {
		name:  jsontest.Name("Bools"),
		inBuf: `[null,false,true]`,
		inVal: new([]bool),
		want:  addr([]bool{false, false, true}),
	}, {
		name:  jsontest.Name("Bools/Named"),
		inBuf: `[null,false,true]`,
		inVal: new([]namedBool),
		want:  addr([]namedBool{false, false, true}),
	}, {
		name:    jsontest.Name("Bools/Invalid/StringifiedFalse"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"false"`,
		inVal:   addr(true),
		want:    addr(true),
		wantErr: EU(nil).withType('"', boolType),
	}, {
		name:    jsontest.Name("Bools/Invalid/StringifiedTrue"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"true"`,
		inVal:   addr(true),
		want:    addr(true),
		wantErr: EU(nil).withType('"', boolType),
	}, {
		name:  jsontest.Name("Bools/StringifiedBool/True"),
		opts:  []Options{jsonflags.StringifyBoolsAndStrings | 1},
		inBuf: `"true"`,
		inVal: addr(false),
		want:  addr(true),
	}, {
		name:  jsontest.Name("Bools/StringifiedBool/False"),
		opts:  []Options{jsonflags.StringifyBoolsAndStrings | 1},
		inBuf: `"false"`,
		inVal: addr(true),
		want:  addr(false),
	}, {
		name:    jsontest.Name("Bools/StringifiedBool/InvalidWhitespace"),
		opts:    []Options{jsonflags.StringifyBoolsAndStrings | 1},
		inBuf:   `"false "`,
		inVal:   addr(true),
		want:    addr(true),
		wantErr: EU(strconv.ErrSyntax).withVal(`"false "`).withType('"', boolType),
	}, {
		name:    jsontest.Name("Bools/StringifiedBool/InvalidBool"),
		opts:    []Options{jsonflags.StringifyBoolsAndStrings | 1},
		inBuf:   `false`,
		inVal:   addr(true),
		want:    addr(true),
		wantErr: EU(nil).withType('f', boolType),
	}, {
		name:    jsontest.Name("Bools/Invalid/Number"),
		inBuf:   `0`,
		inVal:   addr(true),
		want:    addr(true),
		wantErr: EU(nil).withType('0', boolType),
	}, {
		name:    jsontest.Name("Bools/Invalid/String"),
		inBuf:   `""`,
		inVal:   addr(true),
		want:    addr(true),
		wantErr: EU(nil).withType('"', boolType),
	}, {
		name:    jsontest.Name("Bools/Invalid/Object"),
		inBuf:   `{}`,
		inVal:   addr(true),
		want:    addr(true),
		wantErr: EU(nil).withType('{', boolType),
	}, {
		name:    jsontest.Name("Bools/Invalid/Array"),
		inBuf:   `[]`,
		inVal:   addr(true),
		want:    addr(true),
		wantErr: EU(nil).withType('[', boolType),
	}, {
		name:  jsontest.Name("Bools/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `false`,
		inVal: addr(true),
		want:  addr(false),
	}, {
		name:  jsontest.Name("Strings/Null"),
		inBuf: `null`,
		inVal: addr("something"),
		want:  addr(""),
	}, {
		name:  jsontest.Name("Strings"),
		inBuf: `[null,"","hello","世界"]`,
		inVal: new([]string),
		want:  addr([]string{"", "", "hello", "世界"}),
	}, {
		name:  jsontest.Name("Strings/Escaped"),
		inBuf: `[null,"","\u0068\u0065\u006c\u006c\u006f","\u4e16\u754c"]`,
		inVal: new([]string),
		want:  addr([]string{"", "", "hello", "世界"}),
	}, {
		name:  jsontest.Name("Strings/Named"),
		inBuf: `[null,"","hello","世界"]`,
		inVal: new([]namedString),
		want:  addr([]namedString{"", "", "hello", "世界"}),
	}, {
		name:    jsontest.Name("Strings/Invalid/False"),
		inBuf:   `false`,
		inVal:   addr("nochange"),
		want:    addr("nochange"),
		wantErr: EU(nil).withType('f', stringType),
	}, {
		name:    jsontest.Name("Strings/Invalid/True"),
		inBuf:   `true`,
		inVal:   addr("nochange"),
		want:    addr("nochange"),
		wantErr: EU(nil).withType('t', stringType),
	}, {
		name:    jsontest.Name("Strings/Invalid/Object"),
		inBuf:   `{}`,
		inVal:   addr("nochange"),
		want:    addr("nochange"),
		wantErr: EU(nil).withType('{', stringType),
	}, {
		name:    jsontest.Name("Strings/Invalid/Array"),
		inBuf:   `[]`,
		inVal:   addr("nochange"),
		want:    addr("nochange"),
		wantErr: EU(nil).withType('[', stringType),
	}, {
		name:  jsontest.Name("Strings/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `"hello"`,
		inVal: addr("goodbye"),
		want:  addr("hello"),
	}, {
		name:  jsontest.Name("Strings/StringifiedString"),
		opts:  []Options{jsonflags.StringifyBoolsAndStrings | 1},
		inBuf: `"\"foo\""`,
		inVal: new(string),
		want:  addr("foo"),
	}, {
		name:    jsontest.Name("Strings/StringifiedString/InvalidWhitespace"),
		opts:    []Options{jsonflags.StringifyBoolsAndStrings | 1},
		inBuf:   `"\"foo\" "`,
		inVal:   new(string),
		want:    new(string),
		wantErr: EU(newInvalidCharacterError(" ", "after string value", 0, "")).withType('"', stringType),
	}, {
		name:    jsontest.Name("Strings/StringifiedString/InvalidString"),
		opts:    []Options{jsonflags.StringifyBoolsAndStrings | 1},
		inBuf:   `""`,
		inVal:   new(string),
		want:    new(string),
		wantErr: EU(&jsontext.SyntacticError{Err: io.ErrUnexpectedEOF}).withType('"', stringType),
	}, {
		name:  jsontest.Name("Bytes/Null"),
		inBuf: `null`,
		inVal: addr([]byte("something")),
		want:  addr([]byte(nil)),
	}, {
		name:  jsontest.Name("Bytes"),
		inBuf: `[null,"","AQ==","AQI=","AQID"]`,
		inVal: new([][]byte),
		want:  addr([][]byte{nil, {}, {1}, {1, 2}, {1, 2, 3}}),
	}, {
		name:  jsontest.Name("Bytes/Large"),
		inBuf: `"dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wZWQgb3ZlciB0aGUgbGF6eSBkb2cgYW5kIGF0ZSB0aGUgaG9tZXdvcmsgdGhhdCBJIHNwZW50IHNvIG11Y2ggdGltZSBvbi4="`,
		inVal: new([]byte),
		want:  addr([]byte("the quick brown fox jumped over the lazy dog and ate the homework that I spent so much time on.")),
	}, {
		name:  jsontest.Name("Bytes/Reuse"),
		inBuf: `"AQID"`,
		inVal: addr([]byte("changed")),
		want:  addr([]byte{1, 2, 3}),
	}, {
		name:  jsontest.Name("Bytes/Escaped"),
		inBuf: `[null,"","\u0041\u0051\u003d\u003d","\u0041\u0051\u0049\u003d","\u0041\u0051\u0049\u0044"]`,
		inVal: new([][]byte),
		want:  addr([][]byte{nil, {}, {1}, {1, 2}, {1, 2, 3}}),
	}, {
		name:  jsontest.Name("Bytes/Named"),
		inBuf: `[null,"","AQ==","AQI=","AQID"]`,
		inVal: new([]namedBytes),
		want:  addr([]namedBytes{nil, {}, {1}, {1, 2}, {1, 2, 3}}),
	}, {
		name:  jsontest.Name("Bytes/NotStringified"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `[null,"","AQ==","AQI=","AQID"]`,
		inVal: new([][]byte),
		want:  addr([][]byte{nil, {}, {1}, {1, 2}, {1, 2, 3}}),
	}, {
		// NOTE: []namedByte is not assignable to []byte,
		// so the following should be treated as a slice of uints.
		name:  jsontest.Name("Bytes/Invariant"),
		inBuf: `[null,[],[1],[1,2],[1,2,3]]`,
		inVal: new([][]namedByte),
		want:  addr([][]namedByte{nil, {}, {1}, {1, 2}, {1, 2, 3}}),
	}, {
		// NOTE: This differs in behavior from v1,
		// but keeps the representation of slices and arrays more consistent.
		name:  jsontest.Name("Bytes/ByteArray"),
		inBuf: `"aGVsbG8="`,
		inVal: new([5]byte),
		want:  addr([5]byte{'h', 'e', 'l', 'l', 'o'}),
	}, {
		name:  jsontest.Name("Bytes/ByteArray0/Valid"),
		inBuf: `""`,
		inVal: new([0]byte),
		want:  addr([0]byte{}),
	}, {
		name:  jsontest.Name("Bytes/ByteArray0/Invalid"),
		inBuf: `"A"`,
		inVal: new([0]byte),
		want:  addr([0]byte{}),
		wantErr: EU(func() error {
			_, err := base64.StdEncoding.Decode(make([]byte, 0), []byte("A"))
			return err
		}()).withType('"', T[[0]byte]()),
	}, {
		name:    jsontest.Name("Bytes/ByteArray0/Overflow"),
		inBuf:   `"AA=="`,
		inVal:   new([0]byte),
		want:    addr([0]byte{}),
		wantErr: EU(errors.New("decoded length of 1 mismatches array length of 0")).withType('"', T[[0]byte]()),
	}, {
		name:  jsontest.Name("Bytes/ByteArray1/Valid"),
		inBuf: `"AQ=="`,
		inVal: new([1]byte),
		want:  addr([1]byte{1}),
	}, {
		name:  jsontest.Name("Bytes/ByteArray1/Invalid"),
		inBuf: `"$$=="`,
		inVal: new([1]byte),
		want:  addr([1]byte{}),
		wantErr: EU(func() error {
			_, err := base64.StdEncoding.Decode(make([]byte, 1), []byte("$$=="))
			return err
		}()).withType('"', T[[1]byte]()),
	}, {
		name:    jsontest.Name("Bytes/ByteArray1/Underflow"),
		inBuf:   `""`,
		inVal:   new([1]byte),
		want:    addr([1]byte{}),
		wantErr: EU(errors.New("decoded length of 0 mismatches array length of 1")).withType('"', T[[1]byte]()),
	}, {
		name:    jsontest.Name("Bytes/ByteArray1/Overflow"),
		inBuf:   `"AQI="`,
		inVal:   new([1]byte),
		want:    addr([1]byte{1}),
		wantErr: EU(errors.New("decoded length of 2 mismatches array length of 1")).withType('"', T[[1]byte]()),
	}, {
		name:  jsontest.Name("Bytes/ByteArray2/Valid"),
		inBuf: `"AQI="`,
		inVal: new([2]byte),
		want:  addr([2]byte{1, 2}),
	}, {
		name:  jsontest.Name("Bytes/ByteArray2/Invalid"),
		inBuf: `"$$$="`,
		inVal: new([2]byte),
		want:  addr([2]byte{}),
		wantErr: EU(func() error {
			_, err := base64.StdEncoding.Decode(make([]byte, 2), []byte("$$$="))
			return err
		}()).withType('"', T[[2]byte]()),
	}, {
		name:    jsontest.Name("Bytes/ByteArray2/Underflow"),
		inBuf:   `"AQ=="`,
		inVal:   new([2]byte),
		want:    addr([2]byte{1, 0}),
		wantErr: EU(errors.New("decoded length of 1 mismatches array length of 2")).withType('"', T[[2]byte]()),
	}, {
		name:  jsontest.Name("Bytes/ByteArray2/Underflow/Allowed"),
		opts:  []Options{jsonflags.UnmarshalArrayFromAnyLength | 1},
		inBuf: `"AQ=="`,
		inVal: new([2]byte),
		want:  addr([2]byte{1, 0}),
	}, {
		name:    jsontest.Name("Bytes/ByteArray2/Overflow"),
		inBuf:   `"AQID"`,
		inVal:   new([2]byte),
		want:    addr([2]byte{1, 2}),
		wantErr: EU(errors.New("decoded length of 3 mismatches array length of 2")).withType('"', T[[2]byte]()),
	}, {
		name:  jsontest.Name("Bytes/ByteArray2/Overflow/Allowed"),
		opts:  []Options{jsonflags.UnmarshalArrayFromAnyLength | 1},
		inBuf: `"AQID"`,
		inVal: new([2]byte),
		want:  addr([2]byte{1, 2}),
	}, {
		name:  jsontest.Name("Bytes/ByteArray3/Valid"),
		inBuf: `"AQID"`,
		inVal: new([3]byte),
		want:  addr([3]byte{1, 2, 3}),
	}, {
		name:  jsontest.Name("Bytes/ByteArray3/Invalid"),
		inBuf: `"$$$$"`,
		inVal: new([3]byte),
		want:  addr([3]byte{}),
		wantErr: EU(func() error {
			_, err := base64.StdEncoding.Decode(make([]byte, 3), []byte("$$$$"))
			return err
		}()).withType('"', T[[3]byte]()),
	}, {
		name:    jsontest.Name("Bytes/ByteArray3/Underflow"),
		inBuf:   `"AQI="`,
		inVal:   addr([3]byte{0xff, 0xff, 0xff}),
		want:    addr([3]byte{1, 2, 0}),
		wantErr: EU(errors.New("decoded length of 2 mismatches array length of 3")).withType('"', T[[3]byte]()),
	}, {
		name:    jsontest.Name("Bytes/ByteArray3/Overflow"),
		inBuf:   `"AQIDAQ=="`,
		inVal:   new([3]byte),
		want:    addr([3]byte{1, 2, 3}),
		wantErr: EU(errors.New("decoded length of 4 mismatches array length of 3")).withType('"', T[[3]byte]()),
	}, {
		name:  jsontest.Name("Bytes/ByteArray4/Valid"),
		inBuf: `"AQIDBA=="`,
		inVal: new([4]byte),
		want:  addr([4]byte{1, 2, 3, 4}),
	}, {
		name:  jsontest.Name("Bytes/ByteArray4/Invalid"),
		inBuf: `"$$$$$$=="`,
		inVal: new([4]byte),
		want:  addr([4]byte{}),
		wantErr: EU(func() error {
			_, err := base64.StdEncoding.Decode(make([]byte, 4), []byte("$$$$$$=="))
			return err
		}()).withType('"', T[[4]byte]()),
	}, {
		name:    jsontest.Name("Bytes/ByteArray4/Underflow"),
		inBuf:   `"AQID"`,
		inVal:   new([4]byte),
		want:    addr([4]byte{1, 2, 3, 0}),
		wantErr: EU(errors.New("decoded length of 3 mismatches array length of 4")).withType('"', T[[4]byte]()),
	}, {
		name:    jsontest.Name("Bytes/ByteArray4/Overflow"),
		inBuf:   `"AQIDBAU="`,
		inVal:   new([4]byte),
		want:    addr([4]byte{1, 2, 3, 4}),
		wantErr: EU(errors.New("decoded length of 5 mismatches array length of 4")).withType('"', T[[4]byte]()),
	}, {
		// NOTE: []namedByte is not assignable to []byte,
		// so the following should be treated as a array of uints.
		name:  jsontest.Name("Bytes/NamedByteArray"),
		inBuf: `[104,101,108,108,111]`,
		inVal: new([5]namedByte),
		want:  addr([5]namedByte{'h', 'e', 'l', 'l', 'o'}),
	}, {
		name:  jsontest.Name("Bytes/Valid/Denormalized"),
		inBuf: `"AR=="`,
		inVal: new([]byte),
		want:  addr([]byte{1}),
	}, {
		name:  jsontest.Name("Bytes/Invalid/Unpadded1"),
		inBuf: `"AQ="`,
		inVal: addr([]byte("nochange")),
		want:  addr([]byte("nochange")),
		wantErr: EU(func() error {
			_, err := base64.StdEncoding.Decode(make([]byte, 0), []byte("AQ="))
			return err
		}()).withType('"', bytesType),
	}, {
		name:  jsontest.Name("Bytes/Invalid/Unpadded2"),
		inBuf: `"AQ"`,
		inVal: addr([]byte("nochange")),
		want:  addr([]byte("nochange")),
		wantErr: EU(func() error {
			_, err := base64.StdEncoding.Decode(make([]byte, 0), []byte("AQ"))
			return err
		}()).withType('"', bytesType),
	}, {
		name:  jsontest.Name("Bytes/Invalid/Character"),
		inBuf: `"@@@@"`,
		inVal: addr([]byte("nochange")),
		want:  addr([]byte("nochange")),
		wantErr: EU(func() error {
			_, err := base64.StdEncoding.Decode(make([]byte, 3), []byte("@@@@"))
			return err
		}()).withType('"', bytesType),
	}, {
		name:    jsontest.Name("Bytes/Invalid/Bool"),
		inBuf:   `true`,
		inVal:   addr([]byte("nochange")),
		want:    addr([]byte("nochange")),
		wantErr: EU(nil).withType('t', bytesType),
	}, {
		name:    jsontest.Name("Bytes/Invalid/Number"),
		inBuf:   `0`,
		inVal:   addr([]byte("nochange")),
		want:    addr([]byte("nochange")),
		wantErr: EU(nil).withType('0', bytesType),
	}, {
		name:    jsontest.Name("Bytes/Invalid/Object"),
		inBuf:   `{}`,
		inVal:   addr([]byte("nochange")),
		want:    addr([]byte("nochange")),
		wantErr: EU(nil).withType('{', bytesType),
	}, {
		name:    jsontest.Name("Bytes/Invalid/Array"),
		inBuf:   `[]`,
		inVal:   addr([]byte("nochange")),
		want:    addr([]byte("nochange")),
		wantErr: EU(nil).withType('[', bytesType),
	}, {
		name:  jsontest.Name("Bytes/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `"aGVsbG8="`,
		inVal: new([]byte),
		want:  addr([]byte("hello")),
	}, {
		name:  jsontest.Name("Ints/Null"),
		inBuf: `null`,
		inVal: addr(int(1)),
		want:  addr(int(0)),
	}, {
		name:  jsontest.Name("Ints/Int"),
		inBuf: `1`,
		inVal: addr(int(0)),
		want:  addr(int(1)),
	}, {
		name:    jsontest.Name("Ints/Int8/MinOverflow"),
		inBuf:   `-129`,
		inVal:   addr(int8(-1)),
		want:    addr(int8(-1)),
		wantErr: EU(strconv.ErrRange).withVal(`-129`).withType('0', T[int8]()),
	}, {
		name:  jsontest.Name("Ints/Int8/Min"),
		inBuf: `-128`,
		inVal: addr(int8(0)),
		want:  addr(int8(-128)),
	}, {
		name:  jsontest.Name("Ints/Int8/Max"),
		inBuf: `127`,
		inVal: addr(int8(0)),
		want:  addr(int8(127)),
	}, {
		name:    jsontest.Name("Ints/Int8/MaxOverflow"),
		inBuf:   `128`,
		inVal:   addr(int8(-1)),
		want:    addr(int8(-1)),
		wantErr: EU(strconv.ErrRange).withVal(`128`).withType('0', T[int8]()),
	}, {
		name:    jsontest.Name("Ints/Int16/MinOverflow"),
		inBuf:   `-32769`,
		inVal:   addr(int16(-1)),
		want:    addr(int16(-1)),
		wantErr: EU(strconv.ErrRange).withVal(`-32769`).withType('0', T[int16]()),
	}, {
		name:  jsontest.Name("Ints/Int16/Min"),
		inBuf: `-32768`,
		inVal: addr(int16(0)),
		want:  addr(int16(-32768)),
	}, {
		name:  jsontest.Name("Ints/Int16/Max"),
		inBuf: `32767`,
		inVal: addr(int16(0)),
		want:  addr(int16(32767)),
	}, {
		name:    jsontest.Name("Ints/Int16/MaxOverflow"),
		inBuf:   `32768`,
		inVal:   addr(int16(-1)),
		want:    addr(int16(-1)),
		wantErr: EU(strconv.ErrRange).withVal(`32768`).withType('0', T[int16]()),
	}, {
		name:    jsontest.Name("Ints/Int32/MinOverflow"),
		inBuf:   `-2147483649`,
		inVal:   addr(int32(-1)),
		want:    addr(int32(-1)),
		wantErr: EU(strconv.ErrRange).withVal(`-2147483649`).withType('0', T[int32]()),
	}, {
		name:  jsontest.Name("Ints/Int32/Min"),
		inBuf: `-2147483648`,
		inVal: addr(int32(0)),
		want:  addr(int32(-2147483648)),
	}, {
		name:  jsontest.Name("Ints/Int32/Max"),
		inBuf: `2147483647`,
		inVal: addr(int32(0)),
		want:  addr(int32(2147483647)),
	}, {
		name:    jsontest.Name("Ints/Int32/MaxOverflow"),
		inBuf:   `2147483648`,
		inVal:   addr(int32(-1)),
		want:    addr(int32(-1)),
		wantErr: EU(strconv.ErrRange).withVal(`2147483648`).withType('0', T[int32]()),
	}, {
		name:    jsontest.Name("Ints/Int64/MinOverflow"),
		inBuf:   `-9223372036854775809`,
		inVal:   addr(int64(-1)),
		want:    addr(int64(-1)),
		wantErr: EU(strconv.ErrRange).withVal(`-9223372036854775809`).withType('0', T[int64]()),
	}, {
		name:  jsontest.Name("Ints/Int64/Min"),
		inBuf: `-9223372036854775808`,
		inVal: addr(int64(0)),
		want:  addr(int64(-9223372036854775808)),
	}, {
		name:  jsontest.Name("Ints/Int64/Max"),
		inBuf: `9223372036854775807`,
		inVal: addr(int64(0)),
		want:  addr(int64(9223372036854775807)),
	}, {
		name:    jsontest.Name("Ints/Int64/MaxOverflow"),
		inBuf:   `9223372036854775808`,
		inVal:   addr(int64(-1)),
		want:    addr(int64(-1)),
		wantErr: EU(strconv.ErrRange).withVal(`9223372036854775808`).withType('0', T[int64]()),
	}, {
		name:  jsontest.Name("Ints/Named"),
		inBuf: `-6464`,
		inVal: addr(namedInt64(0)),
		want:  addr(namedInt64(-6464)),
	}, {
		name:  jsontest.Name("Ints/Stringified"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `"-6464"`,
		inVal: new(int),
		want:  addr(int(-6464)),
	}, {
		name:    jsontest.Name("Ints/Stringified/Invalid"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `-6464`,
		inVal:   new(int),
		want:    new(int),
		wantErr: EU(nil).withType('0', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Stringified/LeadingZero"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"00"`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"00"`).withType('"', T[int]()),
	}, {
		name:  jsontest.Name("Ints/Escaped"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `"\u002d\u0036\u0034\u0036\u0034"`,
		inVal: new(int),
		want:  addr(int(-6464)),
	}, {
		name:  jsontest.Name("Ints/Valid/NegativeZero"),
		inBuf: `-0`,
		inVal: addr(int(1)),
		want:  addr(int(0)),
	}, {
		name:    jsontest.Name("Ints/Invalid/Fraction"),
		inBuf:   `1.0`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`1.0`).withType('0', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/Exponent"),
		inBuf:   `1e0`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`1e0`).withType('0', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/StringifiedFraction"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"1.0"`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"1.0"`).withType('"', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/StringifiedExponent"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"1e0"`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"1e0"`).withType('"', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/Overflow"),
		inBuf:   `100000000000000000000000000000`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(strconv.ErrRange).withVal(`100000000000000000000000000000`).withType('0', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/OverflowSyntax"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"100000000000000000000000000000x"`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"100000000000000000000000000000x"`).withType('"', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/Whitespace"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"0 "`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"0 "`).withType('"', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/Bool"),
		inBuf:   `true`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(nil).withType('t', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/String"),
		inBuf:   `"0"`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(nil).withType('"', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/Object"),
		inBuf:   `{}`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(nil).withType('{', T[int]()),
	}, {
		name:    jsontest.Name("Ints/Invalid/Array"),
		inBuf:   `[]`,
		inVal:   addr(int(-1)),
		want:    addr(int(-1)),
		wantErr: EU(nil).withType('[', T[int]()),
	}, {
		name:  jsontest.Name("Ints/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `1`,
		inVal: addr(int(0)),
		want:  addr(int(1)),
	}, {
		name:  jsontest.Name("Uints/Null"),
		inBuf: `null`,
		inVal: addr(uint(1)),
		want:  addr(uint(0)),
	}, {
		name:  jsontest.Name("Uints/Uint"),
		inBuf: `1`,
		inVal: addr(uint(0)),
		want:  addr(uint(1)),
	}, {
		name:  jsontest.Name("Uints/Uint8/Min"),
		inBuf: `0`,
		inVal: addr(uint8(1)),
		want:  addr(uint8(0)),
	}, {
		name:  jsontest.Name("Uints/Uint8/Max"),
		inBuf: `255`,
		inVal: addr(uint8(0)),
		want:  addr(uint8(255)),
	}, {
		name:    jsontest.Name("Uints/Uint8/MaxOverflow"),
		inBuf:   `256`,
		inVal:   addr(uint8(1)),
		want:    addr(uint8(1)),
		wantErr: EU(strconv.ErrRange).withVal(`256`).withType('0', T[uint8]()),
	}, {
		name:  jsontest.Name("Uints/Uint16/Min"),
		inBuf: `0`,
		inVal: addr(uint16(1)),
		want:  addr(uint16(0)),
	}, {
		name:  jsontest.Name("Uints/Uint16/Max"),
		inBuf: `65535`,
		inVal: addr(uint16(0)),
		want:  addr(uint16(65535)),
	}, {
		name:    jsontest.Name("Uints/Uint16/MaxOverflow"),
		inBuf:   `65536`,
		inVal:   addr(uint16(1)),
		want:    addr(uint16(1)),
		wantErr: EU(strconv.ErrRange).withVal(`65536`).withType('0', T[uint16]()),
	}, {
		name:  jsontest.Name("Uints/Uint32/Min"),
		inBuf: `0`,
		inVal: addr(uint32(1)),
		want:  addr(uint32(0)),
	}, {
		name:  jsontest.Name("Uints/Uint32/Max"),
		inBuf: `4294967295`,
		inVal: addr(uint32(0)),
		want:  addr(uint32(4294967295)),
	}, {
		name:    jsontest.Name("Uints/Uint32/MaxOverflow"),
		inBuf:   `4294967296`,
		inVal:   addr(uint32(1)),
		want:    addr(uint32(1)),
		wantErr: EU(strconv.ErrRange).withVal(`4294967296`).withType('0', T[uint32]()),
	}, {
		name:  jsontest.Name("Uints/Uint64/Min"),
		inBuf: `0`,
		inVal: addr(uint64(1)),
		want:  addr(uint64(0)),
	}, {
		name:  jsontest.Name("Uints/Uint64/Max"),
		inBuf: `18446744073709551615`,
		inVal: addr(uint64(0)),
		want:  addr(uint64(18446744073709551615)),
	}, {
		name:    jsontest.Name("Uints/Uint64/MaxOverflow"),
		inBuf:   `18446744073709551616`,
		inVal:   addr(uint64(1)),
		want:    addr(uint64(1)),
		wantErr: EU(strconv.ErrRange).withVal(`18446744073709551616`).withType('0', T[uint64]()),
	}, {
		name:  jsontest.Name("Uints/Uintptr"),
		inBuf: `1`,
		inVal: addr(uintptr(0)),
		want:  addr(uintptr(1)),
	}, {
		name:  jsontest.Name("Uints/Named"),
		inBuf: `6464`,
		inVal: addr(namedUint64(0)),
		want:  addr(namedUint64(6464)),
	}, {
		name:  jsontest.Name("Uints/Stringified"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `"6464"`,
		inVal: new(uint),
		want:  addr(uint(6464)),
	}, {
		name:    jsontest.Name("Uints/Stringified/Invalid"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `6464`,
		inVal:   new(uint),
		want:    new(uint),
		wantErr: EU(nil).withType('0', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Stringified/LeadingZero"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"00"`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"00"`).withType('"', T[uint]()),
	}, {
		name:  jsontest.Name("Uints/Escaped"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `"\u0036\u0034\u0036\u0034"`,
		inVal: new(uint),
		want:  addr(uint(6464)),
	}, {
		name:    jsontest.Name("Uints/Invalid/NegativeOne"),
		inBuf:   `-1`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`-1`).withType('0', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/NegativeZero"),
		inBuf:   `-0`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`-0`).withType('0', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/Fraction"),
		inBuf:   `1.0`,
		inVal:   addr(uint(10)),
		want:    addr(uint(10)),
		wantErr: EU(strconv.ErrSyntax).withVal(`1.0`).withType('0', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/Exponent"),
		inBuf:   `1e0`,
		inVal:   addr(uint(10)),
		want:    addr(uint(10)),
		wantErr: EU(strconv.ErrSyntax).withVal(`1e0`).withType('0', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/StringifiedFraction"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"1.0"`,
		inVal:   addr(uint(10)),
		want:    addr(uint(10)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"1.0"`).withType('"', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/StringifiedExponent"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"1e0"`,
		inVal:   addr(uint(10)),
		want:    addr(uint(10)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"1e0"`).withType('"', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/Overflow"),
		inBuf:   `100000000000000000000000000000`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(strconv.ErrRange).withVal(`100000000000000000000000000000`).withType('0', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/OverflowSyntax"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"100000000000000000000000000000x"`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"100000000000000000000000000000x"`).withType('"', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/Whitespace"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"0 "`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"0 "`).withType('"', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/Bool"),
		inBuf:   `true`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(nil).withType('t', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/String"),
		inBuf:   `"0"`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(nil).withType('"', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/Object"),
		inBuf:   `{}`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(nil).withType('{', T[uint]()),
	}, {
		name:    jsontest.Name("Uints/Invalid/Array"),
		inBuf:   `[]`,
		inVal:   addr(uint(1)),
		want:    addr(uint(1)),
		wantErr: EU(nil).withType('[', T[uint]()),
	}, {
		name:  jsontest.Name("Uints/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `1`,
		inVal: addr(uint(0)),
		want:  addr(uint(1)),
	}, {
		name:  jsontest.Name("Floats/Null"),
		inBuf: `null`,
		inVal: addr(float64(64.64)),
		want:  addr(float64(0)),
	}, {
		name:  jsontest.Name("Floats/Float32/Pi"),
		inBuf: `3.14159265358979323846264338327950288419716939937510582097494459`,
		inVal: addr(float32(32.32)),
		want:  addr(float32(math.Pi)),
	}, {
		name:  jsontest.Name("Floats/Float32/Underflow"),
		inBuf: `1e-1000`,
		inVal: addr(float32(32.32)),
		want:  addr(float32(0)),
	}, {
		name:    jsontest.Name("Floats/Float32/Overflow"),
		inBuf:   `-1e1000`,
		inVal:   addr(float32(32.32)),
		want:    addr(float32(-math.MaxFloat32)),
		wantErr: EU(strconv.ErrRange).withVal(`-1e1000`).withType('0', T[float32]()),
	}, {
		name:  jsontest.Name("Floats/Float64/Pi"),
		inBuf: `3.14159265358979323846264338327950288419716939937510582097494459`,
		inVal: addr(float64(64.64)),
		want:  addr(float64(math.Pi)),
	}, {
		name:  jsontest.Name("Floats/Float64/Underflow"),
		inBuf: `1e-1000`,
		inVal: addr(float64(64.64)),
		want:  addr(float64(0)),
	}, {
		name:    jsontest.Name("Floats/Float64/Overflow"),
		inBuf:   `-1e1000`,
		inVal:   addr(float64(64.64)),
		want:    addr(float64(-math.MaxFloat64)),
		wantErr: EU(strconv.ErrRange).withVal(`-1e1000`).withType('0', T[float64]()),
	}, {
		name:    jsontest.Name("Floats/Any/Overflow"),
		inBuf:   `1e1000`,
		inVal:   new(any),
		want:    addr(any(float64(math.MaxFloat64))),
		wantErr: EU(strconv.ErrRange).withVal(`1e1000`).withType('0', T[float64]()),
	}, {
		name:  jsontest.Name("Floats/Named"),
		inBuf: `64.64`,
		inVal: addr(namedFloat64(0)),
		want:  addr(namedFloat64(64.64)),
	}, {
		name:  jsontest.Name("Floats/Stringified"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `"64.64"`,
		inVal: new(float64),
		want:  addr(float64(64.64)),
	}, {
		name:    jsontest.Name("Floats/Stringified/Invalid"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `64.64`,
		inVal:   new(float64),
		want:    new(float64),
		wantErr: EU(nil).withType('0', T[float64]()),
	}, {
		name:  jsontest.Name("Floats/Escaped"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `"\u0036\u0034\u002e\u0036\u0034"`,
		inVal: new(float64),
		want:  addr(float64(64.64)),
	}, {
		name:    jsontest.Name("Floats/Invalid/NaN"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"NaN"`,
		inVal:   addr(float64(64.64)),
		want:    addr(float64(64.64)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"NaN"`).withType('"', float64Type),
	}, {
		name:    jsontest.Name("Floats/Invalid/Infinity"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"Infinity"`,
		inVal:   addr(float64(64.64)),
		want:    addr(float64(64.64)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"Infinity"`).withType('"', float64Type),
	}, {
		name:    jsontest.Name("Floats/Invalid/Whitespace"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"1 "`,
		inVal:   addr(float64(64.64)),
		want:    addr(float64(64.64)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"1 "`).withType('"', float64Type),
	}, {
		name:    jsontest.Name("Floats/Invalid/GoSyntax"),
		opts:    []Options{StringifyNumbers(true)},
		inBuf:   `"1p-2"`,
		inVal:   addr(float64(64.64)),
		want:    addr(float64(64.64)),
		wantErr: EU(strconv.ErrSyntax).withVal(`"1p-2"`).withType('"', float64Type),
	}, {
		name:    jsontest.Name("Floats/Invalid/Bool"),
		inBuf:   `true`,
		inVal:   addr(float64(64.64)),
		want:    addr(float64(64.64)),
		wantErr: EU(nil).withType('t', float64Type),
	}, {
		name:    jsontest.Name("Floats/Invalid/String"),
		inBuf:   `"0"`,
		inVal:   addr(float64(64.64)),
		want:    addr(float64(64.64)),
		wantErr: EU(nil).withType('"', float64Type),
	}, {
		name:    jsontest.Name("Floats/Invalid/Object"),
		inBuf:   `{}`,
		inVal:   addr(float64(64.64)),
		want:    addr(float64(64.64)),
		wantErr: EU(nil).withType('{', float64Type),
	}, {
		name:    jsontest.Name("Floats/Invalid/Array"),
		inBuf:   `[]`,
		inVal:   addr(float64(64.64)),
		want:    addr(float64(64.64)),
		wantErr: EU(nil).withType('[', float64Type),
	}, {
		name:  jsontest.Name("Floats/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `1`,
		inVal: addr(float64(0)),
		want:  addr(float64(1)),
	}, {
		name:  jsontest.Name("Maps/Null"),
		inBuf: `null`,
		inVal: addr(map[string]string{"key": "value"}),
		want:  new(map[string]string),
	}, {
		name:    jsontest.Name("Maps/InvalidKey/Bool"),
		inBuf:   `{"true":"false"}`,
		inVal:   new(map[bool]bool),
		want:    addr(make(map[bool]bool)),
		wantErr: EU(nil).withPos(`{`, "/true").withType('"', boolType),
	}, {
		name:    jsontest.Name("Maps/InvalidKey/NamedBool"),
		inBuf:   `{"true":"false"}`,
		inVal:   new(map[namedBool]bool),
		want:    addr(make(map[namedBool]bool)),
		wantErr: EU(nil).withPos(`{`, "/true").withType('"', T[namedBool]()),
	}, {
		name:    jsontest.Name("Maps/InvalidKey/Array"),
		inBuf:   `{"key":"value"}`,
		inVal:   new(map[[1]string]string),
		want:    addr(make(map[[1]string]string)),
		wantErr: EU(nil).withPos(`{`, "/key").withType('"', T[[1]string]()),
	}, {
		name:    jsontest.Name("Maps/InvalidKey/Channel"),
		inBuf:   `{"key":"value"}`,
		inVal:   new(map[chan string]string),
		want:    addr(make(map[chan string]string)),
		wantErr: EU(nil).withPos(`{`, "").withType(0, T[chan string]()),
	}, {
		name:  jsontest.Name("Maps/ValidKey/Int"),
		inBuf: `{"0":0,"-1":1,"2":2,"-3":3}`,
		inVal: new(map[int]int),
		want:  addr(map[int]int{0: 0, -1: 1, 2: 2, -3: 3}),
	}, {
		name:  jsontest.Name("Maps/ValidKey/NamedInt"),
		inBuf: `{"0":0,"-1":1,"2":2,"-3":3}`,
		inVal: new(map[namedInt64]int),
		want:  addr(map[namedInt64]int{0: 0, -1: 1, 2: 2, -3: 3}),
	}, {
		name:  jsontest.Name("Maps/ValidKey/Uint"),
		inBuf: `{"0":0,"1":1,"2":2,"3":3}`,
		inVal: new(map[uint]uint),
		want:  addr(map[uint]uint{0: 0, 1: 1, 2: 2, 3: 3}),
	}, {
		name:  jsontest.Name("Maps/ValidKey/NamedUint"),
		inBuf: `{"0":0,"1":1,"2":2,"3":3}`,
		inVal: new(map[namedUint64]uint),
		want:  addr(map[namedUint64]uint{0: 0, 1: 1, 2: 2, 3: 3}),
	}, {
		name:  jsontest.Name("Maps/ValidKey/Float"),
		inBuf: `{"1.234":1.234,"12.34":12.34,"123.4":123.4}`,
		inVal: new(map[float64]float64),
		want:  addr(map[float64]float64{1.234: 1.234, 12.34: 12.34, 123.4: 123.4}),
	}, {
		name:    jsontest.Name("Maps/DuplicateName/Int"),
		inBuf:   `{"0":1,"-0":-1}`,
		inVal:   new(map[int]int),
		want:    addr(map[int]int{0: 1}),
		wantErr: newDuplicateNameError("", []byte(`"-0"`), len64(`{"0":1,`)),
	}, {
		name:    jsontest.Name("Maps/DuplicateName/Int/MergeWithLegacySemantics"),
		opts:    []Options{jsonflags.MergeWithLegacySemantics | 1},
		inBuf:   `{"0":1,"-0":-1}`,
		inVal:   new(map[int]int),
		want:    addr(map[int]int{0: 1}),
		wantErr: newDuplicateNameError("", []byte(`"-0"`), len64(`{"0":1,`)),
	}, {
		name:  jsontest.Name("Maps/DuplicateName/Int/AllowDuplicateNames"),
		opts:  []Options{jsontext.AllowDuplicateNames(true)},
		inBuf: `{"0":1,"-0":-1}`,
		inVal: new(map[int]int),
		want:  addr(map[int]int{0: -1}), // latter takes precedence
	}, {
		name:  jsontest.Name("Maps/DuplicateName/Int/OverwriteExisting"),
		inBuf: `{"-0":-1}`,
		inVal: addr(map[int]int{0: 1}),
		want:  addr(map[int]int{0: -1}),
	}, {
		name:    jsontest.Name("Maps/DuplicateName/Float"),
		inBuf:   `{"1.0":"1.0","1":"1","1e0":"1e0"}`,
		inVal:   new(map[float64]string),
		want:    addr(map[float64]string{1: "1.0"}),
		wantErr: newDuplicateNameError("", []byte(`"1"`), len64(`{"1.0":"1.0",`)),
	}, {
		name:  jsontest.Name("Maps/DuplicateName/Float/AllowDuplicateNames"),
		opts:  []Options{jsontext.AllowDuplicateNames(true)},
		inBuf: `{"1.0":"1.0","1":"1","1e0":"1e0"}`,
		inVal: new(map[float64]string),
		want:  addr(map[float64]string{1: "1e0"}), // latter takes precedence
	}, {
		name:  jsontest.Name("Maps/DuplicateName/Float/OverwriteExisting"),
		inBuf: `{"1.0":"1.0"}`,
		inVal: addr(map[float64]string{1: "1"}),
		want:  addr(map[float64]string{1: "1.0"}),
	}, {
		name:    jsontest.Name("Maps/DuplicateName/NoCaseString"),
		inBuf:   `{"hello":"hello","HELLO":"HELLO"}`,
		inVal:   new(map[nocaseString]string),
		want:    addr(map[nocaseString]string{"hello": "hello"}),
		wantErr: newDuplicateNameError("", []byte(`"HELLO"`), len64(`{"hello":"hello",`)),
	}, {
		name:  jsontest.Name("Maps/DuplicateName/NoCaseString/AllowDuplicateNames"),
		opts:  []Options{jsontext.AllowDuplicateNames(true)},
		inBuf: `{"hello":"hello","HELLO":"HELLO"}`,
		inVal: new(map[nocaseString]string),
		want:  addr(map[nocaseString]string{"hello": "HELLO"}), // latter takes precedence
	}, {
		name:  jsontest.Name("Maps/DuplicateName/NoCaseString/OverwriteExisting"),
		opts:  []Options{jsontext.AllowDuplicateNames(true)},
		inBuf: `{"HELLO":"HELLO"}`,
		inVal: addr(map[nocaseString]string{"hello": "hello"}),
		want:  addr(map[nocaseString]string{"hello": "HELLO"}),
	}, {
		name:  jsontest.Name("Maps/ValidKey/Interface"),
		inBuf: `{"false":"false","true":"true","string":"string","0":"0","[]":"[]","{}":"{}"}`,
		inVal: new(map[any]string),
		want: addr(map[any]string{
			"false":  "false",
			"true":   "true",
			"string": "string",
			"0":      "0",
			"[]":     "[]",
			"{}":     "{}",
		}),
	}, {
		name:  jsontest.Name("Maps/InvalidValue/Channel"),
		inBuf: `{"key":"value"}`,
		inVal: new(map[string]chan string),
		want: addr(map[string]chan string{
			"key": nil,
		}),
		wantErr: EU(nil).withPos(`{"key":`, "/key").withType(0, T[chan string]()),
	}, {
		name:  jsontest.Name("Maps/RecursiveMap"),
		inBuf: `{"buzz":{},"fizz":{"bar":{},"foo":{}}}`,
		inVal: new(recursiveMap),
		want: addr(recursiveMap{
			"fizz": {
				"foo": {},
				"bar": {},
			},
			"buzz": {},
		}),
	}, {
		// NOTE: The semantics differs from v1,
		// where existing map entries were not merged into.
		// See https://go.dev/issue/31924.
		name:  jsontest.Name("Maps/Merge"),
		opts:  []Options{jsontext.AllowDuplicateNames(true)},
		inBuf: `{"k1":{"k2":"v2"},"k2":{"k1":"v1"},"k2":{"k2":"v2"}}`,
		inVal: addr(map[string]map[string]string{
			"k1": {"k1": "v1"},
		}),
		want: addr(map[string]map[string]string{
			"k1": {"k1": "v1", "k2": "v2"},
			"k2": {"k1": "v1", "k2": "v2"},
		}),
	}, {
		name:    jsontest.Name("Maps/Invalid/Bool"),
		inBuf:   `true`,
		inVal:   addr(map[string]string{"key": "value"}),
		want:    addr(map[string]string{"key": "value"}),
		wantErr: EU(nil).withType('t', T[map[string]string]()),
	}, {
		name:    jsontest.Name("Maps/Invalid/String"),
		inBuf:   `""`,
		inVal:   addr(map[string]string{"key": "value"}),
		want:    addr(map[string]string{"key": "value"}),
		wantErr: EU(nil).withType('"', T[map[string]string]()),
	}, {
		name:    jsontest.Name("Maps/Invalid/Number"),
		inBuf:   `0`,
		inVal:   addr(map[string]string{"key": "value"}),
		want:    addr(map[string]string{"key": "value"}),
		wantErr: EU(nil).withType('0', T[map[string]string]()),
	}, {
		name:    jsontest.Name("Maps/Invalid/Array"),
		inBuf:   `[]`,
		inVal:   addr(map[string]string{"key": "value"}),
		want:    addr(map[string]string{"key": "value"}),
		wantErr: EU(nil).withType('[', T[map[string]string]()),
	}, {
		name:  jsontest.Name("Maps/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `{"hello":"goodbye"}`,
		inVal: addr(map[string]string{}),
		want:  addr(map[string]string{"hello": "goodbye"}),
	}, {
		name:  jsontest.Name("Structs/Null"),
		inBuf: `null`,
		inVal: addr(structAll{String: "something"}),
		want:  addr(structAll{}),
	}, {
		name:  jsontest.Name("Structs/Empty"),
		inBuf: `{}`,
		inVal: addr(structAll{
			String: "hello",
			Map:    map[string]string{},
			Slice:  []string{},
		}),
		want: addr(structAll{
			String: "hello",
			Map:    map[string]string{},
			Slice:  []string{},
		}),
	}, {
		name: jsontest.Name("Structs/Normal"),
		inBuf: `{
	"Bool": true,
	"String": "hello",
	"Bytes": "AQID",
	"Int": -64,
	"Uint": 64,
	"Float": 3.14159,
	"Map": {"key": "value"},
	"StructScalars": {
		"Bool": true,
		"String": "hello",
		"Bytes": "AQID",
		"Int": -64,
		"Uint": 64,
		"Float": 3.14159
	},
	"StructMaps": {
		"MapBool": {"": true},
		"MapString": {"": "hello"},
		"MapBytes": {"": "AQID"},
		"MapInt": {"": -64},
		"MapUint": {"": 64},
		"MapFloat": {"": 3.14159}
	},
	"StructSlices": {
		"SliceBool": [true],
		"SliceString": ["hello"],
		"SliceBytes": ["AQID"],
		"SliceInt": [-64],
		"SliceUint": [64],
		"SliceFloat": [3.14159]
	},
	"Slice": ["fizz","buzz"],
	"Array": ["goodbye"],
	"Pointer": {},
	"Interface": null
}`,
		inVal: new(structAll),
		want: addr(structAll{
			Bool:   true,
			String: "hello",
			Bytes:  []byte{1, 2, 3},
			Int:    -64,
			Uint:   +64,
			Float:  3.14159,
			Map:    map[string]string{"key": "value"},
			StructScalars: structScalars{
				Bool:   true,
				String: "hello",
				Bytes:  []byte{1, 2, 3},
				Int:    -64,
				Uint:   +64,
				Float:  3.14159,
			},
			StructMaps: structMaps{
				MapBool:   map[string]bool{"": true},
				MapString: map[string]string{"": "hello"},
				MapBytes:  map[string][]byte{"": {1, 2, 3}},
				MapInt:    map[string]int64{"": -64},
				MapUint:   map[string]uint64{"": +64},
				MapFloat:  map[string]float64{"": 3.14159},
			},
			StructSlices: structSlices{
				SliceBool:   []bool{true},
				SliceString: []string{"hello"},
				SliceBytes:  [][]byte{{1, 2, 3}},
				SliceInt:    []int64{-64},
				SliceUint:   []uint64{+64},
				SliceFloat:  []float64{3.14159},
			},
			Slice:   []string{"fizz", "buzz"},
			Array:   [1]string{"goodbye"},
			Pointer: new(structAll),
		}),
	}, {
		name: jsontest.Name("Structs/Merge"),
		inBuf: `{
	"Bool": false,
	"String": "goodbye",
	"Int": -64,
	"Float": 3.14159,
	"Map": {"k2": "v2"},
	"StructScalars": {
		"Bool": true,
		"String": "hello",
		"Bytes": "AQID",
		"Int": -64
	},
	"StructMaps": {
		"MapBool": {"": true},
		"MapString": {"": "hello"},
		"MapBytes": {"": "AQID"},
		"MapInt": {"": -64},
		"MapUint": {"": 64},
		"MapFloat": {"": 3.14159}
	},
	"StructSlices": {
		"SliceString": ["hello"],
		"SliceBytes": ["AQID"],
		"SliceInt": [-64],
		"SliceUint": [64]
	},
	"Slice": ["fizz","buzz"],
	"Array": ["goodbye"],
	"Pointer": {},
	"Interface": {"k2":"v2"}
}`,
		inVal: addr(structAll{
			Bool:   true,
			String: "hello",
			Bytes:  []byte{1, 2, 3},
			Uint:   +64,
			Float:  math.NaN(),
			Map:    map[string]string{"k1": "v1"},
			StructScalars: structScalars{
				String: "hello",
				Bytes:  make([]byte, 2, 4),
				Uint:   +64,
				Float:  3.14159,
			},
			StructMaps: structMaps{
				MapBool:  map[string]bool{"": false},
				MapBytes: map[string][]byte{"": {}},
				MapInt:   map[string]int64{"": 123},
				MapFloat: map[string]float64{"": math.Inf(+1)},
			},
			StructSlices: structSlices{
				SliceBool:  []bool{true},
				SliceBytes: [][]byte{nil, nil},
				SliceInt:   []int64{-123},
				SliceUint:  []uint64{+123},
				SliceFloat: []float64{3.14159},
			},
			Slice:     []string{"buzz", "fizz", "gizz"},
			Array:     [1]string{"hello"},
			Pointer:   new(structAll),
			Interface: map[string]string{"k1": "v1"},
		}),
		want: addr(structAll{
			Bool:   false,
			String: "goodbye",
			Bytes:  []byte{1, 2, 3},
			Int:    -64,
			Uint:   +64,
			Float:  3.14159,
			Map:    map[string]string{"k1": "v1", "k2": "v2"},
			StructScalars: structScalars{
				Bool:   true,
				String: "hello",
				Bytes:  []byte{1, 2, 3},
				Int:    -64,
				Uint:   +64,
				Float:  3.14159,
			},
			StructMaps: structMaps{
				MapBool:   map[string]bool{"": true},
				MapString: map[string]string{"": "hello"},
				MapBytes:  map[string][]byte{"": {1, 2, 3}},
				MapInt:    map[string]int64{"": -64},
				MapUint:   map[string]uint64{"": +64},
				MapFloat:  map[string]float64{"": 3.14159},
			},
			StructSlices: structSlices{
				SliceBool:   []bool{true},
				SliceString: []string{"hello"},
				SliceBytes:  [][]byte{{1, 2, 3}},
				SliceInt:    []int64{-64},
				SliceUint:   []uint64{+64},
				SliceFloat:  []float64{3.14159},
			},
			Slice:     []string{"fizz", "buzz"},
			Array:     [1]string{"goodbye"},
			Pointer:   new(structAll),
			Interface: map[string]string{"k1": "v1", "k2": "v2"},
		}),
	}, {
		name: jsontest.Name("Structs/Stringified/Normal"),
		inBuf: `{
	"Bool": true,
	"String": "hello",
	"Bytes": "AQID",
	"Int": "-64",
	"Uint": "64",
	"Float": "3.14159",
	"Map": {"key": "value"},
	"StructScalars": {
		"Bool": true,
		"String": "hello",
		"Bytes": "AQID",
		"Int": "-64",
		"Uint": "64",
		"Float": "3.14159"
	},
	"StructMaps": {
		"MapBool": {"": true},
		"MapString": {"": "hello"},
		"MapBytes": {"": "AQID"},
		"MapInt": {"": "-64"},
		"MapUint": {"": "64"},
		"MapFloat": {"": "3.14159"}
	},
	"StructSlices": {
		"SliceBool": [true],
		"SliceString": ["hello"],
		"SliceBytes": ["AQID"],
		"SliceInt": ["-64"],
		"SliceUint": ["64"],
		"SliceFloat": ["3.14159"]
	},
	"Slice": ["fizz","buzz"],
	"Array": ["goodbye"],
	"Pointer": {},
	"Interface": null
}`,
		inVal: new(structStringifiedAll),
		want: addr(structStringifiedAll{
			Bool:   true,
			String: "hello",
			Bytes:  []byte{1, 2, 3},
			Int:    -64,     // may be stringified
			Uint:   +64,     // may be stringified
			Float:  3.14159, // may be stringified
			Map:    map[string]string{"key": "value"},
			StructScalars: structScalars{
				Bool:   true,
				String: "hello",
				Bytes:  []byte{1, 2, 3},
				Int:    -64,     // may be stringified
				Uint:   +64,     // may be stringified
				Float:  3.14159, // may be stringified
			},
			StructMaps: structMaps{
				MapBool:   map[string]bool{"": true},
				MapString: map[string]string{"": "hello"},
				MapBytes:  map[string][]byte{"": {1, 2, 3}},
				MapInt:    map[string]int64{"": -64},       // may be stringified
				MapUint:   map[string]uint64{"": +64},      // may be stringified
				MapFloat:  map[string]float64{"": 3.14159}, // may be stringified
			},
			StructSlices: structSlices{
				SliceBool:   []bool{true},
				SliceString: []string{"hello"},
				SliceBytes:  [][]byte{{1, 2, 3}},
				SliceInt:    []int64{-64},       // may be stringified
				SliceUint:   []uint64{+64},      // may be stringified
				SliceFloat:  []float64{3.14159}, // may be stringified
			},
			Slice:   []string{"fizz", "buzz"},
			Array:   [1]string{"goodbye"},
			Pointer: new(structStringifiedAll), // may be stringified
		}),
	}, {
		name: jsontest.Name("Structs/Stringified/String"),
		inBuf: `{
	"Bool": true,
	"String": "hello",
	"Bytes": "AQID",
	"Int": "-64",
	"Uint": "64",
	"Float": "3.14159",
	"Map": {"key": "value"},
	"StructScalars": {
		"Bool": true,
		"String": "hello",
		"Bytes": "AQID",
		"Int": "-64",
		"Uint": "64",
		"Float": "3.14159"
	},
	"StructMaps": {
		"MapBool": {"": true},
		"MapString": {"": "hello"},
		"MapBytes": {"": "AQID"},
		"MapInt": {"": "-64"},
		"MapUint": {"": "64"},
		"MapFloat": {"": "3.14159"}
	},
	"StructSlices": {
		"SliceBool": [true],
		"SliceString": ["hello"],
		"SliceBytes": ["AQID"],
		"SliceInt": ["-64"],
		"SliceUint": ["64"],
		"SliceFloat": ["3.14159"]
	},
	"Slice": ["fizz","buzz"],
	"Array": ["goodbye"],
	"Pointer": {},
	"Interface": null
}`,
		inVal: new(structStringifiedAll),
		want: addr(structStringifiedAll{
			Bool:   true,
			String: "hello",
			Bytes:  []byte{1, 2, 3},
			Int:    -64,     // may be stringified
			Uint:   +64,     // may be stringified
			Float:  3.14159, // may be stringified
			Map:    map[string]string{"key": "value"},
			StructScalars: structScalars{
				Bool:   true,
				String: "hello",
				Bytes:  []byte{1, 2, 3},
				Int:    -64,     // may be stringified
				Uint:   +64,     // may be stringified
				Float:  3.14159, // may be stringified
			},
			StructMaps: structMaps{
				MapBool:   map[string]bool{"": true},
				MapString: map[string]string{"": "hello"},
				MapBytes:  map[string][]byte{"": {1, 2, 3}},
				MapInt:    map[string]int64{"": -64},       // may be stringified
				MapUint:   map[string]uint64{"": +64},      // may be stringified
				MapFloat:  map[string]float64{"": 3.14159}, // may be stringified
			},
			StructSlices: structSlices{
				SliceBool:   []bool{true},
				SliceString: []string{"hello"},
				SliceBytes:  [][]byte{{1, 2, 3}},
				SliceInt:    []int64{-64},       // may be stringified
				SliceUint:   []uint64{+64},      // may be stringified
				SliceFloat:  []float64{3.14159}, // may be stringified
			},
			Slice:   []string{"fizz", "buzz"},
			Array:   [1]string{"goodbye"},
			Pointer: new(structStringifiedAll), // may be stringified
		}),
	}, {
		name:    jsontest.Name("Structs/Stringified/InvalidEmpty"),
		inBuf:   `{"Int":""}`,
		inVal:   new(structStringifiedAll),
		want:    new(structStringifiedAll),
		wantErr: EU(strconv.ErrSyntax).withVal(`""`).withPos(`{"Int":`, "/Int").withType('"', T[int64]()),
	}, {
		name: jsontest.Name("Structs/LegacyStringified"),
		opts: []Options{jsonflags.StringifyWithLegacySemantics | 1},
		inBuf: `{
	"Bool": "true",
	"String": "\"hello\"",
	"Bytes": "AQID",
	"Int": "-64",
	"Uint": "64",
	"Float": "3.14159",
	"Map": {"key": "value"},
	"StructScalars": {
		"Bool": true,
		"String": "hello",
		"Bytes": "AQID",
		"Int": -64,
		"Uint": 64,
		"Float": 3.14159
	},
	"StructMaps": {
		"MapBool": {"": true},
		"MapString": {"": "hello"},
		"MapBytes": {"": "AQID"},
		"MapInt": {"": -64},
		"MapUint": {"": 64},
		"MapFloat": {"": 3.14159}
	},
	"StructSlices": {
		"SliceBool": [true],
		"SliceString": ["hello"],
		"SliceBytes": ["AQID"],
		"SliceInt": [-64],
		"SliceUint": [64],
		"SliceFloat": [3.14159]
	},
	"Slice": ["fizz", "buzz"],
	"Array": ["goodbye"]
}`,
		inVal: new(structStringifiedAll),
		want: addr(structStringifiedAll{
			Bool:   true,
			String: "hello",
			Bytes:  []byte{1, 2, 3},
			Int:    -64,
			Uint:   +64,
			Float:  3.14159,
			Map:    map[string]string{"key": "value"},
			StructScalars: structScalars{
				Bool:   true,
				String: "hello",
				Bytes:  []byte{1, 2, 3},
				Int:    -64,
				Uint:   +64,
				Float:  3.14159,
			},
			StructMaps: structMaps{
				MapBool:   map[string]bool{"": true},
				MapString: map[string]string{"": "hello"},
				MapBytes:  map[string][]byte{"": {1, 2, 3}},
				MapInt:    map[string]int64{"": -64},
				MapUint:   map[string]uint64{"": +64},
				MapFloat:  map[string]float64{"": 3.14159},
			},
			StructSlices: structSlices{
				SliceBool:   []bool{true},
				SliceString: []string{"hello"},
				SliceBytes:  [][]byte{{1, 2, 3}},
				SliceInt:    []int64{-64},
				SliceUint:   []uint64{+64},
				SliceFloat:  []float64{3.14159},
			},
			Slice: []string{"fizz", "buzz"},
			Array: [1]string{"goodbye"},
		}),
	}, {
		name:    jsontest.Name("Structs/LegacyStringified/InvalidBool"),
		opts:    []Options{jsonflags.StringifyWithLegacySemantics | 1},
		inBuf:   `{"Bool": true}`,
		inVal:   new(structStringifiedAll),
		wantErr: EU(nil).withPos(`{"Bool": `, "/Bool").withType('t', T[bool]()),
	}, {
		name:  jsontest.Name("Structs/LegacyStringified/InvalidString"),
		opts:  []Options{jsonflags.StringifyWithLegacySemantics | 1},
		inBuf: `{"String": "string"}`,
		inVal: new(structStringifiedAll),
		wantErr: EU(newInvalidCharacterError("s", "at start of string (expecting '\"')", 0, "")).
			withPos(`{"String": `, "/String").withType('"', T[string]()),
	}, {
		name: jsontest.Name("Structs/Format/Bytes"),
		inBuf: `{
	"Base16": "0123456789abcdef",
	"Base32": "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567",
	"Base32Hex": "0123456789ABCDEFGHIJKLMNOPQRSTUV",
	"Base64": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
	"Base64URL": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
	"Array": [1, 2, 3, 4]
}`,
		inVal: new(structFormatBytes),
		want: addr(structFormatBytes{
			Base16:    []byte("\x01\x23\x45\x67\x89\xab\xcd\xef"),
			Base32:    []byte("\x00D2\x14\xc7BT\xb65τe:V\xd7\xc6u\xbew\xdf"),
			Base32Hex: []byte("\x00D2\x14\xc7BT\xb65τe:V\xd7\xc6u\xbew\xdf"),
			Base64:    []byte("\x00\x10\x83\x10Q\x87 \x92\x8b0ӏA\x14\x93QU\x97a\x96\x9bqן\x82\x18\xa3\x92Y\xa7\xa2\x9a\xab\xb2ۯ\xc3\x1c\xb3\xd3]\xb7㞻\xf3߿"),
			Base64URL: []byte("\x00\x10\x83\x10Q\x87 \x92\x8b0ӏA\x14\x93QU\x97a\x96\x9bqן\x82\x18\xa3\x92Y\xa7\xa2\x9a\xab\xb2ۯ\xc3\x1c\xb3\xd3]\xb7㞻\xf3߿"),
			Array:     []byte{1, 2, 3, 4},
		}),
	}, {
		name: jsontest.Name("Structs/Format/ArrayBytes"),
		inBuf: `{
	"Base16": "01020304",
	"Base32": "AEBAGBA=",
	"Base32Hex": "0410610=",
	"Base64": "AQIDBA==",
	"Base64URL": "AQIDBA==",
	"Array": [1, 2, 3, 4],
	"Default": "AQIDBA=="
}`,
		inVal: new(structFormatArrayBytes),
		want: addr(structFormatArrayBytes{
			Base16:    [4]byte{1, 2, 3, 4},
			Base32:    [4]byte{1, 2, 3, 4},
			Base32Hex: [4]byte{1, 2, 3, 4},
			Base64:    [4]byte{1, 2, 3, 4},
			Base64URL: [4]byte{1, 2, 3, 4},
			Array:     [4]byte{1, 2, 3, 4},
			Default:   [4]byte{1, 2, 3, 4},
		}),
	}, {
		name: jsontest.Name("Structs/Format/ArrayBytes/Legacy"),
		opts: []Options{jsonflags.FormatBytesWithLegacySemantics | 1},
		inBuf: `{
	"Base16": "01020304",
	"Base32": "AEBAGBA=",
	"Base32Hex": "0410610=",
	"Base64": "AQIDBA==",
	"Base64URL": "AQIDBA==",
	"Array": [1, 2, 3, 4],
	"Default": [1, 2, 3, 4]
}`,
		inVal: new(structFormatArrayBytes),
		want: addr(structFormatArrayBytes{
			Base16:    [4]byte{1, 2, 3, 4},
			Base32:    [4]byte{1, 2, 3, 4},
			Base32Hex: [4]byte{1, 2, 3, 4},
			Base64:    [4]byte{1, 2, 3, 4},
			Base64URL: [4]byte{1, 2, 3, 4},
			Array:     [4]byte{1, 2, 3, 4},
			Default:   [4]byte{1, 2, 3, 4},
		}),
	}, {
		name: jsontest.Name("Structs/Format/Bytes/Array"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *byte) error {
				if string(b) == "true" {
					*v = 1
				} else {
					*v = 0
				}
				return nil
			})),
		},
		inBuf: `{"Array":[false,true,false,true,false,true]}`,
		inVal: new(struct {
			Array []byte `json:",format:array"`
		}),
		want: addr(struct {
			Array []byte `json:",format:array"`
		}{
			Array: []byte{0, 1, 0, 1, 0, 1},
		}),
	}, {
		name:    jsontest.Name("Structs/Format/Bytes/Invalid/Base16/WrongKind"),
		inBuf:   `{"Base16": [1,2,3,4]}`,
		inVal:   new(structFormatBytes),
		wantErr: EU(nil).withPos(`{"Base16": `, "/Base16").withType('[', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base16/AllPadding"),
		inBuf: `{"Base16": "===="}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := hex.Decode(make([]byte, 2), []byte("====="))
			return err
		}()).withPos(`{"Base16": `, "/Base16").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base16/EvenPadding"),
		inBuf: `{"Base16": "0123456789abcdef="}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := hex.Decode(make([]byte, 8), []byte("0123456789abcdef="))
			return err
		}()).withPos(`{"Base16": `, "/Base16").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base16/OddPadding"),
		inBuf: `{"Base16": "0123456789abcdef0="}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := hex.Decode(make([]byte, 9), []byte("0123456789abcdef0="))
			return err
		}()).withPos(`{"Base16": `, "/Base16").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base16/NonAlphabet/LineFeed"),
		inBuf: `{"Base16": "aa\naa"}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := hex.Decode(make([]byte, 9), []byte("aa\naa"))
			return err
		}()).withPos(`{"Base16": `, "/Base16").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base16/NonAlphabet/CarriageReturn"),
		inBuf: `{"Base16": "aa\raa"}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := hex.Decode(make([]byte, 9), []byte("aa\raa"))
			return err
		}()).withPos(`{"Base16": `, "/Base16").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base16/NonAlphabet/Space"),
		inBuf: `{"Base16": "aa aa"}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := hex.Decode(make([]byte, 9), []byte("aa aa"))
			return err
		}()).withPos(`{"Base16": `, "/Base16").withType('"', T[[]byte]()),
	}, {
		name: jsontest.Name("Structs/Format/Bytes/Invalid/Base32/Padding"),
		inBuf: `[
			{"Base32": "NA======"},
			{"Base32": "NBSQ===="},
			{"Base32": "NBSWY==="},
			{"Base32": "NBSWY3A="},
			{"Base32": "NBSWY3DP"}
		]`,
		inVal: new([]structFormatBytes),
		want: addr([]structFormatBytes{
			{Base32: []byte("h")},
			{Base32: []byte("he")},
			{Base32: []byte("hel")},
			{Base32: []byte("hell")},
			{Base32: []byte("hello")},
		}),
	}, {
		name: jsontest.Name("Structs/Format/Bytes/Invalid/Base32/Invalid/NoPadding"),
		inBuf: `[
				{"Base32": "NA"},
				{"Base32": "NBSQ"},
				{"Base32": "NBSWY"},
				{"Base32": "NBSWY3A"},
				{"Base32": "NBSWY3DP"}
			]`,
		inVal: new([]structFormatBytes),
		wantErr: EU(func() error {
			_, err := base32.StdEncoding.Decode(make([]byte, 1), []byte("NA"))
			return err
		}()).withPos(`[`+"\n\t\t\t\t"+`{"Base32": `, "/0/Base32").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base32/WrongAlphabet"),
		inBuf: `{"Base32": "0123456789ABCDEFGHIJKLMNOPQRSTUV"}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := base32.StdEncoding.Decode(make([]byte, 20), []byte("0123456789ABCDEFGHIJKLMNOPQRSTUV"))
			return err
		}()).withPos(`{"Base32": `, "/Base32").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base32Hex/WrongAlphabet"),
		inBuf: `{"Base32Hex": "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := base32.HexEncoding.Decode(make([]byte, 20), []byte("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"))
			return err
		}()).withPos(`{"Base32Hex": `, "/Base32Hex").withType('"', T[[]byte]()),
	}, {
		name:    jsontest.Name("Structs/Format/Bytes/Invalid/Base32/NonAlphabet/LineFeed"),
		inBuf:   `{"Base32": "AAAA\nAAAA"}`,
		inVal:   new(structFormatBytes),
		wantErr: EU(errors.New("illegal character '\\n' at offset 4")).withPos(`{"Base32": `, "/Base32").withType('"', T[[]byte]()),
	}, {
		name:    jsontest.Name("Structs/Format/Bytes/Invalid/Base32/NonAlphabet/CarriageReturn"),
		inBuf:   `{"Base32": "AAAA\rAAAA"}`,
		inVal:   new(structFormatBytes),
		wantErr: EU(errors.New("illegal character '\\r' at offset 4")).withPos(`{"Base32": `, "/Base32").withType('"', T[[]byte]()),
	}, {
		name:    jsontest.Name("Structs/Format/Bytes/Invalid/Base32/NonAlphabet/Space"),
		inBuf:   `{"Base32": "AAAA AAAA"}`,
		inVal:   new(structFormatBytes),
		wantErr: EU(base32.CorruptInputError(4)).withPos(`{"Base32": `, "/Base32").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base64/WrongAlphabet"),
		inBuf: `{"Base64": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := base64.StdEncoding.Decode(make([]byte, 48), []byte("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"))
			return err
		}()).withPos(`{"Base64": `, "/Base64").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Invalid/Base64URL/WrongAlphabet"),
		inBuf: `{"Base64URL": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"}`,
		inVal: new(structFormatBytes),
		wantErr: EU(func() error {
			_, err := base64.URLEncoding.Decode(make([]byte, 48), []byte("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"))
			return err
		}()).withPos(`{"Base64URL": `, "/Base64URL").withType('"', T[[]byte]()),
	}, {
		name:    jsontest.Name("Structs/Format/Bytes/Invalid/Base64/NonAlphabet/LineFeed"),
		inBuf:   `{"Base64": "aa=\n="}`,
		inVal:   new(structFormatBytes),
		wantErr: EU(errors.New("illegal character '\\n' at offset 3")).withPos(`{"Base64": `, "/Base64").withType('"', T[[]byte]()),
	}, {
		name:    jsontest.Name("Structs/Format/Bytes/Invalid/Base64/NonAlphabet/CarriageReturn"),
		inBuf:   `{"Base64": "aa=\r="}`,
		inVal:   new(structFormatBytes),
		wantErr: EU(errors.New("illegal character '\\r' at offset 3")).withPos(`{"Base64": `, "/Base64").withType('"', T[[]byte]()),
	}, {
		name:  jsontest.Name("Structs/Format/Bytes/Base64/NonAlphabet/Ignored"),
		opts:  []Options{jsonflags.FormatBytesWithLegacySemantics | 1},
		inBuf: `{"Base64": "aa=\r\n="}`,
		inVal: new(structFormatBytes),
		want:  &structFormatBytes{Base64: []byte{105}},
	}, {
		name:    jsontest.Name("Structs/Format/Bytes/Invalid/Base64/NonAlphabet/Space"),
		inBuf:   `{"Base64": "aa= ="}`,
		inVal:   new(structFormatBytes),
		wantErr: EU(base64.CorruptInputError(2)).withPos(`{"Base64": `, "/Base64").withType('"', T[[]byte]()),
	}, {
		name: jsontest.Name("Structs/Format/Floats"),
		inBuf: `[
	{"NonFinite": 3.141592653589793, "PointerNonFinite": 3.141592653589793},
	{"NonFinite": "-Infinity", "PointerNonFinite": "-Infinity"},
	{"NonFinite": "Infinity", "PointerNonFinite": "Infinity"}
]`,
		inVal: new([]structFormatFloats),
		want: addr([]structFormatFloats{
			{NonFinite: math.Pi, PointerNonFinite: addr(math.Pi)},
			{NonFinite: math.Inf(-1), PointerNonFinite: addr(math.Inf(-1))},
			{NonFinite: math.Inf(+1), PointerNonFinite: addr(math.Inf(+1))},
		}),
	}, {
		name:  jsontest.Name("Structs/Format/Floats/NaN"),
		inBuf: `{"NonFinite": "NaN"}`,
		inVal: new(structFormatFloats),
		// Avoid checking want since reflect.DeepEqual fails for NaNs.
	}, {
		name:    jsontest.Name("Structs/Format/Floats/Invalid/NaN"),
		inBuf:   `{"NonFinite": "nan"}`,
		inVal:   new(structFormatFloats),
		wantErr: EU(nil).withPos(`{"NonFinite": `, "/NonFinite").withType('"', T[float64]()),
	}, {
		name:    jsontest.Name("Structs/Format/Floats/Invalid/PositiveInfinity"),
		inBuf:   `{"NonFinite": "+Infinity"}`,
		inVal:   new(structFormatFloats),
		wantErr: EU(nil).withPos(`{"NonFinite": `, "/NonFinite").withType('"', T[float64]()),
	}, {
		name:    jsontest.Name("Structs/Format/Floats/Invalid/NegativeInfinitySpace"),
		inBuf:   `{"NonFinite": "-Infinity "}`,
		inVal:   new(structFormatFloats),
		wantErr: EU(nil).withPos(`{"NonFinite": `, "/NonFinite").withType('"', T[float64]()),
	}, {
		name: jsontest.Name("Structs/Format/Maps"),
		inBuf: `[
	{"EmitNull": null, "PointerEmitNull": null, "EmitEmpty": null, "PointerEmitEmpty": null, "EmitDefault": null, "PointerEmitDefault": null},
	{"EmitNull": {}, "PointerEmitNull": {}, "EmitEmpty": {}, "PointerEmitEmpty": {}, "EmitDefault": {}, "PointerEmitDefault": {}},
	{"EmitNull": {"k": "v"}, "PointerEmitNull": {"k": "v"}, "EmitEmpty": {"k": "v"}, "PointerEmitEmpty": {"k": "v"}, "EmitDefault": {"k": "v"}, "PointerEmitDefault": {"k": "v"}}
]`,
		inVal: new([]structFormatMaps),
		want: addr([]structFormatMaps{{
			EmitNull: map[string]string(nil), PointerEmitNull: (*map[string]string)(nil),
			EmitEmpty: map[string]string(nil), PointerEmitEmpty: (*map[string]string)(nil),
			EmitDefault: map[string]string(nil), PointerEmitDefault: (*map[string]string)(nil),
		}, {
			EmitNull: map[string]string{}, PointerEmitNull: addr(map[string]string{}),
			EmitEmpty: map[string]string{}, PointerEmitEmpty: addr(map[string]string{}),
			EmitDefault: map[string]string{}, PointerEmitDefault: addr(map[string]string{}),
		}, {
			EmitNull: map[string]string{"k": "v"}, PointerEmitNull: addr(map[string]string{"k": "v"}),
			EmitEmpty: map[string]string{"k": "v"}, PointerEmitEmpty: addr(map[string]string{"k": "v"}),
			EmitDefault: map[string]string{"k": "v"}, PointerEmitDefault: addr(map[string]string{"k": "v"}),
		}}),
	}, {
		name: jsontest.Name("Structs/Format/Slices"),
		inBuf: `[
	{"EmitNull": null, "PointerEmitNull": null, "EmitEmpty": null, "PointerEmitEmpty": null, "EmitDefault": null, "PointerEmitDefault": null},
	{"EmitNull": [], "PointerEmitNull": [], "EmitEmpty": [], "PointerEmitEmpty": [], "EmitDefault": [], "PointerEmitDefault": []},
	{"EmitNull": ["v"], "PointerEmitNull": ["v"], "EmitEmpty": ["v"], "PointerEmitEmpty": ["v"], "EmitDefault": ["v"], "PointerEmitDefault": ["v"]}
]`,
		inVal: new([]structFormatSlices),
		want: addr([]structFormatSlices{{
			EmitNull: []string(nil), PointerEmitNull: (*[]string)(nil),
			EmitEmpty: []string(nil), PointerEmitEmpty: (*[]string)(nil),
			EmitDefault: []string(nil), PointerEmitDefault: (*[]string)(nil),
		}, {
			EmitNull: []string{}, PointerEmitNull: addr([]string{}),
			EmitEmpty: []string{}, PointerEmitEmpty: addr([]string{}),
			EmitDefault: []string{}, PointerEmitDefault: addr([]string{}),
		}, {
			EmitNull: []string{"v"}, PointerEmitNull: addr([]string{"v"}),
			EmitEmpty: []string{"v"}, PointerEmitEmpty: addr([]string{"v"}),
			EmitDefault: []string{"v"}, PointerEmitDefault: addr([]string{"v"}),
		}}),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Bool"),
		inBuf:   `{"Bool":true}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Bool":`, "/Bool").withType(0, T[bool]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/String"),
		inBuf:   `{"String": "string"}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"String": `, "/String").withType(0, T[string]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Bytes"),
		inBuf:   `{"Bytes": "bytes"}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Bytes": `, "/Bytes").withType(0, T[[]byte]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Int"),
		inBuf:   `{"Int":   1}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Int":   `, "/Int").withType(0, T[int64]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Uint"),
		inBuf:   `{"Uint": 1}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Uint": `, "/Uint").withType(0, T[uint64]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Float"),
		inBuf:   `{"Float" : 1}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Float" : `, "/Float").withType(0, T[float64]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Map"),
		inBuf:   `{"Map":{}}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Map":`, "/Map").withType(0, T[map[string]string]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Struct"),
		inBuf:   `{"Struct": {}}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Struct": `, "/Struct").withType(0, T[structAll]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Slice"),
		inBuf:   `{"Slice": {}}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Slice": `, "/Slice").withType(0, T[[]string]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Array"),
		inBuf:   `{"Array": []}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Array": `, "/Array").withType(0, T[[1]string]()),
	}, {
		name:    jsontest.Name("Structs/Format/Invalid/Interface"),
		inBuf:   `{"Interface": "anything"}`,
		inVal:   new(structFormatInvalid),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"Interface": `, "/Interface").withType(0, T[any]()),
	}, {
		name:  jsontest.Name("Structs/Inline/Zero"),
		inBuf: `{"D":""}`,
		inVal: new(structInlined),
		want:  new(structInlined),
	}, {
		name:  jsontest.Name("Structs/Inline/Alloc"),
		inBuf: `{"E":"","F":"","G":"","A":"","B":"","D":""}`,
		inVal: new(structInlined),
		want: addr(structInlined{
			X: structInlinedL1{
				X:            &structInlinedL2{},
				StructEmbed1: StructEmbed1{},
			},
			StructEmbed2: &StructEmbed2{},
		}),
	}, {
		name:  jsontest.Name("Structs/Inline/NonZero"),
		inBuf: `{"E":"E3","F":"F3","G":"G3","A":"A1","B":"B1","D":"D2"}`,
		inVal: new(structInlined),
		want: addr(structInlined{
			X: structInlinedL1{
				X:            &structInlinedL2{A: "A1", B: "B1" /* C: "C1" */},
				StructEmbed1: StructEmbed1{ /* C: "C2" */ D: "D2" /* E: "E2" */},
			},
			StructEmbed2: &StructEmbed2{E: "E3", F: "F3", G: "G3"},
		}),
	}, {
		name:  jsontest.Name("Structs/Inline/Merge"),
		inBuf: `{"E":"E3","F":"F3","G":"G3","A":"A1","B":"B1","D":"D2"}`,
		inVal: addr(structInlined{
			X: structInlinedL1{
				X:            &structInlinedL2{B: "##", C: "C1"},
				StructEmbed1: StructEmbed1{C: "C2", E: "E2"},
			},
			StructEmbed2: &StructEmbed2{E: "##", G: "G3"},
		}),
		want: addr(structInlined{
			X: structInlinedL1{
				X:            &structInlinedL2{A: "A1", B: "B1", C: "C1"},
				StructEmbed1: StructEmbed1{C: "C2", D: "D2", E: "E2"},
			},
			StructEmbed2: &StructEmbed2{E: "E3", F: "F3", G: "G3"},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/Noop"),
		inBuf: `{"A":1,"B":2}`,
		inVal: new(structInlineTextValue),
		want:  addr(structInlineTextValue{A: 1, X: jsontext.Value(nil), B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/MergeN1/Nil"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: new(structInlineTextValue),
		want:  addr(structInlineTextValue{A: 1, X: jsontext.Value(`{"fizz":"buzz"}`), B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/MergeN1/Empty"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: addr(structInlineTextValue{X: jsontext.Value{}}),
		want:  addr(structInlineTextValue{A: 1, X: jsontext.Value(`{"fizz":"buzz"}`), B: 2}),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/MergeN1/Whitespace"),
		inBuf:   `{"A":1,"fizz":"buzz","B":2}`,
		inVal:   addr(structInlineTextValue{X: jsontext.Value("\n\r\t ")}),
		want:    addr(structInlineTextValue{A: 1, X: jsontext.Value("")}),
		wantErr: EU(errRawInlinedNotObject).withPos(`{"A":1,`, "/fizz").withType('"', T[jsontext.Value]()),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/MergeN1/Null"),
		inBuf:   `{"A":1,"fizz":"buzz","B":2}`,
		inVal:   addr(structInlineTextValue{X: jsontext.Value("null")}),
		want:    addr(structInlineTextValue{A: 1, X: jsontext.Value("null")}),
		wantErr: EU(errRawInlinedNotObject).withPos(`{"A":1,`, "/fizz").withType('"', T[jsontext.Value]()),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/MergeN1/ObjectN0"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: addr(structInlineTextValue{X: jsontext.Value(` { } `)}),
		want:  addr(structInlineTextValue{A: 1, X: jsontext.Value(` {"fizz":"buzz"}`), B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/MergeN2/ObjectN1"),
		inBuf: `{"A":1,"fizz":"buzz","B":2,"foo": [ 1 , 2 , 3 ]}`,
		inVal: addr(structInlineTextValue{X: jsontext.Value(` { "fizz" : "buzz" } `)}),
		want:  addr(structInlineTextValue{A: 1, X: jsontext.Value(` { "fizz" : "buzz","fizz":"buzz","foo":[ 1 , 2 , 3 ]}`), B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/Merge/EndObject"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: addr(structInlineTextValue{X: jsontext.Value(` } `)}),
		// NOTE: This produces invalid output,
		// but the value being merged into is already invalid.
		want: addr(structInlineTextValue{A: 1, X: jsontext.Value(`,"fizz":"buzz"}`), B: 2}),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/MergeInvalidValue"),
		inBuf:   `{"A":1,"fizz":nil,"B":2}`,
		inVal:   new(structInlineTextValue),
		want:    addr(structInlineTextValue{A: 1, X: jsontext.Value(`{"fizz":`)}),
		wantErr: newInvalidCharacterError("i", "in literal null (expecting 'u')", len64(`{"A":1,"fizz":n`), "/fizz"),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/CaseSensitive"),
		inBuf: `{"A":1,"fizz":"buzz","B":2,"a":3}`,
		inVal: new(structInlineTextValue),
		want:  addr(structInlineTextValue{A: 1, X: jsontext.Value(`{"fizz":"buzz","a":3}`), B: 2}),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/TextValue/RejectDuplicateNames"),
		opts:    []Options{jsontext.AllowDuplicateNames(false)},
		inBuf:   `{"A":1,"fizz":"buzz","B":2,"fizz":"buzz"}`,
		inVal:   new(structInlineTextValue),
		want:    addr(structInlineTextValue{A: 1, X: jsontext.Value(`{"fizz":"buzz"}`), B: 2}),
		wantErr: newDuplicateNameError("", []byte(`"fizz"`), len64(`{"A":1,"fizz":"buzz","B":2,`)),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/AllowDuplicateNames"),
		opts:  []Options{jsontext.AllowDuplicateNames(true)},
		inBuf: `{"A":1,"fizz":"buzz","B":2,"fizz":"buzz"}`,
		inVal: new(structInlineTextValue),
		want:  addr(structInlineTextValue{A: 1, X: jsontext.Value(`{"fizz":"buzz","fizz":"buzz"}`), B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/Nested/Noop"),
		inBuf: `{}`,
		inVal: new(structInlinePointerInlineTextValue),
		want:  new(structInlinePointerInlineTextValue),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/Nested/Alloc"),
		inBuf: `{"A":1,"fizz":"buzz"}`,
		inVal: new(structInlinePointerInlineTextValue),
		want: addr(structInlinePointerInlineTextValue{
			X: &struct {
				A int
				X jsontext.Value `json:",inline"`
			}{A: 1, X: jsontext.Value(`{"fizz":"buzz"}`)},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/TextValue/Nested/Merge"),
		inBuf: `{"fizz":"buzz"}`,
		inVal: addr(structInlinePointerInlineTextValue{
			X: &struct {
				A int
				X jsontext.Value `json:",inline"`
			}{A: 1},
		}),
		want: addr(structInlinePointerInlineTextValue{
			X: &struct {
				A int
				X jsontext.Value `json:",inline"`
			}{A: 1, X: jsontext.Value(`{"fizz":"buzz"}`)},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/PointerTextValue/Noop"),
		inBuf: `{"A":1,"B":2}`,
		inVal: new(structInlinePointerTextValue),
		want:  addr(structInlinePointerTextValue{A: 1, X: nil, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/PointerTextValue/Alloc"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: new(structInlinePointerTextValue),
		want:  addr(structInlinePointerTextValue{A: 1, X: addr(jsontext.Value(`{"fizz":"buzz"}`)), B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/PointerTextValue/Merge"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: addr(structInlinePointerTextValue{X: addr(jsontext.Value(`{"fizz":"buzz"}`))}),
		want:  addr(structInlinePointerTextValue{A: 1, X: addr(jsontext.Value(`{"fizz":"buzz","fizz":"buzz"}`)), B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/PointerTextValue/Nested/Nil"),
		inBuf: `{"fizz":"buzz"}`,
		inVal: new(structInlineInlinePointerTextValue),
		want: addr(structInlineInlinePointerTextValue{
			X: struct {
				X *jsontext.Value `json:",inline"`
			}{X: addr(jsontext.Value(`{"fizz":"buzz"}`))},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/Noop"),
		inBuf: `{"A":1,"B":2}`,
		inVal: new(structInlineMapStringAny),
		want:  addr(structInlineMapStringAny{A: 1, X: nil, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/MergeN1/Nil"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: new(structInlineMapStringAny),
		want:  addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": "buzz"}, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/MergeN1/Empty"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: addr(structInlineMapStringAny{X: jsonObject{}}),
		want:  addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": "buzz"}, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/MergeN1/ObjectN1"),
		inBuf: `{"A":1,"fizz":{"charlie":"DELTA","echo":"foxtrot"},"B":2}`,
		inVal: addr(structInlineMapStringAny{X: jsonObject{"fizz": jsonObject{
			"alpha":   "bravo",
			"charlie": "delta",
		}}}),
		want: addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": jsonObject{
			"alpha":   "bravo",
			"charlie": "DELTA",
			"echo":    "foxtrot",
		}}, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/MergeN2/ObjectN1"),
		inBuf: `{"A":1,"fizz":"buzz","B":2,"foo": [ 1 , 2 , 3 ]}`,
		inVal: addr(structInlineMapStringAny{X: jsonObject{"fizz": "wuzz"}}),
		want:  addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": "buzz", "foo": jsonArray{1.0, 2.0, 3.0}}, B: 2}),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapStringAny/MergeInvalidValue"),
		inBuf:   `{"A":1,"fizz":nil,"B":2}`,
		inVal:   new(structInlineMapStringAny),
		want:    addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": nil}}),
		wantErr: newInvalidCharacterError("i", "in literal null (expecting 'u')", len64(`{"A":1,"fizz":n`), "/fizz"),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapStringAny/MergeInvalidValue/Existing"),
		inBuf:   `{"A":1,"fizz":nil,"B":2}`,
		inVal:   addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": true}}),
		want:    addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": true}}),
		wantErr: newInvalidCharacterError("i", "in literal null (expecting 'u')", len64(`{"A":1,"fizz":n`), "/fizz"),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/CaseSensitive"),
		inBuf: `{"A":1,"fizz":"buzz","B":2,"a":3}`,
		inVal: new(structInlineMapStringAny),
		want:  addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": "buzz", "a": 3.0}, B: 2}),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapStringAny/RejectDuplicateNames"),
		opts:    []Options{jsontext.AllowDuplicateNames(false)},
		inBuf:   `{"A":1,"fizz":"buzz","B":2,"fizz":"buzz"}`,
		inVal:   new(structInlineMapStringAny),
		want:    addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": "buzz"}, B: 2}),
		wantErr: newDuplicateNameError("", []byte(`"fizz"`), len64(`{"A":1,"fizz":"buzz","B":2,`)),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/AllowDuplicateNames"),
		opts:  []Options{jsontext.AllowDuplicateNames(true)},
		inBuf: `{"A":1,"fizz":{"one":1,"two":-2},"B":2,"fizz":{"two":2,"three":3}}`,
		inVal: new(structInlineMapStringAny),
		want:  addr(structInlineMapStringAny{A: 1, X: jsonObject{"fizz": jsonObject{"one": 1.0, "two": 2.0, "three": 3.0}}, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/Nested/Noop"),
		inBuf: `{}`,
		inVal: new(structInlinePointerInlineMapStringAny),
		want:  new(structInlinePointerInlineMapStringAny),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/Nested/Alloc"),
		inBuf: `{"A":1,"fizz":"buzz"}`,
		inVal: new(structInlinePointerInlineMapStringAny),
		want: addr(structInlinePointerInlineMapStringAny{
			X: &struct {
				A int
				X jsonObject `json:",inline"`
			}{A: 1, X: jsonObject{"fizz": "buzz"}},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringAny/Nested/Merge"),
		inBuf: `{"fizz":"buzz"}`,
		inVal: addr(structInlinePointerInlineMapStringAny{
			X: &struct {
				A int
				X jsonObject `json:",inline"`
			}{A: 1},
		}),
		want: addr(structInlinePointerInlineMapStringAny{
			X: &struct {
				A int
				X jsonObject `json:",inline"`
			}{A: 1, X: jsonObject{"fizz": "buzz"}},
		}),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringInt/UnmarshalFunc"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *any) error {
				var err error
				*v, err = strconv.ParseFloat(string(bytes.Trim(b, `"`)), 64)
				return err
			})),
		},
		inBuf: `{"D":"1.1","E":"2.2","F":"3.3"}`,
		inVal: new(structInlineMapStringAny),
		want:  addr(structInlineMapStringAny{X: jsonObject{"D": 1.1, "E": 2.2, "F": 3.3}}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/PointerMapStringAny/Noop"),
		inBuf: `{"A":1,"B":2}`,
		inVal: new(structInlinePointerMapStringAny),
		want:  addr(structInlinePointerMapStringAny{A: 1, X: nil, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/PointerMapStringAny/Alloc"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: new(structInlinePointerMapStringAny),
		want:  addr(structInlinePointerMapStringAny{A: 1, X: addr(jsonObject{"fizz": "buzz"}), B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/PointerMapStringAny/Merge"),
		inBuf: `{"A":1,"fizz":"wuzz","B":2}`,
		inVal: addr(structInlinePointerMapStringAny{X: addr(jsonObject{"fizz": "buzz"})}),
		want:  addr(structInlinePointerMapStringAny{A: 1, X: addr(jsonObject{"fizz": "wuzz"}), B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/PointerMapStringAny/Nested/Nil"),
		inBuf: `{"fizz":"buzz"}`,
		inVal: new(structInlineInlinePointerMapStringAny),
		want: addr(structInlineInlinePointerMapStringAny{
			X: struct {
				X *jsonObject `json:",inline"`
			}{X: addr(jsonObject{"fizz": "buzz"})},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringInt"),
		inBuf: `{"zero": 0, "one": 1, "two": 2}`,
		inVal: new(structInlineMapStringInt),
		want: addr(structInlineMapStringInt{
			X: map[string]int{"zero": 0, "one": 1, "two": 2},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringInt/Null"),
		inBuf: `{"zero": 0, "one": null, "two": 2}`,
		inVal: new(structInlineMapStringInt),
		want: addr(structInlineMapStringInt{
			X: map[string]int{"zero": 0, "one": 0, "two": 2},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringInt/Invalid"),
		inBuf: `{"zero": 0, "one": {}, "two": 2}`,
		inVal: new(structInlineMapStringInt),
		want: addr(structInlineMapStringInt{
			X: map[string]int{"zero": 0, "one": 0},
		}),
		wantErr: EU(nil).withPos(`{"zero": 0, "one": `, "/one").withType('{', T[int]()),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapStringInt/StringifiedNumbers"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `{"zero": "0", "one": "1", "two": "2"}`,
		inVal: new(structInlineMapStringInt),
		want: addr(structInlineMapStringInt{
			X: map[string]int{"zero": 0, "one": 1, "two": 2},
		}),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapStringInt/UnmarshalFunc"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *int) error {
				i, err := strconv.ParseInt(string(bytes.Trim(b, `"`)), 10, 64)
				if err != nil {
					return err
				}
				*v = int(i)
				return nil
			})),
		},
		inBuf: `{"zero": "0", "one": "1", "two": "2"}`,
		inVal: new(structInlineMapStringInt),
		want: addr(structInlineMapStringInt{
			X: map[string]int{"zero": 0, "one": 1, "two": 2},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringInt"),
		inBuf: `{"zero": 0, "one": 1, "two": 2}`,
		inVal: new(structInlineMapNamedStringInt),
		want: addr(structInlineMapNamedStringInt{
			X: map[namedString]int{"zero": 0, "one": 1, "two": 2},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringInt/Null"),
		inBuf: `{"zero": 0, "one": null, "two": 2}`,
		inVal: new(structInlineMapNamedStringInt),
		want: addr(structInlineMapNamedStringInt{
			X: map[namedString]int{"zero": 0, "one": 0, "two": 2},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringInt/Invalid"),
		inBuf: `{"zero": 0, "one": {}, "two": 2}`,
		inVal: new(structInlineMapNamedStringInt),
		want: addr(structInlineMapNamedStringInt{
			X: map[namedString]int{"zero": 0, "one": 0},
		}),
		wantErr: EU(nil).withPos(`{"zero": 0, "one": `, "/one").withType('{', T[int]()),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringInt/StringifiedNumbers"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `{"zero": "0", "one": 1, "two": "2"}`,
		inVal: new(structInlineMapNamedStringInt),
		want: addr(structInlineMapNamedStringInt{
			X: map[namedString]int{"zero": 0, "one": 0},
		}),
		wantErr: EU(nil).withPos(`{"zero": "0", "one": `, "/one").withType('0', T[int]()),
	}, {
		name: jsontest.Name("Structs/InlinedFallback/MapNamedStringInt/UnmarshalFunc"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *int) error {
				i, err := strconv.ParseInt(string(bytes.Trim(b, `"`)), 10, 64)
				if err != nil {
					return err
				}
				*v = int(i)
				return nil
			})),
		},
		inBuf: `{"zero": "0", "one": "1", "two": "2"}`,
		inVal: new(structInlineMapNamedStringInt),
		want: addr(structInlineMapNamedStringInt{
			X: map[namedString]int{"zero": 0, "one": 1, "two": 2},
		}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/Noop"),
		inBuf: `{"A":1,"B":2}`,
		inVal: new(structInlineMapNamedStringAny),
		want:  addr(structInlineMapNamedStringAny{A: 1, X: nil, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/MergeN1/Nil"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: new(structInlineMapNamedStringAny),
		want:  addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": "buzz"}, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/MergeN1/Empty"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: addr(structInlineMapNamedStringAny{X: map[namedString]any{}}),
		want:  addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": "buzz"}, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/MergeN1/ObjectN1"),
		inBuf: `{"A":1,"fizz":{"charlie":"DELTA","echo":"foxtrot"},"B":2}`,
		inVal: addr(structInlineMapNamedStringAny{X: map[namedString]any{"fizz": jsonObject{
			"alpha":   "bravo",
			"charlie": "delta",
		}}}),
		want: addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": jsonObject{
			"alpha":   "bravo",
			"charlie": "DELTA",
			"echo":    "foxtrot",
		}}, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/MergeN2/ObjectN1"),
		inBuf: `{"A":1,"fizz":"buzz","B":2,"foo": [ 1 , 2 , 3 ]}`,
		inVal: addr(structInlineMapNamedStringAny{X: map[namedString]any{"fizz": "wuzz"}}),
		want:  addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": "buzz", "foo": jsonArray{1.0, 2.0, 3.0}}, B: 2}),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/MergeInvalidValue"),
		inBuf:   `{"A":1,"fizz":nil,"B":2}`,
		inVal:   new(structInlineMapNamedStringAny),
		want:    addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": nil}}),
		wantErr: newInvalidCharacterError("i", "in literal null (expecting 'u')", len64(`{"A":1,"fizz":n`), "/fizz"),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/MergeInvalidValue/Existing"),
		inBuf:   `{"A":1,"fizz":nil,"B":2}`,
		inVal:   addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": true}}),
		want:    addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": true}}),
		wantErr: newInvalidCharacterError("i", "in literal null (expecting 'u')", len64(`{"A":1,"fizz":n`), "/fizz"),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/CaseSensitive"),
		inBuf: `{"A":1,"fizz":"buzz","B":2,"a":3}`,
		inVal: new(structInlineMapNamedStringAny),
		want:  addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": "buzz", "a": 3.0}, B: 2}),
	}, {
		name:    jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/RejectDuplicateNames"),
		opts:    []Options{jsontext.AllowDuplicateNames(false)},
		inBuf:   `{"A":1,"fizz":"buzz","B":2,"fizz":"buzz"}`,
		inVal:   new(structInlineMapNamedStringAny),
		want:    addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": "buzz"}, B: 2}),
		wantErr: newDuplicateNameError("", []byte(`"fizz"`), len64(`{"A":1,"fizz":"buzz","B":2,`)),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/MapNamedStringAny/AllowDuplicateNames"),
		opts:  []Options{jsontext.AllowDuplicateNames(true)},
		inBuf: `{"A":1,"fizz":{"one":1,"two":-2},"B":2,"fizz":{"two":2,"three":3}}`,
		inVal: new(structInlineMapNamedStringAny),
		want:  addr(structInlineMapNamedStringAny{A: 1, X: map[namedString]any{"fizz": map[string]any{"one": 1.0, "two": 2.0, "three": 3.0}}, B: 2}),
	}, {
		name:  jsontest.Name("Structs/InlinedFallback/RejectUnknownMembers"),
		opts:  []Options{RejectUnknownMembers(true)},
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: new(structInlineTextValue),
		// NOTE: DiscardUnknownMembers has no effect since this is "inline".
		want: addr(structInlineTextValue{
			A: 1,
			X: jsontext.Value(`{"fizz":"buzz"}`),
			B: 2,
		}),
	}, {
		name:    jsontest.Name("Structs/UnknownFallback/RejectUnknownMembers"),
		opts:    []Options{RejectUnknownMembers(true)},
		inBuf:   `{"A":1,"fizz":"buzz","B":2}`,
		inVal:   new(structUnknownTextValue),
		want:    addr(structUnknownTextValue{A: 1}),
		wantErr: EU(ErrUnknownName).withPos(`{"A":1,`, "/fizz").withType('"', T[structUnknownTextValue]()),
	}, {
		name:  jsontest.Name("Structs/UnknownFallback"),
		inBuf: `{"A":1,"fizz":"buzz","B":2}`,
		inVal: new(structUnknownTextValue),
		want: addr(structUnknownTextValue{
			A: 1,
			X: jsontext.Value(`{"fizz":"buzz"}`),
			B: 2,
		}),
	}, {
		name:  jsontest.Name("Structs/UnknownIgnored"),
		opts:  []Options{RejectUnknownMembers(false)},
		inBuf: `{"unknown":"fizzbuzz"}`,
		inVal: new(structAll),
		want:  new(structAll),
	}, {
		name:    jsontest.Name("Structs/RejectUnknownMembers"),
		opts:    []Options{RejectUnknownMembers(true)},
		inBuf:   `{"unknown":"fizzbuzz"}`,
		inVal:   new(structAll),
		want:    new(structAll),
		wantErr: EU(ErrUnknownName).withPos(`{`, "/unknown").withType('"', T[structAll]()),
	}, {
		name:  jsontest.Name("Structs/UnexportedIgnored"),
		inBuf: `{"ignored":"unused"}`,
		inVal: new(structUnexportedIgnored),
		want:  new(structUnexportedIgnored),
	}, {
		name:  jsontest.Name("Structs/IgnoredUnexportedEmbedded"),
		inBuf: `{"namedString":"unused"}`,
		inVal: new(structIgnoredUnexportedEmbedded),
		want:  new(structIgnoredUnexportedEmbedded),
	}, {
		name:  jsontest.Name("Structs/WeirdNames"),
		inBuf: `{"":"empty",",":"comma","\"":"quote"}`,
		inVal: new(structWeirdNames),
		want:  addr(structWeirdNames{Empty: "empty", Comma: "comma", Quote: "quote"}),
	}, {
		name:  jsontest.Name("Structs/NoCase/Exact"),
		inBuf: `{"Aaa":"Aaa","AA_A":"AA_A","AaA":"AaA","AAa":"AAa","AAA":"AAA"}`,
		inVal: new(structNoCase),
		want:  addr(structNoCase{AaA: "AaA", AAa: "AAa", Aaa: "Aaa", AAA: "AAA", AA_A: "AA_A"}),
	}, {
		name:  jsontest.Name("Structs/NoCase/CaseInsensitiveDefault"),
		inBuf: `{"aa_a":"aa_a"}`,
		inVal: new(structNoCase),
		want:  addr(structNoCase{AaA: "aa_a"}),
	}, {
		name:  jsontest.Name("Structs/NoCase/MatchCaseSensitiveDelimiter"),
		opts:  []Options{jsonflags.MatchCaseSensitiveDelimiter | 1},
		inBuf: `{"aa_a":"aa_a"}`,
		inVal: new(structNoCase),
		want:  addr(structNoCase{}),
	}, {
		name:  jsontest.Name("Structs/NoCase/MatchCaseInsensitiveNames+MatchCaseSensitiveDelimiter"),
		opts:  []Options{MatchCaseInsensitiveNames(true), jsonflags.MatchCaseSensitiveDelimiter | 1},
		inBuf: `{"aa_a":"aa_a"}`,
		inVal: new(structNoCase),
		want:  addr(structNoCase{AA_A: "aa_a"}),
	}, {
		name:  jsontest.Name("Structs/NoCase/Merge/AllowDuplicateNames"),
		opts:  []Options{jsontext.AllowDuplicateNames(true)},
		inBuf: `{"AaA":"AaA","aaa":"aaa","aAa":"aAa"}`,
		inVal: new(structNoCase),
		want:  addr(structNoCase{AaA: "aAa"}),
	}, {
		name:    jsontest.Name("Structs/NoCase/Merge/RejectDuplicateNames"),
		opts:    []Options{jsontext.AllowDuplicateNames(false)},
		inBuf:   `{"AaA":"AaA","aaa":"aaa"}`,
		inVal:   new(structNoCase),
		want:    addr(structNoCase{AaA: "AaA"}),
		wantErr: newDuplicateNameError("", []byte(`"aaa"`), len64(`{"AaA":"AaA",`)),
	}, {
		name:  jsontest.Name("Structs/CaseSensitive"),
		inBuf: `{"BOOL": true, "STRING": "hello", "BYTES": "AQID", "INT": -64, "UINT": 64, "FLOAT": 3.14159}`,
		inVal: new(structScalars),
		want:  addr(structScalars{}),
	}, {
		name:  jsontest.Name("Structs/DuplicateName/NoCase/ExactDifferent"),
		inBuf: `{"AAA":"AAA","AaA":"AaA","AAa":"AAa","Aaa":"Aaa"}`,
		inVal: addr(structNoCaseInlineTextValue{}),
		want:  addr(structNoCaseInlineTextValue{AAA: "AAA", AaA: "AaA", AAa: "AAa", Aaa: "Aaa"}),
	}, {
		name:    jsontest.Name("Structs/DuplicateName/NoCase/ExactConflict"),
		inBuf:   `{"AAA":"AAA","AAA":"AAA"}`,
		inVal:   addr(structNoCaseInlineTextValue{}),
		want:    addr(structNoCaseInlineTextValue{AAA: "AAA"}),
		wantErr: newDuplicateNameError("", []byte(`"AAA"`), len64(`{"AAA":"AAA",`)),
	}, {
		name:  jsontest.Name("Structs/DuplicateName/NoCase/OverwriteExact"),
		inBuf: `{"AAA":"after"}`,
		inVal: addr(structNoCaseInlineTextValue{AAA: "before"}),
		want:  addr(structNoCaseInlineTextValue{AAA: "after"}),
	}, {
		name:    jsontest.Name("Structs/DuplicateName/NoCase/NoCaseConflict"),
		inBuf:   `{"aaa":"aaa","aaA":"aaA"}`,
		inVal:   addr(structNoCaseInlineTextValue{}),
		want:    addr(structNoCaseInlineTextValue{AaA: "aaa"}),
		wantErr: newDuplicateNameError("", []byte(`"aaA"`), len64(`{"aaa":"aaa",`)),
	}, {
		name:    jsontest.Name("Structs/DuplicateName/NoCase/OverwriteNoCase"),
		inBuf:   `{"aaa":"aaa","aaA":"aaA"}`,
		inVal:   addr(structNoCaseInlineTextValue{}),
		want:    addr(structNoCaseInlineTextValue{AaA: "aaa"}),
		wantErr: newDuplicateNameError("", []byte(`"aaA"`), len64(`{"aaa":"aaa",`)),
	}, {
		name:  jsontest.Name("Structs/DuplicateName/Inline/Unknown"),
		inBuf: `{"unknown":""}`,
		inVal: addr(structNoCaseInlineTextValue{}),
		want:  addr(structNoCaseInlineTextValue{X: jsontext.Value(`{"unknown":""}`)}),
	}, {
		name:  jsontest.Name("Structs/DuplicateName/Inline/UnknownMerge"),
		inBuf: `{"unknown":""}`,
		inVal: addr(structNoCaseInlineTextValue{X: jsontext.Value(`{"unknown":""}`)}),
		want:  addr(structNoCaseInlineTextValue{X: jsontext.Value(`{"unknown":"","unknown":""}`)}),
	}, {
		name:  jsontest.Name("Structs/DuplicateName/Inline/NoCaseOkay"),
		inBuf: `{"b":"","B":""}`,
		inVal: addr(structNoCaseInlineTextValue{}),
		want:  addr(structNoCaseInlineTextValue{X: jsontext.Value(`{"b":"","B":""}`)}),
	}, {
		name:    jsontest.Name("Structs/DuplicateName/Inline/ExactConflict"),
		inBuf:   `{"b":"","b":""}`,
		inVal:   addr(structNoCaseInlineTextValue{}),
		want:    addr(structNoCaseInlineTextValue{X: jsontext.Value(`{"b":""}`)}),
		wantErr: newDuplicateNameError("", []byte(`"b"`), len64(`{"b":"",`)),
	}, {
		name:    jsontest.Name("Structs/Invalid/ErrUnexpectedEOF"),
		inBuf:   ``,
		inVal:   addr(structAll{}),
		want:    addr(structAll{}),
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name:    jsontest.Name("Structs/Invalid/NestedErrUnexpectedEOF"),
		inBuf:   `{"Pointer":`,
		inVal:   addr(structAll{}),
		want:    addr(structAll{Pointer: new(structAll)}),
		wantErr: &jsontext.SyntacticError{ByteOffset: len64(`{"Pointer":`), JSONPointer: "/Pointer", Err: io.ErrUnexpectedEOF},
	}, {
		name:    jsontest.Name("Structs/Invalid/Conflicting"),
		inBuf:   `{}`,
		inVal:   addr(structConflicting{}),
		want:    addr(structConflicting{}),
		wantErr: EU(errors.New(`Go struct fields A and B conflict over JSON object name "conflict"`)).withType('{', T[structConflicting]()),
	}, {
		name:    jsontest.Name("Structs/Invalid/NoneExported"),
		inBuf:   ` {}`,
		inVal:   addr(structNoneExported{}),
		want:    addr(structNoneExported{}),
		wantErr: EU(errNoExportedFields).withPos(` `, "").withType('{', T[structNoneExported]()),
	}, {
		name:    jsontest.Name("Structs/Invalid/MalformedTag"),
		inBuf:   `{}`,
		inVal:   addr(structMalformedTag{}),
		want:    addr(structMalformedTag{}),
		wantErr: EU(errors.New("Go struct field Malformed has malformed `json` tag: invalid character '\"' at start of option (expecting Unicode letter or single quote)")).withType('{', T[structMalformedTag]()),
	}, {
		name:    jsontest.Name("Structs/Invalid/UnexportedTag"),
		inBuf:   `{}`,
		inVal:   addr(structUnexportedTag{}),
		want:    addr(structUnexportedTag{}),
		wantErr: EU(errors.New("unexported Go struct field unexported cannot have non-ignored `json:\"name\"` tag")).withType('{', T[structUnexportedTag]()),
	}, {
		name:    jsontest.Name("Structs/Invalid/ExportedEmbedded"),
		inBuf:   `{"NamedString":"hello"}`,
		inVal:   addr(structExportedEmbedded{}),
		want:    addr(structExportedEmbedded{}),
		wantErr: EU(errors.New("embedded Go struct field NamedString of non-struct type must be explicitly given a JSON name")).withType('{', T[structExportedEmbedded]()),
	}, {
		name:  jsontest.Name("Structs/Valid/ExportedEmbedded"),
		opts:  []Options{jsonflags.ReportErrorsWithLegacySemantics | 1},
		inBuf: `{"NamedString":"hello"}`,
		inVal: addr(structExportedEmbedded{}),
		want:  addr(structExportedEmbedded{"hello"}),
	}, {
		name:  jsontest.Name("Structs/Valid/ExportedEmbeddedTag"),
		inBuf: `{"name":"hello"}`,
		inVal: addr(structExportedEmbeddedTag{}),
		want:  addr(structExportedEmbeddedTag{"hello"}),
	}, {
		name:    jsontest.Name("Structs/Invalid/UnexportedEmbedded"),
		inBuf:   `{}`,
		inVal:   addr(structUnexportedEmbedded{}),
		want:    addr(structUnexportedEmbedded{}),
		wantErr: EU(errors.New("embedded Go struct field namedString of non-struct type must be explicitly given a JSON name")).withType('{', T[structUnexportedEmbedded]()),
	}, {
		name:  jsontest.Name("Structs/UnexportedEmbeddedStruct"),
		inBuf: `{"Bool":true,"FizzBuzz":5,"Addr":"192.168.0.1"}`,
		inVal: addr(structUnexportedEmbeddedStruct{}),
		want:  addr(structUnexportedEmbeddedStruct{structOmitZeroAll{Bool: true}, 5, structNestedAddr{netip.AddrFrom4([4]byte{192, 168, 0, 1})}}),
	}, {
		name:    jsontest.Name("Structs/UnexportedEmbeddedStructPointer/Nil"),
		inBuf:   `{"Bool":true,"FizzBuzz":5}`,
		inVal:   addr(structUnexportedEmbeddedStructPointer{}),
		wantErr: EU(errNilField).withPos(`{"Bool":`, "/Bool").withType(0, T[structUnexportedEmbeddedStructPointer]()),
	}, {
		name:    jsontest.Name("Structs/UnexportedEmbeddedStructPointer/Nil"),
		inBuf:   `{"FizzBuzz":5,"Addr":"192.168.0.1"}`,
		inVal:   addr(structUnexportedEmbeddedStructPointer{}),
		wantErr: EU(errNilField).withPos(`{"FizzBuzz":5,"Addr":`, "/Addr").withType(0, T[structUnexportedEmbeddedStructPointer]()),
	}, {
		name:  jsontest.Name("Structs/UnexportedEmbeddedStructPointer/Nil"),
		inBuf: `{"Bool":true,"FizzBuzz":10,"Addr":"192.168.0.1"}`,
		inVal: addr(structUnexportedEmbeddedStructPointer{&structOmitZeroAll{Int: 5}, 5, &structNestedAddr{netip.AddrFrom4([4]byte{127, 0, 0, 1})}}),
		want:  addr(structUnexportedEmbeddedStructPointer{&structOmitZeroAll{Bool: true, Int: 5}, 10, &structNestedAddr{netip.AddrFrom4([4]byte{192, 168, 0, 1})}}),
	}, {
		name: jsontest.Name("Structs/Unknown"),
		inBuf: `{
	"object0": {},
	"object1": {"key1": "value"},
	"object2": {"key1": "value", "key2": "value"},
	"objects": {"":{"":{"":{}}}},
	"array0": [],
	"array1": ["value1"],
	"array2": ["value1", "value2"],
	"array": [[[]]],
	"scalars": [null, false, true, "string", 12.345]
}`,
		inVal: addr(struct{}{}),
		want:  addr(struct{}{}),
	}, {
		name:  jsontest.Name("Structs/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `{"Field":"Value"}`,
		inVal: addr(struct{ Field string }{}),
		want:  addr(struct{ Field string }{"Value"}),
	}, {
		name:  jsontest.Name("Slices/Null"),
		inBuf: `null`,
		inVal: addr([]string{"something"}),
		want:  addr([]string(nil)),
	}, {
		name:  jsontest.Name("Slices/Bool"),
		inBuf: `[true,false]`,
		inVal: new([]bool),
		want:  addr([]bool{true, false}),
	}, {
		name:  jsontest.Name("Slices/String"),
		inBuf: `["hello","goodbye"]`,
		inVal: new([]string),
		want:  addr([]string{"hello", "goodbye"}),
	}, {
		name:  jsontest.Name("Slices/Bytes"),
		inBuf: `["aGVsbG8=","Z29vZGJ5ZQ=="]`,
		inVal: new([][]byte),
		want:  addr([][]byte{[]byte("hello"), []byte("goodbye")}),
	}, {
		name:  jsontest.Name("Slices/Int"),
		inBuf: `[-2,-1,0,1,2]`,
		inVal: new([]int),
		want:  addr([]int{-2, -1, 0, 1, 2}),
	}, {
		name:  jsontest.Name("Slices/Uint"),
		inBuf: `[0,1,2,3,4]`,
		inVal: new([]uint),
		want:  addr([]uint{0, 1, 2, 3, 4}),
	}, {
		name:  jsontest.Name("Slices/Float"),
		inBuf: `[3.14159,12.34]`,
		inVal: new([]float64),
		want:  addr([]float64{3.14159, 12.34}),
	}, {
		// NOTE: The semantics differs from v1, where the slice length is reset
		// and new elements are appended to the end.
		// See https://go.dev/issue/21092.
		name:  jsontest.Name("Slices/Merge"),
		inBuf: `[{"k3":"v3"},{"k4":"v4"}]`,
		inVal: addr([]map[string]string{{"k1": "v1"}, {"k2": "v2"}}[:1]),
		want:  addr([]map[string]string{{"k3": "v3"}, {"k4": "v4"}}),
	}, {
		name:    jsontest.Name("Slices/Invalid/Channel"),
		inBuf:   `["hello"]`,
		inVal:   new([]chan string),
		want:    addr([]chan string{nil}),
		wantErr: EU(nil).withPos(`[`, "/0").withType(0, T[chan string]()),
	}, {
		name:  jsontest.Name("Slices/RecursiveSlice"),
		inBuf: `[[],[],[[]],[[],[]]]`,
		inVal: new(recursiveSlice),
		want: addr(recursiveSlice{
			{},
			{},
			{{}},
			{{}, {}},
		}),
	}, {
		name:    jsontest.Name("Slices/Invalid/Bool"),
		inBuf:   `true`,
		inVal:   addr([]string{"nochange"}),
		want:    addr([]string{"nochange"}),
		wantErr: EU(nil).withType('t', T[[]string]()),
	}, {
		name:    jsontest.Name("Slices/Invalid/String"),
		inBuf:   `""`,
		inVal:   addr([]string{"nochange"}),
		want:    addr([]string{"nochange"}),
		wantErr: EU(nil).withType('"', T[[]string]()),
	}, {
		name:    jsontest.Name("Slices/Invalid/Number"),
		inBuf:   `0`,
		inVal:   addr([]string{"nochange"}),
		want:    addr([]string{"nochange"}),
		wantErr: EU(nil).withType('0', T[[]string]()),
	}, {
		name:    jsontest.Name("Slices/Invalid/Object"),
		inBuf:   `{}`,
		inVal:   addr([]string{"nochange"}),
		want:    addr([]string{"nochange"}),
		wantErr: EU(nil).withType('{', T[[]string]()),
	}, {
		name:  jsontest.Name("Slices/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `[false,true]`,
		inVal: addr([]bool{true, false}),
		want:  addr([]bool{false, true}),
	}, {
		name:  jsontest.Name("Arrays/Null"),
		inBuf: `null`,
		inVal: addr([1]string{"something"}),
		want:  addr([1]string{}),
	}, {
		name:  jsontest.Name("Arrays/Bool"),
		inBuf: `[true,false]`,
		inVal: new([2]bool),
		want:  addr([2]bool{true, false}),
	}, {
		name:  jsontest.Name("Arrays/String"),
		inBuf: `["hello","goodbye"]`,
		inVal: new([2]string),
		want:  addr([2]string{"hello", "goodbye"}),
	}, {
		name:  jsontest.Name("Arrays/Bytes"),
		inBuf: `["aGVsbG8=","Z29vZGJ5ZQ=="]`,
		inVal: new([2][]byte),
		want:  addr([2][]byte{[]byte("hello"), []byte("goodbye")}),
	}, {
		name:  jsontest.Name("Arrays/Int"),
		inBuf: `[-2,-1,0,1,2]`,
		inVal: new([5]int),
		want:  addr([5]int{-2, -1, 0, 1, 2}),
	}, {
		name:  jsontest.Name("Arrays/Uint"),
		inBuf: `[0,1,2,3,4]`,
		inVal: new([5]uint),
		want:  addr([5]uint{0, 1, 2, 3, 4}),
	}, {
		name:  jsontest.Name("Arrays/Float"),
		inBuf: `[3.14159,12.34]`,
		inVal: new([2]float64),
		want:  addr([2]float64{3.14159, 12.34}),
	}, {
		// NOTE: The semantics differs from v1, where elements are not merged.
		// This is to maintain consistent merge semantics with slices.
		name:  jsontest.Name("Arrays/Merge"),
		inBuf: `[{"k3":"v3"},{"k4":"v4"}]`,
		inVal: addr([2]map[string]string{{"k1": "v1"}, {"k2": "v2"}}),
		want:  addr([2]map[string]string{{"k3": "v3"}, {"k4": "v4"}}),
	}, {
		name:    jsontest.Name("Arrays/Invalid/Channel"),
		inBuf:   `["hello"]`,
		inVal:   new([1]chan string),
		want:    new([1]chan string),
		wantErr: EU(nil).withPos(`[`, "/0").withType(0, T[chan string]()),
	}, {
		name:    jsontest.Name("Arrays/Invalid/Underflow"),
		inBuf:   `{"F":[   ]}`,
		inVal:   new(struct{ F [1]string }),
		want:    addr(struct{ F [1]string }{}),
		wantErr: EU(errArrayUnderflow).withPos(`{"F":[   `, "/F").withType(']', T[[1]string]()),
	}, {
		name:  jsontest.Name("Arrays/Invalid/Underflow/UnmarshalArrayFromAnyLength"),
		opts:  []Options{jsonflags.UnmarshalArrayFromAnyLength | 1},
		inBuf: `[-1,-2]`,
		inVal: addr([4]int{1, 2, 3, 4}),
		want:  addr([4]int{-1, -2, 0, 0}),
	}, {
		name:    jsontest.Name("Arrays/Invalid/Overflow"),
		inBuf:   `["1","2"]`,
		inVal:   new([1]string),
		want:    addr([1]string{"1"}),
		wantErr: EU(errArrayOverflow).withPos(`["1","2"`, "").withType(']', T[[1]string]()),
	}, {
		name:  jsontest.Name("Arrays/Invalid/Overflow/UnmarshalArrayFromAnyLength"),
		opts:  []Options{jsonflags.UnmarshalArrayFromAnyLength | 1},
		inBuf: `[-1,-2,-3,-4,-5,-6]`,
		inVal: addr([4]int{1, 2, 3, 4}),
		want:  addr([4]int{-1, -2, -3, -4}),
	}, {
		name:    jsontest.Name("Arrays/Invalid/Bool"),
		inBuf:   `true`,
		inVal:   addr([1]string{"nochange"}),
		want:    addr([1]string{"nochange"}),
		wantErr: EU(nil).withType('t', T[[1]string]()),
	}, {
		name:    jsontest.Name("Arrays/Invalid/String"),
		inBuf:   `""`,
		inVal:   addr([1]string{"nochange"}),
		want:    addr([1]string{"nochange"}),
		wantErr: EU(nil).withType('"', T[[1]string]()),
	}, {
		name:    jsontest.Name("Arrays/Invalid/Number"),
		inBuf:   `0`,
		inVal:   addr([1]string{"nochange"}),
		want:    addr([1]string{"nochange"}),
		wantErr: EU(nil).withType('0', T[[1]string]()),
	}, {
		name:    jsontest.Name("Arrays/Invalid/Object"),
		inBuf:   `{}`,
		inVal:   addr([1]string{"nochange"}),
		want:    addr([1]string{"nochange"}),
		wantErr: EU(nil).withType('{', T[[1]string]()),
	}, {
		name:  jsontest.Name("Arrays/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `[false,true]`,
		inVal: addr([2]bool{true, false}),
		want:  addr([2]bool{false, true}),
	}, {
		name:  jsontest.Name("Pointers/NullL0"),
		inBuf: `null`,
		inVal: new(*string),
		want:  addr((*string)(nil)),
	}, {
		name:  jsontest.Name("Pointers/NullL1"),
		inBuf: `null`,
		inVal: addr(new(*string)),
		want:  addr((**string)(nil)),
	}, {
		name:  jsontest.Name("Pointers/Bool"),
		inBuf: `true`,
		inVal: addr(new(bool)),
		want:  addr(addr(true)),
	}, {
		name:  jsontest.Name("Pointers/String"),
		inBuf: `"hello"`,
		inVal: addr(new(string)),
		want:  addr(addr("hello")),
	}, {
		name:  jsontest.Name("Pointers/Bytes"),
		inBuf: `"aGVsbG8="`,
		inVal: addr(new([]byte)),
		want:  addr(addr([]byte("hello"))),
	}, {
		name:  jsontest.Name("Pointers/Int"),
		inBuf: `-123`,
		inVal: addr(new(int)),
		want:  addr(addr(int(-123))),
	}, {
		name:  jsontest.Name("Pointers/Uint"),
		inBuf: `123`,
		inVal: addr(new(int)),
		want:  addr(addr(int(123))),
	}, {
		name:  jsontest.Name("Pointers/Float"),
		inBuf: `123.456`,
		inVal: addr(new(float64)),
		want:  addr(addr(float64(123.456))),
	}, {
		name:  jsontest.Name("Pointers/Allocate"),
		inBuf: `"hello"`,
		inVal: addr((*string)(nil)),
		want:  addr(addr("hello")),
	}, {
		name:  jsontest.Name("Points/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `true`,
		inVal: addr(new(bool)),
		want:  addr(addr(true)),
	}, {
		name:  jsontest.Name("Interfaces/Empty/Null"),
		inBuf: `null`,
		inVal: new(any),
		want:  new(any),
	}, {
		name:  jsontest.Name("Interfaces/NonEmpty/Null"),
		inBuf: `null`,
		inVal: new(io.Reader),
		want:  new(io.Reader),
	}, {
		name:    jsontest.Name("Interfaces/NonEmpty/Invalid"),
		inBuf:   `"hello"`,
		inVal:   new(io.Reader),
		want:    new(io.Reader),
		wantErr: EU(errNilInterface).withType(0, T[io.Reader]()),
	}, {
		name:  jsontest.Name("Interfaces/Empty/False"),
		inBuf: `false`,
		inVal: new(any),
		want: func() any {
			var vi any = false
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/Empty/True"),
		inBuf: `true`,
		inVal: new(any),
		want: func() any {
			var vi any = true
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/Empty/String"),
		inBuf: `"string"`,
		inVal: new(any),
		want: func() any {
			var vi any = "string"
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/Empty/Number"),
		inBuf: `3.14159`,
		inVal: new(any),
		want: func() any {
			var vi any = 3.14159
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/Empty/Object"),
		inBuf: `{"k":"v"}`,
		inVal: new(any),
		want: func() any {
			var vi any = map[string]any{"k": "v"}
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/Empty/Array"),
		inBuf: `["v"]`,
		inVal: new(any),
		want: func() any {
			var vi any = []any{"v"}
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/NamedAny/String"),
		inBuf: `"string"`,
		inVal: new(namedAny),
		want: func() namedAny {
			var vi namedAny = "string"
			return &vi
		}(),
	}, {
		name:    jsontest.Name("Interfaces/Invalid"),
		inBuf:   `]`,
		inVal:   new(any),
		want:    new(any),
		wantErr: newInvalidCharacterError("]", "at start of value", 0, ""),
	}, {
		// NOTE: The semantics differs from v1,
		// where existing map entries were not merged into.
		// See https://go.dev/issue/26946.
		// See https://go.dev/issue/33993.
		name:  jsontest.Name("Interfaces/Merge/Map"),
		inBuf: `{"k2":"v2"}`,
		inVal: func() any {
			var vi any = map[string]string{"k1": "v1"}
			return &vi
		}(),
		want: func() any {
			var vi any = map[string]string{"k1": "v1", "k2": "v2"}
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/Merge/Struct"),
		inBuf: `{"Array":["goodbye"]}`,
		inVal: func() any {
			var vi any = structAll{String: "hello"}
			return &vi
		}(),
		want: func() any {
			var vi any = structAll{String: "hello", Array: [1]string{"goodbye"}}
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/Merge/NamedInt"),
		inBuf: `64`,
		inVal: func() any {
			var vi any = namedInt64(-64)
			return &vi
		}(),
		want: func() any {
			var vi any = namedInt64(+64)
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `true`,
		inVal: new(any),
		want: func() any {
			var vi any = true
			return &vi
		}(),
	}, {
		name:  jsontest.Name("Interfaces/Any"),
		inBuf: `{"X":[null,false,true,"",0,{},[]]}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{[]any{nil, false, true, "", 0.0, map[string]any{}, []any{}}}),
	}, {
		name:  jsontest.Name("Interfaces/Any/Named"),
		inBuf: `{"X":[null,false,true,"",0,{},[]]}`,
		inVal: new(struct{ X namedAny }),
		want:  addr(struct{ X namedAny }{[]any{nil, false, true, "", 0.0, map[string]any{}, []any{}}}),
	}, {
		name:  jsontest.Name("Interfaces/Any/Stringified"),
		opts:  []Options{StringifyNumbers(true)},
		inBuf: `{"X":"0"}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{"0"}),
	}, {
		name: jsontest.Name("Interfaces/Any/UnmarshalFunc/Any"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *any) error {
				*v = "called"
				return nil
			})),
		},
		inBuf: `{"X":[null,false,true,"",0,{},[]]}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{"called"}),
	}, {
		name: jsontest.Name("Interfaces/Any/UnmarshalFunc/Bool"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *bool) error {
				*v = string(b) != "true"
				return nil
			})),
		},
		inBuf: `{"X":[null,false,true,"",0,{},[]]}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{[]any{nil, true, false, "", 0.0, map[string]any{}, []any{}}}),
	}, {
		name: jsontest.Name("Interfaces/Any/UnmarshalFunc/String"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *string) error {
				*v = "called"
				return nil
			})),
		},
		inBuf: `{"X":[null,false,true,"",0,{},[]]}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{[]any{nil, false, true, "called", 0.0, map[string]any{}, []any{}}}),
	}, {
		name: jsontest.Name("Interfaces/Any/UnmarshalFunc/Float64"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *float64) error {
				*v = 3.14159
				return nil
			})),
		},
		inBuf: `{"X":[null,false,true,"",0,{},[]]}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{[]any{nil, false, true, "", 3.14159, map[string]any{}, []any{}}}),
	}, {
		name: jsontest.Name("Interfaces/Any/UnmarshalFunc/MapStringAny"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *map[string]any) error {
				*v = map[string]any{"called": nil}
				return nil
			})),
		},
		inBuf: `{"X":[null,false,true,"",0,{},[]]}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{[]any{nil, false, true, "", 0.0, map[string]any{"called": nil}, []any{}}}),
	}, {
		name: jsontest.Name("Interfaces/Any/UnmarshalFunc/SliceAny"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *[]any) error {
				*v = []any{"called"}
				return nil
			})),
		},
		inBuf: `{"X":[null,false,true,"",0,{},[]]}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{[]any{"called"}}),
	}, {
		name:  jsontest.Name("Interfaces/Any/Maps/NonEmpty"),
		inBuf: `{"X":{"fizz":"buzz"}}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{map[string]any{"fizz": "buzz"}}),
	}, {
		name:    jsontest.Name("Interfaces/Any/Maps/RejectDuplicateNames"),
		inBuf:   `{"X":{"fizz":"buzz","fizz":true}}`,
		inVal:   new(struct{ X any }),
		want:    addr(struct{ X any }{map[string]any{"fizz": "buzz"}}),
		wantErr: newDuplicateNameError("/X", []byte(`"fizz"`), len64(`{"X":{"fizz":"buzz",`)),
	}, {
		name:    jsontest.Name("Interfaces/Any/Maps/AllowDuplicateNames"),
		opts:    []Options{jsontext.AllowDuplicateNames(true)},
		inBuf:   `{"X":{"fizz":"buzz","fizz":true}}`,
		inVal:   new(struct{ X any }),
		want:    addr(struct{ X any }{map[string]any{"fizz": "buzz"}}),
		wantErr: EU(nil).withPos(`{"X":{"fizz":"buzz","fizz":`, "/X/fizz").withType('t', T[string]()),
	}, {
		name:  jsontest.Name("Interfaces/Any/Slices/NonEmpty"),
		inBuf: `{"X":["fizz","buzz"]}`,
		inVal: new(struct{ X any }),
		want:  addr(struct{ X any }{[]any{"fizz", "buzz"}}),
	}, {
		name:  jsontest.Name("Methods/NilPointer/Null"),
		inBuf: `{"X":null}`,
		inVal: addr(struct{ X *allMethods }{X: (*allMethods)(nil)}),
		want:  addr(struct{ X *allMethods }{X: (*allMethods)(nil)}), // method should not be called
	}, {
		name:  jsontest.Name("Methods/NilPointer/Value"),
		inBuf: `{"X":"value"}`,
		inVal: addr(struct{ X *allMethods }{X: (*allMethods)(nil)}),
		want:  addr(struct{ X *allMethods }{X: &allMethods{method: "UnmarshalJSONFrom", value: []byte(`"value"`)}}),
	}, {
		name:  jsontest.Name("Methods/NilInterface/Null"),
		inBuf: `{"X":null}`,
		inVal: addr(struct{ X MarshalerTo }{X: (*allMethods)(nil)}),
		want:  addr(struct{ X MarshalerTo }{X: nil}), // interface value itself is nil'd out
	}, {
		name:  jsontest.Name("Methods/NilInterface/Value"),
		inBuf: `{"X":"value"}`,
		inVal: addr(struct{ X MarshalerTo }{X: (*allMethods)(nil)}),
		want:  addr(struct{ X MarshalerTo }{X: &allMethods{method: "UnmarshalJSONFrom", value: []byte(`"value"`)}}),
	}, {
		name:  jsontest.Name("Methods/AllMethods"),
		inBuf: `{"X":"hello"}`,
		inVal: new(struct{ X *allMethods }),
		want:  addr(struct{ X *allMethods }{X: &allMethods{method: "UnmarshalJSONFrom", value: []byte(`"hello"`)}}),
	}, {
		name:  jsontest.Name("Methods/AllMethodsExceptJSONv2"),
		inBuf: `{"X":"hello"}`,
		inVal: new(struct{ X *allMethodsExceptJSONv2 }),
		want:  addr(struct{ X *allMethodsExceptJSONv2 }{X: &allMethodsExceptJSONv2{allMethods: allMethods{method: "UnmarshalJSON", value: []byte(`"hello"`)}}}),
	}, {
		name:  jsontest.Name("Methods/AllMethodsExceptJSONv1"),
		inBuf: `{"X":"hello"}`,
		inVal: new(struct{ X *allMethodsExceptJSONv1 }),
		want:  addr(struct{ X *allMethodsExceptJSONv1 }{X: &allMethodsExceptJSONv1{allMethods: allMethods{method: "UnmarshalJSONFrom", value: []byte(`"hello"`)}}}),
	}, {
		name:  jsontest.Name("Methods/AllMethodsExceptText"),
		inBuf: `{"X":"hello"}`,
		inVal: new(struct{ X *allMethodsExceptText }),
		want:  addr(struct{ X *allMethodsExceptText }{X: &allMethodsExceptText{allMethods: allMethods{method: "UnmarshalJSONFrom", value: []byte(`"hello"`)}}}),
	}, {
		name:  jsontest.Name("Methods/OnlyMethodJSONv2"),
		inBuf: `{"X":"hello"}`,
		inVal: new(struct{ X *onlyMethodJSONv2 }),
		want:  addr(struct{ X *onlyMethodJSONv2 }{X: &onlyMethodJSONv2{allMethods: allMethods{method: "UnmarshalJSONFrom", value: []byte(`"hello"`)}}}),
	}, {
		name:  jsontest.Name("Methods/OnlyMethodJSONv1"),
		inBuf: `{"X":"hello"}`,
		inVal: new(struct{ X *onlyMethodJSONv1 }),
		want:  addr(struct{ X *onlyMethodJSONv1 }{X: &onlyMethodJSONv1{allMethods: allMethods{method: "UnmarshalJSON", value: []byte(`"hello"`)}}}),
	}, {
		name:  jsontest.Name("Methods/OnlyMethodText"),
		inBuf: `{"X":"hello"}`,
		inVal: new(struct{ X *onlyMethodText }),
		want:  addr(struct{ X *onlyMethodText }{X: &onlyMethodText{allMethods: allMethods{method: "UnmarshalText", value: []byte(`hello`)}}}),
	}, {
		name:  jsontest.Name("Methods/Text/Null"),
		inBuf: `{"X":null}`,
		inVal: addr(struct{ X unmarshalTextFunc }{unmarshalTextFunc(func(b []byte) error {
			return errMustNotCall
		})}),
		want: addr(struct{ X unmarshalTextFunc }{nil}),
	}, {
		name:  jsontest.Name("Methods/IP"),
		inBuf: `"192.168.0.100"`,
		inVal: new(net.IP),
		want:  addr(net.IPv4(192, 168, 0, 100)),
	}, {
		// NOTE: Fixes https://go.dev/issue/46516.
		name:  jsontest.Name("Methods/Anonymous"),
		inBuf: `{"X":"hello"}`,
		inVal: new(struct{ X struct{ allMethods } }),
		want:  addr(struct{ X struct{ allMethods } }{X: struct{ allMethods }{allMethods{method: "UnmarshalJSONFrom", value: []byte(`"hello"`)}}}),
	}, {
		// NOTE: Fixes https://go.dev/issue/22967.
		name:  jsontest.Name("Methods/Addressable"),
		inBuf: `{"V":"hello","M":{"K":"hello"},"I":"hello"}`,
		inVal: addr(struct {
			V allMethods
			M map[string]allMethods
			I any
		}{
			I: allMethods{}, // need to initialize with concrete value
		}),
		want: addr(struct {
			V allMethods
			M map[string]allMethods
			I any
		}{
			V: allMethods{method: "UnmarshalJSONFrom", value: []byte(`"hello"`)},
			M: map[string]allMethods{"K": {method: "UnmarshalJSONFrom", value: []byte(`"hello"`)}},
			I: allMethods{method: "UnmarshalJSONFrom", value: []byte(`"hello"`)},
		}),
	}, {
		// NOTE: Fixes https://go.dev/issue/29732.
		name:  jsontest.Name("Methods/MapKey/JSONv2"),
		inBuf: `{"k1":"v1b","k2":"v2"}`,
		inVal: addr(map[structMethodJSONv2]string{{"k1"}: "v1a", {"k3"}: "v3"}),
		want:  addr(map[structMethodJSONv2]string{{"k1"}: "v1b", {"k2"}: "v2", {"k3"}: "v3"}),
	}, {
		// NOTE: Fixes https://go.dev/issue/29732.
		name:  jsontest.Name("Methods/MapKey/JSONv1"),
		inBuf: `{"k1":"v1b","k2":"v2"}`,
		inVal: addr(map[structMethodJSONv1]string{{"k1"}: "v1a", {"k3"}: "v3"}),
		want:  addr(map[structMethodJSONv1]string{{"k1"}: "v1b", {"k2"}: "v2", {"k3"}: "v3"}),
	}, {
		name:  jsontest.Name("Methods/MapKey/Text"),
		inBuf: `{"k1":"v1b","k2":"v2"}`,
		inVal: addr(map[structMethodText]string{{"k1"}: "v1a", {"k3"}: "v3"}),
		want:  addr(map[structMethodText]string{{"k1"}: "v1b", {"k2"}: "v2", {"k3"}: "v3"}),
	}, {
		name:  jsontest.Name("Methods/Invalid/JSONv2/Error"),
		inBuf: `{}`,
		inVal: addr(unmarshalJSONv2Func(func(*jsontext.Decoder) error {
			return errSomeError
		})),
		wantErr: EU(errSomeError).withType(0, T[unmarshalJSONv2Func]()),
	}, {
		name: jsontest.Name("Methods/Invalid/JSONv2/TooFew"),
		inVal: addr(unmarshalJSONv2Func(func(*jsontext.Decoder) error {
			return nil // do nothing
		})),
		wantErr: EU(errNonSingularValue).withType(0, T[unmarshalJSONv2Func]()),
	}, {
		name:  jsontest.Name("Methods/Invalid/JSONv2/TooMany"),
		inBuf: `{}{}`,
		inVal: addr(unmarshalJSONv2Func(func(dec *jsontext.Decoder) error {
			dec.ReadValue()
			dec.ReadValue()
			return nil
		})),
		wantErr: EU(errNonSingularValue).withPos(`{}`, "").withType(0, T[unmarshalJSONv2Func]()),
	}, {
		name:  jsontest.Name("Methods/Invalid/JSONv2/SkipFunc"),
		inBuf: `{}`,
		inVal: addr(unmarshalJSONv2Func(func(*jsontext.Decoder) error {
			return SkipFunc
		})),
		wantErr: EU(wrapSkipFunc(SkipFunc, "unmarshal method")).withType(0, T[unmarshalJSONv2Func]()),
	}, {
		name:  jsontest.Name("Methods/Invalid/JSONv1/Error"),
		inBuf: `{}`,
		inVal: addr(unmarshalJSONv1Func(func([]byte) error {
			return errSomeError
		})),
		wantErr: EU(errSomeError).withType('{', T[unmarshalJSONv1Func]()),
	}, {
		name:  jsontest.Name("Methods/Invalid/JSONv1/SkipFunc"),
		inBuf: `{}`,
		inVal: addr(unmarshalJSONv1Func(func([]byte) error {
			return SkipFunc
		})),
		wantErr: EU(wrapSkipFunc(SkipFunc, "unmarshal method")).withType('{', T[unmarshalJSONv1Func]()),
	}, {
		name:  jsontest.Name("Methods/Invalid/Text/Error"),
		inBuf: `"value"`,
		inVal: addr(unmarshalTextFunc(func([]byte) error {
			return errSomeError
		})),
		wantErr: EU(errSomeError).withType('"', T[unmarshalTextFunc]()),
	}, {
		name:  jsontest.Name("Methods/Invalid/Text/Syntax"),
		inBuf: `{}`,
		inVal: addr(unmarshalTextFunc(func([]byte) error {
			panic("should not be called")
		})),
		wantErr: EU(errNonStringValue).withType('{', T[unmarshalTextFunc]()),
	}, {
		name:  jsontest.Name("Methods/Invalid/Text/SkipFunc"),
		inBuf: `"value"`,
		inVal: addr(unmarshalTextFunc(func([]byte) error {
			return SkipFunc
		})),
		wantErr: EU(wrapSkipFunc(SkipFunc, "unmarshal method")).withType('"', T[unmarshalTextFunc]()),
	}, {
		name: jsontest.Name("Functions/String/V1"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *string) error {
				if string(b) != `""` {
					return fmt.Errorf("got %s, want %s", b, `""`)
				}
				*v = "called"
				return nil
			})),
		},
		inBuf: `""`,
		inVal: addr(""),
		want:  addr("called"),
	}, {
		name:  jsontest.Name("Functions/String/Empty"),
		opts:  []Options{WithUnmarshalers(nil)},
		inBuf: `"hello"`,
		inVal: addr(""),
		want:  addr("hello"),
	}, {
		name: jsontest.Name("Functions/NamedString/V1/NoMatch"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *namedString) error {
				panic("should not be called")
			})),
		},
		inBuf: `""`,
		inVal: addr(""),
		want:  addr(""),
	}, {
		name: jsontest.Name("Functions/NamedString/V1/Match"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *namedString) error {
				if string(b) != `""` {
					return fmt.Errorf("got %s, want %s", b, `""`)
				}
				*v = "called"
				return nil
			})),
		},
		inBuf: `""`,
		inVal: addr(namedString("")),
		want:  addr(namedString("called")),
	}, {
		name: jsontest.Name("Functions/String/V2"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				switch b, err := dec.ReadValue(); {
				case err != nil:
					return err
				case string(b) != `""`:
					return fmt.Errorf("got %s, want %s", b, `""`)
				}
				*v = "called"
				return nil
			})),
		},
		inBuf: `""`,
		inVal: addr(""),
		want:  addr("called"),
	}, {
		name: jsontest.Name("Functions/NamedString/V2/NoMatch"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *namedString) error {
				panic("should not be called")
			})),
		},
		inBuf: `""`,
		inVal: addr(""),
		want:  addr(""),
	}, {
		name: jsontest.Name("Functions/NamedString/V2/Match"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *namedString) error {
				switch t, err := dec.ReadToken(); {
				case err != nil:
					return err
				case t.String() != ``:
					return fmt.Errorf("got %q, want %q", t, ``)
				}
				*v = "called"
				return nil
			})),
		},
		inBuf: `""`,
		inVal: addr(namedString("")),
		want:  addr(namedString("called")),
	}, {
		name: jsontest.Name("Functions/String/Empty1/NoMatch"),
		opts: []Options{
			WithUnmarshalers(new(Unmarshalers)),
		},
		inBuf: `""`,
		inVal: addr(""),
		want:  addr(""),
	}, {
		name: jsontest.Name("Functions/String/Empty2/NoMatch"),
		opts: []Options{
			WithUnmarshalers(JoinUnmarshalers()),
		},
		inBuf: `""`,
		inVal: addr(""),
		want:  addr(""),
	}, {
		name: jsontest.Name("Functions/String/V1/DirectError"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func([]byte, *string) error {
				return errSomeError
			})),
		},
		inBuf:   `""`,
		inVal:   addr(""),
		want:    addr(""),
		wantErr: EU(errSomeError).withType('"', reflect.PointerTo(stringType)),
	}, {
		name: jsontest.Name("Functions/String/V1/SkipError"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func([]byte, *string) error {
				return SkipFunc
			})),
		},
		inBuf:   `""`,
		inVal:   addr(""),
		want:    addr(""),
		wantErr: EU(wrapSkipFunc(SkipFunc, "unmarshal function of type func([]byte, T) error")).withType('"', reflect.PointerTo(stringType)),
	}, {
		name: jsontest.Name("Functions/String/V2/DirectError"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				return errSomeError
			})),
		},
		inBuf:   `""`,
		inVal:   addr(""),
		want:    addr(""),
		wantErr: EU(errSomeError).withType(0, reflect.PointerTo(stringType)),
	}, {
		name: jsontest.Name("Functions/String/V2/TooFew"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				return nil
			})),
		},
		inBuf:   `""`,
		inVal:   addr(""),
		want:    addr(""),
		wantErr: EU(errNonSingularValue).withType(0, reflect.PointerTo(stringType)),
	}, {
		name: jsontest.Name("Functions/String/V2/TooMany"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				if _, err := dec.ReadValue(); err != nil {
					return err
				}
				if _, err := dec.ReadValue(); err != nil {
					return err
				}
				return nil
			})),
		},
		inBuf:   `{"X":["",""]}`,
		inVal:   addr(struct{ X []string }{}),
		want:    addr(struct{ X []string }{[]string{""}}),
		wantErr: EU(errNonSingularValue).withPos(`{"X":["",`, "/X").withType(0, reflect.PointerTo(stringType)),
	}, {
		name: jsontest.Name("Functions/String/V2/Skipped"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				return SkipFunc
			})),
		},
		inBuf: `""`,
		inVal: addr(""),
		want:  addr(""),
	}, {
		name: jsontest.Name("Functions/String/V2/ProcessBeforeSkip"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				if _, err := dec.ReadValue(); err != nil {
					return err
				}
				return SkipFunc
			})),
		},
		inBuf:   `""`,
		inVal:   addr(""),
		want:    addr(""),
		wantErr: EU(errSkipMutation).withType(0, reflect.PointerTo(stringType)),
	}, {
		name: jsontest.Name("Functions/String/V2/WrappedSkipError"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				return fmt.Errorf("wrap: %w", SkipFunc)
			})),
		},
		inBuf:   `""`,
		inVal:   addr(""),
		want:    addr(""),
		wantErr: EU(fmt.Errorf("wrap: %w", SkipFunc)).withType(0, reflect.PointerTo(stringType)),
	}, {
		name: jsontest.Name("Functions/Map/Key/NoCaseString/V1"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *nocaseString) error {
				if string(b) != `"hello"` {
					return fmt.Errorf("got %s, want %s", b, `"hello"`)
				}
				*v = "called"
				return nil
			})),
		},
		inBuf: `{"hello":"world"}`,
		inVal: addr(map[nocaseString]string{}),
		want:  addr(map[nocaseString]string{"called": "world"}),
	}, {
		name: jsontest.Name("Functions/Map/Key/TextMarshaler/V1"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v encoding.TextMarshaler) error {
				if string(b) != `"hello"` {
					return fmt.Errorf("got %s, want %s", b, `"hello"`)
				}
				*v.(*nocaseString) = "called"
				return nil
			})),
		},
		inBuf: `{"hello":"world"}`,
		inVal: addr(map[nocaseString]string{}),
		want:  addr(map[nocaseString]string{"called": "world"}),
	}, {
		name: jsontest.Name("Functions/Map/Key/NoCaseString/V2"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *nocaseString) error {
				switch t, err := dec.ReadToken(); {
				case err != nil:
					return err
				case t.String() != "hello":
					return fmt.Errorf("got %q, want %q", t, "hello")
				}
				*v = "called"
				return nil
			})),
		},
		inBuf: `{"hello":"world"}`,
		inVal: addr(map[nocaseString]string{}),
		want:  addr(map[nocaseString]string{"called": "world"}),
	}, {
		name: jsontest.Name("Functions/Map/Key/TextMarshaler/V2"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v encoding.TextMarshaler) error {
				switch b, err := dec.ReadValue(); {
				case err != nil:
					return err
				case string(b) != `"hello"`:
					return fmt.Errorf("got %s, want %s", b, `"hello"`)
				}
				*v.(*nocaseString) = "called"
				return nil
			})),
		},
		inBuf: `{"hello":"world"}`,
		inVal: addr(map[nocaseString]string{}),
		want:  addr(map[nocaseString]string{"called": "world"}),
	}, {
		name: jsontest.Name("Functions/Map/Key/String/V1/DuplicateName"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				if _, err := dec.ReadValue(); err != nil {
					return err
				}
				xd := export.Decoder(dec)
				*v = fmt.Sprintf("%d-%d", len(xd.Tokens.Stack), xd.Tokens.Last.Length())
				return nil
			})),
		},
		inBuf:   `{"name":"value","name":"value"}`,
		inVal:   addr(map[string]string{}),
		want:    addr(map[string]string{"1-1": "1-2"}),
		wantErr: newDuplicateNameError("", []byte(`"name"`), len64(`{"name":"value",`)),
	}, {
		name: jsontest.Name("Functions/Map/Value/NoCaseString/V1"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *nocaseString) error {
				if string(b) != `"world"` {
					return fmt.Errorf("got %s, want %s", b, `"world"`)
				}
				*v = "called"
				return nil
			})),
		},
		inBuf: `{"hello":"world"}`,
		inVal: addr(map[string]nocaseString{}),
		want:  addr(map[string]nocaseString{"hello": "called"}),
	}, {
		name: jsontest.Name("Functions/Map/Value/TextMarshaler/V1"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v encoding.TextMarshaler) error {
				if string(b) != `"world"` {
					return fmt.Errorf("got %s, want %s", b, `"world"`)
				}
				*v.(*nocaseString) = "called"
				return nil
			})),
		},
		inBuf: `{"hello":"world"}`,
		inVal: addr(map[string]nocaseString{}),
		want:  addr(map[string]nocaseString{"hello": "called"}),
	}, {
		name: jsontest.Name("Functions/Map/Value/NoCaseString/V2"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *nocaseString) error {
				switch t, err := dec.ReadToken(); {
				case err != nil:
					return err
				case t.String() != "world":
					return fmt.Errorf("got %q, want %q", t, "world")
				}
				*v = "called"
				return nil
			})),
		},
		inBuf: `{"hello":"world"}`,
		inVal: addr(map[string]nocaseString{}),
		want:  addr(map[string]nocaseString{"hello": "called"}),
	}, {
		name: jsontest.Name("Functions/Map/Value/TextMarshaler/V2"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v encoding.TextMarshaler) error {
				switch b, err := dec.ReadValue(); {
				case err != nil:
					return err
				case string(b) != `"world"`:
					return fmt.Errorf("got %s, want %s", b, `"world"`)
				}
				*v.(*nocaseString) = "called"
				return nil
			})),
		},
		inBuf: `{"hello":"world"}`,
		inVal: addr(map[string]nocaseString{}),
		want:  addr(map[string]nocaseString{"hello": "called"}),
	}, {
		name: jsontest.Name("Funtions/Struct/Fields"),
		opts: []Options{
			WithUnmarshalers(JoinUnmarshalers(
				UnmarshalFunc(func(b []byte, v *bool) error {
					if string(b) != `"called1"` {
						return fmt.Errorf("got %s, want %s", b, `"called1"`)
					}
					*v = true
					return nil
				}),
				UnmarshalFunc(func(b []byte, v *string) error {
					if string(b) != `"called2"` {
						return fmt.Errorf("got %s, want %s", b, `"called2"`)
					}
					*v = "called2"
					return nil
				}),
				UnmarshalFromFunc(func(dec *jsontext.Decoder, v *[]byte) error {
					switch t, err := dec.ReadToken(); {
					case err != nil:
						return err
					case t.String() != "called3":
						return fmt.Errorf("got %q, want %q", t, "called3")
					}
					*v = []byte("called3")
					return nil
				}),
				UnmarshalFromFunc(func(dec *jsontext.Decoder, v *int64) error {
					switch b, err := dec.ReadValue(); {
					case err != nil:
						return err
					case string(b) != `"called4"`:
						return fmt.Errorf("got %s, want %s", b, `"called4"`)
					}
					*v = 123
					return nil
				}),
			)),
		},
		inBuf: `{"Bool":"called1","String":"called2","Bytes":"called3","Int":"called4","Uint":456,"Float":789}`,
		inVal: addr(structScalars{}),
		want:  addr(structScalars{Bool: true, String: "called2", Bytes: []byte("called3"), Int: 123, Uint: 456, Float: 789}),
	}, {
		name: jsontest.Name("Functions/Struct/Inlined"),
		opts: []Options{
			WithUnmarshalers(JoinUnmarshalers(
				UnmarshalFunc(func([]byte, *structInlinedL1) error {
					panic("should not be called")
				}),
				UnmarshalFromFunc(func(dec *jsontext.Decoder, v *StructEmbed2) error {
					panic("should not be called")
				}),
			)),
		},
		inBuf: `{"E":"E3","F":"F3","G":"G3","A":"A1","B":"B1","D":"D2"}`,
		inVal: new(structInlined),
		want: addr(structInlined{
			X: structInlinedL1{
				X:            &structInlinedL2{A: "A1", B: "B1" /* C: "C1" */},
				StructEmbed1: StructEmbed1{ /* C: "C2" */ D: "D2" /* E: "E2" */},
			},
			StructEmbed2: &StructEmbed2{E: "E3", F: "F3", G: "G3"},
		}),
	}, {
		name: jsontest.Name("Functions/Slice/Elem"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *string) error {
				*v = strings.Trim(strings.ToUpper(string(b)), `"`)
				return nil
			})),
		},
		inBuf: `["hello","World"]`,
		inVal: addr([]string{}),
		want:  addr([]string{"HELLO", "WORLD"}),
	}, {
		name: jsontest.Name("Functions/Array/Elem"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFunc(func(b []byte, v *string) error {
				*v = strings.Trim(strings.ToUpper(string(b)), `"`)
				return nil
			})),
		},
		inBuf: `["hello","World"]`,
		inVal: addr([2]string{}),
		want:  addr([2]string{"HELLO", "WORLD"}),
	}, {
		name: jsontest.Name("Functions/Pointer/Nil"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				t, err := dec.ReadToken()
				*v = strings.ToUpper(t.String())
				return err
			})),
		},
		inBuf: `{"X":"hello"}`,
		inVal: addr(struct{ X *string }{nil}),
		want:  addr(struct{ X *string }{addr("HELLO")}),
	}, {
		name: jsontest.Name("Functions/Pointer/NonNil"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
				t, err := dec.ReadToken()
				*v = strings.ToUpper(t.String())
				return err
			})),
		},
		inBuf: `{"X":"hello"}`,
		inVal: addr(struct{ X *string }{addr("")}),
		want:  addr(struct{ X *string }{addr("HELLO")}),
	}, {
		name: jsontest.Name("Functions/Interface/Nil"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v fmt.Stringer) error {
				panic("should not be called")
			})),
		},
		inBuf:   `{"X":"hello"}`,
		inVal:   addr(struct{ X fmt.Stringer }{nil}),
		want:    addr(struct{ X fmt.Stringer }{nil}),
		wantErr: EU(errNilInterface).withPos(`{"X":`, "/X").withType(0, T[fmt.Stringer]()),
	}, {
		name: jsontest.Name("Functions/Interface/NetIP"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *fmt.Stringer) error {
				*v = net.IP{}
				return SkipFunc
			})),
		},
		inBuf: `{"X":"1.1.1.1"}`,
		inVal: addr(struct{ X fmt.Stringer }{nil}),
		want:  addr(struct{ X fmt.Stringer }{net.IPv4(1, 1, 1, 1)}),
	}, {
		name: jsontest.Name("Functions/Interface/NewPointerNetIP"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *fmt.Stringer) error {
				*v = new(net.IP)
				return SkipFunc
			})),
		},
		inBuf: `{"X":"1.1.1.1"}`,
		inVal: addr(struct{ X fmt.Stringer }{nil}),
		want:  addr(struct{ X fmt.Stringer }{addr(net.IPv4(1, 1, 1, 1))}),
	}, {
		name: jsontest.Name("Functions/Interface/NilPointerNetIP"),
		opts: []Options{
			WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, v *fmt.Stringer) error {
				*v = (*net.IP)(nil)
				return SkipFunc
			})),
		},
		inBuf: `{"X":"1.1.1.1"}`,
		inVal: addr(struct{ X fmt.Stringer }{nil}),
		want:  addr(struct{ X fmt.Stringer }{addr(net.IPv4(1, 1, 1, 1))}),
	}, {
		name: jsontest.Name("Functions/Interface/NilPointerNetIP/Override"),
		opts: []Options{
			WithUnmarshalers(JoinUnmarshalers(
				UnmarshalFromFunc(func(dec *jsontext.Decoder, v *fmt.Stringer) error {
					*v = (*net.IP)(nil)
					return SkipFunc
				}),
				UnmarshalFunc(func(b []byte, v *net.IP) error {
					b = bytes.ReplaceAll(b, []byte(`1`), []byte(`8`))
					return v.UnmarshalText(bytes.Trim(b, `"`))
				}),
			)),
		},
		inBuf: `{"X":"1.1.1.1"}`,
		inVal: addr(struct{ X fmt.Stringer }{nil}),
		want:  addr(struct{ X fmt.Stringer }{addr(net.IPv4(8, 8, 8, 8))}),
	}, {
		name:  jsontest.Name("Functions/Interface/Any"),
		inBuf: `[null,{},{},{},{},{},{},{},{},{},{},{},{},"LAST"]`,
		inVal: addr([...]any{
			nil,                           // nil
			valueStringer{},               // T
			(*valueStringer)(nil),         // *T
			addr(valueStringer{}),         // *T
			(**valueStringer)(nil),        // **T
			addr((*valueStringer)(nil)),   // **T
			addr(addr(valueStringer{})),   // **T
			pointerStringer{},             // T
			(*pointerStringer)(nil),       // *T
			addr(pointerStringer{}),       // *T
			(**pointerStringer)(nil),      // **T
			addr((*pointerStringer)(nil)), // **T
			addr(addr(pointerStringer{})), // **T
			"LAST",
		}),
		opts: []Options{
			WithUnmarshalers(func() *Unmarshalers {
				type P struct {
					D int
					N int64
				}
				type PV struct {
					P P
					V any
				}

				var lastChecks []func() error
				checkLast := func() error {
					for _, fn := range lastChecks {
						if err := fn(); err != nil {
							return err
						}
					}
					return SkipFunc
				}
				makeValueChecker := func(name string, want []PV) func(d *jsontext.Decoder, v any) error {
					checkNext := func(d *jsontext.Decoder, v any) error {
						xd := export.Decoder(d)
						p := P{len(xd.Tokens.Stack), xd.Tokens.Last.Length()}
						rv := reflect.ValueOf(v)
						pv := PV{p, v}
						switch {
						case len(want) == 0:
							return fmt.Errorf("%s: %v: got more values than expected", name, p)
						case !rv.IsValid() || rv.Kind() != reflect.Pointer || rv.IsNil():
							return fmt.Errorf("%s: %v: got %#v, want non-nil pointer type", name, p, v)
						case !reflect.DeepEqual(pv, want[0]):
							return fmt.Errorf("%s:\n\tgot  %#v\n\twant %#v", name, pv, want[0])
						default:
							want = want[1:]
							return SkipFunc
						}
					}
					lastChecks = append(lastChecks, func() error {
						if len(want) > 0 {
							return fmt.Errorf("%s: did not get enough values, want %d more", name, len(want))
						}
						return nil
					})
					return checkNext
				}
				makePositionChecker := func(name string, want []P) func(d *jsontext.Decoder, v any) error {
					checkNext := func(d *jsontext.Decoder, v any) error {
						xd := export.Decoder(d)
						p := P{len(xd.Tokens.Stack), xd.Tokens.Last.Length()}
						switch {
						case len(want) == 0:
							return fmt.Errorf("%s: %v: got more values than wanted", name, p)
						case p != want[0]:
							return fmt.Errorf("%s: got %v, want %v", name, p, want[0])
						default:
							want = want[1:]
							return SkipFunc
						}
					}
					lastChecks = append(lastChecks, func() error {
						if len(want) > 0 {
							return fmt.Errorf("%s: did not get enough values, want %d more", name, len(want))
						}
						return nil
					})
					return checkNext
				}

				// In contrast to marshal, unmarshal automatically allocates for
				// nil pointers, which causes unmarshal to visit more values.
				wantAny := []PV{
					{P{1, 0}, addr(any(nil))},
					{P{1, 1}, addr(any(valueStringer{}))},
					{P{1, 1}, addr(valueStringer{})},
					{P{1, 2}, addr(any((*valueStringer)(nil)))},
					{P{1, 2}, addr((*valueStringer)(nil))},
					{P{1, 2}, addr(valueStringer{})},
					{P{1, 3}, addr(any(addr(valueStringer{})))},
					{P{1, 3}, addr(addr(valueStringer{}))},
					{P{1, 3}, addr(valueStringer{})},
					{P{1, 4}, addr(any((**valueStringer)(nil)))},
					{P{1, 4}, addr((**valueStringer)(nil))},
					{P{1, 4}, addr((*valueStringer)(nil))},
					{P{1, 4}, addr(valueStringer{})},
					{P{1, 5}, addr(any(addr((*valueStringer)(nil))))},
					{P{1, 5}, addr(addr((*valueStringer)(nil)))},
					{P{1, 5}, addr((*valueStringer)(nil))},
					{P{1, 5}, addr(valueStringer{})},
					{P{1, 6}, addr(any(addr(addr(valueStringer{}))))},
					{P{1, 6}, addr(addr(addr(valueStringer{})))},
					{P{1, 6}, addr(addr(valueStringer{}))},
					{P{1, 6}, addr(valueStringer{})},
					{P{1, 7}, addr(any(pointerStringer{}))},
					{P{1, 7}, addr(pointerStringer{})},
					{P{1, 8}, addr(any((*pointerStringer)(nil)))},
					{P{1, 8}, addr((*pointerStringer)(nil))},
					{P{1, 8}, addr(pointerStringer{})},
					{P{1, 9}, addr(any(addr(pointerStringer{})))},
					{P{1, 9}, addr(addr(pointerStringer{}))},
					{P{1, 9}, addr(pointerStringer{})},
					{P{1, 10}, addr(any((**pointerStringer)(nil)))},
					{P{1, 10}, addr((**pointerStringer)(nil))},
					{P{1, 10}, addr((*pointerStringer)(nil))},
					{P{1, 10}, addr(pointerStringer{})},
					{P{1, 11}, addr(any(addr((*pointerStringer)(nil))))},
					{P{1, 11}, addr(addr((*pointerStringer)(nil)))},
					{P{1, 11}, addr((*pointerStringer)(nil))},
					{P{1, 11}, addr(pointerStringer{})},
					{P{1, 12}, addr(any(addr(addr(pointerStringer{}))))},
					{P{1, 12}, addr(addr(addr(pointerStringer{})))},
					{P{1, 12}, addr(addr(pointerStringer{}))},
					{P{1, 12}, addr(pointerStringer{})},
					{P{1, 13}, addr(any("LAST"))},
					{P{1, 13}, addr("LAST")},
				}
				checkAny := makeValueChecker("any", wantAny)
				anyUnmarshaler := UnmarshalFromFunc(func(dec *jsontext.Decoder, v any) error {
					return checkAny(dec, v)
				})

				var wantPointerAny []PV
				for _, v := range wantAny {
					if _, ok := v.V.(*any); ok {
						wantPointerAny = append(wantPointerAny, v)
					}
				}
				checkPointerAny := makeValueChecker("*any", wantPointerAny)
				pointerAnyUnmarshaler := UnmarshalFromFunc(func(dec *jsontext.Decoder, v *any) error {
					return checkPointerAny(dec, v)
				})

				checkNamedAny := makeValueChecker("namedAny", wantAny)
				namedAnyUnmarshaler := UnmarshalFromFunc(func(dec *jsontext.Decoder, v namedAny) error {
					return checkNamedAny(dec, v)
				})

				checkPointerNamedAny := makeValueChecker("*namedAny", nil)
				pointerNamedAnyUnmarshaler := UnmarshalFromFunc(func(dec *jsontext.Decoder, v *namedAny) error {
					return checkPointerNamedAny(dec, v)
				})

				type stringer = fmt.Stringer
				var wantStringer []PV
				for _, v := range wantAny {
					if _, ok := v.V.(stringer); ok {
						wantStringer = append(wantStringer, v)
					}
				}
				checkStringer := makeValueChecker("stringer", wantStringer)
				stringerUnmarshaler := UnmarshalFromFunc(func(dec *jsontext.Decoder, v stringer) error {
					return checkStringer(dec, v)
				})

				checkPointerStringer := makeValueChecker("*stringer", nil)
				pointerStringerUnmarshaler := UnmarshalFromFunc(func(dec *jsontext.Decoder, v *stringer) error {
					return checkPointerStringer(dec, v)
				})

				wantValueStringer := []P{{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}}
				checkPointerValueStringer := makePositionChecker("*valueStringer", wantValueStringer)
				pointerValueStringerUnmarshaler := UnmarshalFromFunc(func(dec *jsontext.Decoder, v *valueStringer) error {
					return checkPointerValueStringer(dec, v)
				})

				wantPointerStringer := []P{{1, 7}, {1, 8}, {1, 9}, {1, 10}, {1, 11}, {1, 12}}
				checkPointerPointerStringer := makePositionChecker("*pointerStringer", wantPointerStringer)
				pointerPointerStringerUnmarshaler := UnmarshalFromFunc(func(dec *jsontext.Decoder, v *pointerStringer) error {
					return checkPointerPointerStringer(dec, v)
				})

				lastUnmarshaler := UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
					return checkLast()
				})

				return JoinUnmarshalers(
					// This is just like unmarshaling into a Go array,
					// but avoids zeroing the element before calling unmarshal.
					UnmarshalFromFunc(func(dec *jsontext.Decoder, v *[14]any) error {
						if _, err := dec.ReadToken(); err != nil {
							return err
						}
						for i := range len(*v) {
							if err := UnmarshalDecode(dec, &(*v)[i]); err != nil {
								return err
							}
						}
						if _, err := dec.ReadToken(); err != nil {
							return err
						}
						return nil
					}),

					anyUnmarshaler,
					pointerAnyUnmarshaler,
					namedAnyUnmarshaler,
					pointerNamedAnyUnmarshaler, // never called
					stringerUnmarshaler,
					pointerStringerUnmarshaler, // never called
					pointerValueStringerUnmarshaler,
					pointerPointerStringerUnmarshaler,
					lastUnmarshaler,
				)
			}()),
		},
	}, {
		name: jsontest.Name("Functions/Precedence/V1First"),
		opts: []Options{
			WithUnmarshalers(JoinUnmarshalers(
				UnmarshalFunc(func(b []byte, v *string) error {
					if string(b) != `"called"` {
						return fmt.Errorf("got %s, want %s", b, `"called"`)
					}
					*v = "called"
					return nil
				}),
				UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
					panic("should not be called")
				}),
			)),
		},
		inBuf: `"called"`,
		inVal: addr(""),
		want:  addr("called"),
	}, {
		name: jsontest.Name("Functions/Precedence/V2First"),
		opts: []Options{
			WithUnmarshalers(JoinUnmarshalers(
				UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
					switch t, err := dec.ReadToken(); {
					case err != nil:
						return err
					case t.String() != "called":
						return fmt.Errorf("got %q, want %q", t, "called")
					}
					*v = "called"
					return nil
				}),
				UnmarshalFunc(func([]byte, *string) error {
					panic("should not be called")
				}),
			)),
		},
		inBuf: `"called"`,
		inVal: addr(""),
		want:  addr("called"),
	}, {
		name: jsontest.Name("Functions/Precedence/V2Skipped"),
		opts: []Options{
			WithUnmarshalers(JoinUnmarshalers(
				UnmarshalFromFunc(func(dec *jsontext.Decoder, v *string) error {
					return SkipFunc
				}),
				UnmarshalFunc(func(b []byte, v *string) error {
					if string(b) != `"called"` {
						return fmt.Errorf("got %s, want %s", b, `"called"`)
					}
					*v = "called"
					return nil
				}),
			)),
		},
		inBuf: `"called"`,
		inVal: addr(""),
		want:  addr("called"),
	}, {
		name: jsontest.Name("Functions/Precedence/NestedFirst"),
		opts: []Options{
			WithUnmarshalers(JoinUnmarshalers(
				JoinUnmarshalers(
					UnmarshalFunc(func(b []byte, v *string) error {
						if string(b) != `"called"` {
							return fmt.Errorf("got %s, want %s", b, `"called"`)
						}
						*v = "called"
						return nil
					}),
				),
				UnmarshalFunc(func([]byte, *string) error {
					panic("should not be called")
				}),
			)),
		},
		inBuf: `"called"`,
		inVal: addr(""),
		want:  addr("called"),
	}, {
		name: jsontest.Name("Functions/Precedence/NestedLast"),
		opts: []Options{
			WithUnmarshalers(JoinUnmarshalers(
				UnmarshalFunc(func(b []byte, v *string) error {
					if string(b) != `"called"` {
						return fmt.Errorf("got %s, want %s", b, `"called"`)
					}
					*v = "called"
					return nil
				}),
				JoinUnmarshalers(
					UnmarshalFunc(func([]byte, *string) error {
						panic("should not be called")
					}),
				),
			)),
		},
		inBuf: `"called"`,
		inVal: addr(""),
		want:  addr("called"),
	}, {
		name:  jsontest.Name("Duration/Null"),
		inBuf: `{"D1":null,"D2":null}`,
		inVal: addr(struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}{1, 1}),
		want: addr(struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}{0, 0}),
	}, {
		name:  jsontest.Name("Duration/Zero"),
		inBuf: `{"D1":"0s","D2":0}`,
		inVal: addr(struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}{1, 1}),
		want: addr(struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}{0, 0}),
	}, {
		name:  jsontest.Name("Duration/Positive"),
		inBuf: `{"D1":"34293h33m9.123456789s","D2":123456789123456789}`,
		inVal: new(struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}),
		want: addr(struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}{
			123456789123456789,
			123456789123456789,
		}),
	}, {
		name:  jsontest.Name("Duration/Negative"),
		inBuf: `{"D1":"-34293h33m9.123456789s","D2":-123456789123456789}`,
		inVal: new(struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}),
		want: addr(struct {
			D1 time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
			D2 time.Duration `json:",format:nano"`
		}{
			-123456789123456789,
			-123456789123456789,
		}),
	}, {
		name:  jsontest.Name("Duration/Nanos/String"),
		inBuf: `{"D":"12345"}`,
		inVal: addr(struct {
			D time.Duration `json:",string,format:nano"`
		}{1}),
		want: addr(struct {
			D time.Duration `json:",string,format:nano"`
		}{12345}),
	}, {
		name:  jsontest.Name("Duration/Nanos/String/Invalid"),
		inBuf: `{"D":"+12345"}`,
		inVal: addr(struct {
			D time.Duration `json:",string,format:nano"`
		}{1}),
		want: addr(struct {
			D time.Duration `json:",string,format:nano"`
		}{1}),
		wantErr: EU(fmt.Errorf(`invalid duration "+12345": %w`, strconv.ErrSyntax)).withPos(`{"D":`, "/D").withType('"', timeDurationType),
	}, {
		name:  jsontest.Name("Duration/Nanos/Mismatch"),
		inBuf: `{"D":"34293h33m9.123456789s"}`,
		inVal: addr(struct {
			D time.Duration `json:",format:nano"`
		}{1}),
		want: addr(struct {
			D time.Duration `json:",format:nano"`
		}{1}),
		wantErr: EU(nil).withPos(`{"D":`, "/D").withType('"', timeDurationType),
	}, {
		name:  jsontest.Name("Duration/Nanos"),
		inBuf: `{"D":1.324}`,
		inVal: addr(struct {
			D time.Duration `json:",format:nano"`
		}{-1}),
		want: addr(struct {
			D time.Duration `json:",format:nano"`
		}{1}),
	}, {
		name:  jsontest.Name("Duration/String/Mismatch"),
		inBuf: `{"D":-123456789123456789}`,
		inVal: addr(struct {
			D time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
		}{1}),
		want: addr(struct {
			D time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
		}{1}),
		wantErr: EU(nil).withPos(`{"D":`, "/D").withType('0', timeDurationType),
	}, {
		name:  jsontest.Name("Duration/String/Invalid"),
		inBuf: `{"D":"5minkutes"}`,
		inVal: addr(struct {
			D time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
		}{1}),
		want: addr(struct {
			D time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
		}{1}),
		wantErr: EU(func() error {
			_, err := time.ParseDuration("5minkutes")
			return err
		}()).withPos(`{"D":`, "/D").withType('"', timeDurationType),
	}, {
		name:  jsontest.Name("Duration/Syntax/Invalid"),
		inBuf: `{"D":x}`,
		inVal: addr(struct {
			D time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
		}{1}),
		want: addr(struct {
			D time.Duration `json:",format:units"` // TODO(https://go.dev/issue/71631): Remove the format flag.
		}{1}),
		wantErr: newInvalidCharacterError("x", "at start of value", len64(`{"D":`), "/D"),
	}, {
		name: jsontest.Name("Duration/Format"),
		inBuf: `{
			"D1": "12h34m56.078090012s",
			"D2": "12h34m56.078090012s",
			"D3": 45296.078090012,
			"D4": "45296.078090012",
			"D5": 45296078.090012,
			"D6": "45296078.090012",
			"D7": 45296078090.012,
			"D8": "45296078090.012",
			"D9": 45296078090012,
			"D10": "45296078090012",
			"D11": "PT12H34M56.078090012S"
        }`,
		inVal: new(structDurationFormat),
		want: addr(structDurationFormat{
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
		}),
	}, {
		name:  jsontest.Name("Duration/Format/Invalid"),
		inBuf: `{"D":"0s"}`,
		inVal: addr(struct {
			D time.Duration `json:",format:invalid"`
		}{1}),
		want: addr(struct {
			D time.Duration `json:",format:invalid"`
		}{1}),
		wantErr: EU(errInvalidFormatFlag).withPos(`{"D":`, "/D").withType(0, timeDurationType),
	}, {
		/* TODO(https://go.dev/issue/71631): Re-enable this test case.
		name:  jsontest.Name("Duration/Format/Legacy"),
		inBuf: `{"D1":45296078090012,"D2":"12h34m56.078090012s"}`,
		opts:  []Options{jsonflags.FormatTimeWithLegacySemantics | 1},
		inVal: new(structDurationFormat),
		want: addr(structDurationFormat{
			D1: 12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
			D2: 12*time.Hour + 34*time.Minute + 56*time.Second + 78*time.Millisecond + 90*time.Microsecond + 12*time.Nanosecond,
		}),
		}, { */
		/* TODO(https://go.dev/issue/71631): Re-enable this test case.
		name:  jsontest.Name("Duration/MapKey"),
		inBuf: `{"1s":""}`,
		inVal: new(map[time.Duration]string),
		want:  addr(map[time.Duration]string{time.Second: ""}),
		}, { */
		name:  jsontest.Name("Duration/MapKey/Legacy"),
		opts:  []Options{jsonflags.FormatTimeWithLegacySemantics | 1},
		inBuf: `{"1000000000":""}`,
		inVal: new(map[time.Duration]string),
		want:  addr(map[time.Duration]string{time.Second: ""}),
	}, {
		/* TODO(https://go.dev/issue/71631): Re-enable this test case.
		name:  jsontest.Name("Duration/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `"1s"`,
		inVal: addr(time.Duration(0)),
		want:  addr(time.Second),
		}, { */
		name:  jsontest.Name("Time/Zero"),
		inBuf: `{"T1":"0001-01-01T00:00:00Z","T2":"01 Jan 01 00:00 UTC","T3":"0001-01-01","T4":"0001-01-01T00:00:00Z","T5":"0001-01-01T00:00:00Z"}`,
		inVal: new(struct {
			T1 time.Time
			T2 time.Time `json:",format:RFC822"`
			T3 time.Time `json:",format:'2006-01-02'"`
			T4 time.Time `json:",omitzero"`
			T5 time.Time `json:",omitempty"`
		}),
		want: addr(struct {
			T1 time.Time
			T2 time.Time `json:",format:RFC822"`
			T3 time.Time `json:",format:'2006-01-02'"`
			T4 time.Time `json:",omitzero"`
			T5 time.Time `json:",omitempty"`
		}{
			mustParseTime(time.RFC3339Nano, "0001-01-01T00:00:00Z"),
			mustParseTime(time.RFC822, "01 Jan 01 00:00 UTC"),
			mustParseTime("2006-01-02", "0001-01-01"),
			mustParseTime(time.RFC3339Nano, "0001-01-01T00:00:00Z"),
			mustParseTime(time.RFC3339Nano, "0001-01-01T00:00:00Z"),
		}),
	}, {
		name: jsontest.Name("Time/Format"),
		inBuf: `{
			"T1": "1234-01-02T03:04:05.000000006Z",
			"T2": "Mon Jan  2 03:04:05 1234",
			"T3": "Mon Jan  2 03:04:05 UTC 1234",
			"T4": "Mon Jan 02 03:04:05 +0000 1234",
			"T5": "02 Jan 34 03:04 UTC",
			"T6": "02 Jan 34 03:04 +0000",
			"T7": "Monday, 02-Jan-34 03:04:05 UTC",
			"T8": "Mon, 02 Jan 1234 03:04:05 UTC",
			"T9": "Mon, 02 Jan 1234 03:04:05 +0000",
			"T10": "1234-01-02T03:04:05Z",
			"T11": "1234-01-02T03:04:05.000000006Z",
			"T12": "3:04AM",
			"T13": "Jan  2 03:04:05",
			"T14": "Jan  2 03:04:05.000",
			"T15": "Jan  2 03:04:05.000000",
			"T16": "Jan  2 03:04:05.000000006",
			"T17": "1234-01-02 03:04:05",
			"T18": "1234-01-02",
			"T19": "03:04:05",
			"T20": "1234-01-02",
			"T21": "\"weird\"1234",
			"T22": -23225777754.999999994,
			"T23": "-23225777754.999999994",
			"T24": -23225777754999.999994,
			"T25": "-23225777754999.999994",
			"T26": -23225777754999999.994,
			"T27": "-23225777754999999.994",
			"T28": -23225777754999999994,
			"T29": "-23225777754999999994"
		}`,
		inVal: new(structTimeFormat),
		want: addr(structTimeFormat{
			mustParseTime(time.RFC3339Nano, "1234-01-02T03:04:05.000000006Z"),
			mustParseTime(time.ANSIC, "Mon Jan  2 03:04:05 1234"),
			mustParseTime(time.UnixDate, "Mon Jan  2 03:04:05 UTC 1234"),
			mustParseTime(time.RubyDate, "Mon Jan 02 03:04:05 +0000 1234"),
			mustParseTime(time.RFC822, "02 Jan 34 03:04 UTC"),
			mustParseTime(time.RFC822Z, "02 Jan 34 03:04 +0000"),
			mustParseTime(time.RFC850, "Monday, 02-Jan-34 03:04:05 UTC"),
			mustParseTime(time.RFC1123, "Mon, 02 Jan 1234 03:04:05 UTC"),
			mustParseTime(time.RFC1123Z, "Mon, 02 Jan 1234 03:04:05 +0000"),
			mustParseTime(time.RFC3339, "1234-01-02T03:04:05Z"),
			mustParseTime(time.RFC3339Nano, "1234-01-02T03:04:05.000000006Z"),
			mustParseTime(time.Kitchen, "3:04AM"),
			mustParseTime(time.Stamp, "Jan  2 03:04:05"),
			mustParseTime(time.StampMilli, "Jan  2 03:04:05.000"),
			mustParseTime(time.StampMicro, "Jan  2 03:04:05.000000"),
			mustParseTime(time.StampNano, "Jan  2 03:04:05.000000006"),
			mustParseTime(time.DateTime, "1234-01-02 03:04:05"),
			mustParseTime(time.DateOnly, "1234-01-02"),
			mustParseTime(time.TimeOnly, "03:04:05"),
			mustParseTime("2006-01-02", "1234-01-02"),
			mustParseTime(`\"weird\"2006`, `\"weird\"1234`),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
		}),
	}, {
		name: jsontest.Name("Time/Format/UnixString/InvalidNumber"),
		inBuf: `{
			"T23": -23225777754.999999994,
			"T25": -23225777754999.999994,
			"T27": -23225777754999999.994,
			"T29": -23225777754999999994
		}`,
		inVal:   new(structTimeFormat),
		want:    new(structTimeFormat),
		wantErr: EU(nil).withPos(`{`+"\n\t\t\t"+`"T23": `, "/T23").withType('0', timeTimeType),
	}, {
		name: jsontest.Name("Time/Format/UnixString/InvalidString"),
		inBuf: `{
			"T22": "-23225777754.999999994",
			"T24": "-23225777754999.999994",
			"T26": "-23225777754999999.994",
			"T28": "-23225777754999999994"
		}`,
		inVal:   new(structTimeFormat),
		want:    new(structTimeFormat),
		wantErr: EU(nil).withPos(`{`+"\n\t\t\t"+`"T22": `, "/T22").withType('"', timeTimeType),
	}, {
		name:  jsontest.Name("Time/Format/Null"),
		inBuf: `{"T1":null,"T2":null,"T3":null,"T4":null,"T5":null,"T6":null,"T7":null,"T8":null,"T9":null,"T10":null,"T11":null,"T12":null,"T13":null,"T14":null,"T15":null,"T16":null,"T17":null,"T18":null,"T19":null,"T20":null,"T21":null,"T22":null,"T23":null,"T24":null,"T25":null,"T26":null,"T27":null,"T28":null,"T29":null}`,
		inVal: addr(structTimeFormat{
			mustParseTime(time.RFC3339Nano, "1234-01-02T03:04:05.000000006Z"),
			mustParseTime(time.ANSIC, "Mon Jan  2 03:04:05 1234"),
			mustParseTime(time.UnixDate, "Mon Jan  2 03:04:05 UTC 1234"),
			mustParseTime(time.RubyDate, "Mon Jan 02 03:04:05 +0000 1234"),
			mustParseTime(time.RFC822, "02 Jan 34 03:04 UTC"),
			mustParseTime(time.RFC822Z, "02 Jan 34 03:04 +0000"),
			mustParseTime(time.RFC850, "Monday, 02-Jan-34 03:04:05 UTC"),
			mustParseTime(time.RFC1123, "Mon, 02 Jan 1234 03:04:05 UTC"),
			mustParseTime(time.RFC1123Z, "Mon, 02 Jan 1234 03:04:05 +0000"),
			mustParseTime(time.RFC3339, "1234-01-02T03:04:05Z"),
			mustParseTime(time.RFC3339Nano, "1234-01-02T03:04:05.000000006Z"),
			mustParseTime(time.Kitchen, "3:04AM"),
			mustParseTime(time.Stamp, "Jan  2 03:04:05"),
			mustParseTime(time.StampMilli, "Jan  2 03:04:05.000"),
			mustParseTime(time.StampMicro, "Jan  2 03:04:05.000000"),
			mustParseTime(time.StampNano, "Jan  2 03:04:05.000000006"),
			mustParseTime(time.DateTime, "1234-01-02 03:04:05"),
			mustParseTime(time.DateOnly, "1234-01-02"),
			mustParseTime(time.TimeOnly, "03:04:05"),
			mustParseTime("2006-01-02", "1234-01-02"),
			mustParseTime(`\"weird\"2006`, `\"weird\"1234`),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
			time.Unix(-23225777755, 6).UTC(),
		}),
		want: new(structTimeFormat),
	}, {
		name:  jsontest.Name("Time/RFC3339/Mismatch"),
		inBuf: `{"T":1234}`,
		inVal: new(struct {
			T time.Time
		}),
		wantErr: EU(nil).withPos(`{"T":`, "/T").withType('0', timeTimeType),
	}, {
		name:  jsontest.Name("Time/RFC3339/ParseError"),
		inBuf: `{"T":"2021-09-29T12:44:52"}`,
		inVal: new(struct {
			T time.Time
		}),
		wantErr: EU(func() error {
			_, err := time.Parse(time.RFC3339, "2021-09-29T12:44:52")
			return err
		}()).withPos(`{"T":`, "/T").withType('"', timeTimeType),
	}, {
		name:  jsontest.Name("Time/Format/Invalid"),
		inBuf: `{"T":""}`,
		inVal: new(struct {
			T time.Time `json:",format:UndefinedConstant"`
		}),
		wantErr: EU(errors.New(`invalid format flag "UndefinedConstant"`)).withPos(`{"T":`, "/T").withType(0, timeTimeType),
	}, {
		name:    jsontest.Name("Time/Format/SingleDigitHour"),
		inBuf:   `{"T":"2000-01-01T1:12:34Z"}`,
		inVal:   new(struct{ T time.Time }),
		wantErr: EU(newParseTimeError(time.RFC3339, "2000-01-01T1:12:34Z", "15", "1", "")).withPos(`{"T":`, "/T").withType('"', timeTimeType),
	}, {
		name:    jsontest.Name("Time/Format/SubsecondComma"),
		inBuf:   `{"T":"2000-01-01T00:00:00,000Z"}`,
		inVal:   new(struct{ T time.Time }),
		wantErr: EU(newParseTimeError(time.RFC3339, "2000-01-01T00:00:00,000Z", ".", ",", "")).withPos(`{"T":`, "/T").withType('"', timeTimeType),
	}, {
		name:    jsontest.Name("Time/Format/TimezoneHourOverflow"),
		inBuf:   `{"T":"2000-01-01T00:00:00+24:00"}`,
		inVal:   new(struct{ T time.Time }),
		wantErr: EU(newParseTimeError(time.RFC3339, "2000-01-01T00:00:00+24:00", "Z07:00", "+24:00", ": timezone hour out of range")).withPos(`{"T":`, "/T").withType('"', timeTimeType),
	}, {
		name:    jsontest.Name("Time/Format/TimezoneMinuteOverflow"),
		inBuf:   `{"T":"2000-01-01T00:00:00+00:60"}`,
		inVal:   new(struct{ T time.Time }),
		wantErr: EU(newParseTimeError(time.RFC3339, "2000-01-01T00:00:00+00:60", "Z07:00", "+00:60", ": timezone minute out of range")).withPos(`{"T":`, "/T").withType('"', timeTimeType),
	}, {
		name:  jsontest.Name("Time/Syntax/Invalid"),
		inBuf: `{"T":x}`,
		inVal: new(struct {
			T time.Time
		}),
		wantErr: newInvalidCharacterError("x", "at start of value", len64(`{"T":`), "/T"),
	}, {
		name:  jsontest.Name("Time/IgnoreInvalidFormat"),
		opts:  []Options{invalidFormatOption},
		inBuf: `"2000-01-01T00:00:00Z"`,
		inVal: addr(time.Time{}),
		want:  addr(time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)),
	}}

	for _, tt := range tests {
		t.Run(tt.name.Name, func(t *testing.T) {
			got := tt.inVal
			gotErr := Unmarshal([]byte(tt.inBuf), got, tt.opts...)
			if !reflect.DeepEqual(got, tt.want) && tt.want != nil {
				t.Errorf("%s: Unmarshal output mismatch:\ngot  %v\nwant %v", tt.name.Where, got, tt.want)
			}
			if !reflect.DeepEqual(gotErr, tt.wantErr) {
				t.Errorf("%s: Unmarshal error mismatch:\ngot  %v\nwant %v", tt.name.Where, gotErr, tt.wantErr)
			}
		})
	}
}

func TestMarshalInvalidNamespace(t *testing.T) {
	tests := []struct {
		name jsontest.CaseName
		val  any
	}{
		{jsontest.Name("Map"), map[string]string{"X": "\xde\xad\xbe\xef"}},
		{jsontest.Name("Struct"), struct{ X string }{"\xde\xad\xbe\xef"}},
	}
	for _, tt := range tests {
		t.Run(tt.name.Name, func(t *testing.T) {
			enc := jsontext.NewEncoder(new(bytes.Buffer))
			if err := MarshalEncode(enc, tt.val); err == nil {
				t.Fatalf("%s: MarshalEncode error is nil, want non-nil", tt.name.Where)
			}
			for _, tok := range []jsontext.Token{
				jsontext.Null, jsontext.String(""), jsontext.Int(0), jsontext.BeginObject, jsontext.EndObject, jsontext.BeginArray, jsontext.EndArray,
			} {
				if err := enc.WriteToken(tok); err == nil {
					t.Fatalf("%s: WriteToken error is nil, want non-nil", tt.name.Where)
				}
			}
			for _, val := range []string{`null`, `""`, `0`, `{}`, `[]`} {
				if err := enc.WriteValue([]byte(val)); err == nil {
					t.Fatalf("%s: WriteToken error is nil, want non-nil", tt.name.Where)
				}
			}
		})
	}
}

func TestUnmarshalInvalidNamespace(t *testing.T) {
	tests := []struct {
		name jsontest.CaseName
		val  any
	}{
		{jsontest.Name("Map"), addr(map[string]int{})},
		{jsontest.Name("Struct"), addr(struct{ X int }{})},
	}
	for _, tt := range tests {
		t.Run(tt.name.Name, func(t *testing.T) {
			dec := jsontext.NewDecoder(strings.NewReader(`{"X":""}`))
			if err := UnmarshalDecode(dec, tt.val); err == nil {
				t.Fatalf("%s: UnmarshalDecode error is nil, want non-nil", tt.name.Where)
			}
			if _, err := dec.ReadToken(); err == nil {
				t.Fatalf("%s: ReadToken error is nil, want non-nil", tt.name.Where)
			}
			if _, err := dec.ReadValue(); err == nil {
				t.Fatalf("%s: ReadValue error is nil, want non-nil", tt.name.Where)
			}
		})
	}
}

func TestUnmarshalReuse(t *testing.T) {
	t.Run("Bytes", func(t *testing.T) {
		in := make([]byte, 3)
		want := &in[0]
		if err := Unmarshal([]byte(`"AQID"`), &in); err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}
		got := &in[0]
		if got != want {
			t.Errorf("input buffer was not reused")
		}
	})
	t.Run("Slices", func(t *testing.T) {
		in := make([]int, 3)
		want := &in[0]
		if err := Unmarshal([]byte(`[0,1,2]`), &in); err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}
		got := &in[0]
		if got != want {
			t.Errorf("input slice was not reused")
		}
	})
	t.Run("Maps", func(t *testing.T) {
		in := make(map[string]string)
		want := reflect.ValueOf(in).Pointer()
		if err := Unmarshal([]byte(`{"key":"value"}`), &in); err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}
		got := reflect.ValueOf(in).Pointer()
		if got != want {
			t.Errorf("input map was not reused")
		}
	})
	t.Run("Pointers", func(t *testing.T) {
		in := addr(addr(addr("hello")))
		want := **in
		if err := Unmarshal([]byte(`"goodbye"`), &in); err != nil {
			t.Fatalf("Unmarshal error: %v", err)
		}
		got := **in
		if got != want {
			t.Errorf("input pointer was not reused")
		}
	})
}

type ReaderFunc func([]byte) (int, error)

func (f ReaderFunc) Read(b []byte) (int, error) { return f(b) }

type WriterFunc func([]byte) (int, error)

func (f WriterFunc) Write(b []byte) (int, error) { return f(b) }

func TestCoderBufferGrowth(t *testing.T) {
	// The growth rate of the internal buffer should be exponential,
	// but should not grow unbounded.
	checkGrowth := func(ns []int) {
		t.Helper()
		var sumBytes, sumRates, numGrows float64
		prev := ns[0]
		for i := 1; i < len(ns)-1; i++ {
			n := ns[i]
			if n != prev {
				sumRates += float64(n) / float64(prev)
				numGrows++
				prev = n
			}
			if n > 1<<20 {
				t.Fatalf("single Read/Write too large: %d", n)
			}
			sumBytes += float64(n)
		}
		if mean := sumBytes / float64(len(ns)); mean < 1<<10 {
			t.Fatalf("average Read/Write too small: %0.1f", mean)
		}
		switch mean := sumRates / numGrows; {
		case mean < 1.25:
			t.Fatalf("average growth rate too slow: %0.3f", mean)
		case mean > 2.00:
			t.Fatalf("average growth rate too fast: %0.3f", mean)
		}
	}

	// bb is identical to bytes.Buffer,
	// but a different type to avoid any optimizations for bytes.Buffer.
	bb := struct{ *bytes.Buffer }{new(bytes.Buffer)}

	var writeSizes []int
	if err := MarshalWrite(WriterFunc(func(b []byte) (int, error) {
		n, err := bb.Write(b)
		writeSizes = append(writeSizes, n)
		return n, err
	}), make([]struct{}, 1e6)); err != nil {
		t.Fatalf("MarshalWrite error: %v", err)
	}
	checkGrowth(writeSizes)

	var readSizes []int
	if err := UnmarshalRead(ReaderFunc(func(b []byte) (int, error) {
		n, err := bb.Read(b)
		readSizes = append(readSizes, n)
		return n, err
	}), new([]struct{})); err != nil {
		t.Fatalf("UnmarshalRead error: %v", err)
	}
	checkGrowth(readSizes)
}

func TestUintSet(t *testing.T) {
	type operation any // has | insert
	type has struct {
		in   uint
		want bool
	}
	type insert struct {
		in   uint
		want bool
	}

	// Sequence of operations to perform (order matters).
	ops := []operation{
		has{0, false},
		has{63, false},
		has{64, false},
		has{1234, false},
		insert{3, true},
		has{2, false},
		has{3, true},
		has{4, false},
		has{63, false},
		insert{3, false},
		insert{63, true},
		has{63, true},
		insert{64, true},
		insert{64, false},
		has{64, true},
		insert{3264, true},
		has{3264, true},
		insert{3, false},
		has{3, true},
	}

	var us uintSet
	for i, op := range ops {
		switch op := op.(type) {
		case has:
			if got := us.has(op.in); got != op.want {
				t.Fatalf("%d: uintSet.has(%v) = %v, want %v", i, op.in, got, op.want)
			}
		case insert:
			if got := us.insert(op.in); got != op.want {
				t.Fatalf("%d: uintSet.insert(%v) = %v, want %v", i, op.in, got, op.want)
			}
		default:
			panic(fmt.Sprintf("unknown operation: %T", op))
		}
	}
}

func TestUnmarshalDecodeOptions(t *testing.T) {
	var calledFuncs int
	var calledOptions Options
	in := strings.NewReader(strings.Repeat("\"\xde\xad\xbe\xef\"\n", 5))
	dec := jsontext.NewDecoder(in,
		jsontext.AllowInvalidUTF8(true), // decoder-specific option
		WithUnmarshalers(UnmarshalFromFunc(func(dec *jsontext.Decoder, _ any) error {
			opts := dec.Options()
			if v, _ := GetOption(opts, jsontext.AllowInvalidUTF8); !v {
				t.Errorf("nested Options.AllowInvalidUTF8 = false, want true")
			}
			calledFuncs++
			calledOptions = opts
			return SkipFunc
		})), // unmarshal-specific option; only relevant for UnmarshalDecode
	)

	if err := UnmarshalDecode(dec, new(string)); err != nil {
		t.Fatalf("UnmarshalDecode: %v", err)
	}
	if calledFuncs != 1 {
		t.Fatalf("calledFuncs = %d, want 1", calledFuncs)
	}
	if err := UnmarshalDecode(dec, new(string), calledOptions); err != nil {
		t.Fatalf("UnmarshalDecode: %v", err)
	}
	if calledFuncs != 2 {
		t.Fatalf("calledFuncs = %d, want 2", calledFuncs)
	}
	if err := UnmarshalDecode(dec, new(string),
		jsontext.AllowInvalidUTF8(false), // should be ignored
		WithUnmarshalers(nil),            // should override
	); err != nil {
		t.Fatalf("UnmarshalDecode: %v", err)
	}
	if calledFuncs != 2 {
		t.Fatalf("calledFuncs = %d, want 2", calledFuncs)
	}
	if err := UnmarshalDecode(dec, new(string)); err != nil {
		t.Fatalf("UnmarshalDecode: %v", err)
	}
	if calledFuncs != 3 {
		t.Fatalf("calledFuncs = %d, want 3", calledFuncs)
	}
	if err := UnmarshalDecode(dec, new(string), JoinOptions(
		jsontext.AllowInvalidUTF8(false), // should be ignored
		WithUnmarshalers(UnmarshalFromFunc(func(_ *jsontext.Decoder, _ any) error {
			opts := dec.Options()
			if v, _ := GetOption(opts, jsontext.AllowInvalidUTF8); !v {
				t.Errorf("nested Options.AllowInvalidUTF8 = false, want true")
			}
			calledFuncs = math.MaxInt
			return SkipFunc
		})), // should override
	)); err != nil {
		t.Fatalf("UnmarshalDecode: %v", err)
	}
	if calledFuncs != math.MaxInt {
		t.Fatalf("calledFuncs = %d, want %d", calledFuncs, math.MaxInt)
	}

	// Reset with the decoder options as part of the arguments should not
	// observe mutations to the options until after Reset is done.
	opts := dec.Options()                                 // AllowInvalidUTF8 is currently true
	dec.Reset(in, jsontext.AllowInvalidUTF8(false), opts) // earlier AllowInvalidUTF8(false) should be overridden by latter AllowInvalidUTF8(true) in opts
	if v, _ := GetOption(dec.Options(), jsontext.AllowInvalidUTF8); v == false {
		t.Errorf("Options.AllowInvalidUTF8 = false, want true")
	}
}

// BenchmarkUnmarshalDecodeOptions is a minimal decode operation to measure
// the overhead options setup before the unmarshal operation.
func BenchmarkUnmarshalDecodeOptions(b *testing.B) {
	var i int
	in := new(bytes.Buffer)
	dec := jsontext.NewDecoder(in)
	makeBench := func(opts ...Options) func(*testing.B) {
		return func(b *testing.B) {
			for range b.N {
				in.WriteString("0 ")
			}
			dec.Reset(in)
			b.ResetTimer()
			for range b.N {
				UnmarshalDecode(dec, &i, opts...)
			}
		}
	}
	b.Run("None", makeBench())
	b.Run("Same", makeBench(&export.Decoder(dec).Struct))
	b.Run("New", makeBench(DefaultOptionsV2()))
}

func TestMarshalEncodeOptions(t *testing.T) {
	var calledFuncs int
	var calledOptions Options
	out := new(bytes.Buffer)
	enc := jsontext.NewEncoder(
		out,
		jsontext.AllowInvalidUTF8(true), // encoder-specific option
		WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, _ any) error {
			opts := enc.Options()
			if v, _ := GetOption(opts, jsontext.AllowInvalidUTF8); !v {
				t.Errorf("nested Options.AllowInvalidUTF8 = false, want true")
			}
			calledFuncs++
			calledOptions = opts
			return SkipFunc
		})), // marshal-specific option; only relevant for MarshalEncode
	)

	if err := MarshalEncode(enc, "\xde\xad\xbe\xef"); err != nil {
		t.Fatalf("MarshalEncode: %v", err)
	}
	if calledFuncs != 1 {
		t.Fatalf("calledFuncs = %d, want 1", calledFuncs)
	}
	if err := MarshalEncode(enc, "\xde\xad\xbe\xef", calledOptions); err != nil {
		t.Fatalf("MarshalEncode: %v", err)
	}
	if calledFuncs != 2 {
		t.Fatalf("calledFuncs = %d, want 2", calledFuncs)
	}
	if err := MarshalEncode(enc, "\xde\xad\xbe\xef",
		jsontext.AllowInvalidUTF8(false), // should be ignored
		WithMarshalers(nil),              // should override
	); err != nil {
		t.Fatalf("MarshalEncode: %v", err)
	}
	if calledFuncs != 2 {
		t.Fatalf("calledFuncs = %d, want 2", calledFuncs)
	}
	if err := MarshalEncode(enc, "\xde\xad\xbe\xef"); err != nil {
		t.Fatalf("MarshalEncode: %v", err)
	}
	if calledFuncs != 3 {
		t.Fatalf("calledFuncs = %d, want 3", calledFuncs)
	}
	if err := MarshalEncode(enc, "\xde\xad\xbe\xef", JoinOptions(
		jsontext.AllowInvalidUTF8(false), // should be ignored
		WithMarshalers(MarshalToFunc(func(enc *jsontext.Encoder, _ any) error {
			opts := enc.Options()
			if v, _ := GetOption(opts, jsontext.AllowInvalidUTF8); !v {
				t.Errorf("nested Options.AllowInvalidUTF8 = false, want true")
			}
			calledFuncs = math.MaxInt
			return SkipFunc
		})), // should override
	)); err != nil {
		t.Fatalf("MarshalEncode: %v", err)
	}
	if calledFuncs != math.MaxInt {
		t.Fatalf("calledFuncs = %d, want %d", calledFuncs, math.MaxInt)
	}
	if out.String() != strings.Repeat("\"\xde\xad\ufffd\ufffd\"\n", 5) {
		t.Fatalf("output mismatch:\n\tgot:  %s\n\twant: %s", out.String(), strings.Repeat("\"\xde\xad\xbe\xef\"\n", 5))
	}

	// Reset with the encoder options as part of the arguments should not
	// observe mutations to the options until after Reset is done.
	opts := enc.Options()                                  // AllowInvalidUTF8 is currently true
	enc.Reset(out, jsontext.AllowInvalidUTF8(false), opts) // earlier AllowInvalidUTF8(false) should be overridden by latter AllowInvalidUTF8(true) in opts
	if v, _ := GetOption(enc.Options(), jsontext.AllowInvalidUTF8); v == false {
		t.Errorf("Options.AllowInvalidUTF8 = false, want true")
	}
}

// BenchmarkMarshalEncodeOptions is a minimal encode operation to measure
// the overhead of options setup before the marshal operation.
func BenchmarkMarshalEncodeOptions(b *testing.B) {
	var i int
	out := new(bytes.Buffer)
	enc := jsontext.NewEncoder(out)
	makeBench := func(opts ...Options) func(*testing.B) {
		return func(b *testing.B) {
			out.Reset()
			enc.Reset(out)
			b.ResetTimer()
			for range b.N {
				MarshalEncode(enc, &i, opts...)
			}
		}
	}
	b.Run("None", makeBench())
	b.Run("Same", makeBench(&export.Encoder(enc).Struct))
	b.Run("New", makeBench(DefaultOptionsV2()))
}
