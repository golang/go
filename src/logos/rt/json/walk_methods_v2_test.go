// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"encoding"
	stdjson "encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"testing"
)

type v2CustomToOnly struct {
	V int
}

var _ jsonv2.MarshalerTo = v2CustomToOnly{}

func (v2CustomToOnly) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteToken(jsontext.String("custom_to_only"))
}

type v2CustomAppendOnly struct {
	V int
}

var _ encoding.TextAppender = v2CustomAppendOnly{}

func (v2CustomAppendOnly) AppendText(b []byte) ([]byte, error) {
	return append(b, "custom_append_only"...), nil
}

type v2PrecedenceBoth struct {
	V int
}

var (
	_ stdjson.Marshaler  = v2PrecedenceBoth{}
	_ jsonv2.MarshalerTo = v2PrecedenceBoth{}
)

func (v2PrecedenceBoth) MarshalJSON() ([]byte, error) {
	return []byte(`"from_marshal_json"`), nil
}

func (v2PrecedenceBoth) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteToken(jsontext.String("from_marshal_json_to"))
}

type v2PrecedencePtrToValJSON struct {
	V int
}

var (
	_ stdjson.Marshaler  = v2PrecedencePtrToValJSON{}
	_ jsonv2.MarshalerTo = (*v2PrecedencePtrToValJSON)(nil)
)

func (v2PrecedencePtrToValJSON) MarshalJSON() ([]byte, error) {
	return []byte(`"from_val_marshal_json"`), nil
}

func (*v2PrecedencePtrToValJSON) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteToken(jsontext.String("from_ptr_marshal_json_to"))
}

type v2PtrReceiverTo struct {
	V int
}

var _ jsonv2.MarshalerTo = (*v2PtrReceiverTo)(nil)

func (*v2PtrReceiverTo) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteToken(jsontext.String("ptr_to"))
}

type v2PtrReceiverAppend struct {
	V int
}

var _ encoding.TextAppender = (*v2PtrReceiverAppend)(nil)

func (*v2PtrReceiverAppend) AppendText(b []byte) ([]byte, error) {
	return append(b, "ptr_append"...), nil
}

func checkWalkExpected(t *testing.T, label string, v any, want string) {
	t.Helper()
	var s Walk
	got, err := s.Marshal(v)
	if err != nil {
		t.Fatalf("%s: Walk.Marshal unexpected error: %v", label, err)
	}
	if string(got) != want {
		t.Fatalf("%s: Walk.Marshal got %s, want %s", label, got, want)
	}
	checkWalkParity(t, label, v)
}

func TestWalkV2MethodDispatch(t *testing.T) {
	toVal := v2PtrReceiverTo{V: 10}
	appVal := v2PtrReceiverAppend{V: 20}

	tests := []struct {
		name string
		in   any
		want string
	}{
		{
			name: "MarshalJSONToOnly/root",
			in:   v2CustomToOnly{V: 1},
			want: `"custom_to_only"`,
		},
		{
			name: "MarshalJSONToOnly/struct_field",
			in: struct {
				A v2CustomToOnly `json:"a"`
			}{A: v2CustomToOnly{V: 1}},
			want: `{"a":"custom_to_only"}`,
		},
		{
			name: "MarshalJSONToOnly/slice",
			in:   []v2CustomToOnly{{V: 1}},
			want: `["custom_to_only"]`,
		},
		{
			name: "MarshalJSONToOnly/interface",
			in:   any(v2CustomToOnly{V: 1}),
			want: `"custom_to_only"`,
		},

		{
			name: "AppendTextOnly/root",
			in:   v2CustomAppendOnly{V: 2},
			want: `"custom_append_only"`,
		},
		{
			name: "AppendTextOnly/struct_field",
			in: struct {
				A v2CustomAppendOnly `json:"a"`
			}{A: v2CustomAppendOnly{V: 2}},
			want: `{"a":"custom_append_only"}`,
		},
		{
			name: "AppendTextOnly/slice",
			in:   []v2CustomAppendOnly{{V: 2}},
			want: `["custom_append_only"]`,
		},
		{
			name: "AppendTextOnly/interface",
			in:   any(v2CustomAppendOnly{V: 2}),
			want: `"custom_append_only"`,
		},

		{
			name: "Precedence/ValueReceiver",
			in: struct {
				A v2PrecedenceBoth `json:"a"`
			}{A: v2PrecedenceBoth{V: 3}},
			want: `{"a":"from_marshal_json_to"}`,
		},
		{
			name: "Precedence/UnaddressablePtrToVsValJSON",
			in: struct {
				A v2PrecedencePtrToValJSON `json:"a"`
			}{A: v2PrecedencePtrToValJSON{V: 4}},
			want: `{"a":"from_val_marshal_json"}`,
		},
		{
			name: "Precedence/AddressablePtrToVsValJSON",
			in: &struct {
				A v2PrecedencePtrToValJSON `json:"a"`
			}{A: v2PrecedencePtrToValJSON{V: 4}},
			want: `{"a":"from_ptr_marshal_json_to"}`,
		},

		{
			name: "PointerReceiver/UnaddressableFieldMarshalJSONTo",
			in: struct {
				A v2PtrReceiverTo `json:"a"`
			}{A: toVal},
			want: `{"a":{"V":10}}`,
		},
		{
			name: "PointerReceiver/AddressableFieldMarshalJSONTo",
			in: &struct {
				A v2PtrReceiverTo `json:"a"`
			}{A: toVal},
			want: `{"a":"ptr_to"}`,
		},
		{
			name: "PointerReceiver/NonNilPointerMarshalJSONTo",
			in: struct {
				A *v2PtrReceiverTo `json:"a"`
			}{A: &toVal},
			want: `{"a":"ptr_to"}`,
		},
		{
			name: "PointerReceiver/NilPointerMarshalJSONTo",
			in: struct {
				A *v2PtrReceiverTo `json:"a"`
			}{A: nil},
			want: `{"a":null}`,
		},
		{
			name: "PointerReceiver/UnaddressableFieldAppendText",
			in: struct {
				A v2PtrReceiverAppend `json:"a"`
			}{A: appVal},
			want: `{"a":{"V":20}}`,
		},
		{
			name: "PointerReceiver/AddressableFieldAppendText",
			in: &struct {
				A v2PtrReceiverAppend `json:"a"`
			}{A: appVal},
			want: `{"a":"ptr_append"}`,
		},
		{
			name: "PointerReceiver/NonNilPointerAppendText",
			in: struct {
				A *v2PtrReceiverAppend `json:"a"`
			}{A: &appVal},
			want: `{"a":"ptr_append"}`,
		},
		{
			name: "PointerReceiver/NilPointerAppendText",
			in: struct {
				A *v2PtrReceiverAppend `json:"a"`
			}{A: nil},
			want: `{"a":null}`,
		},

		{
			name: "StringTag/MarshalJSONTo",
			in: struct {
				A v2CustomToOnly `json:"a,string"`
			}{A: v2CustomToOnly{V: 30}},
			want: `{"a":"custom_to_only"}`,
		},
		{
			name: "StringTag/AppendText",
			in: struct {
				A v2CustomAppendOnly `json:"a,string"`
			}{A: v2CustomAppendOnly{V: 40}},
			want: `{"a":"custom_append_only"}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			checkWalkExpected(t, tc.name, tc.in, tc.want)
		})
	}
}
