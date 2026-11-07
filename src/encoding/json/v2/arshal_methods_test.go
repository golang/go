// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json_test

import (
	"bytes"
	"strings"
	"testing"

	"encoding/json/jsontext"
	"encoding/json/v2"
)

type methodMarshalInner struct{}

func (methodMarshalInner) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteValue(jsontext.Value("null"))
}

type methodMarshalOuter struct {
	enc *jsontext.Encoder
}

func (m methodMarshalOuter) MarshalJSONTo(enc *jsontext.Encoder) error {
	if err := json.MarshalEncode(enc, methodMarshalInner{}); err != nil {
		return err
	}
	m.enc.Reset(&bytes.Buffer{})
	return nil
}

func TestEncoderResetNestedMarshalerTo(t *testing.T) {
	enc := jsontext.NewEncoder(&bytes.Buffer{})
	defer func() {
		if recover() == nil {
			t.Fatal("Reset did not panic inside outer MarshalerTo")
		}
	}()
	if err := json.MarshalEncode(enc, methodMarshalOuter{enc}); err != nil {
		t.Fatal(err)
	}
}

type methodUnmarshalInner struct{}

type methodUnmarshalOuter struct {
	dec *jsontext.Decoder
}

func (*methodUnmarshalInner) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	_, err := dec.ReadValue()
	return err
}

func (u *methodUnmarshalOuter) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	if err := json.UnmarshalDecode(dec, new(methodUnmarshalInner)); err != nil {
		return err
	}
	u.dec.Reset(strings.NewReader(`null`))
	return nil
}

func TestDecoderResetNestedUnmarshalerFrom(t *testing.T) {
	dec := jsontext.NewDecoder(strings.NewReader(`null`))
	defer func() {
		if recover() == nil {
			t.Fatal("Reset did not panic inside outer UnmarshalerFrom")
		}
	}()
	if err := json.UnmarshalDecode(dec, &methodUnmarshalOuter{dec}); err != nil {
		t.Fatal(err)
	}
}
