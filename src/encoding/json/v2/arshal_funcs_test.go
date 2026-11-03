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

type funcMarshalInner int
type funcMarshalOuter int

func TestEncoderResetNestedMarshalToFunc(t *testing.T) {
	enc := jsontext.NewEncoder(&bytes.Buffer{})
	opts := json.WithMarshalers(json.JoinMarshalers(
		json.MarshalToFunc(func(e *jsontext.Encoder, _ funcMarshalInner) error {
			return e.WriteValue(jsontext.Value("null"))
		}),
		json.MarshalToFunc(func(e *jsontext.Encoder, _ funcMarshalOuter) error {
			if err := json.MarshalEncode(e, funcMarshalInner(0)); err != nil {
				return err
			}
			e.Reset(&bytes.Buffer{})
			return nil
		}),
	))
	defer func() {
		if recover() == nil {
			t.Fatal("Reset did not panic inside outer MarshalToFunc")
		}
	}()
	if err := json.MarshalEncode(enc, funcMarshalOuter(0), opts); err != nil {
		t.Fatal(err)
	}
}

type funcUnmarshalInner int
type funcUnmarshalOuter int

func TestDecoderResetNestedUnmarshalFromFunc(t *testing.T) {
	dec := jsontext.NewDecoder(strings.NewReader(`null`))
	opts := json.WithUnmarshalers(json.JoinUnmarshalers(
		json.UnmarshalFromFunc(func(d *jsontext.Decoder, _ *funcUnmarshalInner) error {
			_, err := d.ReadValue()
			return err
		}),
		json.UnmarshalFromFunc(func(d *jsontext.Decoder, _ *funcUnmarshalOuter) error {
			if err := json.UnmarshalDecode(d, new(funcUnmarshalInner)); err != nil {
				return err
			}
			d.Reset(strings.NewReader(`null`))
			return nil
		}),
	))
	defer func() {
		if recover() == nil {
			t.Fatal("Reset did not panic inside outer UnmarshalFromFunc")
		}
	}()
	var out funcUnmarshalOuter
	if err := json.UnmarshalDecode(dec, &out, opts); err != nil {
		t.Fatal(err)
	}
}
