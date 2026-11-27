// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json_test

import (
	jsonv1 "encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"strings"
	"testing"
)

func TestLegacyUnmarshalErrorOptions(t *testing.T) {
	for _, tt := range []struct {
		name string
		opts jsonv2.Options
		want bool
	}{
		{"v1 defaults", jsonv1.DefaultOptionsV1(), true},
		{"legacy error option", jsonv1.ReportErrorsWithLegacySemantics(true), true},
		{"joined", jsonv2.JoinOptions(jsonv2.DefaultOptionsV2(), jsonv1.ReportErrorsWithLegacySemantics(true)), true},
		{"disabled", jsonv2.JoinOptions(jsonv1.DefaultOptionsV1(), jsonv1.ReportErrorsWithLegacySemantics(false)), false},
		{"v2 defaults", jsonv2.JoinOptions(jsonv1.DefaultOptionsV1(), jsonv2.DefaultOptionsV2()), false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			const input = `"hello"`
			var target int
			for name, unmarshal := range map[string]func() error{
				"Unmarshal":     func() error { return jsonv2.Unmarshal([]byte(input), &target, tt.opts) },
				"UnmarshalRead": func() error { return jsonv2.UnmarshalRead(strings.NewReader(input), &target, tt.opts) },
				"UnmarshalDecode": func() error {
					return jsonv2.UnmarshalDecode(jsontext.NewDecoder(strings.NewReader(input), tt.opts), &target)
				},
			} {
				err := unmarshal()
				var legacy *jsonv1.UnmarshalTypeError
				if err == nil || errors.As(err, &legacy) != tt.want {
					t.Errorf("%s returned %T (%v), want legacy error: %t", name, err, err, tt.want)
				}
			}
		})
	}
}

func TestLegacyErrorOptionComparable(t *testing.T) {
	option := jsonv1.ReportErrorsWithLegacySemantics(true)
	byOption := map[jsonv2.Options]string{option: "legacy"}
	if got := byOption[jsonv1.ReportErrorsWithLegacySemantics(true)]; got != "legacy" {
		t.Fatalf("equal legacy option lookup = %q, want legacy", got)
	}
	if got, ok := jsonv2.GetOption(option, jsonv1.ReportErrorsWithLegacySemantics); !ok || !got {
		t.Fatalf("GetOption = (%t, %t), want (true, true)", got, ok)
	}
}

type forwardLegacyEncoderOptions struct{}

func (forwardLegacyEncoderOptions) MarshalJSONTo(enc *jsontext.Encoder) error {
	var target int
	err := jsonv2.Unmarshal([]byte(`"hello"`), &target, enc.Options())
	var legacy *jsonv1.UnmarshalTypeError
	if !errors.As(err, &legacy) {
		return errors.New("encoder options did not preserve legacy unmarshal errors")
	}
	return enc.WriteToken(jsontext.Null)
}

func TestLegacyEncoderOptionsForwarding(t *testing.T) {
	if _, err := jsonv1.Marshal(forwardLegacyEncoderOptions{}); err != nil {
		t.Errorf("Marshal: %v", err)
	}
	var out strings.Builder
	if err := jsonv1.NewEncoder(&out).Encode(forwardLegacyEncoderOptions{}); err != nil {
		t.Errorf("Encode: %v", err)
	}
}
