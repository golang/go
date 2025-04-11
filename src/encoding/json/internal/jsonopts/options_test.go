// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsonopts_test

import (
	"reflect"
	"testing"

	"encoding/json/internal/jsonflags"
	. "encoding/json/internal/jsonopts"
	"encoding/json/jsontext"
	"encoding/json/v2"
)

func makeFlags(f ...jsonflags.Bools) (fs jsonflags.Flags) {
	for _, f := range f {
		fs.Set(f)
	}
	return fs
}

func TestJoin(t *testing.T) {
	tests := []struct {
		in            Options
		excludeCoders bool
		want          *Struct
	}{{
		in:   jsonflags.AllowInvalidUTF8 | 1,
		want: &Struct{Flags: makeFlags(jsonflags.AllowInvalidUTF8 | 1)},
	}, {
		in: jsonflags.Multiline | 0,
		want: &Struct{
			Flags: makeFlags(jsonflags.AllowInvalidUTF8|1, jsonflags.Multiline|0)},
	}, {
		in: Indent("\t"), // implicitly sets Multiline=true
		want: &Struct{
			Flags:       makeFlags(jsonflags.AllowInvalidUTF8 | jsonflags.Multiline | jsonflags.Indent | 1),
			CoderValues: CoderValues{Indent: "\t"},
		},
	}, {
		in: &Struct{
			Flags: makeFlags(jsonflags.Multiline|jsonflags.EscapeForJS|0, jsonflags.AllowInvalidUTF8|1),
		},
		want: &Struct{
			Flags:       makeFlags(jsonflags.AllowInvalidUTF8|jsonflags.Indent|1, jsonflags.Multiline|jsonflags.EscapeForJS|0),
			CoderValues: CoderValues{Indent: "\t"},
		},
	}, {
		in: &DefaultOptionsV1,
		want: func() *Struct {
			v1 := DefaultOptionsV1
			v1.Flags.Set(jsonflags.Indent | 1)
			v1.Flags.Set(jsonflags.Multiline | 0)
			v1.Indent = "\t"
			return &v1
		}(), // v1 fully replaces before (except for whitespace related flags)
	}, {
		in: &DefaultOptionsV2,
		want: func() *Struct {
			v2 := DefaultOptionsV2
			v2.Flags.Set(jsonflags.Indent | 1)
			v2.Flags.Set(jsonflags.Multiline | 0)
			v2.Indent = "\t"
			return &v2
		}(), // v2 fully replaces before (except for whitespace related flags)
	}, {
		in: jsonflags.Deterministic | jsonflags.AllowInvalidUTF8 | 1, excludeCoders: true,
		want: func() *Struct {
			v2 := DefaultOptionsV2
			v2.Flags.Set(jsonflags.Deterministic | 1)
			v2.Flags.Set(jsonflags.Indent | 1)
			v2.Flags.Set(jsonflags.Multiline | 0)
			v2.Indent = "\t"
			return &v2
		}(),
	}, {
		in: jsontext.WithIndentPrefix("    "), excludeCoders: true,
		want: func() *Struct {
			v2 := DefaultOptionsV2
			v2.Flags.Set(jsonflags.Deterministic | 1)
			v2.Flags.Set(jsonflags.Indent | 1)
			v2.Flags.Set(jsonflags.Multiline | 0)
			v2.Indent = "\t"
			return &v2
		}(),
	}, {
		in: jsontext.WithIndentPrefix("    "), excludeCoders: false,
		want: func() *Struct {
			v2 := DefaultOptionsV2
			v2.Flags.Set(jsonflags.Deterministic | 1)
			v2.Flags.Set(jsonflags.Indent | 1)
			v2.Flags.Set(jsonflags.IndentPrefix | 1)
			v2.Flags.Set(jsonflags.Multiline | 1)
			v2.Indent = "\t"
			v2.IndentPrefix = "    "
			return &v2
		}(),
	}, {
		in: &Struct{
			Flags: jsonflags.Flags{
				Presence: uint64(jsonflags.Deterministic | jsonflags.Indent | jsonflags.IndentPrefix),
				Values:   uint64(jsonflags.Indent | jsonflags.IndentPrefix),
			},
			CoderValues: CoderValues{Indent: "  ", IndentPrefix: "  "},
		},
		excludeCoders: true,
		want: func() *Struct {
			v2 := DefaultOptionsV2
			v2.Flags.Set(jsonflags.Indent | 1)
			v2.Flags.Set(jsonflags.IndentPrefix | 1)
			v2.Flags.Set(jsonflags.Multiline | 1)
			v2.Indent = "\t"
			v2.IndentPrefix = "    "
			return &v2
		}(),
	}, {
		in: &Struct{
			Flags: jsonflags.Flags{
				Presence: uint64(jsonflags.Deterministic | jsonflags.Indent | jsonflags.IndentPrefix),
				Values:   uint64(jsonflags.Indent | jsonflags.IndentPrefix),
			},
			CoderValues: CoderValues{Indent: "  ", IndentPrefix: "  "},
		},
		excludeCoders: false,
		want: func() *Struct {
			v2 := DefaultOptionsV2
			v2.Flags.Set(jsonflags.Indent | 1)
			v2.Flags.Set(jsonflags.IndentPrefix | 1)
			v2.Flags.Set(jsonflags.Multiline | 1)
			v2.Indent = "  "
			v2.IndentPrefix = "  "
			return &v2
		}(),
	}}
	got := new(Struct)
	for i, tt := range tests {
		if tt.excludeCoders {
			got.JoinWithoutCoderOptions(tt.in)
		} else {
			got.Join(tt.in)
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Fatalf("%d: Join:\n\tgot:  %+v\n\twant: %+v", i, got, tt.want)
		}
	}
}

func TestGet(t *testing.T) {
	opts := &Struct{
		Flags:        makeFlags(jsonflags.Indent|jsonflags.Deterministic|jsonflags.Marshalers|1, jsonflags.Multiline|0),
		CoderValues:  CoderValues{Indent: "\t"},
		ArshalValues: ArshalValues{Marshalers: new(json.Marshalers)},
	}
	if v, ok := json.GetOption(nil, jsontext.AllowDuplicateNames); v || ok {
		t.Errorf("GetOption(..., AllowDuplicateNames) = (%v, %v), want (false, false)", v, ok)
	}
	if v, ok := json.GetOption(jsonflags.AllowInvalidUTF8|0, jsontext.AllowDuplicateNames); v || ok {
		t.Errorf("GetOption(..., AllowDuplicateNames) = (%v, %v), want (false, false)", v, ok)
	}
	if v, ok := json.GetOption(jsonflags.AllowDuplicateNames|0, jsontext.AllowDuplicateNames); v || !ok {
		t.Errorf("GetOption(..., AllowDuplicateNames) = (%v, %v), want (false, true)", v, ok)
	}
	if v, ok := json.GetOption(jsonflags.AllowDuplicateNames|1, jsontext.AllowDuplicateNames); !v || !ok {
		t.Errorf("GetOption(..., AllowDuplicateNames) = (%v, %v), want (true, true)", v, ok)
	}
	if v, ok := json.GetOption(Indent(""), jsontext.AllowDuplicateNames); v || ok {
		t.Errorf("GetOption(..., AllowDuplicateNames) = (%v, %v), want (false, false)", v, ok)
	}
	if v, ok := json.GetOption(Indent(" "), jsontext.WithIndent); v != " " || !ok {
		t.Errorf(`GetOption(..., WithIndent) = (%q, %v), want (" ", true)`, v, ok)
	}
	if v, ok := json.GetOption(jsonflags.AllowDuplicateNames|1, jsontext.WithIndent); v != "" || ok {
		t.Errorf(`GetOption(..., WithIndent) = (%q, %v), want ("", false)`, v, ok)
	}
	if v, ok := json.GetOption(opts, jsontext.AllowDuplicateNames); v || ok {
		t.Errorf("GetOption(..., AllowDuplicateNames) = (%v, %v), want (false, false)", v, ok)
	}
	if v, ok := json.GetOption(opts, json.Deterministic); !v || !ok {
		t.Errorf("GetOption(..., Deterministic) = (%v, %v), want (true, true)", v, ok)
	}
	if v, ok := json.GetOption(opts, jsontext.Multiline); v || !ok {
		t.Errorf("GetOption(..., Multiline) = (%v, %v), want (false, true)", v, ok)
	}
	if v, ok := json.GetOption(opts, jsontext.AllowInvalidUTF8); v || ok {
		t.Errorf("GetOption(..., AllowInvalidUTF8) = (%v, %v), want (false, false)", v, ok)
	}
	if v, ok := json.GetOption(opts, jsontext.WithIndent); v != "\t" || !ok {
		t.Errorf(`GetOption(..., WithIndent) = (%q, %v), want ("\t", true)`, v, ok)
	}
	if v, ok := json.GetOption(opts, jsontext.WithIndentPrefix); v != "" || ok {
		t.Errorf(`GetOption(..., WithIndentPrefix) = (%q, %v), want ("", false)`, v, ok)
	}
	if v, ok := json.GetOption(opts, json.WithMarshalers); v == nil || !ok {
		t.Errorf(`GetOption(..., WithMarshalers) = (%v, %v), want (non-nil, true)`, v, ok)
	}
	if v, ok := json.GetOption(opts, json.WithUnmarshalers); v != nil || ok {
		t.Errorf(`GetOption(..., WithUnmarshalers) = (%v, %v), want (nil, false)`, v, ok)
	}
}

var sink struct {
	Bool       bool
	String     string
	Marshalers *json.Marshalers
}

func BenchmarkGetBool(b *testing.B) {
	b.ReportAllocs()
	opts := json.DefaultOptionsV2()
	for range b.N {
		sink.Bool, sink.Bool = json.GetOption(opts, jsontext.AllowDuplicateNames)
	}
}

func BenchmarkGetIndent(b *testing.B) {
	b.ReportAllocs()
	opts := json.DefaultOptionsV2()
	for range b.N {
		sink.String, sink.Bool = json.GetOption(opts, jsontext.WithIndent)
	}
}

func BenchmarkGetMarshalers(b *testing.B) {
	b.ReportAllocs()
	opts := json.JoinOptions(json.DefaultOptionsV2(), json.WithMarshalers(nil))
	for range b.N {
		sink.Marshalers, sink.Bool = json.GetOption(opts, json.WithMarshalers)
	}
}
