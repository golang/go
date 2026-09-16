// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"simd/archsimd/_gen/specgen/specexpr"
	"strings"
	"testing"
)

func TestNewSpecTemplate(t *testing.T) {
	tests := []struct {
		tmpl    string
		isTmpl  bool
		wantErr bool
	}{
		{
			tmpl:   "",
			isTmpl: false,
		},
		{
			tmpl:   "Convert",
			isTmpl: false,
		},
		{
			tmpl:   "Convert{{.zL}}To{{.zB}}{{.zN}}",
			isTmpl: true,
		},
		{
			tmpl:    "Convert{{.zL",
			wantErr: true,
		},
		{
			tmpl:    "Convert{{if}}",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		got, err := newSpecTemplate(tc.tmpl)
		if (err != nil) != tc.wantErr {
			t.Errorf("newSpecTemplate(%q) returned error: %v, wantErr: %v", tc.tmpl, err, tc.wantErr)
			continue
		}
		if tc.wantErr {
			continue
		}
		if got.raw != tc.tmpl {
			t.Errorf("newSpecTemplate(%q) raw = %q, want %q", tc.tmpl, got.raw, tc.tmpl)
		}
		if (got.tmpl != nil) != tc.isTmpl {
			t.Errorf("newSpecTemplate(%q) isTmpl = %v, want %v", tc.tmpl, got.tmpl != nil, tc.isTmpl)
		}
	}
}

func TestSpecTemplateExpand(t *testing.T) {
	tmpl, err := newSpecTemplate("Convert{{.zL}}To{{.zB}}{{.zN}}")
	if err != nil {
		t.Fatalf("unexpected error parsing template: %v", err)
	}

	data := map[string]any{
		"zL": specexpr.Int(4),
		"zB": "Float",
		"zN": specexpr.Int(32),
	}

	got, err := tmpl.expand(data)
	if err != nil {
		t.Fatalf("unexpected error expanding template: %v", err)
	}
	want := "Convert4ToFloat32"
	if got != want {
		t.Errorf("expected expanded string %q, got %q", want, got)
	}
}

func TestSpecTemplateTitle(t *testing.T) {
	tmpl, err := newSpecTemplate("ConvertTo{{.zE | title}}")
	if err != nil {
		t.Fatalf("unexpected error parsing template: %v", err)
	}

	data := map[string]any{
		"zE": specexpr.Basic{Base: "float", Bits: 32},
	}

	got, err := tmpl.expand(data)
	if err != nil {
		t.Fatalf("unexpected error expanding template: %v", err)
	}
	want := "ConvertToFloat32"
	if got != want {
		t.Errorf("expected %q, got %q", want, got)
	}
}

func TestSpecTemplateConditionals(t *testing.T) {
	tmplStr := `Doc for {{.Name}}.
{{if lt .zL .xL}}
Upper elements of the result are zeroed.
{{end}}`

	tmpl, err := newSpecTemplate(tmplStr)
	if err != nil {
		t.Fatalf("unexpected error parsing template: %v", err)
	}

	// Case 1: zL < xL
	data1 := map[string]any{
		"Name": "ConvertLo4ToFloat32",
		"zL":   specexpr.Int(4),
		"xL":   specexpr.Int(8),
	}
	got1, err := tmpl.expand(data1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got1, "Upper elements of the result are zeroed.") {
		t.Errorf("expected conditional text when zL < xL, got: %q", got1)
	}

	// Case 2: zL >= xL
	data2 := map[string]any{
		"Name": "ConvertLo8ToFloat32",
		"zL":   specexpr.Int(8),
		"xL":   specexpr.Int(8),
	}
	got2, err := tmpl.expand(data2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(got2, "Upper elements of the result are zeroed.") {
		t.Errorf("expected no conditional text when zL == xL, got: %q", got2)
	}
}

func TestSpecTemplateScalableWidthCompare(t *testing.T) {
	tmpl, err := newSpecTemplate("{{if lt .zL .xL}}smaller{{else}}same-or-larger{{end}}")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// zL = VW/32, xL = VW/16 => zL < xL
	zL, err := specexpr.VW().Div(specexpr.Int(32))
	if err != nil {
		t.Fatal(err)
	}
	xL, err := specexpr.VW().Div(specexpr.Int(16))
	if err != nil {
		t.Fatal(err)
	}

	data := map[string]any{
		"zL": zL,
		"xL": xL,
	}
	got, err := tmpl.expand(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "smaller" {
		t.Errorf("expected 'smaller', got %q", got)
	}
}

func TestSpecTemplateMissingKey(t *testing.T) {
	tmpl, err := newSpecTemplate("Convert{{.missing}}")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, err = tmpl.expand(map[string]any{})
	if err == nil {
		t.Errorf("expected error on missing key, got nil")
	}
}

func TestCleanDocNewlines(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{
			input: "hello\n\n\n\nworld\n",
			want:  "hello\n\nworld\n",
		},
		{
			input: "hello\n\n\n",
			want:  "hello\n",
		},
		{
			input: "",
			want:  "",
		},
	}

	for _, tc := range tests {
		got := cleanDocNewlines(tc.input)
		if got != tc.want {
			t.Errorf("cleanDocNewlines(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
