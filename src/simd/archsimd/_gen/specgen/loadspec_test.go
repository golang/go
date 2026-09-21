// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"strings"
	"sync"
	"testing"
)

var (
	specPkg      *specPackage
	specLoadErr  error
	loadSpecOnce sync.Once
)

func loadSpec(t *testing.T) *specPackage {
	loadSpecOnce.Do(func() {
		var root contextRoot
		ctx := context{root: &root}

		specPkg = loadSpecPackage(ctx, "../../../internal/spec", &LoadOptions{})
		if err := root.gatherErrors(); err != nil {
			specLoadErr = err
		}
	})
	if specLoadErr != nil {
		t.Fatalf("failed to load spec: %v", specLoadErr)
	}
	return specPkg
}

func TestLoadSpec(t *testing.T) {
	pkg := loadSpec(t)

	// Verify we parsed functions
	if len(pkg.Funcs) == 0 {
		t.Errorf("expected parsed functions, got 0")
	}

	var foundAdd, foundExtend bool
	for _, f := range pkg.Funcs {
		if f.Name == "Add" {
			foundAdd = true
			if len(f.TypeParams) != 2 {
				t.Errorf("Add should have 2 type parameters, got %d", len(f.TypeParams))
			} else {
				if f.TypeParams[0].Obj().Name() != "E" || !strings.Contains(f.TypeParams[0].Constraint().String(), "Nums") {
					t.Errorf("unexpected Add type param 0: %s %s", f.TypeParams[0].Obj().Name(), f.TypeParams[0].Constraint())
				}
				if f.TypeParams[1].Obj().Name() != "W" || !strings.Contains(f.TypeParams[1].Constraint().String(), "Width") {
					t.Errorf("unexpected Add type param 1: %s %s", f.TypeParams[1].Obj().Name(), f.TypeParams[1].Constraint())
				}
			}
			if len(f.Params) != 2 {
				t.Errorf("Add should have 2 params, got %d", len(f.Params))
			} else {
				if f.Params[0].Name() != "x" || !strings.Contains(f.Params[0].Type().String(), "Vec[") {
					t.Errorf("unexpected Add param 0: %s %s", f.Params[0].Name(), f.Params[0].Type())
				}
				if f.Params[1].Name() != "y" || !strings.Contains(f.Params[1].Type().String(), "Vec[") {
					t.Errorf("unexpected Add param 1: %s %s", f.Params[1].Name(), f.Params[1].Type())
				}
			}
			if len(f.Results) != 1 {
				t.Errorf("Add should have 1 result, got %d", len(f.Results))
			} else {
				if !strings.Contains(f.Results[0].Type().String(), "Vec[") {
					t.Errorf("unexpected Add result type: %s", f.Results[0].Type())
				}
			}
		}

		if f.Name == "ExtendLoLToZ" {
			foundExtend = true
			if len(f.Requirements) != 2 {
				t.Errorf("expected 2 requirements for ExtendLoLToZ, got %d", len(f.Requirements))
			} else {
				if f.Requirements[0] == nil || f.Requirements[1] == nil {
					t.Errorf("expected non-nil parsed requirements")
				}
			}
		}
	}

	if !foundAdd {
		t.Errorf("failed to find function Add in parsed package")
	}
	if !foundExtend {
		t.Errorf("failed to find function ExtendLoLToZ in parsed package")
	}
}

func TestLoadSpecNameTmpl(t *testing.T) {
	pkg := loadSpec(t)
	var found bool
	for _, f := range pkg.Funcs {
		if f.Name == "MaskFromBits" {
			found = true
			want := "{z}FromBits"
			if f.NameTmpl.tmpl != want {
				t.Errorf("MaskFromBits: expected NameTmpl.tmpl %q, got %q", want, f.NameTmpl.tmpl)
			}
			break
		}
	}
	if !found {
		t.Errorf("failed to find function MaskFromBits in parsed package")
	}
}

func TestNewSpecTemplate(t *testing.T) {
	tests := []struct {
		tmpl    string
		want    specTemplate
		wantErr bool
	}{
		{
			tmpl: "",
			want: specTemplate{tmpl: "", fields: nil},
		},
		{
			tmpl: "Convert",
			want: specTemplate{tmpl: "Convert", fields: nil},
		},
		{
			tmpl: "Convert{zL}To{zB}{zN}",
			want: specTemplate{
				tmpl:   "Convert{zL}To{zB}{zN}",
				fields: [][2]int{{7, 11}, {13, 17}, {17, 21}},
			},
		},
		{
			tmpl:    "Convert{zL",
			wantErr: true,
		},
		{
			tmpl:    "Convert}",
			wantErr: true,
		},
		{
			tmpl:    "Convert{a{b}}",
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
		if got.tmpl != tc.want.tmpl {
			t.Errorf("newSpecTemplate(%q) tmpl = %q, want %q", tc.tmpl, got.tmpl, tc.want.tmpl)
		}
		if len(got.fields) != len(tc.want.fields) {
			t.Errorf("newSpecTemplate(%q) fields len = %d, want %d", tc.tmpl, len(got.fields), len(tc.want.fields))
		} else {
			for i := range got.fields {
				if got.fields[i] != tc.want.fields[i] {
					t.Errorf("newSpecTemplate(%q) fields[%d] = %v, want %v", tc.tmpl, i, got.fields[i], tc.want.fields[i])
				}
			}
		}
	}
}

func TestSpecTemplateExpand(t *testing.T) {
	tmpl, err := newSpecTemplate("Convert{zL}To{zB}{zN}")
	if err != nil {
		t.Fatalf("unexpected error parsing template: %v", err)
	}

	lookup := func(name string) string {
		switch name {
		case "zL":
			return "4"
		case "zB":
			return "Float"
		case "zN":
			return "32"
		}
		return ""
	}

	got := tmpl.expand(lookup)
	want := "Convert4ToFloat32"
	if got != want {
		t.Errorf("expected expanded string %q, got %q", want, got)
	}
}
