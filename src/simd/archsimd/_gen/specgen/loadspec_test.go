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
			want := "{{.z}}FromBits"
			if f.NameTmpl.raw != want {
				t.Errorf("MaskFromBits: expected NameTmpl.raw %q, got %q", want, f.NameTmpl.raw)
			}
			break
		}
	}
	if !found {
		t.Errorf("failed to find function MaskFromBits in parsed package")
	}
}
