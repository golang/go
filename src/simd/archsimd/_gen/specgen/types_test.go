// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"go/types"
	"testing"
)

func TestTypeSet(t *testing.T) {
	pkg := loadSpec(t)

	obj := pkg.Pkg.Scope().Lookup("Nums")
	if obj == nil {
		t.Fatalf("failed to find 'Nums' in spec package scope")
	}

	typeName, ok := obj.(*types.TypeName)
	if !ok {
		t.Fatalf("Nums is not a TypeName, got %T", obj)
	}

	typesList := typeSet(typeName.Type())

	expected := []string{
		"float32", "float64",
		"int8", "int16", "int32", "int64",
		"uint8", "uint16", "uint32", "uint64",
	}

	if len(typesList) != len(expected) {
		t.Errorf("expected %d types, got %d", len(expected), len(typesList))
	}

	found := make(map[string]bool)
	for _, ty := range typesList {
		found[ty.String()] = true
	}

	for _, exp := range expected {
		if !found[exp] {
			t.Errorf("expected type %s not found in satisfying list", exp)
		}
	}
}
