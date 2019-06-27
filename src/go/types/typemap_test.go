// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "testing"

func TestTypeMap(t *testing.T) {
	var m *TypeMap
	if n := m.Len(); n != 0 {
		t.Errorf("got %d entries for zero type map; want 0", n)
	}

	m = new(TypeMap)
	if n := m.Len(); n != 0 {
		t.Errorf("got %d entries for empty type map; want 0", n)
	}

	prev := m.Insert(universeByte, "byte")
	if prev != nil {
		t.Errorf("got %v as prev entry; want nil", prev)
	}
	if n := m.Len(); n != 1 {
		t.Errorf("got %d entries for type map; want 1", n)
	}
	if v := m.At(universeByte); v != "byte" {
		t.Errorf("got %v => %v; want %v", universeByte, v, "byte")
	}

	// TODO(gri) add more tests
}
