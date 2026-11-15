// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"reflect"
	"testing"
)

func TestPR3ReinterpretNilInterface(t *testing.T) {
	var v any = nil
	got := Reinterpret[any](&v)
	if got != nil {
		t.Fatalf("Reinterpret[any](&v) = %v (%T), want nil", got, got)
	}
}

func TestPR3ReinterpretNonNilAnyRoundtrip(t *testing.T) {
	cases := []any{
		"payload",
		12345,
		[]int{1, 2, 3},
	}
	for _, tc := range cases {
		v := tc
		got := Reinterpret[any](&v)
		if !reflect.DeepEqual(got, tc) {
			t.Fatalf("Reinterpret[any](%v) = %v, want %v", tc, got, tc)
		}
	}
}
