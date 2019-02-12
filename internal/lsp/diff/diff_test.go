// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"reflect"
	"testing"
)

func TestDiff(t *testing.T) {
	for _, tt := range []struct {
		a, b       []string
		lines      []*Op
		operations []*Op
	}{
		{
			a: []string{"A", "B", "C", "A", "B", "B", "A"},
			b: []string{"C", "B", "A", "B", "A", "C"},
			operations: []*Op{
				&Op{Kind: Delete, I1: 0, I2: 1, J1: 0, J2: 0},
				&Op{Kind: Delete, I1: 1, I2: 2, J1: 0, J2: 0},
				&Op{Kind: Insert, Content: "B", I1: 3, I2: 3, J1: 1, J2: 2},
				&Op{Kind: Delete, I1: 5, I2: 6, J1: 4, J2: 4},
				&Op{Kind: Insert, Content: "C", I1: 7, I2: 7, J1: 5, J2: 6},
			},
		},
		{
			a: []string{"A", "B"},
			b: []string{"A", "C", ""},
			operations: []*Op{
				&Op{Kind: Delete, I1: 1, I2: 2, J1: 1, J2: 1},
				&Op{Kind: Insert, Content: "C", I1: 2, I2: 2, J1: 1, J2: 2},
				&Op{Kind: Insert, Content: "", I1: 2, I2: 2, J1: 2, J2: 3},
			},
		},
	} {
		ops := Operations(tt.a, tt.b)
		if len(ops) != len(tt.operations) {
			t.Fatalf("expected %v operations, got %v", len(tt.operations), len(ops))
		}
		for i, got := range ops {
			want := tt.operations[i]
			if !reflect.DeepEqual(want, got) {
				t.Errorf("expected %v, got %v", want, got)
			}
		}
		b := ApplyEdits(tt.a, tt.operations)
		for i, want := range tt.b {
			got := b[i]
			if got != want {
				t.Errorf("expected %v got %v", want, got)
			}
		}
	}
}
