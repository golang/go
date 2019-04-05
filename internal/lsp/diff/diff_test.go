// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff

import (
	"reflect"
	"strings"
	"testing"
)

func TestDiff(t *testing.T) {
	for _, test := range []struct {
		a, b       string
		lines      []*Op
		operations []*Op
	}{
		{
			a: "A\nB\nC\nA\nB\nB\nA\n",
			b: "C\nB\nA\nB\nA\nC\n",
			operations: []*Op{
				&Op{Kind: Delete, I1: 0, I2: 1, J1: 0},
				&Op{Kind: Delete, I1: 1, I2: 2, J1: 0},
				&Op{Kind: Insert, Content: []string{"B\n"}, I1: 3, I2: 3, J1: 1},
				&Op{Kind: Delete, I1: 5, I2: 6, J1: 4},
				&Op{Kind: Insert, Content: []string{"C\n"}, I1: 7, I2: 7, J1: 5},
			},
		},
		{
			a: "A\nB\n",
			b: "A\nC\n\n",
			operations: []*Op{
				&Op{Kind: Delete, I1: 1, I2: 2, J1: 1},
				&Op{Kind: Insert, Content: []string{"C\n"}, I1: 2, I2: 2, J1: 1},
				&Op{Kind: Insert, Content: []string{"\n"}, I1: 2, I2: 2, J1: 2},
			},
		},
	} {
		a := strings.SplitAfter(test.a, "\n")
		b := strings.SplitAfter(test.b, "\n")
		ops := Operations(a, b)
		if len(ops) != len(test.operations) {
			t.Fatalf("expected %v operations, got %v", len(test.operations), len(ops))
		}
		for i, got := range ops {
			want := test.operations[i]
			if !reflect.DeepEqual(want, got) {
				t.Errorf("expected %v, got %v", want, got)
			}
		}
		applied := ApplyEdits(a, test.operations)
		for i, want := range applied {
			got := b[i]
			if got != want {
				t.Errorf("expected %v got %v", want, got)
			}
		}
	}
}
