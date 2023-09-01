// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package persistent_test

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/tools/internal/constraints"
	"golang.org/x/tools/internal/persistent"
)

func TestSet(t *testing.T) {
	const (
		add = iota
		remove
	)
	type op struct {
		op int
		v  int
	}

	tests := []struct {
		label string
		ops   []op
		want  []int
	}{
		{"empty", nil, nil},
		{"singleton", []op{{add, 1}}, []int{1}},
		{"add and remove", []op{
			{add, 1},
			{remove, 1},
		}, nil},
		{"interleaved and remove", []op{
			{add, 1},
			{add, 2},
			{remove, 1},
			{add, 3},
		}, []int{2, 3}},
	}

	for _, test := range tests {
		t.Run(test.label, func(t *testing.T) {
			var s persistent.Set[int]
			for _, op := range test.ops {
				switch op.op {
				case add:
					s.Add(op.v)
				case remove:
					s.Remove(op.v)
				}
			}

			if d := diff(&s, test.want); d != "" {
				t.Errorf("unexpected diff:\n%s", d)
			}
		})
	}
}

func TestSet_Clone(t *testing.T) {
	s1 := new(persistent.Set[int])
	s1.Add(1)
	s1.Add(2)
	s2 := s1.Clone()
	s1.Add(3)
	s2.Add(4)
	if d := diff(s1, []int{1, 2, 3}); d != "" {
		t.Errorf("s1: unexpected diff:\n%s", d)
	}
	if d := diff(s2, []int{1, 2, 4}); d != "" {
		t.Errorf("s2: unexpected diff:\n%s", d)
	}
}

func TestSet_AddAll(t *testing.T) {
	s1 := new(persistent.Set[int])
	s1.Add(1)
	s1.Add(2)
	s2 := new(persistent.Set[int])
	s2.Add(2)
	s2.Add(3)
	s2.Add(4)
	s3 := new(persistent.Set[int])

	s := new(persistent.Set[int])
	s.AddAll(s1)
	s.AddAll(s2)
	s.AddAll(s3)

	if d := diff(s1, []int{1, 2}); d != "" {
		t.Errorf("s1: unexpected diff:\n%s", d)
	}
	if d := diff(s2, []int{2, 3, 4}); d != "" {
		t.Errorf("s2: unexpected diff:\n%s", d)
	}
	if d := diff(s3, nil); d != "" {
		t.Errorf("s3: unexpected diff:\n%s", d)
	}
	if d := diff(s, []int{1, 2, 3, 4}); d != "" {
		t.Errorf("s: unexpected diff:\n%s", d)
	}
}

func diff[K constraints.Ordered](got *persistent.Set[K], want []K) string {
	wantSet := make(map[K]struct{})
	for _, w := range want {
		wantSet[w] = struct{}{}
	}
	var diff []string
	got.Range(func(key K) {
		if _, ok := wantSet[key]; !ok {
			diff = append(diff, fmt.Sprintf("+%v", key))
		}
	})
	for key := range wantSet {
		if !got.Contains(key) {
			diff = append(diff, fmt.Sprintf("-%v", key))
		}
	}
	if len(diff) > 0 {
		d := new(strings.Builder)
		for _, l := range diff {
			fmt.Fprintln(d, l)
		}
		return d.String()
	}
	return ""
}
