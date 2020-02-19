// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import "testing"

type fakeCache struct {
	Cache

	id string
}

func (c fakeCache) ID() string {
	return c.id
}

func TestState(t *testing.T) {
	c1 := fakeCache{id: "1"}
	c2 := fakeCache{id: "2"}
	c3 := fakeCache{id: "3"}

	var s State
	s.AddCache(c1)
	s.AddCache(c2)
	s.AddCache(c3)

	compareCaches := func(desc string, want []fakeCache) {
		t.Run(desc, func(t *testing.T) {
			caches := s.Caches()
			if gotLen, wantLen := len(caches), len(want); gotLen != wantLen {
				t.Fatalf("len(Caches) = %d, want %d", gotLen, wantLen)
			}
			for i, got := range caches {
				if got != want[i] {
					t.Errorf("Caches[%d] = %v, want %v", i, got, want[i])
				}
			}
		})
	}

	compareCaches("initial load", []fakeCache{c1, c2, c3})
	s.DropCache(c2)
	compareCaches("dropped cache 2", []fakeCache{c1, c3})
	s.DropCache(c2)
	compareCaches("duplicate drop", []fakeCache{c1, c3})
	s.AddCache(c2)
	compareCaches("re-add cache 2", []fakeCache{c1, c3, c2})
	s.DropCache(c1)
	s.DropCache(c2)
	s.DropCache(c3)
	compareCaches("drop all", []fakeCache{})
}
