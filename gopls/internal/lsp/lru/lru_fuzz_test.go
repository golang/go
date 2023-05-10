// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package lru_test

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/lru"
)

// Simple fuzzing test for consistency.
func FuzzCache(f *testing.F) {
	type op struct {
		set        bool
		key, value byte
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var ops []op
		for len(data) >= 3 {
			ops = append(ops, op{data[0]%2 == 0, data[1], data[2]})
			data = data[3:]
		}
		cache := lru.New(100)
		var reference [256]byte
		for _, op := range ops {
			if op.set {
				reference[op.key] = op.value
				cache.Set(op.key, op.value, 1)
			} else {
				if v := cache.Get(op.key); v != nil && v != reference[op.key] {
					t.Fatalf("cache.Get(%d) = %d, want %d", op.key, v, reference[op.key])
				}
			}
		}
	})
}
