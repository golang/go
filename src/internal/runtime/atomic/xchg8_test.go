// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64 || arm || arm64 || ppc64 || ppc64le

package atomic_test

import (
	"internal/runtime/atomic"
	"testing"
)

func TestXchg8(t *testing.T) {
	var a [16]uint8
	for i := range a {
		next := uint8(i + 50)
		a[i] = next
	}
	b := a

	// Compare behavior against non-atomic implementation. Expect the operation
	// to work at any byte offset and to not clobber neighboring values.
	for i := range a {
		next := uint8(i + 100)
		pa := atomic.Xchg8(&a[i], next)
		pb := b[i]
		b[i] = next
		if pa != pb {
			t.Errorf("atomic.Xchg8(a[%d]); %d != %d", i, pa, pb)
		}
		if a != b {
			t.Errorf("after atomic.Xchg8(a[%d]); %d != %d", i, a, b)
		}
		if t.Failed() {
			break
		}
	}
}
