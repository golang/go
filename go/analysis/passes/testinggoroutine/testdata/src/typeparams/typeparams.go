// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
	"testing"
)

func f[P any](t *testing.T) {
	t.Fatal("failed")
}

func TestBadFatalf[P any](t *testing.T) {
	go f[int](t) // want "call to .+T.+Fatal from a non-test goroutine"
}
