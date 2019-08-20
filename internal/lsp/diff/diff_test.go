// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diff_test

import (
	"testing"

	"golang.org/x/tools/internal/lsp/diff/difftest"
)

func TestDiff(t *testing.T) {
	difftest.DiffTest(t)
}
