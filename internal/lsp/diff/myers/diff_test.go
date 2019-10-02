// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package myers_test

import (
	"testing"

	"golang.org/x/tools/internal/lsp/diff/difftest"
	"golang.org/x/tools/internal/lsp/diff/myers"
)

func TestDiff(t *testing.T) {
	difftest.DiffTest(t, myers.ComputeEdits)
}
