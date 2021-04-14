// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"testing"
)

func TestDump(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}

	// provide a dummy error handler so parsing doesn't stop after first error
	ast, err := ParseFile(*src_, func(error) {}, nil, CheckBranches)
	if err != nil {
		t.Error(err)
	}

	if ast != nil {
		Fdump(testOut(), ast)
	}
}
