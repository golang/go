// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package marker

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// Note: we use a separate package for the marker tests so that we can easily
// compare their performance to the existing marker tests in ./internal/lsp.

// TestMarkers runs the marker tests from the testdata directory.
//
// See RunMarkerTests for details on how marker tests work.
func TestMarkers(t *testing.T) {
	RunMarkerTests(t, "testdata")
}
