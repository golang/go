// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2_test

import (
	"testing"
	"testing/synctest"
)

// synctestSubtest starts a subtest and runs f in a synctest bubble within it.
func synctestSubtest(t *testing.T, name string, f func(*testing.T)) {
	t.Helper()
	t.Run(name, func(t *testing.T) {
		synctest.Test(t, f)
	})
}
