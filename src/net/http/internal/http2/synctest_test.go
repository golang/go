// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2_test

import (
	"testing"
	"testing/synctest"
)

func synctestTest(t *testing.T, f func(t testing.TB)) {
	t.Helper()
	synctest.Test(t, func(t *testing.T) {
		t.Helper()
		f(t)
	})
}

// synctestSubtest starts a subtest and runs f in a synctest bubble within it.
func synctestSubtest(t *testing.T, name string, f func(testing.TB)) {
	t.Helper()
	t.Run(name, func(t *testing.T) {
		synctestTest(t, f)
	})
}
