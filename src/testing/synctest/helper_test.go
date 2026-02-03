// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package synctest_test

import "testing"

// helperLog is a t.Helper which logs.
// Since it is a helper, the log prefix should contain
// the caller's file, not helper_test.go.
func helperLog(t *testing.T, s string) {
	t.Helper()
	t.Log(s)
}
