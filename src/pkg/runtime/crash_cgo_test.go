// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo

package runtime_test

import (
	"testing"
)

func TestCgoCrashHandler(t *testing.T) {
	testCrashHandler(t, &crashTest{Cgo: true})
}
