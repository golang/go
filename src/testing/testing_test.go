// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"os"
	"testing"
)

// This is exactly what a test would do without a TestMain.
// It's here only so that there is at least one package in the
// standard library with a TestMain, so that code is executed.

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
