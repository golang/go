// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"go/build"
	"internal/testenv"
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	build.Default.GOROOT = testenv.GOROOT(nil)
	os.Exit(m.Run())
}
