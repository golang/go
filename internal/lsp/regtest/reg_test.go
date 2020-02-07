// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"os"
	"testing"
	"time"
)

var runner *Runner

func TestMain(m *testing.M) {
	runner = NewTestRunner(AllModes, 30*time.Second)
	defer runner.Close()
	os.Exit(m.Run())
}
