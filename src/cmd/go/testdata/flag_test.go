// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag_test

import (
	"flag"
	"testing"
)

var v = flag.Int("v", 0, "v flag")

// Run this as go test pkg -args -v=7
func TestVFlagIsSet(t *testing.T) {
	if *v != 7 {
		t.Fatal("v flag not set")
	}
}
