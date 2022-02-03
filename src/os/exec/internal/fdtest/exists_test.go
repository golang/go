// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fdtest

import (
	"os"
	"runtime"
	"testing"
)

func TestExists(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Exists not implemented for windows")
	}

	if !Exists(os.Stdout.Fd()) {
		t.Errorf("Exists(%d) got false want true", os.Stdout.Fd())
	}
}
