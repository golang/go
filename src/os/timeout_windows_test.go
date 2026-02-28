// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"os"
	"testing"
)

func init() {
	pipeDeadlinesTestCases = []pipeDeadlineTest{
		{
			"named overlapped pipe",
			func(t *testing.T) (r, w *os.File) {
				name := pipeName()
				w = newBytePipe(t, name, true)
				r = newFileOverlapped(t, name, true)
				return
			},
		},
	}
}
