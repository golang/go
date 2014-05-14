// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intsets

import "testing"

func TestNLZ(t *testing.T) {
	if x := nlz(0x0000801000000000); x != 16 {
		t.Errorf("bad %d", x)
	}
}

// Backdoor for testing.
func (s *Sparse) Check() error { return s.check() }
