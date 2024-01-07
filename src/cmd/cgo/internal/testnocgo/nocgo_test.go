// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nocgo

import "testing"

func TestNop(t *testing.T) {
	i := NoCgo()
	if i != 42 {
		t.Errorf("got %d, want %d", i, 42)
	}
}
