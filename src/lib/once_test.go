// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package once

import (
	"once";
	"testing";
)

var ncall int;
func Call() {
	ncall++
}

export func TestOnce(t *testing.T) {
	ncall = 0;
	once.Do(&Call);
	if ncall != 1 {
		t.Fatalf("once.Do(&Call) didn't Call(): ncall=%d", ncall);
	}
	once.Do(&Call);
	if ncall != 1 {
		t.Fatalf("second once.Do(&Call) did Call(): ncall=%d", ncall);
	}
	once.Do(&Call);
	if ncall != 1 {
		t.Fatalf("third once.Do(&Call) did Call(): ncall=%d", ncall);
	}
}
