// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package once

import (
	"once";
	"testing";
)

var ncall int;
func call() {
	ncall++
}

export func TestOnce(t *testing.T) {
	ncall = 0;
	once.Do(&call);
	if ncall != 1 {
		t.Fatalf("once.Do(&call) didn't call(): ncall=%d", ncall);
	}
	once.Do(&call);
	if ncall != 1 {
		t.Fatalf("second once.Do(&call) did call(): ncall=%d", ncall);
	}
	once.Do(&call);
	if ncall != 1 {
		t.Fatalf("third once.Do(&call) did call(): ncall=%d", ncall);
	}
}
