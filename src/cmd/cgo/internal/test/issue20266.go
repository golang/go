// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20266: use -I with a relative path.

package cgotest

/*
#cgo CFLAGS: -I issue20266 -Iissue20266 -Ddef20266
#include "issue20266.h"
*/
import "C"

import "testing"

func test20266(t *testing.T) {
	if got, want := C.issue20266, 20266; got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}
