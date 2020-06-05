// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue27054

/*
#include "egl.h"
*/
import "C"
import (
	"testing"
)

func Test27054(t *testing.T) {
	var (
		// Note: 0, not nil. That makes sure we use uintptr for these types.
		_ C.EGLDisplay = 0
		_ C.EGLConfig  = 0
	)
}
