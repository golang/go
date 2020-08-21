// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue40954

/*
#include "jni.h"
*/
import "C"
import (
	"testing"
)

func Test40954(t *testing.T) {
	var x1 C.jmethodID = 0 // Note: 0, not nil. That makes sure we use uintptr for these types.
	_ = x1
	var x2 C.jfieldID = 0
	_ = x2
}
