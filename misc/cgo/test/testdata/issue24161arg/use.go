// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package issue24161arg

/*
#cgo LDFLAGS: -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
*/
import "C"
import "testing"

func Test(t *testing.T) {
	a := test24161array()
	C.CFArrayCreateCopy(0, a)
}
