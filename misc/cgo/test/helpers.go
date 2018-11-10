// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// const char *greeting = "hello, world";
import "C"

import (
	"reflect"
	"testing"
	"unsafe"
)

const greeting = "hello, world"

type testPair struct {
	Name      string
	Got, Want interface{}
}

var testPairs = []testPair{
	{"GoString", C.GoString(C.greeting), greeting},
	{"GoStringN", C.GoStringN(C.greeting, 5), greeting[:5]},
	{"GoBytes", C.GoBytes(unsafe.Pointer(C.greeting), 5), []byte(greeting[:5])},
}

func testHelpers(t *testing.T) {
	for _, pair := range testPairs {
		if !reflect.DeepEqual(pair.Got, pair.Want) {
			t.Errorf("%s: got %#v, want %#v", pair.Name, pair.Got, pair.Want)
		}
	}
}
