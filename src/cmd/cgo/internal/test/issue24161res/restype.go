// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package issue24161res

/*
#cgo LDFLAGS: -framework CoreFoundation
#include <CoreFoundation/CoreFoundation.h>
*/
import "C"
import (
	"reflect"
	"testing"
)

func Test(t *testing.T) {
	if k := reflect.TypeOf(C.CFArrayCreate(0, nil, 0, nil)).Kind(); k != reflect.Uintptr {
		t.Fatalf("bad kind %s\n", k)
	}
}
