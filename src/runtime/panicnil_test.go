// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"reflect"
	"runtime"
	"testing"
)

func TestPanicNil(t *testing.T) {
	t.Run("default", func(t *testing.T) {
		checkPanicNil(t, new(runtime.PanicNilError))
	})
	t.Run("GODEBUG=panicnil=0", func(t *testing.T) {
		t.Setenv("GODEBUG", "panicnil=0")
		checkPanicNil(t, new(runtime.PanicNilError))
	})
	t.Run("GODEBUG=panicnil=1", func(t *testing.T) {
		t.Setenv("GODEBUG", "panicnil=1")
		checkPanicNil(t, nil)
	})
}

func checkPanicNil(t *testing.T, want any) {
	defer func() {
		e := recover()
		if reflect.TypeOf(e) != reflect.TypeOf(want) {
			println(e, want)
			t.Errorf("recover() = %v, want %v", e, want)
		}
	}()
	panic(nil)
}
