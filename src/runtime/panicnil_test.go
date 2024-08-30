// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"reflect"
	"runtime"
	"runtime/metrics"
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
	name := "/godebug/non-default-behavior/panicnil:events"
	s := []metrics.Sample{{Name: name}}
	metrics.Read(s)
	v1 := s[0].Value.Uint64()

	defer func() {
		e := recover()
		if reflect.TypeOf(e) != reflect.TypeOf(want) {
			println(e, want)
			t.Errorf("recover() = %v, want %v", e, want)
			panic(e)
		}
		metrics.Read(s)
		v2 := s[0].Value.Uint64()
		if want == nil {
			if v2 != v1+1 {
				t.Errorf("recover() with panicnil=1 did not increment metric %s", name)
			}
		} else {
			if v2 != v1 {
				t.Errorf("recover() with panicnil=0 incremented metric %s: %d -> %d", name, v1, v2)
			}
		}
	}()
	panic(nil)
}
