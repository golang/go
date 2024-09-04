// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique

import (
	"fmt"
	"internal/abi"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
	"unsafe"
)

// Set up special types. Because the internal maps are sharded by type,
// this will ensure that we're not overlapping with other tests.
type testString string
type testIntArray [4]int
type testEface any
type testStringArray [3]string
type testStringStruct struct {
	a string
}
type testStringStructArrayStruct struct {
	s [2]testStringStruct
}
type testStruct struct {
	z float64
	b string
}

func TestHandle(t *testing.T) {
	testHandle[testString](t, "foo")
	testHandle[testString](t, "bar")
	testHandle[testString](t, "")
	testHandle[testIntArray](t, [4]int{7, 77, 777, 7777})
	testHandle[testEface](t, nil)
	testHandle[testStringArray](t, [3]string{"a", "b", "c"})
	testHandle[testStringStruct](t, testStringStruct{"x"})
	testHandle[testStringStructArrayStruct](t, testStringStructArrayStruct{
		s: [2]testStringStruct{testStringStruct{"y"}, testStringStruct{"z"}},
	})
	testHandle[testStruct](t, testStruct{0.5, "184"})
	testHandle[testEface](t, testEface("hello"))
}

func testHandle[T comparable](t *testing.T, value T) {
	name := reflect.TypeFor[T]().Name()
	t.Run(fmt.Sprintf("%s/%#v", name, value), func(t *testing.T) {
		t.Parallel()

		v0 := Make(value)
		v1 := Make(value)

		if v0.Value() != v1.Value() {
			t.Error("v0.Value != v1.Value")
		}
		if v0.Value() != value {
			t.Errorf("v0.Value not %#v", value)
		}
		if v0 != v1 {
			t.Error("v0 != v1")
		}

		drainMaps(t)
		checkMapsFor(t, value)
	})
}

// drainMaps ensures that the internal maps are drained.
func drainMaps(t *testing.T) {
	t.Helper()

	wait := make(chan struct{}, 1)

	// Set up a one-time notification for the next time the cleanup runs.
	// Note: this will only run if there's no other active cleanup, so
	// we can be sure that the next time cleanup runs, it'll see the new
	// notification.
	cleanupMu.Lock()
	cleanupNotify = append(cleanupNotify, func() {
		select {
		case wait <- struct{}{}:
		default:
		}
	})

	runtime.GC()
	cleanupMu.Unlock()

	// Wait until cleanup runs.
	<-wait
}

func checkMapsFor[T comparable](t *testing.T, value T) {
	// Manually load the value out of the map.
	typ := abi.TypeFor[T]()
	a, ok := uniqueMaps.Load(typ)
	if !ok {
		return
	}
	m := a.(*uniqueMap[T])
	wp, ok := m.Load(value)
	if !ok {
		return
	}
	if wp.Strong() != nil {
		t.Errorf("value %v still referenced a handle (or tiny block?) ", value)
		return
	}
	t.Errorf("failed to drain internal maps of %v", value)
}

func TestMakeClonesStrings(t *testing.T) {
	s := strings.Clone("abcdefghijklmnopqrstuvwxyz") // N.B. Must be big enough to not be tiny-allocated.
	ran := make(chan bool)
	runtime.SetFinalizer(unsafe.StringData(s), func(_ *byte) {
		ran <- true
	})
	h := Make(s)

	// Clean up s (hopefully) and run the finalizer.
	runtime.GC()

	select {
	case <-time.After(1 * time.Second):
		t.Fatal("string was improperly retained")
	case <-ran:
	}
	runtime.KeepAlive(h)
}
