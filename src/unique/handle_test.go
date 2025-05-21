// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique

import (
	"fmt"
	"internal/abi"
	"internal/asan"
	"internal/msan"
	"internal/race"
	"internal/testenv"
	"math/rand/v2"
	"reflect"
	"runtime"
	"strconv"
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
type testZeroSize struct{}
type testNestedHandle struct {
	next Handle[testNestedHandle]
	arr  [6]int
}

func TestHandle(t *testing.T) {
	testHandle(t, testString("foo"))
	testHandle(t, testString("bar"))
	testHandle(t, testString(""))
	testHandle(t, testIntArray{7, 77, 777, 7777})
	testHandle(t, testEface(nil))
	testHandle(t, testStringArray{"a", "b", "c"})
	testHandle(t, testStringStruct{"x"})
	testHandle(t, testStringStructArrayStruct{
		s: [2]testStringStruct{{"y"}, {"z"}},
	})
	testHandle(t, testStruct{0.5, "184"})
	testHandle(t, testEface("hello"))
	testHandle(t, testZeroSize(struct{}{}))
}

func testHandle[T comparable](t *testing.T, value T) {
	name := reflect.TypeFor[T]().Name()
	t.Run(fmt.Sprintf("%s/%#v", name, value), func(t *testing.T) {
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

		drainMaps[T](t)
		checkMapsFor(t, value)
	})
}

// drainMaps ensures that the internal maps are drained.
func drainMaps[T comparable](t *testing.T) {
	t.Helper()

	if unsafe.Sizeof(*(new(T))) == 0 {
		return // zero-size types are not inserted.
	}
	drainCleanupQueue(t)
}

func drainCleanupQueue(t *testing.T) {
	t.Helper()

	runtime.GC() // Queue up the cleanups.
	runtime_blockUntilEmptyCleanupQueue(int64(5 * time.Second))
}

func checkMapsFor[T comparable](t *testing.T, value T) {
	// Manually load the value out of the map.
	typ := abi.TypeFor[T]()
	a, ok := uniqueMaps.Load(typ)
	if !ok {
		return
	}
	m := a.(*uniqueMap[T])
	p := m.Load(value)
	if p != nil {
		t.Errorf("value %v still referenced by a handle (or tiny block?): internal pointer %p", value, p)
	}
}

func TestMakeClonesStrings(t *testing.T) {
	s := strings.Clone("abcdefghijklmnopqrstuvwxyz") // N.B. Must be big enough to not be tiny-allocated.
	ran := make(chan bool)
	runtime.AddCleanup(unsafe.StringData(s), func(ch chan bool) {
		ch <- true
	}, ran)
	h := Make(s)

	// Clean up s (hopefully) and run the cleanup.
	runtime.GC()

	select {
	case <-time.After(1 * time.Second):
		t.Fatal("string was improperly retained")
	case <-ran:
	}
	runtime.KeepAlive(h)
}

func TestHandleUnsafeString(t *testing.T) {
	var testData []string
	for i := range 1024 {
		testData = append(testData, strconv.Itoa(i))
	}
	var buf []byte
	var handles []Handle[string]
	for _, s := range testData {
		if len(buf) < len(s) {
			buf = make([]byte, len(s)*2)
		}
		copy(buf, s)
		sbuf := unsafe.String(&buf[0], len(s))
		handles = append(handles, Make(sbuf))
	}
	for i, s := range testData {
		h := Make(s)
		if handles[i].Value() != h.Value() {
			t.Fatal("unsafe string improperly retained internally")
		}
	}
}

func nestHandle(n testNestedHandle) testNestedHandle {
	return testNestedHandle{
		next: Make(n),
		arr:  n.arr,
	}
}

func TestNestedHandle(t *testing.T) {
	n0 := testNestedHandle{arr: [6]int{1, 2, 3, 4, 5, 6}}
	n1 := nestHandle(n0)
	n2 := nestHandle(n1)
	n3 := nestHandle(n2)

	if v := n3.next.Value(); v != n2 {
		t.Errorf("n3.Value != n2: %#v vs. %#v", v, n2)
	}
	if v := n2.next.Value(); v != n1 {
		t.Errorf("n2.Value != n1: %#v vs. %#v", v, n1)
	}
	if v := n1.next.Value(); v != n0 {
		t.Errorf("n1.Value != n0: %#v vs. %#v", v, n0)
	}

	// In a good implementation, the entire chain, down to the bottom-most
	// value, should all be gone after we drain the maps.
	drainMaps[testNestedHandle](t)
	checkMapsFor(t, n0)
}

// Implemented in runtime.
//
// Used only by tests.
//
//go:linkname runtime_blockUntilEmptyCleanupQueue
func runtime_blockUntilEmptyCleanupQueue(timeout int64) bool

var (
	randomNumber = rand.IntN(1000000) + 1000000
	heapBytes    = newHeapBytes()
	heapString   = newHeapString()

	stringHandle Handle[string]
	intHandle    Handle[int]
	anyHandle    Handle[any]
	pairHandle   Handle[[2]string]
)

func TestMakeAllocs(t *testing.T) {
	errorf := t.Errorf
	if race.Enabled || msan.Enabled || asan.Enabled || testenv.OptimizationOff() {
		errorf = t.Logf
	}

	tests := []struct {
		name   string
		allocs int
		f      func()
	}{
		{name: "create heap bytes", allocs: 1, f: func() {
			heapBytes = newHeapBytes()
		}},

		{name: "create heap string", allocs: 2, f: func() {
			heapString = newHeapString()
		}},

		{name: "static string", allocs: 0, f: func() {
			stringHandle = Make("this string is statically allocated")
		}},

		{name: "heap string", allocs: 0, f: func() {
			stringHandle = Make(heapString)
		}},

		{name: "stack string", allocs: 0, f: func() {
			var b [16]byte
			b[8] = 'a'
			stringHandle = Make(string(b[:]))
		}},

		{name: "bytes", allocs: 1, f: func() {
			stringHandle = Make(string(heapBytes))
		}},

		{name: "bytes truncated short", allocs: 0, f: func() {
			stringHandle = Make(string(heapBytes[:16]))
		}},

		{name: "bytes truncated long", allocs: 1, f: func() {
			stringHandle = Make(string(heapBytes[:40]))
		}},

		{name: "string to any", allocs: 1, f: func() {
			anyHandle = Make[any](heapString)
		}},

		{name: "large number", allocs: 0, f: func() {
			intHandle = Make(randomNumber)
		}},

		{name: "large number to any", allocs: 1, f: func() {
			anyHandle = Make[any](randomNumber)
		}},

		{name: "pair", allocs: 0, f: func() {
			pairHandle = Make([2]string{heapString, heapString})
		}},

		{name: "pair from stack", allocs: 0, f: func() {
			var b [16]byte
			b[8] = 'a'
			pairHandle = Make([2]string{string(b[:]), string(b[:])})
		}},

		{name: "pair to any", allocs: 1, f: func() {
			anyHandle = Make[any]([2]string{heapString, heapString})
		}},
	}

	for _, tt := range tests {
		allocs := testing.AllocsPerRun(100, tt.f)
		if allocs != float64(tt.allocs) {
			errorf("%s: got %v allocs, want %v", tt.name, allocs, tt.allocs)
		}
	}
}

//go:noinline
func newHeapBytes() []byte {
	const N = 100
	b := make([]byte, N)
	for i := range b {
		b[i] = byte(i)
	}
	return b
}

//go:noinline
func newHeapString() string {
	return string(newHeapBytes())
}
