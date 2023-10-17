// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intern

import (
	"fmt"
	"runtime"
	"testing"
)

func TestBasics(t *testing.T) {
	clearMap()
	foo := Get("foo")
	bar := Get("bar")
	empty := Get("")
	nilEface := Get(nil)
	i := Get(0x7777777)
	foo2 := Get("foo")
	bar2 := Get("bar")
	empty2 := Get("")
	nilEface2 := Get(nil)
	i2 := Get(0x7777777)
	foo3 := GetByString("foo")
	empty3 := GetByString("")

	if foo.Get() != foo2.Get() {
		t.Error("foo/foo2 values differ")
	}
	if foo.Get() != foo3.Get() {
		t.Error("foo/foo3 values differ")
	}
	if foo.Get() != "foo" {
		t.Error("foo.Get not foo")
	}
	if foo != foo2 {
		t.Error("foo/foo2 pointers differ")
	}
	if foo != foo3 {
		t.Error("foo/foo3 pointers differ")
	}

	if bar.Get() != bar2.Get() {
		t.Error("bar values differ")
	}
	if bar.Get() != "bar" {
		t.Error("bar.Get not bar")
	}
	if bar != bar2 {
		t.Error("bar pointers differ")
	}

	if i.Get() != i.Get() {
		t.Error("i values differ")
	}
	if i.Get() != 0x7777777 {
		t.Error("i.Get not 0x7777777")
	}
	if i != i2 {
		t.Error("i pointers differ")
	}

	if empty.Get() != empty2.Get() {
		t.Error("empty/empty2 values differ")
	}
	if empty.Get() != empty.Get() {
		t.Error("empty/empty3 values differ")
	}
	if empty.Get() != "" {
		t.Error("empty.Get not empty string")
	}
	if empty != empty2 {
		t.Error("empty/empty2 pointers differ")
	}
	if empty != empty3 {
		t.Error("empty/empty3 pointers differ")
	}

	if nilEface.Get() != nilEface2.Get() {
		t.Error("nilEface values differ")
	}
	if nilEface.Get() != nil {
		t.Error("nilEface.Get not nil")
	}
	if nilEface != nilEface2 {
		t.Error("nilEface pointers differ")
	}

	if n := mapLen(); n != 5 {
		t.Errorf("map len = %d; want 4", n)
	}

	wantEmpty(t)
}

func wantEmpty(t testing.TB) {
	t.Helper()
	const gcTries = 5000
	for try := 0; try < gcTries; try++ {
		runtime.GC()
		n := mapLen()
		if n == 0 {
			break
		}
		if try == gcTries-1 {
			t.Errorf("map len = %d after (%d GC tries); want 0, contents: %v", n, gcTries, mapKeys())
		}
	}
}

func TestStress(t *testing.T) {
	iters := 10000
	if testing.Short() {
		iters = 1000
	}
	var sink []byte
	for i := 0; i < iters; i++ {
		_ = Get("foo")
		sink = make([]byte, 1<<20)
	}
	_ = sink
}

func BenchmarkStress(b *testing.B) {
	done := make(chan struct{})
	defer close(done)
	go func() {
		for {
			select {
			case <-done:
				return
			default:
			}
			runtime.GC()
		}
	}()

	clearMap()
	v1 := Get("foo")
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v2 := Get("foo")
			if v1 != v2 {
				b.Fatal("wrong value")
			}
			// And also a key we don't retain:
			_ = Get("bar")
		}
	})
	runtime.GC()
	wantEmpty(b)
}

func mapLen() int {
	mu.Lock()
	defer mu.Unlock()
	return len(valMap)
}

func mapKeys() (keys []string) {
	mu.Lock()
	defer mu.Unlock()
	for k := range valMap {
		keys = append(keys, fmt.Sprint(k))
	}
	return keys
}

func clearMap() {
	mu.Lock()
	defer mu.Unlock()
	clear(valMap)
}

var (
	globalString = "not a constant"
	sink         string
)

func TestGetByStringAllocs(t *testing.T) {
	allocs := int(testing.AllocsPerRun(100, func() {
		GetByString(globalString)
	}))
	if allocs != 0 {
		t.Errorf("GetString allocated %d objects, want 0", allocs)
	}
}

func BenchmarkGetByString(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v := GetByString(globalString)
		sink = v.Get().(string)
	}
}
