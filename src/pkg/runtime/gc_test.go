// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"os"
	"runtime"
	"runtime/debug"
	"testing"
)

func TestGcSys(t *testing.T) {
	if os.Getenv("GOGC") == "off" {
		t.Skip("skipping test; GOGC=off in environment")
	}
	data := struct{ Short bool }{testing.Short()}
	got := executeTest(t, testGCSysSource, &data)
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got %q", want, got)
	}
}

const testGCSysSource = `
package main

import (
	"fmt"
	"runtime"
)

func main() {
	runtime.GOMAXPROCS(1)
	memstats := new(runtime.MemStats)
	runtime.GC()
	runtime.ReadMemStats(memstats)
	sys := memstats.Sys

	runtime.MemProfileRate = 0 // disable profiler

	itercount := 1000000
{{if .Short}}
	itercount = 100000
{{end}}
	for i := 0; i < itercount; i++ {
		workthegc()
	}

	// Should only be using a few MB.
	// We allocated 100 MB or (if not short) 1 GB.
	runtime.ReadMemStats(memstats)
	if sys > memstats.Sys {
		sys = 0
	} else {
		sys = memstats.Sys - sys
	}
	if sys > 16<<20 {
		fmt.Printf("using too much memory: %d bytes\n", sys)
		return
	}
	fmt.Printf("OK\n")
}

func workthegc() []byte {
	return make([]byte, 1029)
}
`

func TestGcDeepNesting(t *testing.T) {
	type T [2][2][2][2][2][2][2][2][2][2]*int
	a := new(T)

	// Prevent the compiler from applying escape analysis.
	// This makes sure new(T) is allocated on heap, not on the stack.
	t.Logf("%p", a)

	a[0][0][0][0][0][0][0][0][0][0] = new(int)
	*a[0][0][0][0][0][0][0][0][0][0] = 13
	runtime.GC()
	if *a[0][0][0][0][0][0][0][0][0][0] != 13 {
		t.Fail()
	}
}

func TestGcHashmapIndirection(t *testing.T) {
	defer debug.SetGCPercent(debug.SetGCPercent(1))
	runtime.GC()
	type T struct {
		a [256]int
	}
	m := make(map[T]T)
	for i := 0; i < 2000; i++ {
		var a T
		a.a[0] = i
		m[a] = T{}
	}
}
