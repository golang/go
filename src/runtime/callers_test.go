// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"strings"
	"testing"
)

func f1(pan bool) []uintptr {
	return f2(pan) // line 14
}

func f2(pan bool) []uintptr {
	return f3(pan) // line 18
}

func f3(pan bool) []uintptr {
	if pan {
		panic("f3") // line 23
	}
	ret := make([]uintptr, 20)
	return ret[:runtime.Callers(0, ret)] // line 26
}

func testCallers(t *testing.T, pcs []uintptr, pan bool) {
	m := make(map[string]int, len(pcs))
	frames := runtime.CallersFrames(pcs)
	for {
		frame, more := frames.Next()
		if frame.Function != "" {
			m[frame.Function] = frame.Line
		}
		if !more {
			break
		}
	}

	var seen []string
	for k := range m {
		seen = append(seen, k)
	}
	t.Logf("functions seen: %s", strings.Join(seen, " "))

	var f3Line int
	if pan {
		f3Line = 23
	} else {
		f3Line = 26
	}
	want := []struct {
		name string
		line int
	}{
		{"f1", 14},
		{"f2", 18},
		{"f3", f3Line},
	}
	for _, w := range want {
		if got := m["runtime_test."+w.name]; got != w.line {
			t.Errorf("%s is line %d, want %d", w.name, got, w.line)
		}
	}
}

func TestCallers(t *testing.T) {
	testCallers(t, f1(false), false)
}

func TestCallersPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("did not panic")
		}
		pcs := make([]uintptr, 20)
		pcs = pcs[:runtime.Callers(0, pcs)]
		testCallers(t, pcs, true)
	}()
	f1(true)
}
