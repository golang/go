// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdeps

import (
	"bufio"
	"strconv"
	"strings"
	"testing"
)

func TestTestLogDedup(t *testing.T) {
	var b strings.Builder
	l := &testLog{w: bufio.NewWriter(&b)}
	l.Stat("a")
	l.Stat("a")
	l.Open("a")
	l.Getenv("HOME")
	l.Getenv("HOME")
	l.Chdir("/x")
	l.Stat("a") // relative to /x now, so it must be logged again
	l.Stat("a")
	l.Getenv("HOME")
	l.Open("")     // dropped
	l.Open("b\nc") // dropped
	if err := l.w.Flush(); err != nil {
		t.Fatal(err)
	}
	want := "stat a\nopen a\ngetenv HOME\nchdir /x\nstat a\ngetenv HOME\n"
	if got := b.String(); got != want {
		t.Errorf("log:\n%s\nwant:\n%s", got, want)
	}
}

// TestTestLogBounded checks that the seen set stays bounded and never
// drops an entry that was not written before, however many there are.
func TestTestLogBounded(t *testing.T) {
	var b strings.Builder
	l := &testLog{w: bufio.NewWriter(&b)}
	const n = 100000
	most := 0
	for i := range n {
		l.Stat("f" + strconv.Itoa(i))
		most = max(most, len(l.seen))
	}
	if most > n/2 {
		t.Errorf("seen grew to %d entries for %d distinct names", most, n)
	}
	if err := l.w.Flush(); err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(b.String(), "\n"); got != n {
		t.Errorf("wrote %d lines for %d distinct entries", got, n)
	}
}
