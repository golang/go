// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buffer

import (
	"internal/race"
	"internal/testenv"
	"testing"
)

func Test(t *testing.T) {
	b := New()
	defer b.Free()
	b.WriteString("hello")
	b.WriteByte(',')
	b.Write([]byte(" world"))

	got := b.String()
	want := "hello, world"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestAlloc(t *testing.T) {
	if race.Enabled {
		t.Skip("skipping test in race mode")
	}
	testenv.SkipIfOptimizationOff(t)
	got := int(testing.AllocsPerRun(5, func() {
		b := New()
		defer b.Free()
		b.WriteString("not 1K worth of bytes")
	}))
	if got != 0 {
		t.Errorf("got %d allocs, want 0", got)
	}
}
