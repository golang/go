// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"strings"
	"testing"

	"simd/archsimd/_gen/gentools"
)

// TestInvariants checks the counts in the report that must stay at zero (see
// [invariant]) against the checked-in tree, so that a change that breaks one
// fails here instead of moving a number in the report that nobody rereads.
//
// Like specstats itself, it needs up-to-date generator output: run
// "go run . -w" in _gen first if the generators' inputs changed.
func TestInvariants(t *testing.T) {
	if testing.Short() {
		t.Skip("loads simd and archsimd for every build configuration")
	}
	genOpts := gentools.RegisterFlags(flag.NewFlagSet("specstats", flag.ContinueOnError))
	m := load(genOpts)
	for _, inv := range m.invariants() {
		if len(inv.items) == 0 {
			continue
		}
		const max = 20
		items := inv.items
		more := ""
		if len(items) > max {
			items, more = items[:max], "\n\t..."
		}
		t.Errorf("%s: %d, want 0 [%s]:\n\t%s%s", inv.name, len(inv.items), inv.tasks,
			strings.Join(items, "\n\t"), more)
	}
}
