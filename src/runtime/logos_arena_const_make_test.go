// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime_test

import (
	"internal/goexperiment"
	"runtime"
	"testing"
)

var arenaConstEffects int

//go:noinline
func arenaConstEffect(n int) int {
	arenaConstEffects++
	return n
}

//go:noinline
func arenaConstBytes128() (zeroed bool, used uintptr) {
	b := make([]byte, 128)
	zeroed = true
	for _, x := range b {
		if x != 0 {
			zeroed = false
		}
	}
	for i := range b {
		b[i] = 0xff
	}
	return zeroed, runtime.LogosArenaStat().NoscanUsed
}

//go:noinline
func arenaConstBytes384() (zeroed bool, used uintptr) {
	b := make([]byte, 384)
	zeroed = true
	for _, x := range b {
		if x != 0 {
			zeroed = false
		}
	}
	for i := range b {
		b[i] = 0xff
	}
	return zeroed, runtime.LogosArenaStat().NoscanUsed
}

func arenaConstSitePlaced(t *testing.T, site func() (bool, uintptr), size uintptr) bool {
	t.Helper()
	var before, during, after uintptr
	inFreshGoroutine(t, func() {
		before = runtime.LogosArenaStat().NoscanUsed
		_, during = site()
		after = runtime.LogosArenaStat().NoscanUsed
	})
	if after != before {
		t.Fatalf("calibration leaked watermark: before = %d, after = %d", before, after)
	}
	return during-before >= size
}

//go:noinline
func arenaConstWords4096() (zeroed bool, used uintptr) {
	b := make([]int64, 4096)
	zeroed = true
	for _, x := range b {
		if x != 0 {
			zeroed = false
		}
	}
	for i := range b {
		b[i] = -1
	}
	return zeroed, runtime.LogosArenaStat().NoscanUsed
}

//go:noinline
func arenaConstCapDynLen(n int) int {
	b := make([]byte, n, 128)
	return len(b) + cap(b)
}

//go:noinline
func arenaConstCapEffect(n int) int {
	b := make([]byte, arenaConstEffect(n), 128)
	return len(b)
}

//go:noinline
func arenaDynEffect(n int) int {
	b := make([]byte, arenaConstEffect(n))
	return len(b)
}

func TestLogosArenaConstMakeBump(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	if runtime.GOARCH != "amd64" {
		t.Skip("skipping test: arena make lowering is amd64-only")
	}
	tiers := []struct {
		name string
		site func() (zeroed bool, used uintptr)
		size uintptr
	}{
		{"128 bytes", arenaConstBytes128, 128},
		{"384 bytes", arenaConstBytes384, 384},
		{"32 KiB", arenaConstWords4096, 4096 * 8},
	}
	placed := make([]bool, len(tiers))
	for i, tc := range tiers {
		placed[i] = arenaConstSitePlaced(t, tc.site, tc.size)
	}
	if !placed[len(tiers)-1] {
		t.Fatalf("32 KiB site not arena-placed: exceeds every tier threshold, want placement")
	}
	for i := 1; i < len(tiers); i++ {
		if placed[i-1] && !placed[i] {
			t.Fatalf("%s placed while larger %s did not: no threshold explains this pattern",
				tiers[i-1].name, tiers[i].name)
		}
	}
	inFreshGoroutine(t, func() {
		for i, tc := range tiers {
			if !placed[i] {
				t.Logf("%s: tier-excluded from the arena, skipping", tc.name)
				continue
			}
			for call := 1; call <= 3; call++ {
				before := runtime.LogosArenaStat().NoscanUsed
				zeroed, during := tc.site()
				after := runtime.LogosArenaStat().NoscanUsed
				if !zeroed {
					t.Errorf("%s call %d: block not zeroed after a dirty frame", tc.name, call)
				}
				if during < before+tc.size {
					t.Errorf("%s call %d: watermark during make = %d, want at least %d", tc.name, call, during, before+tc.size)
				}
				if after != before {
					t.Errorf("%s call %d: watermark after frame exit = %d, want %d", tc.name, call, after, before)
				}
			}
		}
	})
}

func TestLogosArenaConstCapChecks(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	for _, n := range []int{0, 1, 128} {
		if got := arenaConstCapDynLen(n); got != n+128 {
			t.Errorf("make([]byte, %d, 128): len+cap = %d, want %d", n, got, n+128)
		}
	}
	for _, tc := range []struct {
		n    int
		want string
	}{
		{-1, arenaMakeLenPanic},
		{129, arenaMakeCapPanic},
	} {
		msg, panicked := recoverMsg(func() { arenaConstCapDynLen(tc.n) })
		if !panicked || msg != tc.want {
			t.Errorf("make([]byte, %d, 128): panic %q (panicked %v), want %q", tc.n, msg, panicked, tc.want)
		}
	}

	arenaConstEffects = 0
	if got := arenaConstCapEffect(64); got != 64 || arenaConstEffects != 1 {
		t.Errorf("make([]byte, f(), 128): len %d with %d calls of f, want 64 and 1", got, arenaConstEffects)
	}
	arenaConstEffects = 0
	if got := arenaDynEffect(64); got != 64 || arenaConstEffects != 1 {
		t.Errorf("make([]byte, f()): len %d with %d calls of f, want 64 and 1", got, arenaConstEffects)
	}
}
