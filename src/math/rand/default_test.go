// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"fmt"
	"internal/race"
	"internal/testenv"
	. "math/rand"
	"os"
	"runtime"
	"strconv"
	"sync"
	"testing"
)

// Test that racy access to the default functions behaves reasonably.
func TestDefaultRace(t *testing.T) {
	// Skip the test in short mode, but even in short mode run
	// the test if we are using the race detector, because part
	// of this is to see whether the race detector reports any problems.
	if testing.Short() && !race.Enabled {
		t.Skip("skipping starting another executable in short mode")
	}

	const env = "GO_RAND_TEST_HELPER_CODE"
	if v := os.Getenv(env); v != "" {
		doDefaultTest(t, v)
		return
	}

	t.Parallel()

	for i := 0; i < 6; i++ {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			t.Parallel()
			cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestDefaultRace$")
			cmd = testenv.CleanCmdEnv(cmd)
			cmd.Env = append(cmd.Env, fmt.Sprintf("GO_RAND_TEST_HELPER_CODE=%d", i/2))
			if i%2 != 0 {
				cmd.Env = append(cmd.Env, "GODEBUG=randautoseed=0")
			}
			out, err := cmd.CombinedOutput()
			if len(out) > 0 {
				t.Logf("%s", out)
			}
			if err != nil {
				t.Error(err)
			}
		})
	}
}

// doDefaultTest should be run before there have been any calls to the
// top-level math/rand functions. Make sure that we can make concurrent
// calls to top-level functions and to Seed without any duplicate values.
// This will also give the race detector a change to report any problems.
func doDefaultTest(t *testing.T, v string) {
	code, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("internal error: unrecognized code %q", v)
	}

	goroutines := runtime.GOMAXPROCS(0)
	if goroutines < 4 {
		goroutines = 4
	}

	ch := make(chan uint64, goroutines*3)
	var wg sync.WaitGroup

	// The various tests below should not cause race detector reports
	// and should not produce duplicate results.
	//
	// Note: these tests can theoretically fail when using fastrand64
	// in that it is possible to coincidentally get the same random
	// number twice. That could happen something like 1 / 2**64 times,
	// which is rare enough that it may never happen. We don't worry
	// about that case.

	switch code {
	case 0:
		// Call Seed and Uint64 concurrently.
		wg.Add(goroutines)
		for i := 0; i < goroutines; i++ {
			go func(s int64) {
				defer wg.Done()
				Seed(s)
			}(int64(i) + 100)
		}
		wg.Add(goroutines)
		for i := 0; i < goroutines; i++ {
			go func() {
				defer wg.Done()
				ch <- Uint64()
			}()
		}
	case 1:
		// Call Uint64 concurrently with no Seed.
		wg.Add(goroutines)
		for i := 0; i < goroutines; i++ {
			go func() {
				defer wg.Done()
				ch <- Uint64()
			}()
		}
	case 2:
		// Start with Uint64 to pick the fast source, then call
		// Seed and Uint64 concurrently.
		ch <- Uint64()
		wg.Add(goroutines)
		for i := 0; i < goroutines; i++ {
			go func(s int64) {
				defer wg.Done()
				Seed(s)
			}(int64(i) + 100)
		}
		wg.Add(goroutines)
		for i := 0; i < goroutines; i++ {
			go func() {
				defer wg.Done()
				ch <- Uint64()
			}()
		}
	default:
		t.Fatalf("internal error: unrecognized code %d", code)
	}

	go func() {
		wg.Wait()
		close(ch)
	}()

	m := make(map[uint64]bool)
	for i := range ch {
		if m[i] {
			t.Errorf("saw %d twice", i)
		}
		m[i] = true
	}
}
