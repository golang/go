// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Pins the accepted freeze finding: FreezeSlice with a non-nil copier
// walks cap(s) even for zero-sized element storage, which ordinary
// rt.Freeze avoids through the guarded dynamic path. Zero-sized
// elements share addresses, so only len, cap, and nil are pinned here;
// freeze_pr2_regress_test.go pins the nonzero identity cases.
package rt_test

import (
	"context"
	"errors"
	"internal/testenv"
	"math"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"logos/rt"
)

const freezeZeroSizeReexecEnv = "LOGOS_RT_TEST_FREEZE_ZEROSIZE_CHILD"

const (
	freezeZeroSizeReps    = 64
	freezeZeroSizeHugeCap = math.MaxInt32
	freezeZeroSizeLen     = 4
)

func copyZeroPtr(fr *rt.Freezer, dst, src *[0]*int) { *dst = *src }

func TestFreezeZeroSizeSliceBoundedProgress(t *testing.T) {
	if os.Getenv(freezeZeroSizeReexecEnv) != "" {
		freezeZeroSizeChildPayload(t)
		return
	}
	testenv.MustHaveExec(t)
	const backstop = 15 * time.Second
	backstopErr := errors.New("zero-size freeze child backstop elapsed")
	ctx, cancel := context.WithTimeoutCause(t.Context(), backstop, backstopErr)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run", "^"+t.Name()+"$", "-test.v")
	cmd.Env = append(os.Environ(), freezeZeroSizeReexecEnv+"=1")
	out, err := cmd.CombinedOutput()
	if cause := context.Cause(ctx); cause != nil {
		t.Fatalf("zero-size freeze payload did not finish %d FreezeSlice repetitions of cap(%d) [0]*int within %v (FreezeSlice with a non-nil copier traverses cap(s) per repetition for zero-sized elements): %v\n%s", freezeZeroSizeReps, freezeZeroSizeHugeCap, backstop, cause, out)
	}
	if err != nil {
		t.Fatalf("zero-size freeze child did not pass: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "--- PASS: "+t.Name()) {
		t.Fatalf("zero-size freeze child exited cleanly without running the payload:\n%s", out)
	}
}

// A shared Freezer would cache the identical header and hide the
// per-cap work after repetition one.
func freezeZeroSizeChildPayload(t *testing.T) {
	for range freezeZeroSizeReps {
		s := make([][0]*int, freezeZeroSizeLen, freezeZeroSizeHugeCap)
		got := rt.FreezeSlice(&rt.Freezer{}, s, copyZeroPtr)
		if got == nil {
			t.Fatalf("FreezeSlice of non-nil [0]*int len %d cap %d returned nil", len(s), cap(s))
		}
		if len(got) != freezeZeroSizeLen || cap(got) != freezeZeroSizeHugeCap {
			t.Fatalf("frozen [0]*int header len, cap = %d, %d; want %d, %d", len(got), cap(got), freezeZeroSizeLen, freezeZeroSizeHugeCap)
		}
	}
	freezeZeroSizeRootRoute(t)
}

type zeroAny [0]any

type zeroHolder struct {
	zs []zeroAny
}

func copyZeroHolder(fr *rt.Freezer, dst, src *zeroHolder) {
	*dst = *src
	dst.zs = rt.FreezeSlice(fr, src.zs, rt.FreezeDynamic[zeroAny])
}

func freezeZeroSizeRootRoute(t *testing.T) {
	src := &zeroHolder{zs: make([]zeroAny, freezeZeroSizeLen, freezeZeroSizeHugeCap)}
	got := rt.FreezeRoot(src, copyZeroHolder)
	if got == nil {
		t.Fatal("FreezeRoot of non-nil zeroHolder returned nil")
	}
	if len(got.zs) != freezeZeroSizeLen || cap(got.zs) != freezeZeroSizeHugeCap {
		t.Fatalf("FreezeRoot frozen zero-size header len, cap = %d, %d; want %d, %d", len(got.zs), cap(got.zs), freezeZeroSizeLen, freezeZeroSizeHugeCap)
	}
}
