// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(austin): All of these tests are skipped if the debuglog build
// tag isn't provided. That means we basically never test debuglog.
// There are two potential ways around this:
//
// 1. Make these tests re-build the runtime test with the debuglog
// build tag and re-invoke themselves.
//
// 2. Always build the whole debuglog infrastructure and depend on
// linker dead-code elimination to drop it. This is easy for dlog()
// since there won't be any calls to it. For printDebugLog, we can
// make panic call a wrapper that is call printDebugLog if the
// debuglog build tag is set, or otherwise do nothing. Then tests
// could call printDebugLog directly. This is the right answer in
// principle, but currently our linker reads in all symbols
// regardless, so this would slow down and bloat all links. If the
// linker gets more efficient about this, we should revisit this
// approach.

package runtime_test

import (
	"fmt"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func skipDebugLog(t *testing.T) {
	if !runtime.DlogEnabled {
		t.Skip("debug log disabled (rebuild with -tags debuglog)")
	}
}

func dlogCanonicalize(x string) string {
	begin := regexp.MustCompile(`(?m)^>> begin log \d+ <<\n`)
	x = begin.ReplaceAllString(x, "")
	prefix := regexp.MustCompile(`(?m)^\[[^]]+\]`)
	x = prefix.ReplaceAllString(x, "[]")
	return x
}

func TestDebugLog(t *testing.T) {
	skipDebugLog(t)
	runtime.ResetDebugLog()
	runtime.Dlog().S("testing").End()
	got := dlogCanonicalize(runtime.DumpDebugLog())
	if want := "[] testing\n"; got != want {
		t.Fatalf("want %q, got %q", want, got)
	}
}

func TestDebugLogTypes(t *testing.T) {
	skipDebugLog(t)
	runtime.ResetDebugLog()
	var varString = strings.Repeat("a", 4)
	runtime.Dlog().B(true).B(false).I(-42).I16(0x7fff).U64(^uint64(0)).Hex(0xfff).P(nil).S(varString).S("const string").End()
	got := dlogCanonicalize(runtime.DumpDebugLog())
	if want := "[] true false -42 32767 18446744073709551615 0xfff 0x0 aaaa const string\n"; got != want {
		t.Fatalf("want %q, got %q", want, got)
	}
}

func TestDebugLogSym(t *testing.T) {
	skipDebugLog(t)
	runtime.ResetDebugLog()
	pc, _, _, _ := runtime.Caller(0)
	runtime.Dlog().PC(pc).End()
	got := dlogCanonicalize(runtime.DumpDebugLog())
	want := regexp.MustCompile(`\[\] 0x[0-9a-f]+ \[runtime_test\.TestDebugLogSym\+0x[0-9a-f]+ .*/debuglog_test\.go:[0-9]+\]\n`)
	if !want.MatchString(got) {
		t.Fatalf("want matching %s, got %q", want, got)
	}
}

func TestDebugLogInterleaving(t *testing.T) {
	skipDebugLog(t)
	runtime.ResetDebugLog()
	var wg sync.WaitGroup
	done := int32(0)
	wg.Add(1)
	go func() {
		// Encourage main goroutine to move around to
		// different Ms and Ps.
		for atomic.LoadInt32(&done) == 0 {
			runtime.Gosched()
		}
		wg.Done()
	}()
	var want strings.Builder
	for i := 0; i < 1000; i++ {
		runtime.Dlog().I(i).End()
		fmt.Fprintf(&want, "[] %d\n", i)
		runtime.Gosched()
	}
	atomic.StoreInt32(&done, 1)
	wg.Wait()

	gotFull := runtime.DumpDebugLog()
	got := dlogCanonicalize(gotFull)
	if got != want.String() {
		// Since the timestamps are useful in understand
		// failures of this test, we print the uncanonicalized
		// output.
		t.Fatalf("want %q, got (uncanonicalized) %q", want.String(), gotFull)
	}
}

func TestDebugLogWraparound(t *testing.T) {
	skipDebugLog(t)

	// Make sure we don't switch logs so it's easier to fill one up.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	runtime.ResetDebugLog()
	var longString = strings.Repeat("a", 128)
	var want strings.Builder
	for i, j := 0, 0; j < 2*runtime.DebugLogBytes; i, j = i+1, j+len(longString) {
		runtime.Dlog().I(i).S(longString).End()
		fmt.Fprintf(&want, "[] %d %s\n", i, longString)
	}
	log := runtime.DumpDebugLog()

	// Check for "lost" message.
	lost := regexp.MustCompile(`^>> begin log \d+; lost first \d+KB <<\n`)
	if !lost.MatchString(log) {
		t.Fatalf("want matching %s, got %q", lost, log)
	}
	idx := lost.FindStringIndex(log)
	// Strip lost message.
	log = dlogCanonicalize(log[idx[1]:])

	// Check log.
	if !strings.HasSuffix(want.String(), log) {
		t.Fatalf("wrong suffix:\n%s", log)
	}
}

func TestDebugLogLongString(t *testing.T) {
	skipDebugLog(t)

	runtime.ResetDebugLog()
	var longString = strings.Repeat("a", runtime.DebugLogStringLimit+1)
	runtime.Dlog().S(longString).End()
	got := dlogCanonicalize(runtime.DumpDebugLog())
	want := "[] " + strings.Repeat("a", runtime.DebugLogStringLimit) + " ..(1 more bytes)..\n"
	if got != want {
		t.Fatalf("want %q, got %q", want, got)
	}
}
