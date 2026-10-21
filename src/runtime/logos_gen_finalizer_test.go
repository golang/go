// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosgen

package runtime_test

import (
	"runtime"
	"sync"
	"testing"
	"time"
	"unsafe"
	"weak"
)

// TestLogosGenFinalizerKeepsYoungChildAlive is a regression test for the
// resurrected-finalizer promotion defect: an object the sweep resurrects
// for finalization promotes with clean cards, so the next minor cycle
// frees the young objects it still points to.
//
// Sequence: a pointer-bearing holder that owns its span survives one
// minor (age 1); a young child is stored into it and a blocking finalizer
// is installed; the external root is dropped; the next minor promotes the
// holder span but sweeps the white holder back to life and queues the
// finalizer; while the finalizer is blocked (keeping the holder alive
// through its own frame, without touching the child), another minor must
// not clear the child's weak pointer. The finalizer samples the holder
// span's age only after the test confirms the resurrecting cycle returned,
// so the observation cannot race the sweep tail that matures the span.
func TestLogosGenFinalizerKeepsYoungChildAlive(t *testing.T) {
	genQuiet(t)
	if !runtime.LogosGenActive() {
		t.Fatal("logosgen is built in but the collector is not generational")
	}

	started := make(chan struct{})
	sample := make(chan struct{})
	sampled := make(chan struct{})
	release := make(chan struct{})
	finished := make(chan struct{})
	ageCh := make(chan int, 1)
	releaseFin := sync.OnceFunc(func() { close(release) })
	signalSample := sync.OnceFunc(func() { close(sample) })
	installed := false
	defer func() {
		genFinalizerDropRoot()
		signalSample()
		if installed {
			releaseFin()
			if !genFinalizerWait(t, finished) {
				t.Errorf("blocked finalizer did not exit after release")
			}
		}
	}()

	runtime.GC()
	genRoot = new(genHolder)
	runtime.LogosGenMinorCycle()
	if age := runtime.LogosGenSpanAge(unsafe.Pointer(genRoot)); age != 1 {
		t.Fatalf("holder span age = %d after one survived minor cycle, want 1", age)
	}

	w := genStoreBarrier(genRoot)
	ctl := &genFinalizerCtl{
		started:  started,
		sample:   sample,
		sampled:  sampled,
		release:  release,
		finished: finished,
		age:      ageCh,
	}
	genFinalizerInstall(genRoot, ctl)
	installed = true
	genFinalizerDropRoot()

	// Resurrecting minor: the holder is white, so promotion rebuilds its
	// cards from marked objects only; the sweep brings the holder back to
	// life, matures its span, and queues the finalizer.
	runtime.LogosGenMinorCycle()

	if !genFinalizerWait(t, started) {
		t.Fatalf("finalizer did not start after the resurrecting minor cycle")
	}
	// The resurrecting cycle has returned, so the sweep that queued the
	// finalizer is complete and the span has aged; only now let the
	// finalizer sample it.
	signalSample()
	if !genFinalizerWait(t, sampled) {
		t.Fatalf("finalizer did not report the resurrected holder span age")
	}
	if age := <-ageCh; age != runtime.LogosGenMatureAge() {
		t.Fatalf("resurrected holder span age = %d, want mature age %d", age, runtime.LogosGenMatureAge())
	}
	if !genFinalizerWeakAlive(w) {
		t.Fatalf("young child was freed by the resurrecting minor cycle itself")
	}

	// While the finalizer is blocked on release, the holder stays alive
	// through the finalizer's frame; a minor must retain the child via
	// the holder's card instead of freeing it.
	runtime.LogosGenMinorCycle()
	if !genFinalizerWeakAlive(w) {
		t.Fatalf("young child reachable only from a resurrected finalizer object was freed by a minor cycle")
	}

	releaseFin()
	if !genFinalizerWait(t, finished) {
		t.Fatalf("blocked finalizer did not exit after release")
	}
}

// genFinalizerCtl carries the finalizer handshake channels. The closure
// captures only this control block; the holder arrives as the finalizer
// argument.
type genFinalizerCtl struct {
	started  chan struct{}
	sample   chan struct{}
	sampled  chan struct{}
	release  chan struct{}
	finished chan struct{}
	age      chan int
}

// genFinalizerInstall attaches a finalizer that reports the holder span's
// maturity only after the test closes sample, then blocks on release
// without reading the holder's slots, and keeps the holder alive past the
// block.
//
//go:noinline
func genFinalizerInstall(h *genHolder, ctl *genFinalizerCtl) {
	runtime.SetFinalizer(h, func(fin *genHolder) {
		close(ctl.started)
		<-ctl.sample
		ctl.age <- runtime.LogosGenSpanAge(unsafe.Pointer(fin))
		close(ctl.sampled)
		<-ctl.release
		runtime.KeepAlive(fin)
		close(ctl.finished)
	})
}

// genFinalizerDropRoot drops the external holder root without keeping a
// copy on the caller's stack.
//
//go:noinline
func genFinalizerDropRoot() {
	genRoot = nil
}

// genFinalizerWeakAlive reports whether the weak pointer still resolves,
// isolating the transient strong reference to this frame.
//
//go:noinline
func genFinalizerWeakAlive(w weak.Pointer[[64]byte]) bool {
	return w.Value() != nil
}

// genFinalizerWait waits for ch with Gosched instead of sleeps and reports
// whether it fired before the deadline. Callers own their timeout messages.
func genFinalizerWait(t *testing.T, ch <-chan struct{}) bool {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		select {
		case <-ch:
			return true
		default:
		}
		if time.Now().After(deadline) {
			return false
		}
		runtime.Gosched()
	}
}
