// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: This test could be implemented on all (most?) UNIXes if we
// added syscall.Tgkill more widely.

// We skip all of these tests under race mode because our test thread
// spends all of its time in the race runtime, which isn't a safe
// point.

// +build amd64
// +build linux
// +build !race

package runtime_test

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sync/atomic"
	"syscall"
	"testing"
)

func startDebugCallWorker(t *testing.T) (g *runtime.G, after func()) {
	// This can deadlock if there aren't enough threads or if a GC
	// tries to interrupt an atomic loop (see issue #10958).
	ogomaxprocs := runtime.GOMAXPROCS(2)
	ogcpercent := debug.SetGCPercent(-1)

	ready := make(chan *runtime.G)
	var stop uint32
	done := make(chan error)
	go debugCallWorker(ready, &stop, done)
	g = <-ready
	return g, func() {
		atomic.StoreUint32(&stop, 1)
		err := <-done
		if err != nil {
			t.Fatal(err)
		}
		runtime.GOMAXPROCS(ogomaxprocs)
		debug.SetGCPercent(ogcpercent)
	}
}

func debugCallWorker(ready chan<- *runtime.G, stop *uint32, done chan<- error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	ready <- runtime.Getg()

	x := 2
	debugCallWorker2(stop, &x)
	if x != 1 {
		done <- fmt.Errorf("want x = 2, got %d; register pointer not adjusted?", x)
	}
	close(done)
}

func debugCallWorker2(stop *uint32, x *int) {
	for atomic.LoadUint32(stop) == 0 {
		// Strongly encourage x to live in a register so we
		// can test pointer register adjustment.
		*x++
	}
	*x = 1
}

func debugCallTKill(tid int) error {
	return syscall.Tgkill(syscall.Getpid(), tid, syscall.SIGTRAP)
}

func TestDebugCall(t *testing.T) {
	g, after := startDebugCallWorker(t)
	defer after()

	// Inject a call into the debugCallWorker goroutine and test
	// basic argument and result passing.
	var args struct {
		x    int
		yRet int
	}
	fn := func(x int) (yRet int) {
		return x + 1
	}
	args.x = 42
	if _, err := runtime.InjectDebugCall(g, fn, &args, debugCallTKill); err != nil {
		t.Fatal(err)
	}
	if args.yRet != 43 {
		t.Fatalf("want 43, got %d", args.yRet)
	}
}

func TestDebugCallLarge(t *testing.T) {
	g, after := startDebugCallWorker(t)
	defer after()

	// Inject a call with a large call frame.
	const N = 128
	var args struct {
		in  [N]int
		out [N]int
	}
	fn := func(in [N]int) (out [N]int) {
		for i := range in {
			out[i] = in[i] + 1
		}
		return
	}
	var want [N]int
	for i := range args.in {
		args.in[i] = i
		want[i] = i + 1
	}
	if _, err := runtime.InjectDebugCall(g, fn, &args, debugCallTKill); err != nil {
		t.Fatal(err)
	}
	if want != args.out {
		t.Fatalf("want %v, got %v", want, args.out)
	}
}

func TestDebugCallGC(t *testing.T) {
	g, after := startDebugCallWorker(t)
	defer after()

	// Inject a call that performs a GC.
	if _, err := runtime.InjectDebugCall(g, runtime.GC, nil, debugCallTKill); err != nil {
		t.Fatal(err)
	}
}

func TestDebugCallGrowStack(t *testing.T) {
	g, after := startDebugCallWorker(t)
	defer after()

	// Inject a call that grows the stack. debugCallWorker checks
	// for stack pointer breakage.
	if _, err := runtime.InjectDebugCall(g, func() { growStack(nil) }, nil, debugCallTKill); err != nil {
		t.Fatal(err)
	}
}

//go:nosplit
func debugCallUnsafePointWorker(gpp **runtime.G, ready, stop *uint32) {
	// The nosplit causes this function to not contain safe-points
	// except at calls.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	*gpp = runtime.Getg()

	for atomic.LoadUint32(stop) == 0 {
		atomic.StoreUint32(ready, 1)
	}
}

func TestDebugCallUnsafePoint(t *testing.T) {
	// This can deadlock if there aren't enough threads or if a GC
	// tries to interrupt an atomic loop (see issue #10958).
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	defer debug.SetGCPercent(debug.SetGCPercent(-1))

	// Test that the runtime refuses call injection at unsafe points.
	var g *runtime.G
	var ready, stop uint32
	defer atomic.StoreUint32(&stop, 1)
	go debugCallUnsafePointWorker(&g, &ready, &stop)
	for atomic.LoadUint32(&ready) == 0 {
		runtime.Gosched()
	}

	_, err := runtime.InjectDebugCall(g, func() {}, nil, debugCallTKill)
	if msg := "call not at safe point"; err == nil || err.Error() != msg {
		t.Fatalf("want %q, got %s", msg, err)
	}
}

func TestDebugCallPanic(t *testing.T) {
	// This can deadlock if there aren't enough threads.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))

	ready := make(chan *runtime.G)
	var stop uint32
	defer atomic.StoreUint32(&stop, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		ready <- runtime.Getg()
		for atomic.LoadUint32(&stop) == 0 {
		}
	}()
	g := <-ready

	p, err := runtime.InjectDebugCall(g, func() { panic("test") }, nil, debugCallTKill)
	if err != nil {
		t.Fatal(err)
	}
	if ps, ok := p.(string); !ok || ps != "test" {
		t.Fatalf("wanted panic %v, got %v", "test", p)
	}
}
