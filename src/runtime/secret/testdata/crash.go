// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/secret"
	"sync"
	"syscall"
	"time"
	_ "unsafe"
	"weak"
)

// Same secret as in ../../crash_test.go
var secretStore = [8]byte{
	0x00,
	0x81,
	0xa0,
	0xc6,
	0xb3,
	0x01,
	0x66,
	0x53,
}

func main() {
	enableCore()
	useSecretProc()
	// clear out secret. That way we don't have
	// to figure out which secret is the allowed
	// source
	clear(secretStore[:])
	panic("terminate")
}

// Copied from runtime/runtime-gdb_unix_test.go
func enableCore() {
	debug.SetTraceback("crash")

	var lim syscall.Rlimit
	err := syscall.Getrlimit(syscall.RLIMIT_CORE, &lim)
	if err != nil {
		panic(fmt.Sprintf("error getting rlimit: %v", err))
	}
	lim.Cur = lim.Max
	fmt.Fprintf(os.Stderr, "Setting RLIMIT_CORE = %+#v\n", lim)
	err = syscall.Setrlimit(syscall.RLIMIT_CORE, &lim)
	if err != nil {
		panic(fmt.Sprintf("error setting rlimit: %v", err))
	}
}

// useSecretProc does 5 seconds of work, using the secret value
// inside secret.Do in a bunch of ways.
func useSecretProc() {
	stop := make(chan bool)
	var wg sync.WaitGroup

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			time.Sleep(1 * time.Second)
			for {
				select {
				case <-stop:
					wg.Done()
					return
				default:
					secret.Do(func() {
						// Copy key into a variable-sized heap allocation.
						// This both puts secrets in heap objects,
						// and more generally just causes allocation,
						// which forces garbage collection, which
						// requires interrupts and the like.
						s := bytes.Repeat(secretStore[:], 1+i*2)
						// Also spam the secret across all registers.
						useSecret(s)
					})
				}
			}
		}()
	}

	// Send some allocations over a channel. This does 2 things:
	// 1) forces some GCs to happen
	// 2) causes more scheduling noise (Gs moving between Ms, etc.)
	c := make(chan []byte)
	wg.Add(2)
	go func() {
		for {
			select {
			case <-stop:
				wg.Done()
				return
			case c <- make([]byte, 256):
			}
		}
	}()
	go func() {
		for {
			select {
			case <-stop:
				wg.Done()
				return
			case <-c:
			}
		}
	}()

	time.Sleep(5 * time.Second)
	close(stop)
	wg.Wait()
	// use a weak reference for ensuring that the GC has cleared everything
	// Use a large value to avoid the tiny allocator.
	w := weak.Make(new([2048]byte))
	// 20 seems like a decent amount?
	for i := 0; i < 20; i++ {
		runtime.GC() // GC should clear any secret heap objects and clear out scheduling buffers.
		if w.Value() == nil {
			fmt.Fprintf(os.Stderr, "number of GCs %v\n", i+1)
			return
		}
	}
	fmt.Fprintf(os.Stderr, "GC didn't clear out in time\n")
	// This will cause the core dump to happen with the sentinel value still in memory
	// so we will detect the fault.
	panic("fault")
}
