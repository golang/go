// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
static unsigned char observed;

static void consume(void *p) {
	if (p != 0) {
		observed = *(unsigned char *)p;
	}
}

static unsigned char read_observed(void) {
	return observed;
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"strings"
	"unsafe"
)

type packet struct {
	header [16]byte
	next   *int
}

func expectUnpinned(f func()) {
	defer func() {
		p := recover()
		if p == nil {
			panic("saved interior pointer bypassed the allocation check")
		}
		err, ok := p.(error)
		if !ok || !strings.Contains(err.Error(), "Go pointer to unpinned Go pointer") {
			panic(fmt.Sprintf("unexpected cgo pointer panic: %v", p))
		}
	}()
	f()
}

func checkObserved(want byte) {
	if got := byte(C.read_observed()); got != want {
		panic(fmt.Sprintf("C observed %d, want %d", got, want))
	}
}

func deferred(p *packet, evaluate func(*packet) *packet) {
	defer C.consume(unsafe.Pointer(&evaluate(p).header))
	p.header[0] = 29
}

func main() {
	// Avoid tiny-allocation pin sharing with an unrelated object.
	storage := new([32]int)
	p := &packet{header: [16]byte{17}, next: &storage[0]}
	C.consume(unsafe.Pointer(&p.header))
	checkObserved(17)
	expectUnpinned(func() {
		addr := unsafe.Pointer(&p.header)
		C.consume(addr)
	})
	calls := 0
	deferred(p, func(p *packet) *packet {
		calls++
		return p
	})
	if calls != 1 {
		panic(fmt.Sprintf("deferred address evaluated %d times, want 1", calls))
	}
	checkObserved(29)

	var pin runtime.Pinner
	pin.Pin(p.next)
	defer pin.Unpin()
	runtime.GC()
	p.header[0] = 23
	addr := unsafe.Pointer(&p.header)
	C.consume(addr)
	checkObserved(23)
	p.header[0] = 31
	C.consume(unsafe.Pointer(&p.header))
	checkObserved(31)
	runtime.KeepAlive(p)
	fmt.Println("typed-address, pinning, and deferred-evaluation checks ok")
}
