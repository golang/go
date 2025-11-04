// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"internal/strconv"
	"unsafe"
)

// The compiler knows that a print of a value of this type
// should use printhex instead of printuint (decimal).
type hex uint64

func bytes(s string) (ret []byte) {
	rp := (*slice)(unsafe.Pointer(&ret))
	sp := stringStructOf(&s)
	rp.array = sp.str
	rp.len = sp.len
	rp.cap = sp.len
	return
}

var (
	// printBacklog is a circular buffer of messages written with the builtin
	// print* functions, for use in postmortem analysis of core dumps.
	printBacklog      [512]byte
	printBacklogIndex int
)

// recordForPanic maintains a circular buffer of messages written by the
// runtime leading up to a process crash, allowing the messages to be
// extracted from a core dump.
//
// The text written during a process crash (following "panic" or "fatal
// error") is not saved, since the goroutine stacks will generally be readable
// from the runtime data structures in the core file.
func recordForPanic(b []byte) {
	printlock()

	if panicking.Load() == 0 {
		// Not actively crashing: maintain circular buffer of print output.
		for i := 0; i < len(b); {
			n := copy(printBacklog[printBacklogIndex:], b[i:])
			i += n
			printBacklogIndex += n
			printBacklogIndex %= len(printBacklog)
		}
	}

	printunlock()
}

var debuglock mutex

// The compiler emits calls to printlock and printunlock around
// the multiple calls that implement a single Go print or println
// statement. Some of the print helpers (printslice, for example)
// call print recursively. There is also the problem of a crash
// happening during the print routines and needing to acquire
// the print lock to print information about the crash.
// For both these reasons, let a thread acquire the printlock 'recursively'.

func printlock() {
	mp := getg().m
	mp.locks++ // do not reschedule between printlock++ and lock(&debuglock).
	mp.printlock++
	if mp.printlock == 1 {
		lock(&debuglock)
	}
	mp.locks-- // now we know debuglock is held and holding up mp.locks for us.
}

func printunlock() {
	mp := getg().m
	mp.printlock--
	if mp.printlock == 0 {
		unlock(&debuglock)
	}
}

// write to goroutine-local buffer if diverting output,
// or else standard error.
func gwrite(b []byte) {
	if len(b) == 0 {
		return
	}
	recordForPanic(b)
	gp := getg()
	// Don't use the writebuf if gp.m is dying. We want anything
	// written through gwrite to appear in the terminal rather
	// than be written to in some buffer, if we're in a panicking state.
	// Note that we can't just clear writebuf in the gp.m.dying case
	// because a panic isn't allowed to have any write barriers.
	if gp == nil || gp.writebuf == nil || gp.m.dying > 0 {
		writeErr(b)
		return
	}

	n := copy(gp.writebuf[len(gp.writebuf):cap(gp.writebuf)], b)
	gp.writebuf = gp.writebuf[:len(gp.writebuf)+n]
}

func printsp() {
	printstring(" ")
}

func printnl() {
	printstring("\n")
}

func printbool(v bool) {
	if v {
		printstring("true")
	} else {
		printstring("false")
	}
}

func printfloat64(v float64) {
	var buf [20]byte
	gwrite(strconv.AppendFloat(buf[:0], v, 'g', -1, 64))
}

func printfloat32(v float32) {
	var buf [20]byte
	gwrite(strconv.AppendFloat(buf[:0], float64(v), 'g', -1, 32))
}

func printcomplex128(c complex128) {
	var buf [44]byte
	gwrite(strconv.AppendComplex(buf[:0], c, 'g', -1, 128))
}

func printcomplex64(c complex64) {
	var buf [44]byte
	gwrite(strconv.AppendComplex(buf[:0], complex128(c), 'g', -1, 64))
}

func printuint(v uint64) {
	// Note: Avoiding strconv.AppendUint so that it's clearer
	// that there are no allocations in this routine.
	// cmd/link/internal/ld.TestAbstractOriginSanity
	// sees the append and doesn't realize it doesn't allocate.
	var buf [20]byte
	i := strconv.RuntimeFormatBase10(buf[:], v)
	gwrite(buf[i:])
}

func printint(v int64) {
	// Note: Avoiding strconv.AppendUint so that it's clearer
	// that there are no allocations in this routine.
	// cmd/link/internal/ld.TestAbstractOriginSanity
	// sees the append and doesn't realize it doesn't allocate.
	neg := v < 0
	u := uint64(v)
	if neg {
		u = -u
	}
	var buf [20]byte
	i := strconv.RuntimeFormatBase10(buf[:], u)
	if neg {
		i--
		buf[i] = '-'
	}
	gwrite(buf[i:])
}

var minhexdigits = 0 // protected by printlock

func printhex(v uint64) {
	const dig = "0123456789abcdef"
	var buf [100]byte
	i := len(buf)
	for i--; i > 0; i-- {
		buf[i] = dig[v%16]
		if v < 16 && len(buf)-i >= minhexdigits {
			break
		}
		v /= 16
	}
	i--
	buf[i] = 'x'
	i--
	buf[i] = '0'
	gwrite(buf[i:])
}

func printpointer(p unsafe.Pointer) {
	printhex(uint64(uintptr(p)))
}
func printuintptr(p uintptr) {
	printhex(uint64(p))
}

func printstring(s string) {
	gwrite(bytes(s))
}

func printslice(s []byte) {
	sp := (*slice)(unsafe.Pointer(&s))
	print("[", len(s), "/", cap(s), "]")
	printpointer(sp.array)
}

func printeface(e eface) {
	print("(", e._type, ",", e.data, ")")
}

func printiface(i iface) {
	print("(", i.tab, ",", i.data, ")")
}

// hexdumpWords prints a word-oriented hex dump of [p, end).
//
// If mark != nil, it will be called with each printed word's address
// and should return a character mark to appear just before that
// word's value. It can return 0 to indicate no mark.
func hexdumpWords(p, end uintptr, mark func(uintptr) byte) {
	printlock()
	var markbuf [1]byte
	markbuf[0] = ' '
	minhexdigits = int(unsafe.Sizeof(uintptr(0)) * 2)
	for i := uintptr(0); p+i < end; i += goarch.PtrSize {
		if i%16 == 0 {
			if i != 0 {
				println()
			}
			print(hex(p+i), ": ")
		}

		if mark != nil {
			markbuf[0] = mark(p + i)
			if markbuf[0] == 0 {
				markbuf[0] = ' '
			}
		}
		gwrite(markbuf[:])
		val := *(*uintptr)(unsafe.Pointer(p + i))
		print(hex(val))
		print(" ")

		// Can we symbolize val?
		fn := findfunc(val)
		if fn.valid() {
			print("<", funcname(fn), "+", hex(val-fn.entry()), "> ")
		}
	}
	minhexdigits = 0
	println()
	printunlock()
}
