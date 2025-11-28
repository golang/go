// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/strconv"
	"unsafe"
)

// The compiler knows that a print of a value of this type
// should use printhex instead of printuint (decimal).
type hex uint64

// The compiler knows that a print of a value of this type should use
// printquoted instead of printstring.
type quoted string

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

func printhexopts(include0x bool, mindigits int, v uint64) {
	const dig = "0123456789abcdef"
	var buf [100]byte
	i := len(buf)
	for i--; i > 0; i-- {
		buf[i] = dig[v%16]
		if v < 16 && len(buf)-i >= mindigits {
			break
		}
		v /= 16
	}
	if include0x {
		i--
		buf[i] = 'x'
		i--
		buf[i] = '0'
	}
	gwrite(buf[i:])
}

func printhex(v uint64) {
	printhexopts(true, minhexdigits, v)
}

func printquoted(s string) {
	printlock()
	gwrite([]byte(`"`))
	for _, r := range s {
		switch r {
		case '\n':
			gwrite([]byte(`\n`))
			continue
		case '\r':
			gwrite([]byte(`\r`))
			continue
		case '\t':
			gwrite([]byte(`\t`))
			print()
			continue
		case '\\', '"':
			gwrite([]byte{byte('\\'), byte(r)})
			continue
		}
		// For now, only allow basic printable ascii through unescaped
		if r >= ' ' && r <= '~' {
			gwrite([]byte{byte(r)})
		} else if r < 127 {
			gwrite(bytes(`\x`))
			printhexopts(false, 2, uint64(r))
		} else if r < 0x1_0000 {
			gwrite(bytes(`\u`))
			printhexopts(false, 4, uint64(r))
		} else {
			gwrite(bytes(`\U`))
			printhexopts(false, 8, uint64(r))
		}
	}
	gwrite([]byte{byte('"')})
	printunlock()
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
