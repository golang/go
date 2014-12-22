// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:noescape
func pread(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32

//go:noescape
func pwrite(fd int32, buf unsafe.Pointer, nbytes int32, offset int64) int32

func seek(fd int32, offset int64, whence int32) int64

//go:noescape
func exits(msg *byte)

//go:noescape
func brk_(addr unsafe.Pointer) int32

func sleep(ms int32) int32

func rfork(flags int32) int32

//go:noescape
func plan9_semacquire(addr *uint32, block int32) int32

//go:noescape
func plan9_tsemacquire(addr *uint32, ms int32) int32

//go:noescape
func plan9_semrelease(addr *uint32, count int32) int32

//go:noescape
func notify(fn unsafe.Pointer) int32

func noted(mode int32) int32

//go:noescape
func nsec(*int64) int64

//go:noescape
func sigtramp(ureg, msg unsafe.Pointer)

func setfpmasks()

//go:noescape
func tstart_plan9(newm *m)

func errstr() string

type _Plink uintptr

//go:linkname os_sigpipe os.sigpipe
func os_sigpipe() {
	gothrow("too many writes on closed pipe")
}

func sigpanic() {
	g := getg()
	if !canpanic(g) {
		gothrow("unexpected signal during runtime execution")
	}

	note := gostringnocopy((*byte)(unsafe.Pointer(g.m.notesig)))
	switch g.sig {
	case _SIGRFAULT, _SIGWFAULT:
		addr := note[index(note, "addr=")+5:]
		g.sigcode1 = uintptr(atolwhex(addr))
		if g.sigcode1 < 0x1000 || g.paniconfault {
			panicmem()
		}
		print("unexpected fault address ", hex(g.sigcode1), "\n")
		gothrow("fault")
	case _SIGTRAP:
		if g.paniconfault {
			panicmem()
		}
		gothrow(note)
	case _SIGINTDIV:
		panicdivide()
	case _SIGFLOAT:
		panicfloat()
	default:
		panic(errorString(note))
	}
}

func atolwhex(p string) int64 {
	for hasprefix(p, " ") || hasprefix(p, "\t") {
		p = p[1:]
	}
	neg := false
	if hasprefix(p, "-") || hasprefix(p, "+") {
		neg = p[0] == '-'
		p = p[1:]
		for hasprefix(p, " ") || hasprefix(p, "\t") {
			p = p[1:]
		}
	}
	var n int64
	switch {
	case hasprefix(p, "0x"), hasprefix(p, "0X"):
		p = p[2:]
		for ; len(p) > 0; p = p[1:] {
			if '0' <= p[0] && p[0] <= '9' {
				n = n*16 + int64(p[0]-'0')
			} else if 'a' <= p[0] && p[0] <= 'f' {
				n = n*16 + int64(p[0]-'a'+10)
			} else if 'A' <= p[0] && p[0] <= 'F' {
				n = n*16 + int64(p[0]-'A'+10)
			} else {
				break
			}
		}
	case hasprefix(p, "0"):
		for ; len(p) > 0 && '0' <= p[0] && p[0] <= '7'; p = p[1:] {
			n = n*8 + int64(p[0]-'0')
		}
	default:
		for ; len(p) > 0 && '0' <= p[0] && p[0] <= '9'; p = p[1:] {
			n = n*10 + int64(p[0]-'0')
		}
	}
	if neg {
		n = -n
	}
	return n
}
