// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd openbsd

// Berkeley packet filter for BSD variants

package syscall

import (
	"unsafe"
)

func BpfStmt(code, k int) *BpfInsn {
	return &BpfInsn{Code: uint16(code), K: uint32(k)}
}

func BpfJump(code, k, jt, jf int) *BpfInsn {
	return &BpfInsn{Code: uint16(code), Jt: uint8(jt), Jf: uint8(jf), K: uint32(k)}
}

func BpfBuflen(fd int) (int, int) {
	var l int
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCGBLEN, uintptr(unsafe.Pointer(&l)))
	if e := int(ep); e != 0 {
		return 0, e
	}
	return l, 0
}

func SetBpfBuflen(fd, l int) (int, int) {
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCSBLEN, uintptr(unsafe.Pointer(&l)))
	if e := int(ep); e != 0 {
		return 0, e
	}
	return l, 0
}

func BpfDatalink(fd int) (int, int) {
	var t int
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCGDLT, uintptr(unsafe.Pointer(&t)))
	if e := int(ep); e != 0 {
		return 0, e
	}
	return t, 0
}

func SetBpfDatalink(fd, t int) (int, int) {
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCSDLT, uintptr(unsafe.Pointer(&t)))
	if e := int(ep); e != 0 {
		return 0, e
	}
	return t, 0
}

func SetBpfPromisc(fd, m int) int {
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCPROMISC, uintptr(unsafe.Pointer(&m)))
	if e := int(ep); e != 0 {
		return e
	}
	return 0
}

func FlushBpf(fd int) int {
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCFLUSH, 0)
	if e := int(ep); e != 0 {
		return e
	}
	return 0
}

type ivalue struct {
	name  [IFNAMSIZ]byte
	value int16
}

func BpfInterface(fd int, name string) (string, int) {
	var iv ivalue
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCGETIF, uintptr(unsafe.Pointer(&iv)))
	if e := int(ep); e != 0 {
		return "", e
	}
	return name, 0
}

func SetBpfInterface(fd int, name string) int {
	var iv ivalue
	copy(iv.name[:], []byte(name))
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCSETIF, uintptr(unsafe.Pointer(&iv)))
	if e := int(ep); e != 0 {
		return e
	}
	return 0
}

func BpfTimeout(fd int) (*Timeval, int) {
	var tv Timeval
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCGRTIMEOUT, uintptr(unsafe.Pointer(&tv)))
	if e := int(ep); e != 0 {
		return nil, e
	}
	return &tv, 0
}

func SetBpfTimeout(fd int, tv *Timeval) int {
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCSRTIMEOUT, uintptr(unsafe.Pointer(tv)))
	if e := int(ep); e != 0 {
		return e
	}
	return 0
}

func BpfStats(fd int) (*BpfStat, int) {
	var s BpfStat
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCGSTATS, uintptr(unsafe.Pointer(&s)))
	if e := int(ep); e != 0 {
		return nil, e
	}
	return &s, 0
}

func SetBpfImmediate(fd, m int) int {
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCIMMEDIATE, uintptr(unsafe.Pointer(&m)))
	if e := int(ep); e != 0 {
		return e
	}
	return 0
}

func SetBpf(fd int, i []BpfInsn) int {
	var p BpfProgram
	p.Len = uint32(len(i))
	p.Insns = (*BpfInsn)(unsafe.Pointer(&i[0]))
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCSETF, uintptr(unsafe.Pointer(&p)))
	if e := int(ep); e != 0 {
		return e
	}
	return 0
}

func CheckBpfVersion(fd int) int {
	var v BpfVersion
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCVERSION, uintptr(unsafe.Pointer(&v)))
	if e := int(ep); e != 0 {
		return e
	}
	if v.Major != BPF_MAJOR_VERSION || v.Minor != BPF_MINOR_VERSION {
		return EINVAL
	}
	return 0
}

func BpfHeadercmpl(fd int) (int, int) {
	var f int
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCGHDRCMPLT, uintptr(unsafe.Pointer(&f)))
	if e := int(ep); e != 0 {
		return 0, e
	}
	return f, 0
}

func SetBpfHeadercmpl(fd, f int) int {
	_, _, ep := Syscall(SYS_IOCTL, uintptr(fd), BIOCSHDRCMPLT, uintptr(unsafe.Pointer(&f)))
	if e := int(ep); e != 0 {
		return e
	}
	return 0
}
