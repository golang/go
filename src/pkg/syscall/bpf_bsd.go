// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd netbsd openbsd

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

func BpfBuflen(fd int) (int, error) {
	var l int
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCGBLEN, uintptr(unsafe.Pointer(&l)))
	if err != 0 {
		return 0, Errno(err)
	}
	return l, nil
}

func SetBpfBuflen(fd, l int) (int, error) {
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCSBLEN, uintptr(unsafe.Pointer(&l)))
	if err != 0 {
		return 0, Errno(err)
	}
	return l, nil
}

func BpfDatalink(fd int) (int, error) {
	var t int
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCGDLT, uintptr(unsafe.Pointer(&t)))
	if err != 0 {
		return 0, Errno(err)
	}
	return t, nil
}

func SetBpfDatalink(fd, t int) (int, error) {
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCSDLT, uintptr(unsafe.Pointer(&t)))
	if err != 0 {
		return 0, Errno(err)
	}
	return t, nil
}

func SetBpfPromisc(fd, m int) error {
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCPROMISC, uintptr(unsafe.Pointer(&m)))
	if err != 0 {
		return Errno(err)
	}
	return nil
}

func FlushBpf(fd int) error {
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCFLUSH, 0)
	if err != 0 {
		return Errno(err)
	}
	return nil
}

type ivalue struct {
	name  [IFNAMSIZ]byte
	value int16
}

func BpfInterface(fd int, name string) (string, error) {
	var iv ivalue
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCGETIF, uintptr(unsafe.Pointer(&iv)))
	if err != 0 {
		return "", Errno(err)
	}
	return name, nil
}

func SetBpfInterface(fd int, name string) error {
	var iv ivalue
	copy(iv.name[:], []byte(name))
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCSETIF, uintptr(unsafe.Pointer(&iv)))
	if err != 0 {
		return Errno(err)
	}
	return nil
}

func BpfTimeout(fd int) (*Timeval, error) {
	var tv Timeval
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCGRTIMEOUT, uintptr(unsafe.Pointer(&tv)))
	if err != 0 {
		return nil, Errno(err)
	}
	return &tv, nil
}

func SetBpfTimeout(fd int, tv *Timeval) error {
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCSRTIMEOUT, uintptr(unsafe.Pointer(tv)))
	if err != 0 {
		return Errno(err)
	}
	return nil
}

func BpfStats(fd int) (*BpfStat, error) {
	var s BpfStat
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCGSTATS, uintptr(unsafe.Pointer(&s)))
	if err != 0 {
		return nil, Errno(err)
	}
	return &s, nil
}

func SetBpfImmediate(fd, m int) error {
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCIMMEDIATE, uintptr(unsafe.Pointer(&m)))
	if err != 0 {
		return Errno(err)
	}
	return nil
}

func SetBpf(fd int, i []BpfInsn) error {
	var p BpfProgram
	p.Len = uint32(len(i))
	p.Insns = (*BpfInsn)(unsafe.Pointer(&i[0]))
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCSETF, uintptr(unsafe.Pointer(&p)))
	if err != 0 {
		return Errno(err)
	}
	return nil
}

func CheckBpfVersion(fd int) error {
	var v BpfVersion
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCVERSION, uintptr(unsafe.Pointer(&v)))
	if err != 0 {
		return Errno(err)
	}
	if v.Major != BPF_MAJOR_VERSION || v.Minor != BPF_MINOR_VERSION {
		return EINVAL
	}
	return nil
}

func BpfHeadercmpl(fd int) (int, error) {
	var f int
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCGHDRCMPLT, uintptr(unsafe.Pointer(&f)))
	if err != 0 {
		return 0, Errno(err)
	}
	return f, nil
}

func SetBpfHeadercmpl(fd, f int) error {
	_, _, err := Syscall(SYS_IOCTL, uintptr(fd), BIOCSHDRCMPLT, uintptr(unsafe.Pointer(&f)))
	if err != 0 {
		return Errno(err)
	}
	return nil
}
