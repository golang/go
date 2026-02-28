// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

// Berkeley packet filter for BSD variants

package syscall

import (
	"unsafe"
)

// Deprecated: Use golang.org/x/net/bpf instead.
func BpfStmt(code, k int) *BpfInsn {
	return &BpfInsn{Code: uint16(code), K: uint32(k)}
}

// Deprecated: Use golang.org/x/net/bpf instead.
func BpfJump(code, k, jt, jf int) *BpfInsn {
	return &BpfInsn{Code: uint16(code), Jt: uint8(jt), Jf: uint8(jf), K: uint32(k)}
}

// Deprecated: Use golang.org/x/net/bpf instead.
func BpfBuflen(fd int) (int, error) {
	var l int
	err := ioctlPtr(fd, BIOCGBLEN, unsafe.Pointer(&l))
	if err != nil {
		return 0, err
	}
	return l, nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func SetBpfBuflen(fd, l int) (int, error) {
	err := ioctlPtr(fd, BIOCSBLEN, unsafe.Pointer(&l))
	if err != nil {
		return 0, err
	}
	return l, nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func BpfDatalink(fd int) (int, error) {
	var t int
	err := ioctlPtr(fd, BIOCGDLT, unsafe.Pointer(&t))
	if err != nil {
		return 0, err
	}
	return t, nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func SetBpfDatalink(fd, t int) (int, error) {
	err := ioctlPtr(fd, BIOCSDLT, unsafe.Pointer(&t))
	if err != nil {
		return 0, err
	}
	return t, nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func SetBpfPromisc(fd, m int) error {
	err := ioctlPtr(fd, BIOCPROMISC, unsafe.Pointer(&m))
	if err != nil {
		return err
	}
	return nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func FlushBpf(fd int) error {
	err := ioctlPtr(fd, BIOCFLUSH, nil)
	if err != nil {
		return err
	}
	return nil
}

type ivalue struct {
	name  [IFNAMSIZ]byte
	value int16
}

// Deprecated: Use golang.org/x/net/bpf instead.
func BpfInterface(fd int, name string) (string, error) {
	var iv ivalue
	err := ioctlPtr(fd, BIOCGETIF, unsafe.Pointer(&iv))
	if err != nil {
		return "", err
	}
	return name, nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func SetBpfInterface(fd int, name string) error {
	var iv ivalue
	copy(iv.name[:], []byte(name))
	err := ioctlPtr(fd, BIOCSETIF, unsafe.Pointer(&iv))
	if err != nil {
		return err
	}
	return nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func BpfTimeout(fd int) (*Timeval, error) {
	var tv Timeval
	err := ioctlPtr(fd, BIOCGRTIMEOUT, unsafe.Pointer(&tv))
	if err != nil {
		return nil, err
	}
	return &tv, nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func SetBpfTimeout(fd int, tv *Timeval) error {
	err := ioctlPtr(fd, BIOCSRTIMEOUT, unsafe.Pointer(tv))
	if err != nil {
		return err
	}
	return nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func BpfStats(fd int) (*BpfStat, error) {
	var s BpfStat
	err := ioctlPtr(fd, BIOCGSTATS, unsafe.Pointer(&s))
	if err != nil {
		return nil, err
	}
	return &s, nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func SetBpfImmediate(fd, m int) error {
	err := ioctlPtr(fd, BIOCIMMEDIATE, unsafe.Pointer(&m))
	if err != nil {
		return err
	}
	return nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func SetBpf(fd int, i []BpfInsn) error {
	var p BpfProgram
	p.Len = uint32(len(i))
	p.Insns = (*BpfInsn)(unsafe.Pointer(&i[0]))
	err := ioctlPtr(fd, BIOCSETF, unsafe.Pointer(&p))
	if err != nil {
		return err
	}
	return nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func CheckBpfVersion(fd int) error {
	var v BpfVersion
	err := ioctlPtr(fd, BIOCVERSION, unsafe.Pointer(&v))
	if err != nil {
		return err
	}
	if v.Major != BPF_MAJOR_VERSION || v.Minor != BPF_MINOR_VERSION {
		return EINVAL
	}
	return nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func BpfHeadercmpl(fd int) (int, error) {
	var f int
	err := ioctlPtr(fd, BIOCGHDRCMPLT, unsafe.Pointer(&f))
	if err != nil {
		return 0, err
	}
	return f, nil
}

// Deprecated: Use golang.org/x/net/bpf instead.
func SetBpfHeadercmpl(fd, f int) error {
	err := ioctlPtr(fd, BIOCSHDRCMPLT, unsafe.Pointer(&f))
	if err != nil {
		return err
	}
	return nil
}
