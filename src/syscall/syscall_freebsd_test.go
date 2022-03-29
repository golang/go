// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd

package syscall_test

import (
	"fmt"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

func TestConvertFromDirent11(t *testing.T) {
	const (
		filenameFmt  = "%04d"
		numFiles     = 64
		fixedHdrSize = int(unsafe.Offsetof(syscall.Dirent_freebsd11{}.Name))
	)

	namlen := len(fmt.Sprintf(filenameFmt, 0))
	reclen := syscall.Roundup(fixedHdrSize+namlen+1, 4)
	old := make([]byte, numFiles*reclen)
	for i := 0; i < numFiles; i++ {
		dent := syscall.Dirent_freebsd11{
			Fileno: uint32(i + 1),
			Reclen: uint16(reclen),
			Type:   syscall.DT_REG,
			Namlen: uint8(namlen),
		}
		rec := make([]byte, reclen)
		copy(rec, (*[fixedHdrSize]byte)(unsafe.Pointer(&dent))[:])
		copy(rec[fixedHdrSize:], fmt.Sprintf(filenameFmt, i+1))
		copy(old[i*reclen:], rec)
	}

	buf := make([]byte, 2*len(old))
	n := syscall.ConvertFromDirents11(buf, old)

	names := make([]string, 0, numFiles)
	_, _, names = syscall.ParseDirent(buf[:n], -1, names)

	if len(names) != numFiles {
		t.Errorf("expected %d files, have %d; names: %q", numFiles, len(names), names)
	}

	for i, name := range names {
		if expected := fmt.Sprintf(filenameFmt, i+1); name != expected {
			t.Errorf("expected names[%d] to be %q; got %q", i, expected, name)
		}
	}
}

func TestMain(m *testing.M) {
	if os.Getenv("GO_DEATHSIG_PARENT") == "1" {
		deathSignalParent()
	} else if os.Getenv("GO_DEATHSIG_CHILD") == "1" {
		deathSignalChild()
	}

	os.Exit(m.Run())
}
