// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux darwin dragonfly freebsd openbsd netbsd solaris

package tar

import (
	"io"
	"os"
	"runtime"
	"syscall"
)

func init() {
	sysSparseDetect = sparseDetectUnix
}

func sparseDetectUnix(f *os.File) (sph sparseHoles, err error) {
	// SEEK_DATA and SEEK_HOLE originated from Solaris and support for it
	// has been added to most of the other major Unix systems.
	var seekData, seekHole = 3, 4 // SEEK_DATA/SEEK_HOLE from unistd.h

	if runtime.GOOS == "darwin" {
		// Darwin has the constants swapped, compared to all other UNIX.
		seekData, seekHole = 4, 3
	}

	// Check for seekData/seekHole support.
	// Different OS and FS may differ in the exact errno that is returned when
	// there is no support. Rather than special-casing every possible errno
	// representing "not supported", just assume that a non-nil error means
	// that seekData/seekHole is not supported.
	if _, err := f.Seek(0, seekHole); err != nil {
		return nil, nil
	}

	// Populate the SparseHoles.
	var last, pos int64 = -1, 0
	for {
		// Get the location of the next hole section.
		if pos, err = fseek(f, pos, seekHole); pos == last || err != nil {
			return sph, err
		}
		offset := pos
		last = pos

		// Get the location of the next data section.
		if pos, err = fseek(f, pos, seekData); pos == last || err != nil {
			return sph, err
		}
		length := pos - offset
		last = pos

		if length > 0 {
			sph = append(sph, SparseEntry{offset, length})
		}
	}
}

func fseek(f *os.File, pos int64, whence int) (int64, error) {
	pos, err := f.Seek(pos, whence)
	if errno(err) == syscall.ENXIO {
		// SEEK_DATA returns ENXIO when past the last data fragment,
		// which makes determining the size of the last hole difficult.
		pos, err = f.Seek(0, io.SeekEnd)
	}
	return pos, err
}

func errno(err error) error {
	if perr, ok := err.(*os.PathError); ok {
		return perr.Err
	}
	return err
}
