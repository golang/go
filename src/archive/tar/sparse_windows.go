// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package tar

import (
	"os"
	"syscall"
	"unsafe"
)

var errInvalidFunc = syscall.Errno(1) // ERROR_INVALID_FUNCTION from WinError.h

func init() {
	sysSparseDetect = sparseDetectWindows
	sysSparsePunch = sparsePunchWindows
}

func sparseDetectWindows(f *os.File) (sph sparseHoles, err error) {
	const queryAllocRanges = 0x000940CF                  // FSCTL_QUERY_ALLOCATED_RANGES from WinIoCtl.h
	type allocRangeBuffer struct{ offset, length int64 } // FILE_ALLOCATED_RANGE_BUFFER from WinIoCtl.h

	s, err := f.Stat()
	if err != nil {
		return nil, err
	}

	queryRange := allocRangeBuffer{0, s.Size()}
	allocRanges := make([]allocRangeBuffer, 64)

	// Repeatedly query for ranges until the input buffer is large enough.
	var bytesReturned uint32
	for {
		err := syscall.DeviceIoControl(
			syscall.Handle(f.Fd()), queryAllocRanges,
			(*byte)(unsafe.Pointer(&queryRange)), uint32(unsafe.Sizeof(queryRange)),
			(*byte)(unsafe.Pointer(&allocRanges[0])), uint32(len(allocRanges)*int(unsafe.Sizeof(allocRanges[0]))),
			&bytesReturned, nil,
		)
		if err == syscall.ERROR_MORE_DATA {
			allocRanges = make([]allocRangeBuffer, 2*len(allocRanges))
			continue
		}
		if err == errInvalidFunc {
			return nil, nil // Sparse file not supported on this FS
		}
		if err != nil {
			return nil, err
		}
		break
	}
	n := bytesReturned / uint32(unsafe.Sizeof(allocRanges[0]))
	allocRanges = append(allocRanges[:n], allocRangeBuffer{s.Size(), 0})

	// Invert the data fragments into hole fragments.
	var pos int64
	for _, r := range allocRanges {
		if r.offset > pos {
			sph = append(sph, SparseEntry{pos, r.offset - pos})
		}
		pos = r.offset + r.length
	}
	return sph, nil
}

func sparsePunchWindows(f *os.File, sph sparseHoles) error {
	const setSparse = 0x000900C4                 // FSCTL_SET_SPARSE from WinIoCtl.h
	const setZeroData = 0x000980C8               // FSCTL_SET_ZERO_DATA from WinIoCtl.h
	type zeroDataInfo struct{ start, end int64 } // FILE_ZERO_DATA_INFORMATION from WinIoCtl.h

	// Set the file as being sparse.
	var bytesReturned uint32
	devErr := syscall.DeviceIoControl(
		syscall.Handle(f.Fd()), setSparse,
		nil, 0, nil, 0,
		&bytesReturned, nil,
	)
	if devErr != nil && devErr != errInvalidFunc {
		return devErr
	}

	// Set the file to the right size.
	var size int64
	if len(sph) > 0 {
		size = sph[len(sph)-1].endOffset()
	}
	if err := f.Truncate(size); err != nil {
		return err
	}
	if devErr == errInvalidFunc {
		// Sparse file not supported on this FS.
		// Call sparsePunchManual since SetEndOfFile does not guarantee that
		// the extended space is filled with zeros.
		return sparsePunchManual(f, sph)
	}

	// Punch holes for all relevant fragments.
	for _, s := range sph {
		zdi := zeroDataInfo{s.Offset, s.endOffset()}
		err := syscall.DeviceIoControl(
			syscall.Handle(f.Fd()), setZeroData,
			(*byte)(unsafe.Pointer(&zdi)), uint32(unsafe.Sizeof(zdi)),
			nil, 0,
			&bytesReturned, nil,
		)
		if err != nil {
			return err
		}
	}
	return nil
}

// sparsePunchManual writes zeros into each hole.
func sparsePunchManual(f *os.File, sph sparseHoles) error {
	const chunkSize = 32 << 10
	zbuf := make([]byte, chunkSize)
	for _, s := range sph {
		for pos := s.Offset; pos < s.endOffset(); pos += chunkSize {
			n := min(chunkSize, s.endOffset()-pos)
			if _, err := f.WriteAt(zbuf[:n], pos); err != nil {
				return err
			}
		}
	}
	return nil
}
