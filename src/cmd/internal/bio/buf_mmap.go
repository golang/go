// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package bio

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
)

// mmapLimit is the maximum number of mmaped regions to create before
// falling back to reading into a heap-allocated slice. This exists
// because some operating systems place a limit on the number of
// distinct mapped regions per process. As of this writing:
//
//	Darwin    unlimited
//	DragonFly   1000000 (vm.max_proc_mmap)
//	FreeBSD   unlimited
//	Linux         reads from /proc/sys/vm/max_map_count
//	NetBSD    unlimited
//	OpenBSD   unlimited
var mmapLimit int32 = 1<<31 - 1

func init() {
	// Linux is the only practically concerning OS.
	if runtime.GOOS == "linux" {
		mmapLimit = getLinuxMaxMapCount()
	}
}

func getLinuxMaxMapCount() int32 {
	data, err := os.ReadFile("/proc/sys/vm/max_map_count")
	if err != nil {
		return 1<<31 - 1
	}

	maxMapCount, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return 1<<31 - 1
	}

	return int32(maxMapCount)
}

func (r *Reader) sliceOS(length uint64) ([]byte, bool) {
	// For small slices, don't bother with the overhead of a
	// mapping, especially since we have no way to unmap it.
	const threshold = 16 << 10
	if length < threshold {
		return nil, false
	}

	// Have we reached the mmap limit?
	if atomic.AddInt32(&mmapLimit, -1) < 0 {
		atomic.AddInt32(&mmapLimit, 1)
		return nil, false
	}

	// Page-align the offset.
	off := r.Offset()
	align := syscall.Getpagesize()
	aoff := off &^ int64(align-1)

	data, err := syscall.Mmap(int(r.f.Fd()), aoff, int(length+uint64(off-aoff)), syscall.PROT_READ, syscall.MAP_SHARED|syscall.MAP_FILE)
	if err != nil {
		return nil, false
	}

	data = data[off-aoff:]
	r.MustSeek(int64(length), 1)
	return data, true
}
