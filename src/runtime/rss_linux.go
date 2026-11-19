// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/runtime/syscall/linux"

var (
	procSelfStatm = []byte("/proc/self/statm\x00")
	// procSelfStatmFD is the file descriptor for reading /proc/self/statm
	// to compute RSS. sysRSSInit opens it at startup and it remains open
	// for the entire process lifetime. sysRSS rereads it from the
	// beginning.
	//
	// We open the file at startup rather than on each call because sysRSS
	// is called on every GC cycle, and opening files at unpredictable times
	// during execution is a problem for some programs: sandboxed programs
	// may install seccomp filters after startup that forbid unexpected
	// opens, and file descriptor leak detection is difficult to implement
	// in a stable way if the runtime opens new files periodically. See
	// defaultGOMAXPROCSInit for another example of this pattern.
	procSelfStatmFD = -1
)

func sysRSSInit() {
	for {
		fd, errno := linux.Open(&procSelfStatm[0], linux.O_RDONLY|linux.O_CLOEXEC, 0)
		if errno == _EINTR {
			continue
		}
		if errno == 0 {
			procSelfStatmFD = fd
		}
		return
	}
}

// sysRSS returns the current resident set size of this process in bytes.
// Linux's statm RSS accounting is asynchronous, so the result is approximate.
func sysRSS() (uint64, bool) {
	fd := procSelfStatmFD
	if fd < 0 {
		return 0, false
	}

	// We only need the first two fields of /proc/self/statm. Each is an
	// unsigned long, so 20 decimal digits per field plus their separating
	// space is enough even on 64-bit systems.
	var buf [2*20 + 1]byte
	for {
		// Read from the beginning to get a fresh value.
		n, errno := linux.Pread(fd, buf[:], 0)
		if errno == _EINTR {
			continue
		}
		if errno != 0 || n <= 0 {
			return 0, false
		}
		return parseProcStatmRSS(buf[:n], uint64(physPageSize))
	}
}

// parseProcStatmRSS parses the first two fields of /proc/self/statm and
// returns the resident page count converted to bytes.
func parseProcStatmRSS(data []byte, pageSize uint64) (uint64, bool) {
	if pageSize == 0 {
		return 0, false
	}

	// Skip the first field. Its value is not needed, but it must contain at
	// least one digit and be followed by a space.
	i := 0
	for i < len(data) && data[i] >= '0' && data[i] <= '9' {
		i++
	}
	if i == 0 || i == len(data) || data[i] != ' ' {
		return 0, false
	}
	i++

	// Parse the resident page count.
	start := i
	var resident uint64
	for i < len(data) && data[i] >= '0' && data[i] <= '9' {
		digit := uint64(data[i] - '0')
		if resident > (^uint64(0)-digit)/10 {
			return 0, false
		}
		resident = resident*10 + digit
		i++
	}
	if i == start || (i < len(data) && data[i] != ' ') {
		return 0, false
	}
	if resident > ^uint64(0)/pageSize {
		return 0, false
	}
	return resident * pageSize, true
}
