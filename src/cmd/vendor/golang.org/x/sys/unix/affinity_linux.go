// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CPU affinity functions

package unix

import (
	"unsafe"
)

const cpuSetSize = _CPU_SETSIZE / _NCPUBITS

// CPUSet represents a CPU affinity mask.
type CPUSet [cpuSetSize]cpuMask

func schedAffinity(trap uintptr, pid int, set *CPUSet) error {
	_, _, e := RawSyscall(trap, uintptr(pid), uintptr(unsafe.Sizeof(*set)), uintptr(unsafe.Pointer(set)))
	if e != 0 {
		return errnoErr(e)
	}
	return nil
}

// SchedGetaffinity gets the CPU affinity mask of the thread specified by pid.
// If pid is 0 the calling thread is used.
func SchedGetaffinity(pid int, set *CPUSet) error {
	return schedAffinity(SYS_SCHED_GETAFFINITY, pid, set)
}

// SchedSetaffinity sets the CPU affinity mask of the thread specified by pid.
// If pid is 0 the calling thread is used.
func SchedSetaffinity(pid int, set *CPUSet) error {
	return schedAffinity(SYS_SCHED_SETAFFINITY, pid, set)
}

// Zero clears the set s, so that it contains no CPUs.
func (s *CPUSet) Zero() {
	for i := range s {
		s[i] = 0
	}
}

func cpuBitsIndex(cpu int) int {
	return cpu / _NCPUBITS
}

func cpuBitsMask(cpu int) cpuMask {
	return cpuMask(1 << (uint(cpu) % _NCPUBITS))
}

// Set adds cpu to the set s.
func (s *CPUSet) Set(cpu int) {
	i := cpuBitsIndex(cpu)
	if i < len(s) {
		s[i] |= cpuBitsMask(cpu)
	}
}

// Clear removes cpu from the set s.
func (s *CPUSet) Clear(cpu int) {
	i := cpuBitsIndex(cpu)
	if i < len(s) {
		s[i] &^= cpuBitsMask(cpu)
	}
}

// IsSet reports whether cpu is in the set s.
func (s *CPUSet) IsSet(cpu int) bool {
	i := cpuBitsIndex(cpu)
	if i < len(s) {
		return s[i]&cpuBitsMask(cpu) != 0
	}
	return false
}

// Count returns the number of CPUs in the set s.
func (s *CPUSet) Count() int {
	c := 0
	for _, b := range s {
		c += onesCount64(uint64(b))
	}
	return c
}

// onesCount64 is a copy of Go 1.9's math/bits.OnesCount64.
// Once this package can require Go 1.9, we can delete this
// and update the caller to use bits.OnesCount64.
func onesCount64(x uint64) int {
	const m0 = 0x5555555555555555 // 01010101 ...
	const m1 = 0x3333333333333333 // 00110011 ...
	const m2 = 0x0f0f0f0f0f0f0f0f // 00001111 ...
	const m3 = 0x00ff00ff00ff00ff // etc.
	const m4 = 0x0000ffff0000ffff

	// Implementation: Parallel summing of adjacent bits.
	// See "Hacker's Delight", Chap. 5: Counting Bits.
	// The following pattern shows the general approach:
	//
	//   x = x>>1&(m0&m) + x&(m0&m)
	//   x = x>>2&(m1&m) + x&(m1&m)
	//   x = x>>4&(m2&m) + x&(m2&m)
	//   x = x>>8&(m3&m) + x&(m3&m)
	//   x = x>>16&(m4&m) + x&(m4&m)
	//   x = x>>32&(m5&m) + x&(m5&m)
	//   return int(x)
	//
	// Masking (& operations) can be left away when there's no
	// danger that a field's sum will carry over into the next
	// field: Since the result cannot be > 64, 8 bits is enough
	// and we can ignore the masks for the shifts by 8 and up.
	// Per "Hacker's Delight", the first line can be simplified
	// more, but it saves at best one instruction, so we leave
	// it alone for clarity.
	const m = 1<<64 - 1
	x = x>>1&(m0&m) + x&(m0&m)
	x = x>>2&(m1&m) + x&(m1&m)
	x = (x>>4 + x) & (m2 & m)
	x += x >> 8
	x += x >> 16
	x += x >> 32
	return int(x) & (1<<7 - 1)
}
