// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CPU affinity functions

package unix

import (
	"math/bits"
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
		c += bits.OnesCount64(uint64(b))
	}
	return c
}
