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

// CPUSet represents a bit mask of CPUs, to be used with [SchedGetaffinity], [SchedSetaffinity],
// and [SetMemPolicy].
//
// Note this type can only represent CPU IDs 0 through 1023.
// Use [CPUSetDynamic]/[NewCPUSet] instead to avoid this limit.
type CPUSet [cpuSetSize]cpuMask

// CPUSetDynamic represents a bit mask of CPUs, to be used with [SchedGetaffinityDynamic],
// [SchedSetaffinityDynamic], and [SetMemPolicyDynamic]. Use [NewCPUSet] to allocate.
type CPUSetDynamic []cpuMask

func schedAffinity(trap uintptr, pid int, size uintptr, ptr unsafe.Pointer) error {
	_, _, e := RawSyscall(trap, uintptr(pid), uintptr(size), uintptr(ptr))
	if e != 0 {
		return errnoErr(e)
	}
	return nil
}

// SchedGetaffinity gets the CPU affinity mask of the thread specified by pid.
// If pid is 0 the calling thread is used.
func SchedGetaffinity(pid int, set *CPUSet) error {
	return schedAffinity(SYS_SCHED_GETAFFINITY, pid, unsafe.Sizeof(*set), unsafe.Pointer(set))
}

// SchedSetaffinity sets the CPU affinity mask of the thread specified by pid.
// If pid is 0 the calling thread is used.
func SchedSetaffinity(pid int, set *CPUSet) error {
	return schedAffinity(SYS_SCHED_SETAFFINITY, pid, unsafe.Sizeof(*set), unsafe.Pointer(set))
}

// Zero clears the set s, so that it contains no CPUs.
func (s *CPUSet) Zero() {
	clear(s[:])
}

// Fill adds all possible CPU bits to the set s. On Linux, [SchedSetaffinity]
// will silently ignore any invalid CPU bits in [CPUSet] so this is an
// efficient way of resetting the CPU affinity of a process.
func (s *CPUSet) Fill() {
	cpuMaskFill(s[:])
}

func cpuBitsIndex(cpu int) int {
	return cpu / _NCPUBITS
}

func cpuBitsMask(cpu int) cpuMask {
	return cpuMask(1 << (uint(cpu) % _NCPUBITS))
}

func cpuMaskFill(s []cpuMask) {
	for i := range s {
		s[i] = ^cpuMask(0)
	}
}

func cpuMaskSet(s []cpuMask, cpu int) {
	i := cpuBitsIndex(cpu)
	if i < len(s) {
		s[i] |= cpuBitsMask(cpu)
	}
}

func cpuMaskClear(s []cpuMask, cpu int) {
	i := cpuBitsIndex(cpu)
	if i < len(s) {
		s[i] &^= cpuBitsMask(cpu)
	}
}

func cpuMaskIsSet(s []cpuMask, cpu int) bool {
	i := cpuBitsIndex(cpu)
	if i < len(s) {
		return s[i]&cpuBitsMask(cpu) != 0
	}
	return false
}

func cpuMaskCount(s []cpuMask) int {
	c := 0
	for _, b := range s {
		c += bits.OnesCount64(uint64(b))
	}
	return c
}

// Set adds cpu to the set s. If cpu is out of bounds for s, no action is taken.
func (s *CPUSet) Set(cpu int) {
	cpuMaskSet(s[:], cpu)
}

// Clear removes cpu from the set s. If cpu is out of bounds for s, no action is taken.
func (s *CPUSet) Clear(cpu int) {
	cpuMaskClear(s[:], cpu)
}

// IsSet reports whether cpu is in the set s.
func (s *CPUSet) IsSet(cpu int) bool {
	return cpuMaskIsSet(s[:], cpu)
}

// Count returns the number of CPUs in the set s.
func (s *CPUSet) Count() int {
	return cpuMaskCount(s[:])
}

// NewCPUSet creates a CPU affinity mask capable of representing CPU IDs
// up to maxCPU (exclusive).
func NewCPUSet(maxCPU int) CPUSetDynamic {
	numMasks := (maxCPU + _NCPUBITS - 1) / _NCPUBITS
	if numMasks == 0 {
		numMasks = 1
	}
	return make(CPUSetDynamic, numMasks)
}

// Zero clears the set s, so that it contains no CPUs.
func (s CPUSetDynamic) Zero() {
	clear(s)
}

// Fill adds all possible CPU bits to the set s. On Linux, [SchedSetaffinityDynamic]
// will silently ignore any invalid CPU bits in [CPUSetDynamic] so this is an
// efficient way of resetting the CPU affinity of a process.
func (s CPUSetDynamic) Fill() {
	cpuMaskFill(s)
}

// Set adds cpu to the set s. If cpu is out of bounds for s, no action is taken.
func (s CPUSetDynamic) Set(cpu int) {
	cpuMaskSet(s, cpu)
}

// Clear removes cpu from the set s. If cpu is out of bounds for s, no action is taken.
func (s CPUSetDynamic) Clear(cpu int) {
	cpuMaskClear(s, cpu)
}

// IsSet reports whether cpu is in the set s.
func (s CPUSetDynamic) IsSet(cpu int) bool {
	return cpuMaskIsSet(s, cpu)
}

// Count returns the number of CPUs in the set s.
func (s CPUSetDynamic) Count() int {
	return cpuMaskCount(s)
}

func (s CPUSetDynamic) size() uintptr {
	return uintptr(len(s)) * unsafe.Sizeof(cpuMask(0))
}

func (s CPUSetDynamic) pointer() unsafe.Pointer {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Pointer(&s[0])
}

// SchedGetaffinityDynamic gets the CPU affinity mask of the thread specified by pid.
// If pid is 0 the calling thread is used.
//
// If the set is smaller than the size of the affinity mask used by the kernel,
// [EINVAL] is returned.
func SchedGetaffinityDynamic(pid int, set CPUSetDynamic) error {
	return schedAffinity(SYS_SCHED_GETAFFINITY, pid, set.size(), set.pointer())
}

// SchedSetaffinityDynamic sets the CPU affinity mask of the thread specified by pid.
// If pid is 0 the calling thread is used.
func SchedSetaffinityDynamic(pid int, set CPUSetDynamic) error {
	return schedAffinity(SYS_SCHED_SETAFFINITY, pid, set.size(), set.pointer())
}
