// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import (
	"syscall"
	"unsafe"
)

// Minimal copy of functionality from x/sys/unix so the cpu package can call
// sysctl without depending on x/sys/unix.

const (
	_CTL_QUERY = -2

	_SYSCTL_VERS_1 = 0x1000000
)

var _zero uintptr

func sysctl(mib []int32, old *byte, oldlen *uintptr, new *byte, newlen uintptr) (err error) {
	var _p0 unsafe.Pointer
	if len(mib) > 0 {
		_p0 = unsafe.Pointer(&mib[0])
	} else {
		_p0 = unsafe.Pointer(&_zero)
	}
	_, _, errno := syscall.Syscall6(
		syscall.SYS___SYSCTL,
		uintptr(_p0),
		uintptr(len(mib)),
		uintptr(unsafe.Pointer(old)),
		uintptr(unsafe.Pointer(oldlen)),
		uintptr(unsafe.Pointer(new)),
		uintptr(newlen))
	if errno != 0 {
		return errno
	}
	return nil
}

type sysctlNode struct {
	Flags          uint32
	Num            int32
	Name           [32]int8
	Ver            uint32
	__rsvd         uint32
	Un             [16]byte
	_sysctl_size   [8]byte
	_sysctl_func   [8]byte
	_sysctl_parent [8]byte
	_sysctl_desc   [8]byte
}

func sysctlNodes(mib []int32) ([]sysctlNode, error) {
	var olen uintptr

	// Get a list of all sysctl nodes below the given MIB by performing
	// a sysctl for the given MIB with CTL_QUERY appended.
	mib = append(mib, _CTL_QUERY)
	qnode := sysctlNode{Flags: _SYSCTL_VERS_1}
	qp := (*byte)(unsafe.Pointer(&qnode))
	sz := unsafe.Sizeof(qnode)
	if err := sysctl(mib, nil, &olen, qp, sz); err != nil {
		return nil, err
	}

	// Now that we know the size, get the actual nodes.
	nodes := make([]sysctlNode, olen/sz)
	np := (*byte)(unsafe.Pointer(&nodes[0]))
	if err := sysctl(mib, np, &olen, qp, sz); err != nil {
		return nil, err
	}

	return nodes, nil
}

func nametomib(name string) ([]int32, error) {
	// Split name into components.
	var parts []string
	last := 0
	for i := 0; i < len(name); i++ {
		if name[i] == '.' {
			parts = append(parts, name[last:i])
			last = i + 1
		}
	}
	parts = append(parts, name[last:])

	mib := []int32{}
	// Discover the nodes and construct the MIB OID.
	for partno, part := range parts {
		nodes, err := sysctlNodes(mib)
		if err != nil {
			return nil, err
		}
		for _, node := range nodes {
			n := make([]byte, 0)
			for i := range node.Name {
				if node.Name[i] != 0 {
					n = append(n, byte(node.Name[i]))
				}
			}
			if string(n) == part {
				mib = append(mib, int32(node.Num))
				break
			}
		}
		if len(mib) != partno+1 {
			return nil, err
		}
	}

	return mib, nil
}

// aarch64SysctlCPUID is struct aarch64_sysctl_cpu_id from NetBSD's <aarch64/armreg.h>
type aarch64SysctlCPUID struct {
	midr      uint64 /* Main ID Register */
	revidr    uint64 /* Revision ID Register */
	mpidr     uint64 /* Multiprocessor Affinity Register */
	aa64dfr0  uint64 /* A64 Debug Feature Register 0 */
	aa64dfr1  uint64 /* A64 Debug Feature Register 1 */
	aa64isar0 uint64 /* A64 Instruction Set Attribute Register 0 */
	aa64isar1 uint64 /* A64 Instruction Set Attribute Register 1 */
	aa64mmfr0 uint64 /* A64 Memory Model Feature Register 0 */
	aa64mmfr1 uint64 /* A64 Memory Model Feature Register 1 */
	aa64mmfr2 uint64 /* A64 Memory Model Feature Register 2 */
	aa64pfr0  uint64 /* A64 Processor Feature Register 0 */
	aa64pfr1  uint64 /* A64 Processor Feature Register 1 */
	aa64zfr0  uint64 /* A64 SVE Feature ID Register 0 */
	mvfr0     uint32 /* Media and VFP Feature Register 0 */
	mvfr1     uint32 /* Media and VFP Feature Register 1 */
	mvfr2     uint32 /* Media and VFP Feature Register 2 */
	pad       uint32
	clidr     uint64 /* Cache Level ID Register */
	ctr       uint64 /* Cache Type Register */
}

func sysctlCPUID(name string) (*aarch64SysctlCPUID, error) {
	mib, err := nametomib(name)
	if err != nil {
		return nil, err
	}

	out := aarch64SysctlCPUID{}
	n := unsafe.Sizeof(out)
	_, _, errno := syscall.Syscall6(
		syscall.SYS___SYSCTL,
		uintptr(unsafe.Pointer(&mib[0])),
		uintptr(len(mib)),
		uintptr(unsafe.Pointer(&out)),
		uintptr(unsafe.Pointer(&n)),
		uintptr(0),
		uintptr(0))
	if errno != 0 {
		return nil, errno
	}
	return &out, nil
}

func doinit() {
	cpuid, err := sysctlCPUID("machdep.cpu0.cpu_id")
	if err != nil {
		setMinimalFeatures()
		return
	}
	parseARM64SystemRegisters(cpuid.aa64isar0, cpuid.aa64isar1, cpuid.aa64pfr0)

	Initialized = true
}
