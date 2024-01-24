// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"os"
	"unsafe"
)

func isExecutable(protection int32) bool {
	return (protection&_VM_PROT_EXECUTE) != 0 && (protection&_VM_PROT_READ) != 0
}

// machVMInfo uses the mach_vm_region region system call to add mapping entries
// for the text region of the running process.
func machVMInfo(addMapping func(lo, hi, offset uint64, file, buildID string)) bool {
	added := false
	var addr uint64 = 0x1
	for {
		var memRegionSize uint64
		var info machVMRegionBasicInfoData
		// Get the first address and page size.
		kr := mach_vm_region(
			&addr,
			&memRegionSize,
			unsafe.Pointer(&info))
		if kr != 0 {
			if kr == _MACH_SEND_INVALID_DEST {
				// No more memory regions.
				return true
			}
			return added // return true if at least one mapping was added
		}
		if isExecutable(info.Protection) {
			// NOTE: the meaning/value of Offset is unclear. However,
			// this likely doesn't matter as the text segment's file
			// offset is usually 0.
			addMapping(addr,
				addr+memRegionSize,
				read64(&info.Offset),
				regionFilename(addr),
				"")
			added = true
		}
		addr += memRegionSize
	}
}

func read64(p *[8]byte) uint64 {
	// all supported darwin platforms are little endian
	return uint64(p[0]) | uint64(p[1])<<8 | uint64(p[2])<<16 | uint64(p[3])<<24 | uint64(p[4])<<32 | uint64(p[5])<<40 | uint64(p[6])<<48 | uint64(p[7])<<56
}

func regionFilename(address uint64) string {
	buf := make([]byte, _MAXPATHLEN)
	r := proc_regionfilename(
		os.Getpid(),
		address,
		unsafe.SliceData(buf),
		int64(cap(buf)))
	if r == 0 {
		return ""
	}
	return string(buf[:r])
}

// mach_vm_region and proc_regionfilename are implemented by
// the runtime package (runtime/sys_darwin.go).
//
//go:noescape
func mach_vm_region(address, region_size *uint64, info unsafe.Pointer) int32

//go:noescape
func proc_regionfilename(pid int, address uint64, buf *byte, buflen int64) int32
