// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

const (
	_AT_NULL     = 0
	_AT_PLATFORM = 15 //  introduced in at least 2.6.11
	_AT_HWCAP    = 16 // introduced in at least 2.6.11
	_AT_RANDOM   = 25 // introduced in 2.6.29

	_HWCAP_VFP   = 1 << 6  // introduced in at least 2.6.11
	_HWCAP_VFPv3 = 1 << 13 // introduced in 2.6.30
)

var randomNumber uint32
var armArch uint8 = 6 // we default to ARMv6
var hwcap uint32      // set by setup_auxv
var goarm uint8       // set by 5l

func checkgoarm() {
	if goarm > 5 && hwcap&_HWCAP_VFP == 0 {
		print("runtime: this CPU has no floating point hardware, so it cannot run\n")
		print("this GOARM=", goarm, " binary. Recompile using GOARM=5.\n")
		exit(1)
	}
	if goarm > 6 && hwcap&_HWCAP_VFPv3 == 0 {
		print("runtime: this CPU has no VFPv3 floating point hardware, so it cannot run\n")
		print("this GOARM=", goarm, " binary. Recompile using GOARM=5.\n")
		exit(1)
	}
}

func sysargs(argc int32, argv **byte) {
	// skip over argv, envv to get to auxv
	n := argc + 1
	for argv_index(argv, n) != nil {
		n++
	}
	n++
	auxv := (*[1 << 28]uint32)(add(unsafe.Pointer(argv), uintptr(n)*ptrSize))

	for i := 0; auxv[i] != _AT_NULL; i += 2 {
		switch auxv[i] {
		case _AT_RANDOM: // kernel provides a pointer to 16-bytes worth of random data
			startupRandomData = (*[16]byte)(unsafe.Pointer(uintptr(auxv[i+1])))[:]
			// the pointer provided may not be word alined, so we must to treat it
			// as a byte array.
			randomNumber = uint32(startupRandomData[4]) | uint32(startupRandomData[5])<<8 |
				uint32(startupRandomData[6])<<16 | uint32(startupRandomData[7])<<24

		case _AT_PLATFORM: // v5l, v6l, v7l
			t := *(*uint8)(unsafe.Pointer(uintptr(auxv[i+1] + 1)))
			if '5' <= t && t <= '7' {
				armArch = t - '0'
			}

		case _AT_HWCAP: // CPU capability bit flags
			hwcap = auxv[i+1]
		}
	}
}

func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed fastrand1().
	// nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// randomNumber provides better seeding of fastrand1.
	return nanotime() + int64(randomNumber)
}
