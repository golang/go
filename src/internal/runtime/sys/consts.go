// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

import (
	"internal/goarch"
	"internal/goos"
)

// AIX and OpenBSD require a larger stack for syscalls.
// The race build also needs more stack. See issue 54291.
// This arithmetic must match that in cmd/internal/objabi/stack.go:stackGuardMultiplier.
const StackGuardMultiplier = 1 + goos.IsAix + goos.IsOpenbsd + isRace

// DefaultPhysPageSize is the default physical page size.
const DefaultPhysPageSize = goarch.DefaultPhysPageSize

// PCQuantum is the minimal unit for a program counter (1 on x86, 4 on most other systems).
// The various PC tables record PC deltas pre-divided by PCQuantum.
const PCQuantum = goarch.PCQuantum

// Int64Align is the required alignment for a 64-bit integer.
const Int64Align = goarch.Int64Align

// MinFrameSize is the size of the system-reserved words at the bottom
// of a frame (just above the architectural stack pointer).
// It is zero on x86 and PtrSize on most non-x86 (LR-based) systems.
// On PowerPC it is larger, to cover three more reserved words:
// the compiler word, the link editor word, and the TOC save word.
const MinFrameSize = goarch.MinFrameSize

// StackAlign is the required alignment of the SP register.
// The stack must be at least word aligned, but some architectures require more.
const StackAlign = goarch.StackAlign

// is64bit is 1 on 64-bit systems, 0 on 32-bit systems.
const is64bit = goarch.PtrSize / 8

// HeapAddrBits is the number of bits in a heap address. On
// amd64, addresses are sign-extended beyond HeapAddrBits. On
// other arches, they are zero-extended.
//
// On most 64-bit platforms, we limit this to 48 bits based on a
// combination of hardware and OS limitations.
//
// amd64 hardware limits addresses to 48 bits, sign-extended
// to 64 bits. Addresses where the top 16 bits are not either
// all 0 or all 1 are "non-canonical" and invalid. Because of
// these "negative" addresses, we offset addresses by 1<<47
// (arenaBaseOffset) on amd64 before computing indexes into
// the heap arenas index. In 2017, amd64 hardware added
// support for 57 bit addresses; however, currently only Linux
// supports this extension and the kernel will never choose an
// address above 1<<47 unless mmap is called with a hint
// address above 1<<47 (which we never do).
//
// arm64 hardware (as of ARMv8) limits user addresses to 48
// bits, in the range [0, 1<<48).
//
// ppc64, mips64, and s390x support arbitrary 64 bit addresses
// in hardware. On Linux, Go leans on stricter OS limits. Based
// on Linux's processor.h, the user address space is limited as
// follows on 64-bit architectures:
//
// Architecture  Name              Maximum Value (exclusive)
// ---------------------------------------------------------------------
// amd64         TASK_SIZE_MAX     0x007ffffffff000 (47 bit addresses)
// arm64         TASK_SIZE_64      0x01000000000000 (48 bit addresses)
// ppc64{,le}    TASK_SIZE_USER64  0x00400000000000 (46 bit addresses)
// mips64{,le}   TASK_SIZE64       0x00010000000000 (40 bit addresses)
// s390x         TASK_SIZE         1<<64 (64 bit addresses)
//
// These limits may increase over time, but are currently at
// most 48 bits except on s390x. On all architectures, Linux
// starts placing mmap'd regions at addresses that are
// significantly below 48 bits, so even if it's possible to
// exceed Go's 48 bit limit, it's extremely unlikely in
// practice.
//
// On 32-bit platforms, we accept the full 32-bit address
// space because doing so is cheap.
// mips32 only has access to the low 2GB of virtual memory, so
// we further limit it to 31 bits.
//
// On ios/arm64, although 64-bit pointers are presumably
// available, pointers are truncated to 33 bits in iOS <14.
// Furthermore, only the top 4 GiB of the address space are
// actually available to the application. In iOS >=14, more
// of the address space is available, and the OS can now
// provide addresses outside of those 33 bits. Pick 40 bits
// as a reasonable balance between address space usage by the
// page allocator, and flexibility for what mmap'd regions
// we'll accept for the heap. We can't just move to the full
// 48 bits because this uses too much address space for older
// iOS versions.
// TODO(mknyszek): Once iOS <14 is deprecated, promote ios/arm64
// to a 48-bit address space like every other arm64 platform.
//
// WebAssembly currently has a limit of 4GB linear memory.
//
// OpenBSD/arm64 is limited to 39 bits, according to:
// https://github.com/openbsd/src/blob/65a9769f762528944cb8c228592381427b188551/sys/arch/arm64/include/vmparam.h#L93
const HeapAddrBits = (is64bit*(1-goarch.IsWasm)*(1-goos.IsIos*goarch.IsArm64)*(1-goos.IsOpenbsd*goarch.IsArm64))*48 +
	(1-is64bit+goarch.IsWasm)*(32-(goarch.IsMips+goarch.IsMipsle)) +
	40*goos.IsIos*goarch.IsArm64 +
	39*goos.IsOpenbsd*goarch.IsArm64

// MaxAlloc is the maximum size of an allocation. On 64-bit,
// it's theoretically possible to allocate 1<<HeapAddrBits bytes. On
// 32-bit, however, this is one less than 1<<32 because the
// number of bytes in the address space doesn't actually fit
// in a uintptr.
const MaxAlloc = (1 << HeapAddrBits) - (1-is64bit)*1
