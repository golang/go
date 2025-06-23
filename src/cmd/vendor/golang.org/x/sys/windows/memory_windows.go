// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

const (
	MEM_COMMIT      = 0x00001000
	MEM_RESERVE     = 0x00002000
	MEM_DECOMMIT    = 0x00004000
	MEM_RELEASE     = 0x00008000
	MEM_RESET       = 0x00080000
	MEM_TOP_DOWN    = 0x00100000
	MEM_WRITE_WATCH = 0x00200000
	MEM_PHYSICAL    = 0x00400000
	MEM_RESET_UNDO  = 0x01000000
	MEM_LARGE_PAGES = 0x20000000

	PAGE_NOACCESS          = 0x00000001
	PAGE_READONLY          = 0x00000002
	PAGE_READWRITE         = 0x00000004
	PAGE_WRITECOPY         = 0x00000008
	PAGE_EXECUTE           = 0x00000010
	PAGE_EXECUTE_READ      = 0x00000020
	PAGE_EXECUTE_READWRITE = 0x00000040
	PAGE_EXECUTE_WRITECOPY = 0x00000080
	PAGE_GUARD             = 0x00000100
	PAGE_NOCACHE           = 0x00000200
	PAGE_WRITECOMBINE      = 0x00000400
	PAGE_TARGETS_INVALID   = 0x40000000
	PAGE_TARGETS_NO_UPDATE = 0x40000000

	QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
	QUOTA_LIMITS_HARDWS_MIN_ENABLE  = 0x00000001
	QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x00000008
	QUOTA_LIMITS_HARDWS_MAX_ENABLE  = 0x00000004
)

type MemoryBasicInformation struct {
	BaseAddress       uintptr
	AllocationBase    uintptr
	AllocationProtect uint32
	PartitionId       uint16
	RegionSize        uintptr
	State             uint32
	Protect           uint32
	Type              uint32
}
