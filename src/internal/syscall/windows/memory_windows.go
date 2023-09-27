// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

type MemoryBasicInformation struct {
	// A pointer to the base address of the region of pages.
	BaseAddress uintptr
	// A pointer to the base address of a range of pages allocated by the VirtualAlloc function.
	// The page pointed to by the BaseAddress member is contained within this allocation range.
	AllocationBase uintptr
	// The memory protection option when the region was initially allocated
	AllocationProtect uint32
	PartitionId       uint16
	// The size of the region beginning at the base address in which all pages have identical attributes, in bytes.
	RegionSize uintptr
	// The state of the pages in the region.
	State uint32
	// The access protection of the pages in the region.
	Protect uint32
	// The type of pages in the region.
	Type uint32
}
