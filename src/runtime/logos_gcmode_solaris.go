// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build solaris

package runtime

// From Solaris and illumos <sys/unistd.h>.
const logosSCPhysPages = 500

// logosPhysicalMemory returns _SC_PHYS_PAGES times the page size in bytes, or
// 0 when sysconf fails.
func logosPhysicalMemory() uint64 {
	pages, size := sysconf(logosSCPhysPages), sysconf(__SC_PAGESIZE)
	if pages <= 0 || size <= 0 {
		return 0
	}
	return uint64(pages) * uint64(size)
}
