// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scan

import (
	"internal/goarch"
	"internal/runtime/gc"
)

// ExpandReference is a reference implementation of an expander function
// that translates object mark bits into a bitmap of one bit per word of
// marked object, assuming the object is of the provided size class.
func ExpandReference(sizeClass int, packed *gc.ObjMask, unpacked *gc.PtrMask) {
	// Look up the size and derive the number of objects in a span.
	// We're only concerned with small objects in single-page spans,
	// and gc.PtrMask enforces this by being statically sized to
	// accomodate only such spans.
	size := uintptr(gc.SizeClassToSize[sizeClass])
	nObj := uintptr(gc.SizeClassToNPages[sizeClass]) * gc.PageSize / size

	// f is the expansion factor. For example, if our objects are of size 48,
	// then each mark bit will translate into 6 (48/8 = 6) set bits in the
	// pointer bitmap.
	f := size / goarch.PtrSize
	for i := range nObj {
		// Check if the object is marked.
		if packed[i/goarch.PtrBits]&(uintptr(1)<<(i%goarch.PtrBits)) == 0 {
			continue
		}
		// Propagate that mark into the destination into one bit per the
		// expansion factor f, offset to the object's offset within the span.
		for j := range f {
			b := i*f + j // i*f is the start bit for the object, j indexes into each corresponding word after.
			unpacked[b/goarch.PtrBits] |= uintptr(1) << (b % goarch.PtrBits)
		}
	}
}
