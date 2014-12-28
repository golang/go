// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Malloc small size classes.
//
// See malloc.h for overview.
//
// The size classes are chosen so that rounding an allocation
// request up to the next size class wastes at most 12.5% (1.125x).
//
// Each size class has its own page count that gets allocated
// and chopped up when new objects of the size class are needed.
// That page count is chosen so that chopping up the run of
// pages into objects of the given size wastes at most 12.5% (1.125x)
// of the memory.  It is not necessary that the cutoff here be
// the same as above.
//
// The two sources of waste multiply, so the worst possible case
// for the above constraints would be that allocations of some
// size might have a 26.6% (1.266x) overhead.
// In practice, only one of the wastes comes into play for a
// given size (sizes < 512 waste mainly on the round-up,
// sizes > 512 waste mainly on the page chopping).
//
// TODO(rsc): Compute max waste for any given size.

package runtime

//var class_to_size [_NumSizeClasses]int32
//var class_to_allocnpages [_NumSizeClasses]int32

// The SizeToClass lookup is implemented using two arrays,
// one mapping sizes <= 1024 to their class and one mapping
// sizes >= 1024 and <= MaxSmallSize to their class.
// All objects are 8-aligned, so the first array is indexed by
// the size divided by 8 (rounded up).  Objects >= 1024 bytes
// are 128-aligned, so the second array is indexed by the
// size divided by 128 (rounded up).  The arrays are filled in
// by InitSizes.
//var size_to_class8 [1024/8 + 1]int8
//var size_to_class128 [(_MaxSmallSize-1024)/128 + 1]int8

func sizeToClass(size int32) int32 {
	if size > _MaxSmallSize {
		throw("SizeToClass - invalid size")
	}
	if size > 1024-8 {
		return int32(size_to_class128[(size-1024+127)>>7])
	}
	return int32(size_to_class8[(size+7)>>3])
}

func initSizes() {
	// Initialize the runtime·class_to_size table (and choose class sizes in the process).
	class_to_size[0] = 0
	sizeclass := 1 // 0 means no class
	align := 8
	for size := align; size <= _MaxSmallSize; size += align {
		if size&(size-1) == 0 { // bump alignment once in a while
			if size >= 2048 {
				align = 256
			} else if size >= 128 {
				align = size / 8
			} else if size >= 16 {
				align = 16 // required for x86 SSE instructions, if we want to use them
			}
		}
		if align&(align-1) != 0 {
			throw("InitSizes - bug")
		}

		// Make the allocnpages big enough that
		// the leftover is less than 1/8 of the total,
		// so wasted space is at most 12.5%.
		allocsize := _PageSize
		for allocsize%size > allocsize/8 {
			allocsize += _PageSize
		}
		npages := allocsize >> _PageShift

		// If the previous sizeclass chose the same
		// allocation size and fit the same number of
		// objects into the page, we might as well
		// use just this size instead of having two
		// different sizes.
		if sizeclass > 1 && npages == int(class_to_allocnpages[sizeclass-1]) && allocsize/size == allocsize/int(class_to_size[sizeclass-1]) {
			class_to_size[sizeclass-1] = int32(size)
			continue
		}

		class_to_allocnpages[sizeclass] = int32(npages)
		class_to_size[sizeclass] = int32(size)
		sizeclass++
	}
	if sizeclass != _NumSizeClasses {
		print("sizeclass=", sizeclass, " NumSizeClasses=", _NumSizeClasses, "\n")
		throw("InitSizes - bad NumSizeClasses")
	}

	// Initialize the size_to_class tables.
	nextsize := 0
	for sizeclass = 1; sizeclass < _NumSizeClasses; sizeclass++ {
		for ; nextsize < 1024 && nextsize <= int(class_to_size[sizeclass]); nextsize += 8 {
			size_to_class8[nextsize/8] = int8(sizeclass)
		}
		if nextsize >= 1024 {
			for ; nextsize <= int(class_to_size[sizeclass]); nextsize += 128 {
				size_to_class128[(nextsize-1024)/128] = int8(sizeclass)
			}
		}
	}

	// Double-check SizeToClass.
	if false {
		for n := int32(0); n < _MaxSmallSize; n++ {
			sizeclass := sizeToClass(n)
			if sizeclass < 1 || sizeclass >= _NumSizeClasses || class_to_size[sizeclass] < n {
				print("size=", n, " sizeclass=", sizeclass, " runtime·class_to_size=", class_to_size[sizeclass], "\n")
				print("incorrect SizeToClass\n")
				goto dump
			}
			if sizeclass > 1 && class_to_size[sizeclass-1] >= n {
				print("size=", n, " sizeclass=", sizeclass, " runtime·class_to_size=", class_to_size[sizeclass], "\n")
				print("SizeToClass too big\n")
				goto dump
			}
		}
	}

	testdefersizes()

	// Copy out for statistics table.
	for i := 0; i < len(class_to_size); i++ {
		memstats.by_size[i].size = uint32(class_to_size[i])
	}
	return

dump:
	if true {
		print("NumSizeClasses=", _NumSizeClasses, "\n")
		print("runtime·class_to_size:")
		for sizeclass = 0; sizeclass < _NumSizeClasses; sizeclass++ {
			print(" ", class_to_size[sizeclass], "")
		}
		print("\n\n")
		print("size_to_class8:")
		for i := 0; i < len(size_to_class8); i++ {
			print(" ", i*8, "=>", size_to_class8[i], "(", class_to_size[size_to_class8[i]], ")\n")
		}
		print("\n")
		print("size_to_class128:")
		for i := 0; i < len(size_to_class128); i++ {
			print(" ", i*128, "=>", size_to_class128[i], "(", class_to_size[size_to_class128[i]], ")\n")
		}
		print("\n")
	}
	throw("InitSizes failed")
}

// Returns size of the memory block that mallocgc will allocate if you ask for the size.
func roundupsize(size uintptr) uintptr {
	if size < _MaxSmallSize {
		if size <= 1024-8 {
			return uintptr(class_to_size[size_to_class8[(size+7)>>3]])
		} else {
			return uintptr(class_to_size[size_to_class128[(size-1024+127)>>7]])
		}
	}
	if size+_PageSize < size {
		return size
	}
	return round(size, _PageSize)
}
