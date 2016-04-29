// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Malloc small size classes.
//
// See malloc.go for overview.
//
// The size classes are chosen so that rounding an allocation
// request up to the next size class wastes at most 12.5% (1.125x).
//
// Each size class has its own page count that gets allocated
// and chopped up when new objects of the size class are needed.
// That page count is chosen so that chopping up the run of
// pages into objects of the given size wastes at most 12.5% (1.125x)
// of the memory. It is not necessary that the cutoff here be
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

// Size classes. Computed and initialized by InitSizes.
//
// SizeToClass(0 <= n <= MaxSmallSize) returns the size class,
//	1 <= sizeclass < NumSizeClasses, for n.
//	Size class 0 is reserved to mean "not small".
//
// class_to_size[i] = largest size in class i
// class_to_allocnpages[i] = number of pages to allocate when
//	making new objects in class i

// The SizeToClass lookup is implemented using two arrays,
// one mapping sizes <= 1024 to their class and one mapping
// sizes >= 1024 and <= MaxSmallSize to their class.
// All objects are 8-aligned, so the first array is indexed by
// the size divided by 8 (rounded up).  Objects >= 1024 bytes
// are 128-aligned, so the second array is indexed by the
// size divided by 128 (rounded up).  The arrays are filled in
// by InitSizes.

var class_to_size [_NumSizeClasses]int32
var class_to_allocnpages [_NumSizeClasses]int32
var class_to_divmagic [_NumSizeClasses]divMagic

var size_to_class8 [1024/8 + 1]int8
var size_to_class128 [(_MaxSmallSize-1024)/128 + 1]int8

func sizeToClass(size int32) int32 {
	if size > _MaxSmallSize {
		throw("invalid size")
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
			throw("incorrect alignment")
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
		print("runtime: sizeclass=", sizeclass, " NumSizeClasses=", _NumSizeClasses, "\n")
		throw("bad NumSizeClasses")
	}
	// Check maxObjsPerSpan => number of objects invariant.
	for i, size := range class_to_size {
		if size != 0 && class_to_allocnpages[i]*pageSize/size > maxObjsPerSpan {
			throw("span contains too many objects")
		}
		if size == 0 && i != 0 {
			throw("size is 0 but class is not 0")
		}
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
				print("runtime: size=", n, " sizeclass=", sizeclass, " runtime·class_to_size=", class_to_size[sizeclass], "\n")
				print("incorrect SizeToClass\n")
				goto dump
			}
			if sizeclass > 1 && class_to_size[sizeclass-1] >= n {
				print("runtime: size=", n, " sizeclass=", sizeclass, " runtime·class_to_size=", class_to_size[sizeclass], "\n")
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

	for i := 1; i < len(class_to_size); i++ {
		class_to_divmagic[i] = computeDivMagic(uint32(class_to_size[i]))
	}

	return

dump:
	if true {
		print("runtime: NumSizeClasses=", _NumSizeClasses, "\n")
		print("runtime·class_to_size:")
		for sizeclass = 0; sizeclass < _NumSizeClasses; sizeclass++ {
			print(" ", class_to_size[sizeclass], "")
		}
		print("\n\n")
		print("runtime: size_to_class8:")
		for i := 0; i < len(size_to_class8); i++ {
			print(" ", i*8, "=>", size_to_class8[i], "(", class_to_size[size_to_class8[i]], ")\n")
		}
		print("\n")
		print("runtime: size_to_class128:")
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

// divMagic holds magic constants to implement division
// by a particular constant as a shift, multiply, and shift.
// That is, given
//	m = computeMagic(d)
// then
//	n/d == ((n>>m.shift) * m.mul) >> m.shift2
//
// The magic computation picks m such that
//	d = d₁*d₂
//	d₂= 2^m.shift
//	m.mul = ⌈2^m.shift2 / d₁⌉
//
// The magic computation here is tailored for malloc block sizes
// and does not handle arbitrary d correctly. Malloc block sizes d are
// always even, so the first shift implements the factors of 2 in d
// and then the mul and second shift implement the odd factor
// that remains. Because the first shift divides n by at least 2 (actually 8)
// before the multiply gets involved, the huge corner cases that
// require additional adjustment are impossible, so the usual
// fixup is not needed.
//
// For more details see Hacker's Delight, Chapter 10, and
// http://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
// http://ridiculousfish.com/blog/posts/labor-of-division-episode-iii.html
type divMagic struct {
	shift    uint8
	mul      uint32
	shift2   uint8
	baseMask uintptr
}

func computeDivMagic(d uint32) divMagic {
	var m divMagic

	// If the size is a power of two, heapBitsForObject can divide even faster by masking.
	// Compute this mask.
	if d&(d-1) == 0 {
		// It is a power of 2 (assuming dinptr != 1)
		m.baseMask = ^(uintptr(d) - 1)
	} else {
		m.baseMask = 0
	}

	// Compute pre-shift by factoring power of 2 out of d.
	for d&1 == 0 {
		m.shift++
		d >>= 1
	}

	// Compute largest k such that ⌈2^k / d⌉ fits in a 32-bit int.
	// This is always a good enough approximation.
	// We could use smaller k for some divisors but there's no point.
	k := uint8(63)
	d64 := uint64(d)
	for ((1<<k)+d64-1)/d64 >= 1<<32 {
		k--
	}
	m.mul = uint32(((1 << k) + d64 - 1) / d64) //  ⌈2^k / d⌉
	m.shift2 = k

	return m
}
