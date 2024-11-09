// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// Generate tables for small malloc size classes.
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
// For really small sizes, alignment constraints force the
// overhead higher.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io"
	"log"
	"math"
	"math/bits"
	"os"
)

// Generate msize.go

var stdout = flag.Bool("stdout", false, "write to stdout instead of sizeclasses.go")

func main() {
	flag.Parse()

	var b bytes.Buffer
	fmt.Fprintln(&b, "// Code generated by mksizeclasses.go; DO NOT EDIT.")
	fmt.Fprintln(&b, "//go:generate go run mksizeclasses.go")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "package runtime")
	classes := makeClasses()

	printComment(&b, classes)

	printClasses(&b, classes)

	out, err := format.Source(b.Bytes())
	if err != nil {
		log.Fatal(err)
	}
	if *stdout {
		_, err = os.Stdout.Write(out)
	} else {
		err = os.WriteFile("sizeclasses.go", out, 0666)
	}
	if err != nil {
		log.Fatal(err)
	}
}

const (
	// Constants that we use and will transfer to the runtime.
	minHeapAlign = 8
	maxSmallSize = 32 << 10
	smallSizeDiv = 8
	smallSizeMax = 1024
	largeSizeDiv = 128
	pageShift    = 13

	// Derived constants.
	pageSize = 1 << pageShift
)

type class struct {
	size   int // max size
	npages int // number of pages
}

func powerOfTwo(x int) bool {
	return x != 0 && x&(x-1) == 0
}

func makeClasses() []class {
	var classes []class

	classes = append(classes, class{}) // class #0 is a dummy entry

	align := minHeapAlign
	for size := align; size <= maxSmallSize; size += align {
		if powerOfTwo(size) { // bump alignment once in a while
			if size >= 2048 {
				align = 256
			} else if size >= 128 {
				align = size / 8
			} else if size >= 32 {
				align = 16 // heap bitmaps assume 16 byte alignment for allocations >= 32 bytes.
			}
		}
		if !powerOfTwo(align) {
			panic("incorrect alignment")
		}

		// Make the allocnpages big enough that
		// the leftover is less than 1/8 of the total,
		// so wasted space is at most 12.5%.
		allocsize := pageSize
		for allocsize%size > allocsize/8 {
			allocsize += pageSize
		}
		npages := allocsize / pageSize

		// If the previous sizeclass chose the same
		// allocation size and fit the same number of
		// objects into the page, we might as well
		// use just this size instead of having two
		// different sizes.
		if len(classes) > 1 && npages == classes[len(classes)-1].npages && allocsize/size == allocsize/classes[len(classes)-1].size {
			classes[len(classes)-1].size = size
			continue
		}
		classes = append(classes, class{size: size, npages: npages})
	}

	// Increase object sizes if we can fit the same number of larger objects
	// into the same number of pages. For example, we choose size 8448 above
	// with 6 objects in 7 pages. But we can well use object size 9472,
	// which is also 6 objects in 7 pages but +1024 bytes (+12.12%).
	// We need to preserve at least largeSizeDiv alignment otherwise
	// sizeToClass won't work.
	for i := range classes {
		if i == 0 {
			continue
		}
		c := &classes[i]
		psize := c.npages * pageSize
		new_size := (psize / (psize / c.size)) &^ (largeSizeDiv - 1)
		if new_size > c.size {
			c.size = new_size
		}
	}

	if len(classes) != 68 {
		panic("number of size classes has changed")
	}

	for i := range classes {
		computeDivMagic(&classes[i])
	}

	return classes
}

// computeDivMagic checks that the division required to compute object
// index from span offset can be computed using 32-bit multiplication.
// n / c.size is implemented as (n * (^uint32(0)/uint32(c.size) + 1)) >> 32
// for all 0 <= n <= c.npages * pageSize
func computeDivMagic(c *class) {
	// divisor
	d := c.size
	if d == 0 {
		return
	}

	// maximum input value for which the formula needs to work.
	max := c.npages * pageSize

	// As reported in [1], if n and d are unsigned N-bit integers, we
	// can compute n / d as ⌊n * c / 2^F⌋, where c is ⌈2^F / d⌉ and F is
	// computed with:
	//
	// 	Algorithm 2: Algorithm to select the number of fractional bits
	// 	and the scaled approximate reciprocal in the case of unsigned
	// 	integers.
	//
	// 	if d is a power of two then
	// 		Let F ← log₂(d) and c = 1.
	// 	else
	// 		Let F ← N + L where L is the smallest integer
	// 		such that d ≤ (2^(N+L) mod d) + 2^L.
	// 	end if
	//
	// [1] "Faster Remainder by Direct Computation: Applications to
	// Compilers and Software Libraries" Daniel Lemire, Owen Kaser,
	// Nathan Kurz arXiv:1902.01961
	//
	// To minimize the risk of introducing errors, we implement the
	// algorithm exactly as stated, rather than trying to adapt it to
	// fit typical Go idioms.
	N := bits.Len(uint(max))
	var F int
	if powerOfTwo(d) {
		F = int(math.Log2(float64(d)))
		if d != 1<<F {
			panic("imprecise log2")
		}
	} else {
		for L := 0; ; L++ {
			if d <= ((1<<(N+L))%d)+(1<<L) {
				F = N + L
				break
			}
		}
	}

	// Also, noted in the paper, F is the smallest number of fractional
	// bits required. We use 32 bits, because it works for all size
	// classes and is fast on all CPU architectures that we support.
	if F > 32 {
		fmt.Printf("d=%d max=%d N=%d F=%d\n", c.size, max, N, F)
		panic("size class requires more than 32 bits of precision")
	}

	// Brute force double-check with the exact computation that will be
	// done by the runtime.
	m := ^uint32(0)/uint32(c.size) + 1
	for n := 0; n <= max; n++ {
		if uint32((uint64(n)*uint64(m))>>32) != uint32(n/c.size) {
			fmt.Printf("d=%d max=%d m=%d n=%d\n", d, max, m, n)
			panic("bad 32-bit multiply magic")
		}
	}
}

func printComment(w io.Writer, classes []class) {
	fmt.Fprintf(w, "// %-5s  %-9s  %-10s  %-7s  %-10s  %-9s  %-9s\n", "class", "bytes/obj", "bytes/span", "objects", "tail waste", "max waste", "min align")
	prevSize := 0
	var minAligns [pageShift + 1]int
	for i, c := range classes {
		if i == 0 {
			continue
		}
		spanSize := c.npages * pageSize
		objects := spanSize / c.size
		tailWaste := spanSize - c.size*(spanSize/c.size)
		maxWaste := float64((c.size-prevSize-1)*objects+tailWaste) / float64(spanSize)
		alignBits := bits.TrailingZeros(uint(c.size))
		if alignBits > pageShift {
			// object alignment is capped at page alignment
			alignBits = pageShift
		}
		for i := range minAligns {
			if i > alignBits {
				minAligns[i] = 0
			} else if minAligns[i] == 0 {
				minAligns[i] = c.size
			}
		}
		prevSize = c.size
		fmt.Fprintf(w, "// %5d  %9d  %10d  %7d  %10d  %8.2f%%  %9d\n", i, c.size, spanSize, objects, tailWaste, 100*maxWaste, 1<<alignBits)
	}
	fmt.Fprintf(w, "\n")

	fmt.Fprintf(w, "// %-9s  %-4s  %-12s\n", "alignment", "bits", "min obj size")
	for bits, size := range minAligns {
		if size == 0 {
			break
		}
		if bits+1 < len(minAligns) && size == minAligns[bits+1] {
			continue
		}
		fmt.Fprintf(w, "// %9d  %4d  %12d\n", 1<<bits, bits, size)
	}
	fmt.Fprintf(w, "\n")
}

func printClasses(w io.Writer, classes []class) {
	fmt.Fprintln(w, "const (")
	fmt.Fprintf(w, "minHeapAlign = %d\n", minHeapAlign)
	fmt.Fprintf(w, "_MaxSmallSize = %d\n", maxSmallSize)
	fmt.Fprintf(w, "smallSizeDiv = %d\n", smallSizeDiv)
	fmt.Fprintf(w, "smallSizeMax = %d\n", smallSizeMax)
	fmt.Fprintf(w, "largeSizeDiv = %d\n", largeSizeDiv)
	fmt.Fprintf(w, "_NumSizeClasses = %d\n", len(classes))
	fmt.Fprintf(w, "_PageShift = %d\n", pageShift)
	fmt.Fprintln(w, ")")

	fmt.Fprint(w, "var class_to_size = [_NumSizeClasses]uint16 {")
	for _, c := range classes {
		fmt.Fprintf(w, "%d,", c.size)
	}
	fmt.Fprintln(w, "}")

	fmt.Fprint(w, "var class_to_allocnpages = [_NumSizeClasses]uint8 {")
	for _, c := range classes {
		fmt.Fprintf(w, "%d,", c.npages)
	}
	fmt.Fprintln(w, "}")

	fmt.Fprint(w, "var class_to_divmagic = [_NumSizeClasses]uint32 {")
	for _, c := range classes {
		if c.size == 0 {
			fmt.Fprintf(w, "0,")
			continue
		}
		fmt.Fprintf(w, "^uint32(0)/%d+1,", c.size)
	}
	fmt.Fprintln(w, "}")

	// map from size to size class, for small sizes.
	sc := make([]int, smallSizeMax/smallSizeDiv+1)
	for i := range sc {
		size := i * smallSizeDiv
		for j, c := range classes {
			if c.size >= size {
				sc[i] = j
				break
			}
		}
	}
	fmt.Fprint(w, "var size_to_class8 = [smallSizeMax/smallSizeDiv+1]uint8 {")
	for _, v := range sc {
		fmt.Fprintf(w, "%d,", v)
	}
	fmt.Fprintln(w, "}")

	// map from size to size class, for large sizes.
	sc = make([]int, (maxSmallSize-smallSizeMax)/largeSizeDiv+1)
	for i := range sc {
		size := smallSizeMax + i*largeSizeDiv
		for j, c := range classes {
			if c.size >= size {
				sc[i] = j
				break
			}
		}
	}
	fmt.Fprint(w, "var size_to_class128 = [(_MaxSmallSize-smallSizeMax)/largeSizeDiv+1]uint8 {")
	for _, v := range sc {
		fmt.Fprintf(w, "%d,", v)
	}
	fmt.Fprintln(w, "}")
}
