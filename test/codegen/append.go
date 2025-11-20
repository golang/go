// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func Append1(n int) []int {
	var r []int
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	// amd64:`.*moveSliceNoCapNoScan`
	return r
}

func Append2(n int) (r []int) {
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	// amd64:`.*moveSliceNoCapNoScan`
	return
}

func Append3(n int) (r []int) {
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	// amd64:`.*moveSliceNoCapNoScan`
	return r
}

func Append4(n int) []int {
	var r []int
	for i := range n {
		// amd64:`.*growsliceBuf`
		r = append(r, i)
	}
	println(cap(r))
	// amd64:`.*moveSliceNoScan`
	return r
}

func Append5(n int) []int {
	var r []int
	for i := range n {
		// amd64:`.*growsliceBuf`
		r = append(r, i)
	}
	useSlice(r)
	// amd64:`.*moveSliceNoScan`
	return r
}

func Append6(n int) []*int {
	var r []*int
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, new(i))
	}
	// amd64:`.*moveSliceNoCap`
	return r
}

func Append7(n int) []*int {
	var r []*int
	for i := range n {
		// amd64:`.*growsliceBuf`
		r = append(r, new(i))
	}
	println(cap(r))
	// amd64:`.*moveSlice`
	return r
}

func Append8(n int, p *[]int) {
	var r []int
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	// amd64:`.*moveSliceNoCapNoScan`
	*p = r
}

func Append9(n int) []int {
	var r []int
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	println(len(r))
	// amd64:`.*moveSliceNoCapNoScan`
	return r
}

func Append10(n int) []int {
	var r []int
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	println(r[3])
	// amd64:`.*moveSliceNoCapNoScan`
	return r
}

func Append11(n int) []int {
	var r []int
	for i := range n {
		// amd64:`.*growsliceBuf`
		r = append(r, i)
	}
	r = r[3:5]
	// amd64:`.*moveSliceNoScan`
	return r
}

func Append12(n int) []int {
	var r []int
	r = nil
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	// amd64:`.*moveSliceNoCapNoScan`
	return r
}

func Append13(n int) []int {
	var r []int
	r, r = nil, nil
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	// amd64:`.*moveSliceNoCapNoScan`
	return r
}

func Append14(n int) []int {
	var r []int
	r = []int{3, 4, 5}
	for i := range n {
		// amd64:`.*growsliceBuf`
		r = append(r, i)
	}
	// amd64:`.*moveSliceNoScan`
	return r
}

func Append15(n int) []int {
	r := []int{3, 4, 5}
	for i := range n {
		// amd64:`.*growsliceBuf`
		r = append(r, i)
	}
	// amd64:`.*moveSliceNoScan`
	return r
}

func Append16(r []int, n int) []int {
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	// amd64:`.*moveSliceNoCapNoScan`
	return r
}

func Append17(n int) []int {
	var r []int
	for i := range n {
		// amd64:`.*growslice`
		r = append(r, i)
	}
	for i, x := range r {
		println(i, x)
	}
	// amd64:`.*moveSliceNoCapNoScan`
	return r
}

func Append18(n int, p *[]int) {
	var r []int
	for i := range n {
		// amd64:-`.*moveSliceNoCapNoScan`
		*p = r
		// amd64:`.*growslice`
		r = append(r, i)
	}
}

func Append19(n int, p [][]int) {
	for j := range p {
		var r []int
		for i := range n {
			// amd64:`.*growslice`
			r = append(r, i)
		}
		// amd64:`.*moveSliceNoCapNoScan`
		p[j] = r
	}
}

func Append20(n int, p [][]int) {
	for j := range p {
		var r []int
		// amd64:`.*growslice`
		r = append(r, 0)
		// amd64:-`.*moveSliceNoCapNoScan`
		p[j] = r
	}
}

//go:noinline
func useSlice(s []int) {
}
