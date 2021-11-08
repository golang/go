// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	ff := []func(){lt_f1, lt_f2, lt_f3, lt_f4, lt_f5, lt_f6, lt_f7, lt_f8, lt_f9,
		gt_f1, gt_f2, gt_f3, le_f1, le_f2, le_f3, ge_f1, ge_f2, ge_f3}

	for _, f := range ff {
		f()
	}
}

func lt_f1() {
	const c = 1
	var a = 0
	var v *int = &a
	if *v-c < len([]int{}) {
	} else {
		panic("bad")
	}
}

func lt_f2() {
	const c = 10
	var a = 0
	var v *int = &a
	if *v+c < len([]int{}) {
		panic("bad")
	}
}

func lt_f3() {
	const c = -10
	var a = 0
	var v *int = &a
	if *v|0xff+c < len([]int{}) {
		panic("bad")
	}
}

func lt_f4() {
	const c = 10
	var a = 0
	var v *int = &a
	if *v|0x0f+c < len([]int{}) {
		panic("bad")
	}
}

func lt_f5() {
	const c int32 = 1
	var a int32 = 0
	var v *int32 = &a
	if *v-c < int32(len([]int32{})) {
	} else {
		panic("bad")
	}
}

func lt_f6() {
	const c int32 = 10
	var a int32 = 0
	var v *int32 = &a
	if *v+c < int32(len([]int32{})) {
		panic("bad")
	}
}

func lt_f7() {
	const c int32 = -10
	var a int32 = 0
	var v *int32 = &a
	if *v|0xff+c < int32(len([]int{})) {
		panic("bad")
	}
}

func lt_f8() {
	const c int32 = 10
	var a int32 = 0
	var v *int32 = &a
	if *v|0x0f+c < int32(len([]int{})) {
		panic("bad")
	}
}

func lt_f9() {
	const c int32 = -10
	var a int32 = 0
	var v *int32 = &a
	if *v|0x0a+c < int32(len([]int{})) {
		panic("bad")
	}
}

func gt_f1() {
	const c = 1
	var a = 0
	var v *int = &a
	if len([]int{}) > *v-c {
	} else {
		panic("bad")
	}
}

func gt_f2() {
	const c = 10
	var a = 0
	var v *int = &a
	if len([]int{}) > *v|0x0f+c {
		panic("bad")
	}
}

func gt_f3() {
	const c int32 = 10
	var a int32 = 0
	var v *int32 = &a
	if int32(len([]int{})) > *v|0x0f+c {
		panic("bad")
	}
}

func le_f1() {
	const c = -10
	var a = 0
	var v *int = &a
	if *v|0xff+c <= len([]int{}) {
		panic("bad")
	}
}

func le_f2() {
	const c = 0xf
	var a = 0
	var v *int = &a
	if *v|0xf-c <= len([]int{}) {
	} else {
		panic("bad")
	}
}

func le_f3() {
	const c int32 = -10
	var a int32 = 0
	var v *int32 = &a
	if *v|0xff+c <= int32(len([]int{})) {
		panic("bad")
	}
}

func ge_f1() {
	const c = -10
	var a = 0
	var v *int = &a
	if len([]int{}) >= *v|0xff+c {
		panic("bad")
	}
}

func ge_f2() {
	const c int32 = 10
	var a int32 = 0
	var v *int32 = &a
	if int32(len([]int{})) >= *v|0x0f+c {
		panic("bad")
	}
}

func ge_f3() {
	const c = -10
	var a = 0
	var v *int = &a
	if len([]int{}) >= *v|0x0a+c {
	} else {
		panic("bad")
	}
}
