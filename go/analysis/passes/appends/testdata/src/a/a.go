// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the appends checker.

package a

func badAppendSlice1() {
	sli := []string{"a", "b", "c"}
	sli = append(sli) // want "append with no values"
}

func badAppendSlice2() {
	_ = append([]string{"a"}) // want "append with no values"
}

func goodAppendSlice1() {
	sli := []string{"a", "b", "c"}
	sli = append(sli, "d")
}

func goodAppendSlice2() {
	sli1 := []string{"a", "b", "c"}
	sli2 := []string{"d", "e", "f"}
	sli1 = append(sli1, sli2...)
}

func goodAppendSlice3() {
	sli := []string{"a", "b", "c"}
	sli = append(sli, "d", "e", "f")
}
