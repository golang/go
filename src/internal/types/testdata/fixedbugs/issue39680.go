// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Embedding stand-alone type parameters is not permitted for now. Disabled.

/*
import "fmt"

// Minimal test case.
func _[T interface{~T}](x T) T{
	return x
}

// Test case from issue.
type constr[T any] interface {
	~T
}

func Print[T constr[T]](s []T) {
	for _, v := range s {
		fmt.Print(v)
	}
}

func f() {
	Print([]string{"Hello, ", "playground\n"})
}
*/
