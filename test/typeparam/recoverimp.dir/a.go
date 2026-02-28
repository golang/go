// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "fmt"

func F[T any](a T) {
	defer func() {
		if x := recover(); x != nil {
			fmt.Printf("panic: %v\n", x)
		}
	}()
	panic(a)
}
