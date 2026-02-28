// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func F[T any](c, d chan T) T {
	select {
	case x := <- c:
		return x
	case x := <- d:
		return x
	}
}

