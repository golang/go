// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 48916: expand_calls pass crashes due to a (dead)
// use of an OpInvalid value.

package p

type T struct {
	num int64
}

func foo(vs map[T]struct{}, d T) error {
	_, ok := vs[d]
	if !ok {
		return nil
	}

	switch d.num {
	case 0:
	case 1:
	case 2:
	case 3:
	case 4:
	case 5:
	case 6:
		var num float64
		if num != 0 {
			return nil
		}
	}

	return nil
}
