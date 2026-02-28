// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func Unique[T comparable](set []T) []T {
	nset := make([]T, 0, 8)

loop:
	for _, s := range set {
		for _, e := range nset {
			if s == e {
				continue loop
			}
		}

		nset = append(nset, s)
	}

	return nset
}
