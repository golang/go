// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Package struct {
	name string
}

type Future struct {
	result chan struct {
		*Package
		error
	}
}

func (t *Future) Result() (*Package, error) {
	result := <-t.result
	t.result <- result
	return result.Package, result.error
}
