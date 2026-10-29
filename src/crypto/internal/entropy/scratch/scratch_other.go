// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !unix

package scratch

import "errors"

func mmap(n int) ([]byte, error) {
	return nil, errors.ErrUnsupported
}

func munmap(b []byte) {
	panic("unreachable")
}
