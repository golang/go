// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux

package os

import "io"

func (f *File) readFrom(r io.Reader) (n int64, handled bool, err error) {
	return 0, false, nil
}
