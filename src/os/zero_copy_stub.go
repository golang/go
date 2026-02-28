// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !freebsd && !linux && !solaris

package os

import "io"

func (f *File) writeTo(w io.Writer) (written int64, handled bool, err error) {
	return 0, false, nil
}

func (f *File) readFrom(r io.Reader) (n int64, handled bool, err error) {
	return 0, false, nil
}
