// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "io"

type Reader struct {
	buf []byte
}
type Token *[16]byte

func Read[T interface{ ~*[16]byte }](r *Reader) (t T, err error) {
	if n := len(t); len(r.buf) >= n {
		t = T(r.buf[:n])
		r.buf = r.buf[n:]
		return
	}
	err = io.EOF
	return
}

func main() {
	r := &Reader{buf: []byte("0123456789abcdef")}
	token, err := Read[Token](r)
	_, _ = token, err
}
