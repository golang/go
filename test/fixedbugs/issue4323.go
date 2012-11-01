// compile

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4323: inlining of functions with local variables
// forgets to typecheck the declarations in the inlined copy.

package main

type reader struct {
	C chan T
}

type T struct{ C chan []byte }

var r = newReader()

func newReader() *reader { return new(reader) }

func (r *reader) Read(n int) ([]byte, error) {
	req := T{C: make(chan []byte)}
	r.C <- req
	return <-req.C, nil
}

func main() {
	s, err := r.Read(1)
	_, _ = s, err
}
