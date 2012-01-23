// $G $D/$F.go || echo "Bug398"

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2582
package foo
    
type T struct {}
func (T) cplx() complex128 {
	for false {}  // avoid inlining
	return complex(1,0)
}

type I interface {
	cplx() complex128
}

func f(e float32, t T) {

    	_ = real(t.cplx())
    	_ = imag(t.cplx())

	var i I
	i = t
    	_ = real(i.cplx())
    	_ = imag(i.cplx())
}