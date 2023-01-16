// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func Real[P ~complex128](x P) {
	_ = real(x /* ERROR not supported */ )
}

func Imag[P ~complex128](x P) {
	_ = imag(x /* ERROR not supported */ )
}

func Complex[P ~float64](x P) {
	_ = complex(x /* ERROR not supported */ , 0)
	_ = complex(0 /* ERROR not supported */ , x)
	_ = complex(x /* ERROR not supported */ , x)
}
