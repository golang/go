// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cases

import "golang.org/x/text/transform"

type caseFolder struct{ transform.NopResetter }

// caseFolder implements the Transformer interface for doing case folding.
func (t *caseFolder) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	c := context{dst: dst, src: src, atEOF: atEOF}
	for c.next() {
		foldFull(&c)
		c.checkpoint()
	}
	return c.ret()
}

func (t *caseFolder) Span(src []byte, atEOF bool) (n int, err error) {
	c := context{src: src, atEOF: atEOF}
	for c.next() && isFoldFull(&c) {
		c.checkpoint()
	}
	return c.retSpan()
}

func makeFold(o options) transform.SpanningTransformer {
	// TODO: Special case folding, through option Language, Special/Turkic, or
	// both.
	// TODO: Implement Compact options.
	return &caseFolder{}
}
