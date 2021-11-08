// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import (
	"unicode/utf8"

	"golang.org/x/text/transform"
)

// Reset implements the Reset method of the transform.Transformer interface.
func (Form) Reset() {}

// Transform implements the Transform method of the transform.Transformer
// interface. It may need to write segments of up to MaxSegmentSize at once.
// Users should either catch ErrShortDst and allow dst to grow or have dst be at
// least of size MaxTransformChunkSize to be guaranteed of progress.
func (f Form) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	// Cap the maximum number of src bytes to check.
	b := src
	eof := atEOF
	if ns := len(dst); ns < len(b) {
		err = transform.ErrShortDst
		eof = false
		b = b[:ns]
	}
	i, ok := formTable[f].quickSpan(inputBytes(b), 0, len(b), eof)
	n := copy(dst, b[:i])
	if !ok {
		nDst, nSrc, err = f.transform(dst[n:], src[n:], atEOF)
		return nDst + n, nSrc + n, err
	}

	if err == nil && n < len(src) && !atEOF {
		err = transform.ErrShortSrc
	}
	return n, n, err
}

func flushTransform(rb *reorderBuffer) bool {
	// Write out (must fully fit in dst, or else it is an ErrShortDst).
	if len(rb.out) < rb.nrune*utf8.UTFMax {
		return false
	}
	rb.out = rb.out[rb.flushCopy(rb.out):]
	return true
}

var errs = []error{nil, transform.ErrShortDst, transform.ErrShortSrc}

// transform implements the transform.Transformer interface. It is only called
// when quickSpan does not pass for a given string.
func (f Form) transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	// TODO: get rid of reorderBuffer. See CL 23460044.
	rb := reorderBuffer{}
	rb.init(f, src)
	for {
		// Load segment into reorder buffer.
		rb.setFlusher(dst[nDst:], flushTransform)
		end := decomposeSegment(&rb, nSrc, atEOF)
		if end < 0 {
			return nDst, nSrc, errs[-end]
		}
		nDst = len(dst) - len(rb.out)
		nSrc = end

		// Next quickSpan.
		end = rb.nsrc
		eof := atEOF
		if n := nSrc + len(dst) - nDst; n < end {
			err = transform.ErrShortDst
			end = n
			eof = false
		}
		end, ok := rb.f.quickSpan(rb.src, nSrc, end, eof)
		n := copy(dst[nDst:], rb.src.bytes[nSrc:end])
		nSrc += n
		nDst += n
		if ok {
			if err == nil && n < rb.nsrc && !atEOF {
				err = transform.ErrShortSrc
			}
			return nDst, nSrc, err
		}
	}
}
