// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package width

import (
	"unicode/utf8"

	"golang_org/x/text/transform"
)

type foldTransform struct {
	transform.NopResetter
}

func (foldTransform) Span(src []byte, atEOF bool) (n int, err error) {
	for n < len(src) {
		if src[n] < utf8.RuneSelf {
			// ASCII fast path.
			for n++; n < len(src) && src[n] < utf8.RuneSelf; n++ {
			}
			continue
		}
		v, size := trie.lookup(src[n:])
		if size == 0 { // incomplete UTF-8 encoding
			if !atEOF {
				err = transform.ErrShortSrc
			} else {
				n = len(src)
			}
			break
		}
		if elem(v)&tagNeedsFold != 0 {
			err = transform.ErrEndOfSpan
			break
		}
		n += size
	}
	return n, err
}

func (foldTransform) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	for nSrc < len(src) {
		if src[nSrc] < utf8.RuneSelf {
			// ASCII fast path.
			start, end := nSrc, len(src)
			if d := len(dst) - nDst; d < end-start {
				end = nSrc + d
			}
			for nSrc++; nSrc < end && src[nSrc] < utf8.RuneSelf; nSrc++ {
			}
			n := copy(dst[nDst:], src[start:nSrc])
			if nDst += n; nDst == len(dst) {
				nSrc = start + n
				if nSrc == len(src) {
					return nDst, nSrc, nil
				}
				if src[nSrc] < utf8.RuneSelf {
					return nDst, nSrc, transform.ErrShortDst
				}
			}
			continue
		}
		v, size := trie.lookup(src[nSrc:])
		if size == 0 { // incomplete UTF-8 encoding
			if !atEOF {
				return nDst, nSrc, transform.ErrShortSrc
			}
			size = 1 // gobble 1 byte
		}
		if elem(v)&tagNeedsFold == 0 {
			if size != copy(dst[nDst:], src[nSrc:nSrc+size]) {
				return nDst, nSrc, transform.ErrShortDst
			}
			nDst += size
		} else {
			data := inverseData[byte(v)]
			if len(dst)-nDst < int(data[0]) {
				return nDst, nSrc, transform.ErrShortDst
			}
			i := 1
			for end := int(data[0]); i < end; i++ {
				dst[nDst] = data[i]
				nDst++
			}
			dst[nDst] = data[i] ^ src[nSrc+size-1]
			nDst++
		}
		nSrc += size
	}
	return nDst, nSrc, nil
}

type narrowTransform struct {
	transform.NopResetter
}

func (narrowTransform) Span(src []byte, atEOF bool) (n int, err error) {
	for n < len(src) {
		if src[n] < utf8.RuneSelf {
			// ASCII fast path.
			for n++; n < len(src) && src[n] < utf8.RuneSelf; n++ {
			}
			continue
		}
		v, size := trie.lookup(src[n:])
		if size == 0 { // incomplete UTF-8 encoding
			if !atEOF {
				err = transform.ErrShortSrc
			} else {
				n = len(src)
			}
			break
		}
		if k := elem(v).kind(); byte(v) == 0 || k != EastAsianFullwidth && k != EastAsianWide && k != EastAsianAmbiguous {
		} else {
			err = transform.ErrEndOfSpan
			break
		}
		n += size
	}
	return n, err
}

func (narrowTransform) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	for nSrc < len(src) {
		if src[nSrc] < utf8.RuneSelf {
			// ASCII fast path.
			start, end := nSrc, len(src)
			if d := len(dst) - nDst; d < end-start {
				end = nSrc + d
			}
			for nSrc++; nSrc < end && src[nSrc] < utf8.RuneSelf; nSrc++ {
			}
			n := copy(dst[nDst:], src[start:nSrc])
			if nDst += n; nDst == len(dst) {
				nSrc = start + n
				if nSrc == len(src) {
					return nDst, nSrc, nil
				}
				if src[nSrc] < utf8.RuneSelf {
					return nDst, nSrc, transform.ErrShortDst
				}
			}
			continue
		}
		v, size := trie.lookup(src[nSrc:])
		if size == 0 { // incomplete UTF-8 encoding
			if !atEOF {
				return nDst, nSrc, transform.ErrShortSrc
			}
			size = 1 // gobble 1 byte
		}
		if k := elem(v).kind(); byte(v) == 0 || k != EastAsianFullwidth && k != EastAsianWide && k != EastAsianAmbiguous {
			if size != copy(dst[nDst:], src[nSrc:nSrc+size]) {
				return nDst, nSrc, transform.ErrShortDst
			}
			nDst += size
		} else {
			data := inverseData[byte(v)]
			if len(dst)-nDst < int(data[0]) {
				return nDst, nSrc, transform.ErrShortDst
			}
			i := 1
			for end := int(data[0]); i < end; i++ {
				dst[nDst] = data[i]
				nDst++
			}
			dst[nDst] = data[i] ^ src[nSrc+size-1]
			nDst++
		}
		nSrc += size
	}
	return nDst, nSrc, nil
}

type wideTransform struct {
	transform.NopResetter
}

func (wideTransform) Span(src []byte, atEOF bool) (n int, err error) {
	for n < len(src) {
		// TODO: Consider ASCII fast path. Special-casing ASCII handling can
		// reduce the ns/op of BenchmarkWideASCII by about 30%. This is probably
		// not enough to warrant the extra code and complexity.
		v, size := trie.lookup(src[n:])
		if size == 0 { // incomplete UTF-8 encoding
			if !atEOF {
				err = transform.ErrShortSrc
			} else {
				n = len(src)
			}
			break
		}
		if k := elem(v).kind(); byte(v) == 0 || k != EastAsianHalfwidth && k != EastAsianNarrow {
		} else {
			err = transform.ErrEndOfSpan
			break
		}
		n += size
	}
	return n, err
}

func (wideTransform) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	for nSrc < len(src) {
		// TODO: Consider ASCII fast path. Special-casing ASCII handling can
		// reduce the ns/op of BenchmarkWideASCII by about 30%. This is probably
		// not enough to warrant the extra code and complexity.
		v, size := trie.lookup(src[nSrc:])
		if size == 0 { // incomplete UTF-8 encoding
			if !atEOF {
				return nDst, nSrc, transform.ErrShortSrc
			}
			size = 1 // gobble 1 byte
		}
		if k := elem(v).kind(); byte(v) == 0 || k != EastAsianHalfwidth && k != EastAsianNarrow {
			if size != copy(dst[nDst:], src[nSrc:nSrc+size]) {
				return nDst, nSrc, transform.ErrShortDst
			}
			nDst += size
		} else {
			data := inverseData[byte(v)]
			if len(dst)-nDst < int(data[0]) {
				return nDst, nSrc, transform.ErrShortDst
			}
			i := 1
			for end := int(data[0]); i < end; i++ {
				dst[nDst] = data[i]
				nDst++
			}
			dst[nDst] = data[i] ^ src[nSrc+size-1]
			nDst++
		}
		nSrc += size
	}
	return nDst, nSrc, nil
}
