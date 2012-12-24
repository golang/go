// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import (
	"fmt"
	"unicode/utf8"
)

const MaxSegmentSize = maxByteBufferSize

// An Iter iterates over a string or byte slice, while normalizing it
// to a given Form.
type Iter struct {
	rb     reorderBuffer
	buf    [maxByteBufferSize]byte
	info   Properties // first character saved from previous iteration
	next   iterFunc   // implementation of next depends on form
	asciiF iterFunc

	p        int    // current position in input source
	multiSeg []byte // remainder of multi-segment decomposition
}

type iterFunc func(*Iter) []byte

// Init initializes i to iterate over src after normalizing it to Form f.
func (i *Iter) Init(f Form, src []byte) {
	i.p = 0
	if len(src) == 0 {
		i.setDone()
		i.rb.nsrc = 0
		return
	}
	i.multiSeg = nil
	i.rb.init(f, src)
	i.next = i.rb.f.nextMain
	i.asciiF = nextASCIIBytes
	i.info = i.rb.f.info(i.rb.src, i.p)
}

// InitString initializes i to iterate over src after normalizing it to Form f.
func (i *Iter) InitString(f Form, src string) {
	i.p = 0
	if len(src) == 0 {
		i.setDone()
		i.rb.nsrc = 0
		return
	}
	i.multiSeg = nil
	i.rb.initString(f, src)
	i.next = i.rb.f.nextMain
	i.asciiF = nextASCIIString
	i.info = i.rb.f.info(i.rb.src, i.p)
}

// Seek sets the segment to be returned by the next call to Next to start
// at position p.  It is the responsibility of the caller to set p to the
// start of a UTF8 rune.
func (i *Iter) Seek(offset int64, whence int) (int64, error) {
	var abs int64
	switch whence {
	case 0:
		abs = offset
	case 1:
		abs = int64(i.p) + offset
	case 2:
		abs = int64(i.rb.nsrc) + offset
	default:
		return 0, fmt.Errorf("norm: invalid whence")
	}
	if abs < 0 {
		return 0, fmt.Errorf("norm: negative position")
	}
	if int(abs) >= i.rb.nsrc {
		i.setDone()
		return int64(i.p), nil
	}
	i.p = int(abs)
	i.multiSeg = nil
	i.next = i.rb.f.nextMain
	i.info = i.rb.f.info(i.rb.src, i.p)
	return abs, nil
}

// returnSlice returns a slice of the underlying input type as a byte slice.
// If the underlying is of type []byte, it will simply return a slice.
// If the underlying is of type string, it will copy the slice to the buffer
// and return that.
func (i *Iter) returnSlice(a, b int) []byte {
	if i.rb.src.bytes == nil {
		return i.buf[:copy(i.buf[:], i.rb.src.str[a:b])]
	}
	return i.rb.src.bytes[a:b]
}

// Pos returns the byte position at which the next call to Next will commence processing.
func (i *Iter) Pos() int {
	return i.p
}

func (i *Iter) setDone() {
	i.next = nextDone
	i.p = i.rb.nsrc
}

// Done returns true if there is no more input to process.
func (i *Iter) Done() bool {
	return i.p >= i.rb.nsrc
}

// Next returns f(i.input[i.Pos():n]), where n is a boundary of i.input.
// For any input a and b for which f(a) == f(b), subsequent calls
// to Next will return the same segments.
// Modifying runes are grouped together with the preceding starter, if such a starter exists.
// Although not guaranteed, n will typically be the smallest possible n.
func (i *Iter) Next() []byte {
	return i.next(i)
}

func nextASCIIBytes(i *Iter) []byte {
	p := i.p + 1
	if p >= i.rb.nsrc {
		i.setDone()
		return i.rb.src.bytes[i.p:p]
	}
	if i.rb.src.bytes[p] < utf8.RuneSelf {
		p0 := i.p
		i.p = p
		return i.rb.src.bytes[p0:p]
	}
	i.info = i.rb.f.info(i.rb.src, i.p)
	i.next = i.rb.f.nextMain
	return i.next(i)
}

func nextASCIIString(i *Iter) []byte {
	p := i.p + 1
	if p >= i.rb.nsrc {
		i.buf[0] = i.rb.src.str[i.p]
		i.setDone()
		return i.buf[:1]
	}
	if i.rb.src.str[p] < utf8.RuneSelf {
		i.buf[0] = i.rb.src.str[i.p]
		i.p = p
		return i.buf[:1]
	}
	i.info = i.rb.f.info(i.rb.src, i.p)
	i.next = i.rb.f.nextMain
	return i.next(i)
}

func nextHangul(i *Iter) []byte {
	if r := i.rb.src.hangul(i.p); r != 0 {
		i.p += hangulUTF8Size
		if i.p >= i.rb.nsrc {
			i.setDone()
		}
		return i.buf[:decomposeHangul(i.buf[:], r)]
	}
	i.info = i.rb.f.info(i.rb.src, i.p)
	i.next = i.rb.f.nextMain
	return i.next(i)
}

func nextDone(i *Iter) []byte {
	return nil
}

// nextMulti is used for iterating over multi-segment decompositions
// for decomposing normal forms.
func nextMulti(i *Iter) []byte {
	j := 0
	d := i.multiSeg
	// skip first rune
	for j = 1; j < len(d) && !utf8.RuneStart(d[j]); j++ {
	}
	for j < len(d) {
		info := i.rb.f.info(input{bytes: d}, j)
		if info.ccc == 0 {
			i.multiSeg = d[j:]
			return d[:j]
		}
		j += int(info.size)
	}
	// treat last segment as normal decomposition
	i.next = i.rb.f.nextMain
	return i.next(i)
}

// nextMultiNorm is used for iterating over multi-segment decompositions
// for composing normal forms.
func nextMultiNorm(i *Iter) []byte {
	j := 0
	d := i.multiSeg
	// skip first rune
	for j = 1; j < len(d) && !utf8.RuneStart(d[j]); j++ {
	}
	for j < len(d) {
		info := i.rb.f.info(input{bytes: d}, j)
		if info.ccc == 0 {
			i.multiSeg = d[j:]
			return d[:j]
		}
		j += int(info.size)
	}
	i.multiSeg = nil
	i.next = nextComposed
	i.p++ // restore old valud of i.p. See nextComposed.
	if i.p >= i.rb.nsrc {
		i.setDone()
	}
	return d
}

// nextDecomposed is the implementation of Next for forms NFD and NFKD.
func nextDecomposed(i *Iter) (next []byte) {
	startp, outp := i.p, 0
	inCopyStart, outCopyStart := i.p, 0
	for {
		if sz := int(i.info.size); sz <= 1 {
			p := i.p
			i.p++ // ASCII or illegal byte.  Either way, advance by 1.
			if i.p >= i.rb.nsrc {
				i.setDone()
				return i.returnSlice(p, i.p)
			} else if i.rb.src._byte(i.p) < utf8.RuneSelf {
				i.next = i.asciiF
				return i.returnSlice(p, i.p)
			}
			outp++
		} else if d := i.info.Decomposition(); d != nil {
			// Note: If leading CCC != 0, then len(d) == 2 and last is also non-zero.
			// Case 1: there is a leftover to copy.  In this case the decomposition
			// must begin with a modifier and should always be appended.
			// Case 2: no leftover. Simply return d if followed by a ccc == 0 value.
			p := outp + len(d)
			if outp > 0 {
				i.rb.src.copySlice(i.buf[outCopyStart:], inCopyStart, i.p)
				if p > len(i.buf) {
					return i.buf[:outp]
				}
			} else if i.info.multiSegment() {
				// outp must be 0 as multi-segment decompositions always
				// start a new segment.
				if i.multiSeg == nil {
					i.multiSeg = d
					i.next = nextMulti
					return nextMulti(i)
				}
				// We are in the last segment.  Treat as normal decomposition.
				d = i.multiSeg
				i.multiSeg = nil
				p = len(d)
			}
			prevCC := i.info.tccc
			if i.p += sz; i.p >= i.rb.nsrc {
				i.setDone()
				i.info = Properties{} // Force BoundaryBefore to succeed.
			} else {
				i.info = i.rb.f.info(i.rb.src, i.p)
			}
			if i.info.BoundaryBefore() {
				if outp > 0 {
					copy(i.buf[outp:], d)
					return i.buf[:p]
				}
				return d
			}
			copy(i.buf[outp:], d)
			outp = p
			inCopyStart, outCopyStart = i.p, outp
			if i.info.ccc < prevCC {
				goto doNorm
			}
			continue
		} else if r := i.rb.src.hangul(i.p); r != 0 {
			i.next = nextHangul
			i.p += hangulUTF8Size
			if i.p >= i.rb.nsrc {
				i.setDone()
			}
			return i.buf[:decomposeHangul(i.buf[:], r)]
		} else {
			p := outp + sz
			if p > len(i.buf) {
				break
			}
			outp = p
			i.p += sz
		}
		if i.p >= i.rb.nsrc {
			i.setDone()
			break
		}
		prevCC := i.info.tccc
		i.info = i.rb.f.info(i.rb.src, i.p)
		if i.info.BoundaryBefore() {
			break
		} else if i.info.ccc < prevCC {
			goto doNorm
		}
	}
	if outCopyStart == 0 {
		return i.returnSlice(inCopyStart, i.p)
	} else if inCopyStart < i.p {
		i.rb.src.copySlice(i.buf[outCopyStart:], inCopyStart, i.p)
	}
	return i.buf[:outp]
doNorm:
	// Insert what we have decomposed so far in the reorderBuffer.
	// As we will only reorder, there will always be enough room.
	i.rb.src.copySlice(i.buf[outCopyStart:], inCopyStart, i.p)
	if !i.rb.insertDecomposed(i.buf[0:outp]) {
		// Start over to prevent decompositions from crossing segment boundaries.
		// This is a rare occurrence.
		i.p = startp
		i.info = i.rb.f.info(i.rb.src, i.p)
	}
	for {
		if !i.rb.insert(i.rb.src, i.p, i.info) {
			break
		}
		if i.p += int(i.info.size); i.p >= i.rb.nsrc {
			i.setDone()
			break
		}
		i.info = i.rb.f.info(i.rb.src, i.p)
		if i.info.ccc == 0 {
			break
		}
	}
	// new segment or too many combining characters: exit normalization
	return i.buf[:i.rb.flushCopy(i.buf[:])]
}

// nextComposed is the implementation of Next for forms NFC and NFKC.
func nextComposed(i *Iter) []byte {
	outp, startp := 0, i.p
	var prevCC uint8
	for {
		if !i.info.isYesC() {
			goto doNorm
		}
		if cc := i.info.ccc; cc == 0 && outp > 0 {
			break
		} else if cc < prevCC {
			goto doNorm
		}
		prevCC = i.info.tccc
		sz := int(i.info.size)
		if sz == 0 {
			sz = 1 // illegal rune: copy byte-by-byte
		}
		p := outp + sz
		if p > len(i.buf) {
			break
		}
		outp = p
		i.p += sz
		if i.p >= i.rb.nsrc {
			i.setDone()
			break
		} else if i.rb.src._byte(i.p) < utf8.RuneSelf {
			i.next = i.asciiF
			break
		}
		i.info = i.rb.f.info(i.rb.src, i.p)
	}
	return i.returnSlice(startp, i.p)
doNorm:
	multi := false
	i.p = startp
	i.info = i.rb.f.info(i.rb.src, i.p)
	for {
		if !i.rb.insert(i.rb.src, i.p, i.info) {
			break
		}
		multi = multi || i.info.multiSegment()
		if i.p += int(i.info.size); i.p >= i.rb.nsrc {
			i.setDone()
			break
		}
		i.info = i.rb.f.info(i.rb.src, i.p)
		if i.info.BoundaryBefore() {
			break
		}
	}
	i.rb.compose()
	seg := i.buf[:i.rb.flushCopy(i.buf[:])]
	if multi {
		i.p-- // fake not being done yet
		i.multiSeg = seg
		i.next = nextMultiNorm
		return nextMultiNorm(i)
	}
	return seg
}
