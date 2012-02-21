// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

const MaxSegmentSize = maxByteBufferSize

// An Iter iterates over a string or byte slice, while normalizing it
// to a given Form.
type Iter struct {
	rb   reorderBuffer
	info runeInfo // first character saved from previous iteration
	next iterFunc // implementation of next depends on form

	p        int // current position in input source
	outStart int // start of current segment in output buffer
	inStart  int // start of current segment in input source
	maxp     int // position in output buffer after which not to start a new segment
	maxseg   int // for tracking an excess of combining characters

	tccc uint8
	done bool
}

type iterFunc func(*Iter, []byte) int

// SetInput initializes i to iterate over src after normalizing it to Form f.
func (i *Iter) SetInput(f Form, src []byte) {
	i.rb.init(f, src)
	if i.rb.f.composing {
		i.next = nextComposed
	} else {
		i.next = nextDecomposed
	}
	i.p = 0
	if i.done = len(src) == 0; !i.done {
		i.info = i.rb.f.info(i.rb.src, i.p)
	}
}

// SetInputString initializes i to iterate over src after normalizing it to Form f.
func (i *Iter) SetInputString(f Form, src string) {
	i.rb.initString(f, src)
	if i.rb.f.composing {
		i.next = nextComposed
	} else {
		i.next = nextDecomposed
	}
	i.p = 0
	if i.done = len(src) == 0; !i.done {
		i.info = i.rb.f.info(i.rb.src, i.p)
	}
}

// Pos returns the byte position at which the next call to Next will commence processing.
func (i *Iter) Pos() int {
	return i.p
}

// Done returns true if there is no more input to process.
func (i *Iter) Done() bool {
	return i.done
}

// Next writes f(i.input[i.Pos():n]...) to buffer buf, where n is the
// largest boundary of i.input such that the result fits in buf.  
// It returns the number of bytes written to buf.
// len(buf) should be at least MaxSegmentSize. 
// Done must be false before calling Next.
func (i *Iter) Next(buf []byte) int {
	return i.next(i, buf)
}

func (i *Iter) initNext(outn, inStart int) {
	i.outStart = 0
	i.inStart = inStart
	i.maxp = outn - MaxSegmentSize
	i.maxseg = MaxSegmentSize
}

// setStart resets the start of the new segment to the given position.
// It returns true if there is not enough room for the new segment.
func (i *Iter) setStart(outp, inp int) bool {
	if outp > i.maxp {
		return true
	}
	i.outStart = outp
	i.inStart = inp
	i.maxseg = outp + MaxSegmentSize
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// nextDecomposed is the implementation of Next for forms NFD and NFKD.
func nextDecomposed(i *Iter, out []byte) int {
	var outp int
	i.initNext(len(out), i.p)
doFast:
	inCopyStart, outCopyStart := i.p, outp // invariant xCopyStart <= i.xStart
	for {
		if sz := int(i.info.size); sz <= 1 {
			// ASCII or illegal byte.  Either way, advance by 1.
			i.p++
			outp++
			max := min(i.rb.nsrc, len(out)-outp+i.p)
			if np := i.rb.src.skipASCII(i.p, max); np > i.p {
				outp += np - i.p
				i.p = np
				if i.p >= i.rb.nsrc {
					break
				}
				// ASCII may combine with consecutive runes.
				if i.setStart(outp-1, i.p-1) {
					i.p--
					outp--
					i.info.size = 1
					break
				}
			}
		} else if d := i.info.decomposition(); d != nil {
			i.rb.src.copySlice(out[outCopyStart:], inCopyStart, i.p)
			p := outp + len(d)
			if p > i.maxseg && i.setStart(outp, i.p) {
				return outp
			}
			copy(out[outp:], d)
			outp = p
			i.p += sz
			inCopyStart, outCopyStart = i.p, outp
		} else if r := i.rb.src.hangul(i.p); r != 0 {
			i.rb.src.copySlice(out[outCopyStart:], inCopyStart, i.p)
			for {
				outp += decomposeHangul(out[outp:], r)
				i.p += hangulUTF8Size
				if r = i.rb.src.hangul(i.p); r == 0 {
					break
				}
				if i.setStart(outp, i.p) {
					return outp
				}
			}
			inCopyStart, outCopyStart = i.p, outp
		} else {
			p := outp + sz
			if p > i.maxseg && i.setStart(outp, i.p) {
				break
			}
			outp = p
			i.p += sz
		}
		if i.p >= i.rb.nsrc {
			break
		}
		prevCC := i.info.tccc
		i.info = i.rb.f.info(i.rb.src, i.p)
		if cc := i.info.ccc; cc == 0 {
			if i.setStart(outp, i.p) {
				break
			}
		} else if cc < prevCC {
			goto doNorm
		}
	}
	if inCopyStart != i.p {
		i.rb.src.copySlice(out[outCopyStart:], inCopyStart, i.p)
	}
	i.done = i.p >= i.rb.nsrc
	return outp
doNorm:
	// Insert what we have decomposed so far in the reorderBuffer.
	// As we will only reorder, there will always be enough room.
	i.rb.src.copySlice(out[outCopyStart:], inCopyStart, i.p)
	if !i.rb.insertDecomposed(out[i.outStart:outp]) {
		// Start over to prevent decompositions from crossing segment boundaries.
		// This is a rare occurance.
		i.p = i.inStart
		i.info = i.rb.f.info(i.rb.src, i.p)
	}
	outp = i.outStart
	for {
		if !i.rb.insert(i.rb.src, i.p, i.info) {
			break
		}
		if i.p += int(i.info.size); i.p >= i.rb.nsrc {
			outp += i.rb.flushCopy(out[outp:])
			i.done = true
			return outp
		}
		i.info = i.rb.f.info(i.rb.src, i.p)
		if i.info.ccc == 0 {
			break
		}
	}
	// new segment or too many combining characters: exit normalization
	if outp += i.rb.flushCopy(out[outp:]); i.setStart(outp, i.p) {
		return outp
	}
	goto doFast
}

// nextComposed is the implementation of Next for forms NFC and NFKC.
func nextComposed(i *Iter, out []byte) int {
	var outp int
	i.initNext(len(out), i.p)
doFast:
	inCopyStart, outCopyStart := i.p, outp // invariant xCopyStart <= i.xStart
	var prevCC uint8
	for {
		if !i.info.isYesC() {
			goto doNorm
		}
		if cc := i.info.ccc; cc == 0 {
			if i.setStart(outp, i.p) {
				break
			}
		} else if cc < prevCC {
			goto doNorm
		}
		prevCC = i.info.tccc
		sz := int(i.info.size)
		if sz == 0 {
			sz = 1 // illegal rune: copy byte-by-byte
		}
		p := outp + sz
		if p > i.maxseg && i.setStart(outp, i.p) {
			break
		}
		outp = p
		i.p += sz
		max := min(i.rb.nsrc, len(out)-outp+i.p)
		if np := i.rb.src.skipASCII(i.p, max); np > i.p {
			outp += np - i.p
			i.p = np
			if i.p >= i.rb.nsrc {
				break
			}
			// ASCII may combine with consecutive runes.
			if i.setStart(outp-1, i.p-1) {
				i.p--
				outp--
				i.info = runeInfo{size: 1}
				break
			}
		}
		if i.p >= i.rb.nsrc {
			break
		}
		i.info = i.rb.f.info(i.rb.src, i.p)
	}
	if inCopyStart != i.p {
		i.rb.src.copySlice(out[outCopyStart:], inCopyStart, i.p)
	}
	i.done = i.p >= i.rb.nsrc
	return outp
doNorm:
	i.rb.src.copySlice(out[outCopyStart:], inCopyStart, i.inStart)
	outp, i.p = i.outStart, i.inStart
	i.info = i.rb.f.info(i.rb.src, i.p)
	for {
		if !i.rb.insert(i.rb.src, i.p, i.info) {
			break
		}
		if i.p += int(i.info.size); i.p >= i.rb.nsrc {
			i.rb.compose()
			outp += i.rb.flushCopy(out[outp:])
			i.done = true
			return outp
		}
		i.info = i.rb.f.info(i.rb.src, i.p)
		if i.info.boundaryBefore() {
			break
		}
	}
	i.rb.compose()
	if outp += i.rb.flushCopy(out[outp:]); i.setStart(outp, i.p) {
		return outp
	}
	goto doFast
}
