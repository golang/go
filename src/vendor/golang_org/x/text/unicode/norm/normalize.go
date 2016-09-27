// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run maketables.go triegen.go
//go:generate go run maketables.go triegen.go -test

// Package norm contains types and functions for normalizing Unicode strings.
package norm // import "golang.org/x/text/unicode/norm"

import (
	"unicode/utf8"

	"golang_org/x/text/transform"
)

// A Form denotes a canonical representation of Unicode code points.
// The Unicode-defined normalization and equivalence forms are:
//
//   NFC   Unicode Normalization Form C
//   NFD   Unicode Normalization Form D
//   NFKC  Unicode Normalization Form KC
//   NFKD  Unicode Normalization Form KD
//
// For a Form f, this documentation uses the notation f(x) to mean
// the bytes or string x converted to the given form.
// A position n in x is called a boundary if conversion to the form can
// proceed independently on both sides:
//   f(x) == append(f(x[0:n]), f(x[n:])...)
//
// References: http://unicode.org/reports/tr15/ and
// http://unicode.org/notes/tn5/.
type Form int

const (
	NFC Form = iota
	NFD
	NFKC
	NFKD
)

// Bytes returns f(b). May return b if f(b) = b.
func (f Form) Bytes(b []byte) []byte {
	src := inputBytes(b)
	ft := formTable[f]
	n, ok := ft.quickSpan(src, 0, len(b), true)
	if ok {
		return b
	}
	out := make([]byte, n, len(b))
	copy(out, b[0:n])
	rb := reorderBuffer{f: *ft, src: src, nsrc: len(b), out: out, flushF: appendFlush}
	return doAppendInner(&rb, n)
}

// String returns f(s).
func (f Form) String(s string) string {
	src := inputString(s)
	ft := formTable[f]
	n, ok := ft.quickSpan(src, 0, len(s), true)
	if ok {
		return s
	}
	out := make([]byte, n, len(s))
	copy(out, s[0:n])
	rb := reorderBuffer{f: *ft, src: src, nsrc: len(s), out: out, flushF: appendFlush}
	return string(doAppendInner(&rb, n))
}

// IsNormal returns true if b == f(b).
func (f Form) IsNormal(b []byte) bool {
	src := inputBytes(b)
	ft := formTable[f]
	bp, ok := ft.quickSpan(src, 0, len(b), true)
	if ok {
		return true
	}
	rb := reorderBuffer{f: *ft, src: src, nsrc: len(b)}
	rb.setFlusher(nil, cmpNormalBytes)
	for bp < len(b) {
		rb.out = b[bp:]
		if bp = decomposeSegment(&rb, bp, true); bp < 0 {
			return false
		}
		bp, _ = rb.f.quickSpan(rb.src, bp, len(b), true)
	}
	return true
}

func cmpNormalBytes(rb *reorderBuffer) bool {
	b := rb.out
	for i := 0; i < rb.nrune; i++ {
		info := rb.rune[i]
		if int(info.size) > len(b) {
			return false
		}
		p := info.pos
		pe := p + info.size
		for ; p < pe; p++ {
			if b[0] != rb.byte[p] {
				return false
			}
			b = b[1:]
		}
	}
	return true
}

// IsNormalString returns true if s == f(s).
func (f Form) IsNormalString(s string) bool {
	src := inputString(s)
	ft := formTable[f]
	bp, ok := ft.quickSpan(src, 0, len(s), true)
	if ok {
		return true
	}
	rb := reorderBuffer{f: *ft, src: src, nsrc: len(s)}
	rb.setFlusher(nil, func(rb *reorderBuffer) bool {
		for i := 0; i < rb.nrune; i++ {
			info := rb.rune[i]
			if bp+int(info.size) > len(s) {
				return false
			}
			p := info.pos
			pe := p + info.size
			for ; p < pe; p++ {
				if s[bp] != rb.byte[p] {
					return false
				}
				bp++
			}
		}
		return true
	})
	for bp < len(s) {
		if bp = decomposeSegment(&rb, bp, true); bp < 0 {
			return false
		}
		bp, _ = rb.f.quickSpan(rb.src, bp, len(s), true)
	}
	return true
}

// patchTail fixes a case where a rune may be incorrectly normalized
// if it is followed by illegal continuation bytes. It returns the
// patched buffer and whether the decomposition is still in progress.
func patchTail(rb *reorderBuffer) bool {
	info, p := lastRuneStart(&rb.f, rb.out)
	if p == -1 || info.size == 0 {
		return true
	}
	end := p + int(info.size)
	extra := len(rb.out) - end
	if extra > 0 {
		// Potentially allocating memory. However, this only
		// happens with ill-formed UTF-8.
		x := make([]byte, 0)
		x = append(x, rb.out[len(rb.out)-extra:]...)
		rb.out = rb.out[:end]
		decomposeToLastBoundary(rb)
		rb.doFlush()
		rb.out = append(rb.out, x...)
		return false
	}
	buf := rb.out[p:]
	rb.out = rb.out[:p]
	decomposeToLastBoundary(rb)
	if s := rb.ss.next(info); s == ssStarter {
		rb.doFlush()
		rb.ss.first(info)
	} else if s == ssOverflow {
		rb.doFlush()
		rb.insertCGJ()
		rb.ss = 0
	}
	rb.insertUnsafe(inputBytes(buf), 0, info)
	return true
}

func appendQuick(rb *reorderBuffer, i int) int {
	if rb.nsrc == i {
		return i
	}
	end, _ := rb.f.quickSpan(rb.src, i, rb.nsrc, true)
	rb.out = rb.src.appendSlice(rb.out, i, end)
	return end
}

// Append returns f(append(out, b...)).
// The buffer out must be nil, empty, or equal to f(out).
func (f Form) Append(out []byte, src ...byte) []byte {
	return f.doAppend(out, inputBytes(src), len(src))
}

func (f Form) doAppend(out []byte, src input, n int) []byte {
	if n == 0 {
		return out
	}
	ft := formTable[f]
	// Attempt to do a quickSpan first so we can avoid initializing the reorderBuffer.
	if len(out) == 0 {
		p, _ := ft.quickSpan(src, 0, n, true)
		out = src.appendSlice(out, 0, p)
		if p == n {
			return out
		}
		rb := reorderBuffer{f: *ft, src: src, nsrc: n, out: out, flushF: appendFlush}
		return doAppendInner(&rb, p)
	}
	rb := reorderBuffer{f: *ft, src: src, nsrc: n}
	return doAppend(&rb, out, 0)
}

func doAppend(rb *reorderBuffer, out []byte, p int) []byte {
	rb.setFlusher(out, appendFlush)
	src, n := rb.src, rb.nsrc
	doMerge := len(out) > 0
	if q := src.skipContinuationBytes(p); q > p {
		// Move leading non-starters to destination.
		rb.out = src.appendSlice(rb.out, p, q)
		p = q
		doMerge = patchTail(rb)
	}
	fd := &rb.f
	if doMerge {
		var info Properties
		if p < n {
			info = fd.info(src, p)
			if !info.BoundaryBefore() || info.nLeadingNonStarters() > 0 {
				if p == 0 {
					decomposeToLastBoundary(rb)
				}
				p = decomposeSegment(rb, p, true)
			}
		}
		if info.size == 0 {
			rb.doFlush()
			// Append incomplete UTF-8 encoding.
			return src.appendSlice(rb.out, p, n)
		}
		if rb.nrune > 0 {
			return doAppendInner(rb, p)
		}
	}
	p = appendQuick(rb, p)
	return doAppendInner(rb, p)
}

func doAppendInner(rb *reorderBuffer, p int) []byte {
	for n := rb.nsrc; p < n; {
		p = decomposeSegment(rb, p, true)
		p = appendQuick(rb, p)
	}
	return rb.out
}

// AppendString returns f(append(out, []byte(s))).
// The buffer out must be nil, empty, or equal to f(out).
func (f Form) AppendString(out []byte, src string) []byte {
	return f.doAppend(out, inputString(src), len(src))
}

// QuickSpan returns a boundary n such that b[0:n] == f(b[0:n]).
// It is not guaranteed to return the largest such n.
func (f Form) QuickSpan(b []byte) int {
	n, _ := formTable[f].quickSpan(inputBytes(b), 0, len(b), true)
	return n
}

// Span implements transform.SpanningTransformer. It returns a boundary n such
// that b[0:n] == f(b[0:n]). It is not guaranteed to return the largest such n.
func (f Form) Span(b []byte, atEOF bool) (n int, err error) {
	n, ok := formTable[f].quickSpan(inputBytes(b), 0, len(b), atEOF)
	if n < len(b) {
		if !ok {
			err = transform.ErrEndOfSpan
		} else {
			err = transform.ErrShortSrc
		}
	}
	return n, err
}

// SpanString returns a boundary n such that s[0:n] == f(s[0:n]).
// It is not guaranteed to return the largest such n.
func (f Form) SpanString(s string, atEOF bool) (n int, err error) {
	n, ok := formTable[f].quickSpan(inputString(s), 0, len(s), atEOF)
	if n < len(s) {
		if !ok {
			err = transform.ErrEndOfSpan
		} else {
			err = transform.ErrShortSrc
		}
	}
	return n, err
}

// quickSpan returns a boundary n such that src[0:n] == f(src[0:n]) and
// whether any non-normalized parts were found. If atEOF is false, n will
// not point past the last segment if this segment might be become
// non-normalized by appending other runes.
func (f *formInfo) quickSpan(src input, i, end int, atEOF bool) (n int, ok bool) {
	var lastCC uint8
	ss := streamSafe(0)
	lastSegStart := i
	for n = end; i < n; {
		if j := src.skipASCII(i, n); i != j {
			i = j
			lastSegStart = i - 1
			lastCC = 0
			ss = 0
			continue
		}
		info := f.info(src, i)
		if info.size == 0 {
			if atEOF {
				// include incomplete runes
				return n, true
			}
			return lastSegStart, true
		}
		// This block needs to be before the next, because it is possible to
		// have an overflow for runes that are starters (e.g. with U+FF9E).
		switch ss.next(info) {
		case ssStarter:
			ss.first(info)
			lastSegStart = i
		case ssOverflow:
			return lastSegStart, false
		case ssSuccess:
			if lastCC > info.ccc {
				return lastSegStart, false
			}
		}
		if f.composing {
			if !info.isYesC() {
				break
			}
		} else {
			if !info.isYesD() {
				break
			}
		}
		lastCC = info.ccc
		i += int(info.size)
	}
	if i == n {
		if !atEOF {
			n = lastSegStart
		}
		return n, true
	}
	return lastSegStart, false
}

// QuickSpanString returns a boundary n such that s[0:n] == f(s[0:n]).
// It is not guaranteed to return the largest such n.
func (f Form) QuickSpanString(s string) int {
	n, _ := formTable[f].quickSpan(inputString(s), 0, len(s), true)
	return n
}

// FirstBoundary returns the position i of the first boundary in b
// or -1 if b contains no boundary.
func (f Form) FirstBoundary(b []byte) int {
	return f.firstBoundary(inputBytes(b), len(b))
}

func (f Form) firstBoundary(src input, nsrc int) int {
	i := src.skipContinuationBytes(0)
	if i >= nsrc {
		return -1
	}
	fd := formTable[f]
	ss := streamSafe(0)
	// We should call ss.first here, but we can't as the first rune is
	// skipped already. This means FirstBoundary can't really determine
	// CGJ insertion points correctly. Luckily it doesn't have to.
	for {
		info := fd.info(src, i)
		if info.size == 0 {
			return -1
		}
		if s := ss.next(info); s != ssSuccess {
			return i
		}
		i += int(info.size)
		if i >= nsrc {
			if !info.BoundaryAfter() && !ss.isMax() {
				return -1
			}
			return nsrc
		}
	}
}

// FirstBoundaryInString returns the position i of the first boundary in s
// or -1 if s contains no boundary.
func (f Form) FirstBoundaryInString(s string) int {
	return f.firstBoundary(inputString(s), len(s))
}

// NextBoundary reports the index of the boundary between the first and next
// segment in b or -1 if atEOF is false and there are not enough bytes to
// determine this boundary.
func (f Form) NextBoundary(b []byte, atEOF bool) int {
	return f.nextBoundary(inputBytes(b), len(b), atEOF)
}

// NextBoundaryInString reports the index of the boundary between the first and
// next segment in b or -1 if atEOF is false and there are not enough bytes to
// determine this boundary.
func (f Form) NextBoundaryInString(s string, atEOF bool) int {
	return f.nextBoundary(inputString(s), len(s), atEOF)
}

func (f Form) nextBoundary(src input, nsrc int, atEOF bool) int {
	if nsrc == 0 {
		if atEOF {
			return 0
		}
		return -1
	}
	fd := formTable[f]
	info := fd.info(src, 0)
	if info.size == 0 {
		if atEOF {
			return 1
		}
		return -1
	}
	ss := streamSafe(0)
	ss.first(info)

	for i := int(info.size); i < nsrc; i += int(info.size) {
		info = fd.info(src, i)
		if info.size == 0 {
			if atEOF {
				return i
			}
			return -1
		}
		if s := ss.next(info); s != ssSuccess {
			return i
		}
	}
	if !atEOF && !info.BoundaryAfter() && !ss.isMax() {
		return -1
	}
	return nsrc
}

// LastBoundary returns the position i of the last boundary in b
// or -1 if b contains no boundary.
func (f Form) LastBoundary(b []byte) int {
	return lastBoundary(formTable[f], b)
}

func lastBoundary(fd *formInfo, b []byte) int {
	i := len(b)
	info, p := lastRuneStart(fd, b)
	if p == -1 {
		return -1
	}
	if info.size == 0 { // ends with incomplete rune
		if p == 0 { // starts with incomplete rune
			return -1
		}
		i = p
		info, p = lastRuneStart(fd, b[:i])
		if p == -1 { // incomplete UTF-8 encoding or non-starter bytes without a starter
			return i
		}
	}
	if p+int(info.size) != i { // trailing non-starter bytes: illegal UTF-8
		return i
	}
	if info.BoundaryAfter() {
		return i
	}
	ss := streamSafe(0)
	v := ss.backwards(info)
	for i = p; i >= 0 && v != ssStarter; i = p {
		info, p = lastRuneStart(fd, b[:i])
		if v = ss.backwards(info); v == ssOverflow {
			break
		}
		if p+int(info.size) != i {
			if p == -1 { // no boundary found
				return -1
			}
			return i // boundary after an illegal UTF-8 encoding
		}
	}
	return i
}

// decomposeSegment scans the first segment in src into rb. It inserts 0x034f
// (Grapheme Joiner) when it encounters a sequence of more than 30 non-starters
// and returns the number of bytes consumed from src or iShortDst or iShortSrc.
func decomposeSegment(rb *reorderBuffer, sp int, atEOF bool) int {
	// Force one character to be consumed.
	info := rb.f.info(rb.src, sp)
	if info.size == 0 {
		return 0
	}
	if rb.nrune > 0 {
		if s := rb.ss.next(info); s == ssStarter {
			goto end
		} else if s == ssOverflow {
			rb.insertCGJ()
			goto end
		}
	} else {
		rb.ss.first(info)
	}
	if err := rb.insertFlush(rb.src, sp, info); err != iSuccess {
		return int(err)
	}
	for {
		sp += int(info.size)
		if sp >= rb.nsrc {
			if !atEOF && !info.BoundaryAfter() {
				return int(iShortSrc)
			}
			break
		}
		info = rb.f.info(rb.src, sp)
		if info.size == 0 {
			if !atEOF {
				return int(iShortSrc)
			}
			break
		}
		if s := rb.ss.next(info); s == ssStarter {
			break
		} else if s == ssOverflow {
			rb.insertCGJ()
			break
		}
		if err := rb.insertFlush(rb.src, sp, info); err != iSuccess {
			return int(err)
		}
	}
end:
	if !rb.doFlush() {
		return int(iShortDst)
	}
	return sp
}

// lastRuneStart returns the runeInfo and position of the last
// rune in buf or the zero runeInfo and -1 if no rune was found.
func lastRuneStart(fd *formInfo, buf []byte) (Properties, int) {
	p := len(buf) - 1
	for ; p >= 0 && !utf8.RuneStart(buf[p]); p-- {
	}
	if p < 0 {
		return Properties{}, -1
	}
	return fd.info(inputBytes(buf), p), p
}

// decomposeToLastBoundary finds an open segment at the end of the buffer
// and scans it into rb. Returns the buffer minus the last segment.
func decomposeToLastBoundary(rb *reorderBuffer) {
	fd := &rb.f
	info, i := lastRuneStart(fd, rb.out)
	if int(info.size) != len(rb.out)-i {
		// illegal trailing continuation bytes
		return
	}
	if info.BoundaryAfter() {
		return
	}
	var add [maxNonStarters + 1]Properties // stores runeInfo in reverse order
	padd := 0
	ss := streamSafe(0)
	p := len(rb.out)
	for {
		add[padd] = info
		v := ss.backwards(info)
		if v == ssOverflow {
			// Note that if we have an overflow, it the string we are appending to
			// is not correctly normalized. In this case the behavior is undefined.
			break
		}
		padd++
		p -= int(info.size)
		if v == ssStarter || p < 0 {
			break
		}
		info, i = lastRuneStart(fd, rb.out[:p])
		if int(info.size) != p-i {
			break
		}
	}
	rb.ss = ss
	// Copy bytes for insertion as we may need to overwrite rb.out.
	var buf [maxBufferSize * utf8.UTFMax]byte
	cp := buf[:copy(buf[:], rb.out[p:])]
	rb.out = rb.out[:p]
	for padd--; padd >= 0; padd-- {
		info = add[padd]
		rb.insertUnsafe(inputBytes(cp), 0, info)
		cp = cp[info.size:]
	}
}
