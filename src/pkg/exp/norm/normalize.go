// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package norm contains types and functions for normalizing Unicode strings.
package norm

import "unicode/utf8"

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
	n := f.QuickSpan(b)
	if n == len(b) {
		return b
	}
	out := make([]byte, n, len(b))
	copy(out, b[0:n])
	return f.Append(out, b[n:]...)
}

// String returns f(s).
func (f Form) String(s string) string {
	n := f.QuickSpanString(s)
	if n == len(s) {
		return s
	}
	out := make([]byte, 0, len(s))
	copy(out, s[0:n])
	return string(f.AppendString(out, s[n:]))
}

// IsNormal returns true if b == f(b).
func (f Form) IsNormal(b []byte) bool {
	rb := reorderBuffer{}
	rb.init(f, b)
	bp := quickSpan(&rb, 0)
	if bp == len(b) {
		return true
	}
	for bp < len(b) {
		decomposeSegment(&rb, bp)
		if rb.f.composing {
			rb.compose()
		}
		for i := 0; i < rb.nrune; i++ {
			info := rb.rune[i]
			if bp+int(info.size) > len(b) {
				return false
			}
			p := info.pos
			pe := p + info.size
			for ; p < pe; p++ {
				if b[bp] != rb.byte[p] {
					return false
				}
				bp++
			}
		}
		rb.reset()
		bp = quickSpan(&rb, bp)
	}
	return true
}

// IsNormalString returns true if s == f(s).
func (f Form) IsNormalString(s string) bool {
	rb := reorderBuffer{}
	rb.initString(f, s)
	bp := quickSpan(&rb, 0)
	if bp == len(s) {
		return true
	}
	for bp < len(s) {
		decomposeSegment(&rb, bp)
		if rb.f.composing {
			rb.compose()
		}
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
		rb.reset()
		bp = quickSpan(&rb, bp)
	}
	return true
}

// patchTail fixes a case where a rune may be incorrectly normalized
// if it is followed by illegal continuation bytes. It returns the
// patched buffer and the number of trailing continuation bytes that
// have been dropped.
func patchTail(rb *reorderBuffer, buf []byte) ([]byte, int) {
	info, p := lastRuneStart(&rb.f, buf)
	if p == -1 || info.size == 0 {
		return buf, 0
	}
	end := p + int(info.size)
	extra := len(buf) - end
	if extra > 0 {
		buf = decomposeToLastBoundary(rb, buf[:end])
		if rb.f.composing {
			rb.compose()
		}
		return rb.flush(buf), extra
	}
	return buf, 0
}

func appendQuick(rb *reorderBuffer, dst []byte, i int) ([]byte, int) {
	if rb.nsrc == i {
		return dst, i
	}
	end := quickSpan(rb, i)
	return rb.src.appendSlice(dst, i, end), end
}

// Append returns f(append(out, b...)).
// The buffer out must be nil, empty, or equal to f(out).
func (f Form) Append(out []byte, src ...byte) []byte {
	if len(src) == 0 {
		return out
	}
	rb := reorderBuffer{}
	rb.init(f, src)
	return doAppend(&rb, out)
}

func doAppend(rb *reorderBuffer, out []byte) []byte {
	src, n := rb.src, rb.nsrc
	doMerge := len(out) > 0
	p := 0
	if p = src.skipNonStarter(); p > 0 {
		// Move leading non-starters to destination.
		out = src.appendSlice(out, 0, p)
		buf, ndropped := patchTail(rb, out)
		if ndropped > 0 {
			out = src.appendSlice(buf, p-ndropped, p)
			doMerge = false // no need to merge, ends with illegal UTF-8
		} else {
			out = decomposeToLastBoundary(rb, buf) // force decomposition
		}
	}
	fd := &rb.f
	if doMerge {
		var info runeInfo
		if p < n {
			info = fd.info(src, p)
			if p == 0 && !fd.boundaryBefore(fd, info) {
				out = decomposeToLastBoundary(rb, out)
			}
		}
		if info.size == 0 || fd.boundaryBefore(fd, info) {
			if fd.composing {
				rb.compose()
			}
			out = rb.flush(out)
			if info.size == 0 {
				// Append incomplete UTF-8 encoding.
				return src.appendSlice(out, p, n)
			}
		}
	}
	if rb.nrune == 0 {
		out, p = appendQuick(rb, out, p)
	}
	for p < n {
		p = decomposeSegment(rb, p)
		if fd.composing {
			rb.compose()
		}
		out = rb.flush(out)
		out, p = appendQuick(rb, out, p)
	}
	return out
}

// AppendString returns f(append(out, []byte(s))).
// The buffer out must be nil, empty, or equal to f(out).
func (f Form) AppendString(out []byte, src string) []byte {
	if len(src) == 0 {
		return out
	}
	rb := reorderBuffer{}
	rb.initString(f, src)
	return doAppend(&rb, out)
}

// QuickSpan returns a boundary n such that b[0:n] == f(b[0:n]).
// It is not guaranteed to return the largest such n.
func (f Form) QuickSpan(b []byte) int {
	rb := reorderBuffer{}
	rb.init(f, b)
	return quickSpan(&rb, 0)
}

func quickSpan(rb *reorderBuffer, i int) int {
	var lastCC uint8
	var nc int
	lastSegStart := i
	src, n := rb.src, rb.nsrc
	for i < n {
		if j := src.skipASCII(i); i != j {
			i = j
			lastSegStart = i - 1
			lastCC = 0
			nc = 0
			continue
		}
		info := rb.f.info(src, i)
		if info.size == 0 {
			// include incomplete runes
			return n
		}
		cc := info.ccc
		if rb.f.composing {
			if !info.flags.isYesC() {
				break
			}
		} else {
			if !info.flags.isYesD() {
				break
			}
		}
		if cc == 0 {
			lastSegStart = i
			nc = 0
		} else {
			if nc >= maxCombiningChars {
				lastSegStart = i
				lastCC = cc
				nc = 1
			} else {
				if lastCC > cc {
					return lastSegStart
				}
				nc++
			}
		}
		lastCC = cc
		i += int(info.size)
	}
	if i == n {
		return n
	}
	if rb.f.composing {
		return lastSegStart
	}
	return i
}

// QuickSpanString returns a boundary n such that b[0:n] == f(s[0:n]).
// It is not guaranteed to return the largest such n.
func (f Form) QuickSpanString(s string) int {
	rb := reorderBuffer{}
	rb.initString(f, s)
	return quickSpan(&rb, 0)
}

// FirstBoundary returns the position i of the first boundary in b
// or -1 if b contains no boundary.
func (f Form) FirstBoundary(b []byte) int {
	rb := reorderBuffer{}
	rb.init(f, b)
	return firstBoundary(&rb)
}

func firstBoundary(rb *reorderBuffer) int {
	src, nsrc := rb.src, rb.nsrc
	i := src.skipNonStarter()
	if i >= nsrc {
		return -1
	}
	fd := &rb.f
	info := fd.info(src, i)
	for n := 0; info.size != 0 && !fd.boundaryBefore(fd, info); {
		i += int(info.size)
		if n++; n >= maxCombiningChars {
			return i
		}
		if i >= nsrc {
			if !fd.boundaryAfter(fd, info) {
				return -1
			}
			return nsrc
		}
		info = fd.info(src, i)
	}
	if info.size == 0 {
		return -1
	}
	return i
}

// FirstBoundaryInString returns the position i of the first boundary in s
// or -1 if s contains no boundary.
func (f Form) FirstBoundaryInString(s string) int {
	rb := reorderBuffer{}
	rb.initString(f, s)
	return firstBoundary(&rb)
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
		if p == 0 { // starts wtih incomplete rune
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
	if fd.boundaryAfter(fd, info) {
		return i
	}
	i = p
	for n := 0; i >= 0 && !fd.boundaryBefore(fd, info); {
		info, p = lastRuneStart(fd, b[:i])
		if n++; n >= maxCombiningChars {
			return len(b)
		}
		if p+int(info.size) != i {
			if p == -1 { // no boundary found
				return -1
			}
			return i // boundary after an illegal UTF-8 encoding
		}
		i = p
	}
	return i
}

// decomposeSegment scans the first segment in src into rb.
// It returns the number of bytes consumed from src.
// TODO(mpvl): consider inserting U+034f (Combining Grapheme Joiner)
// when we detect a sequence of 30+ non-starter chars.
func decomposeSegment(rb *reorderBuffer, sp int) int {
	// Force one character to be consumed.
	info := rb.f.info(rb.src, sp)
	if info.size == 0 {
		return 0
	}
	for rb.insert(rb.src, sp, info) {
		sp += int(info.size)
		if sp >= rb.nsrc {
			break
		}
		info = rb.f.info(rb.src, sp)
		bound := rb.f.boundaryBefore(&rb.f, info)
		if bound || info.size == 0 {
			break
		}
	}
	return sp
}

// lastRuneStart returns the runeInfo and position of the last
// rune in buf or the zero runeInfo and -1 if no rune was found.
func lastRuneStart(fd *formInfo, buf []byte) (runeInfo, int) {
	p := len(buf) - 1
	for ; p >= 0 && !utf8.RuneStart(buf[p]); p-- {
	}
	if p < 0 {
		return runeInfo{0, 0, 0, 0}, -1
	}
	return fd.info(inputBytes(buf), p), p
}

// decomposeToLastBoundary finds an open segment at the end of the buffer
// and scans it into rb. Returns the buffer minus the last segment.
func decomposeToLastBoundary(rb *reorderBuffer, buf []byte) []byte {
	fd := &rb.f
	info, i := lastRuneStart(fd, buf)
	if int(info.size) != len(buf)-i {
		// illegal trailing continuation bytes
		return buf
	}
	if rb.f.boundaryAfter(fd, info) {
		return buf
	}
	var add [maxBackRunes]runeInfo // stores runeInfo in reverse order
	add[0] = info
	padd := 1
	n := 1
	p := len(buf) - int(info.size)
	for ; p >= 0 && !rb.f.boundaryBefore(fd, info); p -= int(info.size) {
		info, i = lastRuneStart(fd, buf[:p])
		if int(info.size) != p-i {
			break
		}
		// Check that decomposition doesn't result in overflow.
		if info.flags.hasDecomposition() {
			dcomp := rb.f.decompose(inputBytes(buf), p-int(info.size))
			for i := 0; i < len(dcomp); {
				inf := rb.f.info(inputBytes(dcomp), i)
				i += int(inf.size)
				n++
			}
		} else {
			n++
		}
		if n > maxBackRunes {
			break
		}
		add[padd] = info
		padd++
	}
	pp := p
	for padd--; padd >= 0; padd-- {
		info = add[padd]
		rb.insert(inputBytes(buf), pp, info)
		pp += int(info.size)
	}
	return buf[:p]
}
