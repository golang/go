// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package transform provides reader and writer wrappers that transform the
// bytes passing through as well as various transformations. Example
// transformations provided by other packages include normalization and
// conversion between character sets.
package transform // import "golang.org/x/text/transform"

import (
	"bytes"
	"errors"
	"io"
	"unicode/utf8"
)

var (
	// ErrShortDst means that the destination buffer was too short to
	// receive all of the transformed bytes.
	ErrShortDst = errors.New("transform: short destination buffer")

	// ErrShortSrc means that the source buffer has insufficient data to
	// complete the transformation.
	ErrShortSrc = errors.New("transform: short source buffer")

	// ErrEndOfSpan means that the input and output (the transformed input)
	// are not identical.
	ErrEndOfSpan = errors.New("transform: input and output are not identical")

	// errInconsistentByteCount means that Transform returned success (nil
	// error) but also returned nSrc inconsistent with the src argument.
	errInconsistentByteCount = errors.New("transform: inconsistent byte count returned")

	// errShortInternal means that an internal buffer is not large enough
	// to make progress and the Transform operation must be aborted.
	errShortInternal = errors.New("transform: short internal buffer")
)

// Transformer transforms bytes.
type Transformer interface {
	// Transform writes to dst the transformed bytes read from src, and
	// returns the number of dst bytes written and src bytes read. The
	// atEOF argument tells whether src represents the last bytes of the
	// input.
	//
	// Callers should always process the nDst bytes produced and account
	// for the nSrc bytes consumed before considering the error err.
	//
	// A nil error means that all of the transformed bytes (whether freshly
	// transformed from src or left over from previous Transform calls)
	// were written to dst. A nil error can be returned regardless of
	// whether atEOF is true. If err is nil then nSrc must equal len(src);
	// the converse is not necessarily true.
	//
	// ErrShortDst means that dst was too short to receive all of the
	// transformed bytes. ErrShortSrc means that src had insufficient data
	// to complete the transformation. If both conditions apply, then
	// either error may be returned. Other than the error conditions listed
	// here, implementations are free to report other errors that arise.
	Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error)

	// Reset resets the state and allows a Transformer to be reused.
	Reset()
}

// SpanningTransformer extends the Transformer interface with a Span method
// that determines how much of the input already conforms to the Transformer.
type SpanningTransformer interface {
	Transformer

	// Span returns a position in src such that transforming src[:n] results in
	// identical output src[:n] for these bytes. It does not necessarily return
	// the largest such n. The atEOF argument tells whether src represents the
	// last bytes of the input.
	//
	// Callers should always account for the n bytes consumed before
	// considering the error err.
	//
	// A nil error means that all input bytes are known to be identical to the
	// output produced by the Transformer. A nil error can be returned
	// regardless of whether atEOF is true. If err is nil, then n must
	// equal len(src); the converse is not necessarily true.
	//
	// ErrEndOfSpan means that the Transformer output may differ from the
	// input after n bytes. Note that n may be len(src), meaning that the output
	// would contain additional bytes after otherwise identical output.
	// ErrShortSrc means that src had insufficient data to determine whether the
	// remaining bytes would change. Other than the error conditions listed
	// here, implementations are free to report other errors that arise.
	//
	// Calling Span can modify the Transformer state as a side effect. In
	// effect, it does the transformation just as calling Transform would, only
	// without copying to a destination buffer and only up to a point it can
	// determine the input and output bytes are the same. This is obviously more
	// limited than calling Transform, but can be more efficient in terms of
	// copying and allocating buffers. Calls to Span and Transform may be
	// interleaved.
	Span(src []byte, atEOF bool) (n int, err error)
}

// NopResetter can be embedded by implementations of Transformer to add a nop
// Reset method.
type NopResetter struct{}

// Reset implements the Reset method of the Transformer interface.
func (NopResetter) Reset() {}

// Reader wraps another io.Reader by transforming the bytes read.
type Reader struct {
	r   io.Reader
	t   Transformer
	err error

	// dst[dst0:dst1] contains bytes that have been transformed by t but
	// not yet copied out via Read.
	dst        []byte
	dst0, dst1 int

	// src[src0:src1] contains bytes that have been read from r but not
	// yet transformed through t.
	src        []byte
	src0, src1 int

	// transformComplete is whether the transformation is complete,
	// regardless of whether or not it was successful.
	transformComplete bool
}

const defaultBufSize = 4096

// NewReader returns a new Reader that wraps r by transforming the bytes read
// via t. It calls Reset on t.
func NewReader(r io.Reader, t Transformer) *Reader {
	t.Reset()
	return &Reader{
		r:   r,
		t:   t,
		dst: make([]byte, defaultBufSize),
		src: make([]byte, defaultBufSize),
	}
}

// Read implements the io.Reader interface.
func (r *Reader) Read(p []byte) (int, error) {
	n, err := 0, error(nil)
	for {
		// Copy out any transformed bytes and return the final error if we are done.
		if r.dst0 != r.dst1 {
			n = copy(p, r.dst[r.dst0:r.dst1])
			r.dst0 += n
			if r.dst0 == r.dst1 && r.transformComplete {
				return n, r.err
			}
			return n, nil
		} else if r.transformComplete {
			return 0, r.err
		}

		// Try to transform some source bytes, or to flush the transformer if we
		// are out of source bytes. We do this even if r.r.Read returned an error.
		// As the io.Reader documentation says, "process the n > 0 bytes returned
		// before considering the error".
		if r.src0 != r.src1 || r.err != nil {
			r.dst0 = 0
			r.dst1, n, err = r.t.Transform(r.dst, r.src[r.src0:r.src1], r.err == io.EOF)
			r.src0 += n

			switch {
			case err == nil:
				if r.src0 != r.src1 {
					r.err = errInconsistentByteCount
				}
				// The Transform call was successful; we are complete if we
				// cannot read more bytes into src.
				r.transformComplete = r.err != nil
				continue
			case err == ErrShortDst && (r.dst1 != 0 || n != 0):
				// Make room in dst by copying out, and try again.
				continue
			case err == ErrShortSrc && r.src1-r.src0 != len(r.src) && r.err == nil:
				// Read more bytes into src via the code below, and try again.
			default:
				r.transformComplete = true
				// The reader error (r.err) takes precedence over the
				// transformer error (err) unless r.err is nil or io.EOF.
				if r.err == nil || r.err == io.EOF {
					r.err = err
				}
				continue
			}
		}

		// Move any untransformed source bytes to the start of the buffer
		// and read more bytes.
		if r.src0 != 0 {
			r.src0, r.src1 = 0, copy(r.src, r.src[r.src0:r.src1])
		}
		n, r.err = r.r.Read(r.src[r.src1:])
		r.src1 += n
	}
}

// TODO: implement ReadByte (and ReadRune??).

// Writer wraps another io.Writer by transforming the bytes read.
// The user needs to call Close to flush unwritten bytes that may
// be buffered.
type Writer struct {
	w   io.Writer
	t   Transformer
	dst []byte

	// src[:n] contains bytes that have not yet passed through t.
	src []byte
	n   int
}

// NewWriter returns a new Writer that wraps w by transforming the bytes written
// via t. It calls Reset on t.
func NewWriter(w io.Writer, t Transformer) *Writer {
	t.Reset()
	return &Writer{
		w:   w,
		t:   t,
		dst: make([]byte, defaultBufSize),
		src: make([]byte, defaultBufSize),
	}
}

// Write implements the io.Writer interface. If there are not enough
// bytes available to complete a Transform, the bytes will be buffered
// for the next write. Call Close to convert the remaining bytes.
func (w *Writer) Write(data []byte) (n int, err error) {
	src := data
	if w.n > 0 {
		// Append bytes from data to the last remainder.
		// TODO: limit the amount copied on first try.
		n = copy(w.src[w.n:], data)
		w.n += n
		src = w.src[:w.n]
	}
	for {
		nDst, nSrc, err := w.t.Transform(w.dst, src, false)
		if _, werr := w.w.Write(w.dst[:nDst]); werr != nil {
			return n, werr
		}
		src = src[nSrc:]
		if w.n == 0 {
			n += nSrc
		} else if len(src) <= n {
			// Enough bytes from w.src have been consumed. We make src point
			// to data instead to reduce the copying.
			w.n = 0
			n -= len(src)
			src = data[n:]
			if n < len(data) && (err == nil || err == ErrShortSrc) {
				continue
			}
		}
		switch err {
		case ErrShortDst:
			// This error is okay as long as we are making progress.
			if nDst > 0 || nSrc > 0 {
				continue
			}
		case ErrShortSrc:
			if len(src) < len(w.src) {
				m := copy(w.src, src)
				// If w.n > 0, bytes from data were already copied to w.src and n
				// was already set to the number of bytes consumed.
				if w.n == 0 {
					n += m
				}
				w.n = m
				err = nil
			} else if nDst > 0 || nSrc > 0 {
				// Not enough buffer to store the remainder. Keep processing as
				// long as there is progress. Without this case, transforms that
				// require a lookahead larger than the buffer may result in an
				// error. This is not something one may expect to be common in
				// practice, but it may occur when buffers are set to small
				// sizes during testing.
				continue
			}
		case nil:
			if w.n > 0 {
				err = errInconsistentByteCount
			}
		}
		return n, err
	}
}

// Close implements the io.Closer interface.
func (w *Writer) Close() error {
	src := w.src[:w.n]
	for {
		nDst, nSrc, err := w.t.Transform(w.dst, src, true)
		if _, werr := w.w.Write(w.dst[:nDst]); werr != nil {
			return werr
		}
		if err != ErrShortDst {
			return err
		}
		src = src[nSrc:]
	}
}

type nop struct{ NopResetter }

func (nop) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	n := copy(dst, src)
	if n < len(src) {
		err = ErrShortDst
	}
	return n, n, err
}

func (nop) Span(src []byte, atEOF bool) (n int, err error) {
	return len(src), nil
}

type discard struct{ NopResetter }

func (discard) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	return 0, len(src), nil
}

var (
	// Discard is a Transformer for which all Transform calls succeed
	// by consuming all bytes and writing nothing.
	Discard Transformer = discard{}

	// Nop is a SpanningTransformer that copies src to dst.
	Nop SpanningTransformer = nop{}
)

// chain is a sequence of links. A chain with N Transformers has N+1 links and
// N+1 buffers. Of those N+1 buffers, the first and last are the src and dst
// buffers given to chain.Transform and the middle N-1 buffers are intermediate
// buffers owned by the chain. The i'th link transforms bytes from the i'th
// buffer chain.link[i].b at read offset chain.link[i].p to the i+1'th buffer
// chain.link[i+1].b at write offset chain.link[i+1].n, for i in [0, N).
type chain struct {
	link []link
	err  error
	// errStart is the index at which the error occurred plus 1. Processing
	// errStart at this level at the next call to Transform. As long as
	// errStart > 0, chain will not consume any more source bytes.
	errStart int
}

func (c *chain) fatalError(errIndex int, err error) {
	if i := errIndex + 1; i > c.errStart {
		c.errStart = i
		c.err = err
	}
}

type link struct {
	t Transformer
	// b[p:n] holds the bytes to be transformed by t.
	b []byte
	p int
	n int
}

func (l *link) src() []byte {
	return l.b[l.p:l.n]
}

func (l *link) dst() []byte {
	return l.b[l.n:]
}

// Chain returns a Transformer that applies t in sequence.
func Chain(t ...Transformer) Transformer {
	if len(t) == 0 {
		return nop{}
	}
	c := &chain{link: make([]link, len(t)+1)}
	for i, tt := range t {
		c.link[i].t = tt
	}
	// Allocate intermediate buffers.
	b := make([][defaultBufSize]byte, len(t)-1)
	for i := range b {
		c.link[i+1].b = b[i][:]
	}
	return c
}

// Reset resets the state of Chain. It calls Reset on all the Transformers.
func (c *chain) Reset() {
	for i, l := range c.link {
		if l.t != nil {
			l.t.Reset()
		}
		c.link[i].p, c.link[i].n = 0, 0
	}
}

// TODO: make chain use Span (is going to be fun to implement!)

// Transform applies the transformers of c in sequence.
func (c *chain) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	// Set up src and dst in the chain.
	srcL := &c.link[0]
	dstL := &c.link[len(c.link)-1]
	srcL.b, srcL.p, srcL.n = src, 0, len(src)
	dstL.b, dstL.n = dst, 0
	var lastFull, needProgress bool // for detecting progress

	// i is the index of the next Transformer to apply, for i in [low, high].
	// low is the lowest index for which c.link[low] may still produce bytes.
	// high is the highest index for which c.link[high] has a Transformer.
	// The error returned by Transform determines whether to increase or
	// decrease i. We try to completely fill a buffer before converting it.
	for low, i, high := c.errStart, c.errStart, len(c.link)-2; low <= i && i <= high; {
		in, out := &c.link[i], &c.link[i+1]
		nDst, nSrc, err0 := in.t.Transform(out.dst(), in.src(), atEOF && low == i)
		out.n += nDst
		in.p += nSrc
		if i > 0 && in.p == in.n {
			in.p, in.n = 0, 0
		}
		needProgress, lastFull = lastFull, false
		switch err0 {
		case ErrShortDst:
			// Process the destination buffer next. Return if we are already
			// at the high index.
			if i == high {
				return dstL.n, srcL.p, ErrShortDst
			}
			if out.n != 0 {
				i++
				// If the Transformer at the next index is not able to process any
				// source bytes there is nothing that can be done to make progress
				// and the bytes will remain unprocessed. lastFull is used to
				// detect this and break out of the loop with a fatal error.
				lastFull = true
				continue
			}
			// The destination buffer was too small, but is completely empty.
			// Return a fatal error as this transformation can never complete.
			c.fatalError(i, errShortInternal)
		case ErrShortSrc:
			if i == 0 {
				// Save ErrShortSrc in err. All other errors take precedence.
				err = ErrShortSrc
				break
			}
			// Source bytes were depleted before filling up the destination buffer.
			// Verify we made some progress, move the remaining bytes to the errStart
			// and try to get more source bytes.
			if needProgress && nSrc == 0 || in.n-in.p == len(in.b) {
				// There were not enough source bytes to proceed while the source
				// buffer cannot hold any more bytes. Return a fatal error as this
				// transformation can never complete.
				c.fatalError(i, errShortInternal)
				break
			}
			// in.b is an internal buffer and we can make progress.
			in.p, in.n = 0, copy(in.b, in.src())
			fallthrough
		case nil:
			// if i == low, we have depleted the bytes at index i or any lower levels.
			// In that case we increase low and i. In all other cases we decrease i to
			// fetch more bytes before proceeding to the next index.
			if i > low {
				i--
				continue
			}
		default:
			c.fatalError(i, err0)
		}
		// Exhausted level low or fatal error: increase low and continue
		// to process the bytes accepted so far.
		i++
		low = i
	}

	// If c.errStart > 0, this means we found a fatal error.  We will clear
	// all upstream buffers. At this point, no more progress can be made
	// downstream, as Transform would have bailed while handling ErrShortDst.
	if c.errStart > 0 {
		for i := 1; i < c.errStart; i++ {
			c.link[i].p, c.link[i].n = 0, 0
		}
		err, c.errStart, c.err = c.err, 0, nil
	}
	return dstL.n, srcL.p, err
}

// Deprecated: Use runes.Remove instead.
func RemoveFunc(f func(r rune) bool) Transformer {
	return removeF(f)
}

type removeF func(r rune) bool

func (removeF) Reset() {}

// Transform implements the Transformer interface.
func (t removeF) Transform(dst, src []byte, atEOF bool) (nDst, nSrc int, err error) {
	for r, sz := rune(0), 0; len(src) > 0; src = src[sz:] {

		if r = rune(src[0]); r < utf8.RuneSelf {
			sz = 1
		} else {
			r, sz = utf8.DecodeRune(src)

			if sz == 1 {
				// Invalid rune.
				if !atEOF && !utf8.FullRune(src) {
					err = ErrShortSrc
					break
				}
				// We replace illegal bytes with RuneError. Not doing so might
				// otherwise turn a sequence of invalid UTF-8 into valid UTF-8.
				// The resulting byte sequence may subsequently contain runes
				// for which t(r) is true that were passed unnoticed.
				if !t(r) {
					if nDst+3 > len(dst) {
						err = ErrShortDst
						break
					}
					nDst += copy(dst[nDst:], "\uFFFD")
				}
				nSrc++
				continue
			}
		}

		if !t(r) {
			if nDst+sz > len(dst) {
				err = ErrShortDst
				break
			}
			nDst += copy(dst[nDst:], src[:sz])
		}
		nSrc += sz
	}
	return
}

// grow returns a new []byte that is longer than b, and copies the first n bytes
// of b to the start of the new slice.
func grow(b []byte, n int) []byte {
	m := len(b)
	if m <= 32 {
		m = 64
	} else if m <= 256 {
		m *= 2
	} else {
		m += m >> 1
	}
	buf := make([]byte, m)
	copy(buf, b[:n])
	return buf
}

const initialBufSize = 128

// String returns a string with the result of converting s[:n] using t, where
// n <= len(s). If err == nil, n will be len(s). It calls Reset on t.
func String(t Transformer, s string) (result string, n int, err error) {
	t.Reset()
	if s == "" {
		// Fast path for the common case for empty input. Results in about a
		// 86% reduction of running time for BenchmarkStringLowerEmpty.
		if _, _, err := t.Transform(nil, nil, true); err == nil {
			return "", 0, nil
		}
	}

	// Allocate only once. Note that both dst and src escape when passed to
	// Transform.
	buf := [2 * initialBufSize]byte{}
	dst := buf[:initialBufSize:initialBufSize]
	src := buf[initialBufSize : 2*initialBufSize]

	// The input string s is transformed in multiple chunks (starting with a
	// chunk size of initialBufSize). nDst and nSrc are per-chunk (or
	// per-Transform-call) indexes, pDst and pSrc are overall indexes.
	nDst, nSrc := 0, 0
	pDst, pSrc := 0, 0

	// pPrefix is the length of a common prefix: the first pPrefix bytes of the
	// result will equal the first pPrefix bytes of s. It is not guaranteed to
	// be the largest such value, but if pPrefix, len(result) and len(s) are
	// all equal after the final transform (i.e. calling Transform with atEOF
	// being true returned nil error) then we don't need to allocate a new
	// result string.
	pPrefix := 0
	for {
		// Invariant: pDst == pPrefix && pSrc == pPrefix.

		n := copy(src, s[pSrc:])
		nDst, nSrc, err = t.Transform(dst, src[:n], pSrc+n == len(s))
		pDst += nDst
		pSrc += nSrc

		// TODO:  let transformers implement an optional Spanner interface, akin
		// to norm's QuickSpan. This would even allow us to avoid any allocation.
		if !bytes.Equal(dst[:nDst], src[:nSrc]) {
			break
		}
		pPrefix = pSrc
		if err == ErrShortDst {
			// A buffer can only be short if a transformer modifies its input.
			break
		} else if err == ErrShortSrc {
			if nSrc == 0 {
				// No progress was made.
				break
			}
			// Equal so far and !atEOF, so continue checking.
		} else if err != nil || pPrefix == len(s) {
			return string(s[:pPrefix]), pPrefix, err
		}
	}
	// Post-condition: pDst == pPrefix + nDst && pSrc == pPrefix + nSrc.

	// We have transformed the first pSrc bytes of the input s to become pDst
	// transformed bytes. Those transformed bytes are discontiguous: the first
	// pPrefix of them equal s[:pPrefix] and the last nDst of them equal
	// dst[:nDst]. We copy them around, into a new dst buffer if necessary, so
	// that they become one contiguous slice: dst[:pDst].
	if pPrefix != 0 {
		newDst := dst
		if pDst > len(newDst) {
			newDst = make([]byte, len(s)+nDst-nSrc)
		}
		copy(newDst[pPrefix:pDst], dst[:nDst])
		copy(newDst[:pPrefix], s[:pPrefix])
		dst = newDst
	}

	// Prevent duplicate Transform calls with atEOF being true at the end of
	// the input. Also return if we have an unrecoverable error.
	if (err == nil && pSrc == len(s)) ||
		(err != nil && err != ErrShortDst && err != ErrShortSrc) {
		return string(dst[:pDst]), pSrc, err
	}

	// Transform the remaining input, growing dst and src buffers as necessary.
	for {
		n := copy(src, s[pSrc:])
		atEOF := pSrc+n == len(s)
		nDst, nSrc, err := t.Transform(dst[pDst:], src[:n], atEOF)
		pDst += nDst
		pSrc += nSrc

		// If we got ErrShortDst or ErrShortSrc, do not grow as long as we can
		// make progress. This may avoid excessive allocations.
		if err == ErrShortDst {
			if nDst == 0 {
				dst = grow(dst, pDst)
			}
		} else if err == ErrShortSrc {
			if atEOF {
				return string(dst[:pDst]), pSrc, err
			}
			if nSrc == 0 {
				src = grow(src, 0)
			}
		} else if err != nil || pSrc == len(s) {
			return string(dst[:pDst]), pSrc, err
		}
	}
}

// Bytes returns a new byte slice with the result of converting b[:n] using t,
// where n <= len(b). If err == nil, n will be len(b). It calls Reset on t.
func Bytes(t Transformer, b []byte) (result []byte, n int, err error) {
	return doAppend(t, 0, make([]byte, len(b)), b)
}

// Append appends the result of converting src[:n] using t to dst, where
// n <= len(src), If err == nil, n will be len(src). It calls Reset on t.
func Append(t Transformer, dst, src []byte) (result []byte, n int, err error) {
	if len(dst) == cap(dst) {
		n := len(src) + len(dst) // It is okay for this to be 0.
		b := make([]byte, n)
		dst = b[:copy(b, dst)]
	}
	return doAppend(t, len(dst), dst[:cap(dst)], src)
}

func doAppend(t Transformer, pDst int, dst, src []byte) (result []byte, n int, err error) {
	t.Reset()
	pSrc := 0
	for {
		nDst, nSrc, err := t.Transform(dst[pDst:], src[pSrc:], true)
		pDst += nDst
		pSrc += nSrc
		if err != ErrShortDst {
			return dst[:pDst], pSrc, err
		}

		// Grow the destination buffer, but do not grow as long as we can make
		// progress. This may avoid excessive allocations.
		if nDst == 0 {
			dst = grow(dst, pDst)
		}
	}
}
