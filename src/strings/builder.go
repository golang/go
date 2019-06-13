// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import (
	"sync/atomic"
	"unicode/utf8"
	"unsafe"
)

// A Builder is used to efficiently build a string using Write methods.
// It minimizes memory copying. The zero value is ready to use.
// Do not copy a non-zero Builder.
type Builder struct {
	addr *Builder // of receiver, to detect copies by value
	buf  []byte
}

// noescape hides a pointer from escape analysis.  noescape is
// the identity function but escape analysis doesn't think the
// output depends on the input. noescape is inlined and currently
// compiles down to zero instructions.
// USE CAREFULLY!
// This was copied from the runtime; see issues 23382 and 7921.
//go:nosplit
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}

func (b *Builder) copyCheck() {
	if b.addr == nil {
		// This hack works around a failing of Go's escape analysis
		// that was causing b to escape and be heap allocated.
		// See issue 23382.
		// TODO: once issue 7921 is fixed, this should be reverted to
		// just "b.addr = b".
		b.addr = (*Builder)(noescape(unsafe.Pointer(b)))
	} else if b.addr != b {
		panic("strings: illegal use of non-zero Builder copied by value")
	}
}

// String returns the accumulated string.
func (b *Builder) String() string {
	return bytes2String(b.buf)
}

func bytes2String(bytes []byte) string {
	return *(*string)(unsafe.Pointer(&bytes))
}

// Len returns the number of accumulated bytes; b.Len() == len(b.String()).
func (b *Builder) Len() int { return len(b.buf) }

// Cap returns the capacity of the builder's underlying byte slice. It is the
// total space allocated for the string being built and includes any bytes
// already written.
func (b *Builder) Cap() int { return cap(b.buf) }

// Reset resets the Builder to be empty.
func (b *Builder) Reset() {
	b.addr = nil
	b.buf = nil
}

// grow copies the buffer to a new, larger buffer so that there are at least n
// bytes of capacity beyond len(b.buf).
func (b *Builder) grow(n int) {
	buf := make([]byte, len(b.buf), 2*cap(b.buf)+n)
	copy(buf, b.buf)
	b.buf = buf
}

// Grow grows b's capacity, if necessary, to guarantee space for
// another n bytes. After Grow(n), at least n bytes can be written to b
// without another allocation. If n is negative, Grow panics.
func (b *Builder) Grow(n int) {
	b.copyCheck()
	if n < 0 {
		panic("strings.Builder.Grow: negative count")
	}
	if cap(b.buf)-len(b.buf) < n {
		b.grow(n)
	}
}

// Write appends the contents of p to b's buffer.
// Write always returns len(p), nil.
func (b *Builder) Write(p []byte) (int, error) {
	b.copyCheck()
	b.buf = append(b.buf, p...)
	return len(p), nil
}

// WriteByte appends the byte c to b's buffer.
// The returned error is always nil.
func (b *Builder) WriteByte(c byte) error {
	b.copyCheck()
	b.buf = append(b.buf, c)
	return nil
}

// WriteRune appends the UTF-8 encoding of Unicode code point r to b's buffer.
// It returns the length of r and a nil error.
func (b *Builder) WriteRune(r rune) (int, error) {
	b.copyCheck()
	if r < utf8.RuneSelf {
		b.buf = append(b.buf, byte(r))
		return 1, nil
	}
	l := len(b.buf)
	if cap(b.buf)-l < utf8.UTFMax {
		b.grow(utf8.UTFMax)
	}
	n := utf8.EncodeRune(b.buf[l:l+utf8.UTFMax], r)
	b.buf = b.buf[:l+n]
	return n, nil
}

// WriteString appends the contents of s to b's buffer.
// It returns the length of s and a nil error.
func (b *Builder) WriteString(s string) (int, error) {
	b.copyCheck()
	b.buf = append(b.buf, s...)
	return len(s), nil
}

const (
	// DefaultFactoryPoolSize is the default pool size for the Factory.
	DefaultFactoryPoolSize = 4096
)

// Factory represents the factory object for generating immutable strings.
type Factory struct {
	b Builder
}

// NewFactory generate a string factory.
func NewFactory() *Factory {
	return NewFactoryWithPoolSize(DefaultFactoryPoolSize)
}

// NewFactoryWithPoolSize specify a pool size for the factory to generate
// strings, the pool size is only for the memory fragmentation preventation.
func NewFactoryWithPoolSize(size int) *Factory {
	f := &Factory{}
	f.b.Grow(size)
	return f
}

// NewString generate a string from bytes content.
func (f *Factory) New(content []byte) string {

	bCap := f.b.Cap()
	bLen := f.b.Len()

	if len(content)*2 > bCap {
		return string(content)
	}

	if len(content) > bCap-bLen {
		f.b.Reset()
		f.b.Grow(bCap)
	}

	preLen := f.b.Len()
	f.b.Write(content)
	return f.b.String()[preLen:]
}

// for internal using, see globalFactory usage
type syncTape struct {
	tape [DefaultFactoryPoolSize]byte
	tPtr int64
}

func (st *syncTape) alloc(size int) ([]byte, bool) {

	end := atomic.AddInt64(&st.tPtr, int64(size))
	if end > int64(len(st.tape)) {
		// to prevent overflow
		atomic.StoreInt64(&st.tPtr, int64(len(st.tape)))
		return nil, false
	}

	return st.tape[end-int64(size) : end], true
}

var globalFactory atomic.Value

// New generate an immutable string from mutable bytes
func New(content []byte) string {

	if len(content)*2 > DefaultFactoryPoolSize {
		return string(content)
	}

	gf := globalFactory.Load()
	if gf != nil {
		tape := gf.(*syncTape)
		frag, ok := tape.alloc(len(content))
		if ok {
			copy(frag, content)
			return bytes2String(frag)
		}
	}

	tape := &syncTape{}
	frag, _ := tape.alloc(len(content))
	globalFactory.Store(tape)
	copy(frag, content)
	return bytes2String(frag)
}
