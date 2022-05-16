// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptobyte

import (
	"errors"
	"fmt"
)

// A Builder builds byte strings from fixed-length and length-prefixed values.
// Builders either allocate space as needed, or are ‘fixed’, which means that
// they write into a given buffer and produce an error if it's exhausted.
//
// The zero value is a usable Builder that allocates space as needed.
//
// Simple values are marshaled and appended to a Builder using methods on the
// Builder. Length-prefixed values are marshaled by providing a
// BuilderContinuation, which is a function that writes the inner contents of
// the value to a given Builder. See the documentation for BuilderContinuation
// for details.
type Builder struct {
	err            error
	result         []byte
	fixedSize      bool
	child          *Builder
	offset         int
	pendingLenLen  int
	pendingIsASN1  bool
	inContinuation *bool
}

// NewBuilder creates a Builder that appends its output to the given buffer.
// Like append(), the slice will be reallocated if its capacity is exceeded.
// Use Bytes to get the final buffer.
func NewBuilder(buffer []byte) *Builder {
	return &Builder{
		result: buffer,
	}
}

// NewFixedBuilder creates a Builder that appends its output into the given
// buffer. This builder does not reallocate the output buffer. Writes that
// would exceed the buffer's capacity are treated as an error.
func NewFixedBuilder(buffer []byte) *Builder {
	return &Builder{
		result:    buffer,
		fixedSize: true,
	}
}

// SetError sets the value to be returned as the error from Bytes. Writes
// performed after calling SetError are ignored.
func (b *Builder) SetError(err error) {
	b.err = err
}

// Bytes returns the bytes written by the builder or an error if one has
// occurred during building.
func (b *Builder) Bytes() ([]byte, error) {
	if b.err != nil {
		return nil, b.err
	}
	return b.result[b.offset:], nil
}

// BytesOrPanic returns the bytes written by the builder or panics if an error
// has occurred during building.
func (b *Builder) BytesOrPanic() []byte {
	if b.err != nil {
		panic(b.err)
	}
	return b.result[b.offset:]
}

// AddUint8 appends an 8-bit value to the byte string.
func (b *Builder) AddUint8(v uint8) {
	b.add(byte(v))
}

// AddUint16 appends a big-endian, 16-bit value to the byte string.
func (b *Builder) AddUint16(v uint16) {
	b.add(byte(v>>8), byte(v))
}

// AddUint24 appends a big-endian, 24-bit value to the byte string. The highest
// byte of the 32-bit input value is silently truncated.
func (b *Builder) AddUint24(v uint32) {
	b.add(byte(v>>16), byte(v>>8), byte(v))
}

// AddUint32 appends a big-endian, 32-bit value to the byte string.
func (b *Builder) AddUint32(v uint32) {
	b.add(byte(v>>24), byte(v>>16), byte(v>>8), byte(v))
}

// AddBytes appends a sequence of bytes to the byte string.
func (b *Builder) AddBytes(v []byte) {
	b.add(v...)
}

// BuilderContinuation is a continuation-passing interface for building
// length-prefixed byte sequences. Builder methods for length-prefixed
// sequences (AddUint8LengthPrefixed etc) will invoke the BuilderContinuation
// supplied to them. The child builder passed to the continuation can be used
// to build the content of the length-prefixed sequence. For example:
//
//	parent := cryptobyte.NewBuilder()
//	parent.AddUint8LengthPrefixed(func (child *Builder) {
//	  child.AddUint8(42)
//	  child.AddUint8LengthPrefixed(func (grandchild *Builder) {
//	    grandchild.AddUint8(5)
//	  })
//	})
//
// It is an error to write more bytes to the child than allowed by the reserved
// length prefix. After the continuation returns, the child must be considered
// invalid, i.e. users must not store any copies or references of the child
// that outlive the continuation.
//
// If the continuation panics with a value of type BuildError then the inner
// error will be returned as the error from Bytes. If the child panics
// otherwise then Bytes will repanic with the same value.
type BuilderContinuation func(child *Builder)

// BuildError wraps an error. If a BuilderContinuation panics with this value,
// the panic will be recovered and the inner error will be returned from
// Builder.Bytes.
type BuildError struct {
	Err error
}

// AddUint8LengthPrefixed adds a 8-bit length-prefixed byte sequence.
func (b *Builder) AddUint8LengthPrefixed(f BuilderContinuation) {
	b.addLengthPrefixed(1, false, f)
}

// AddUint16LengthPrefixed adds a big-endian, 16-bit length-prefixed byte sequence.
func (b *Builder) AddUint16LengthPrefixed(f BuilderContinuation) {
	b.addLengthPrefixed(2, false, f)
}

// AddUint24LengthPrefixed adds a big-endian, 24-bit length-prefixed byte sequence.
func (b *Builder) AddUint24LengthPrefixed(f BuilderContinuation) {
	b.addLengthPrefixed(3, false, f)
}

// AddUint32LengthPrefixed adds a big-endian, 32-bit length-prefixed byte sequence.
func (b *Builder) AddUint32LengthPrefixed(f BuilderContinuation) {
	b.addLengthPrefixed(4, false, f)
}

func (b *Builder) callContinuation(f BuilderContinuation, arg *Builder) {
	if !*b.inContinuation {
		*b.inContinuation = true

		defer func() {
			*b.inContinuation = false

			r := recover()
			if r == nil {
				return
			}

			if buildError, ok := r.(BuildError); ok {
				b.err = buildError.Err
			} else {
				panic(r)
			}
		}()
	}

	f(arg)
}

func (b *Builder) addLengthPrefixed(lenLen int, isASN1 bool, f BuilderContinuation) {
	// Subsequent writes can be ignored if the builder has encountered an error.
	if b.err != nil {
		return
	}

	offset := len(b.result)
	b.add(make([]byte, lenLen)...)

	if b.inContinuation == nil {
		b.inContinuation = new(bool)
	}

	b.child = &Builder{
		result:         b.result,
		fixedSize:      b.fixedSize,
		offset:         offset,
		pendingLenLen:  lenLen,
		pendingIsASN1:  isASN1,
		inContinuation: b.inContinuation,
	}

	b.callContinuation(f, b.child)
	b.flushChild()
	if b.child != nil {
		panic("cryptobyte: internal error")
	}
}

func (b *Builder) flushChild() {
	if b.child == nil {
		return
	}
	b.child.flushChild()
	child := b.child
	b.child = nil

	if child.err != nil {
		b.err = child.err
		return
	}

	length := len(child.result) - child.pendingLenLen - child.offset

	if length < 0 {
		panic("cryptobyte: internal error") // result unexpectedly shrunk
	}

	if child.pendingIsASN1 {
		// For ASN.1, we reserved a single byte for the length. If that turned out
		// to be incorrect, we have to move the contents along in order to make
		// space.
		if child.pendingLenLen != 1 {
			panic("cryptobyte: internal error")
		}
		var lenLen, lenByte uint8
		if int64(length) > 0xfffffffe {
			b.err = errors.New("pending ASN.1 child too long")
			return
		} else if length > 0xffffff {
			lenLen = 5
			lenByte = 0x80 | 4
		} else if length > 0xffff {
			lenLen = 4
			lenByte = 0x80 | 3
		} else if length > 0xff {
			lenLen = 3
			lenByte = 0x80 | 2
		} else if length > 0x7f {
			lenLen = 2
			lenByte = 0x80 | 1
		} else {
			lenLen = 1
			lenByte = uint8(length)
			length = 0
		}

		// Insert the initial length byte, make space for successive length bytes,
		// and adjust the offset.
		child.result[child.offset] = lenByte
		extraBytes := int(lenLen - 1)
		if extraBytes != 0 {
			child.add(make([]byte, extraBytes)...)
			childStart := child.offset + child.pendingLenLen
			copy(child.result[childStart+extraBytes:], child.result[childStart:])
		}
		child.offset++
		child.pendingLenLen = extraBytes
	}

	l := length
	for i := child.pendingLenLen - 1; i >= 0; i-- {
		child.result[child.offset+i] = uint8(l)
		l >>= 8
	}
	if l != 0 {
		b.err = fmt.Errorf("cryptobyte: pending child length %d exceeds %d-byte length prefix", length, child.pendingLenLen)
		return
	}

	if b.fixedSize && &b.result[0] != &child.result[0] {
		panic("cryptobyte: BuilderContinuation reallocated a fixed-size buffer")
	}

	b.result = child.result
}

func (b *Builder) add(bytes ...byte) {
	if b.err != nil {
		return
	}
	if b.child != nil {
		panic("cryptobyte: attempted write while child is pending")
	}
	if len(b.result)+len(bytes) < len(bytes) {
		b.err = errors.New("cryptobyte: length overflow")
	}
	if b.fixedSize && len(b.result)+len(bytes) > cap(b.result) {
		b.err = errors.New("cryptobyte: Builder is exceeding its fixed-size buffer")
		return
	}
	b.result = append(b.result, bytes...)
}

// Unwrite rolls back n bytes written directly to the Builder. An attempt by a
// child builder passed to a continuation to unwrite bytes from its parent will
// panic.
func (b *Builder) Unwrite(n int) {
	if b.err != nil {
		return
	}
	if b.child != nil {
		panic("cryptobyte: attempted unwrite while child is pending")
	}
	length := len(b.result) - b.pendingLenLen - b.offset
	if length < 0 {
		panic("cryptobyte: internal error")
	}
	if n > length {
		panic("cryptobyte: attempted to unwrite more than was written")
	}
	b.result = b.result[:len(b.result)-n]
}

// A MarshalingValue marshals itself into a Builder.
type MarshalingValue interface {
	// Marshal is called by Builder.AddValue. It receives a pointer to a builder
	// to marshal itself into. It may return an error that occurred during
	// marshaling, such as unset or invalid values.
	Marshal(b *Builder) error
}

// AddValue calls Marshal on v, passing a pointer to the builder to append to.
// If Marshal returns an error, it is set on the Builder so that subsequent
// appends don't have an effect.
func (b *Builder) AddValue(v MarshalingValue) {
	err := v.Marshal(b)
	if err != nil {
		b.err = err
	}
}
