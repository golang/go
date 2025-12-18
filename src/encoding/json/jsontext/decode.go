// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"bytes"
	"errors"
	"io"

	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
	"encoding/json/internal/jsonwire"
)

// NOTE: The logic for decoding is complicated by the fact that reading from
// an io.Reader into a temporary buffer means that the buffer may contain a
// truncated portion of some valid input, requiring the need to fetch more data.
//
// This file is structured in the following way:
//
//   - consumeXXX functions parse an exact JSON token from a []byte.
//     If the buffer appears truncated, then it returns io.ErrUnexpectedEOF.
//     The consumeSimpleXXX functions are so named because they only handle
//     a subset of the grammar for the JSON token being parsed.
//     They do not handle the full grammar to keep these functions inlinable.
//
//   - Decoder.consumeXXX methods parse the next JSON token from Decoder.buf,
//     automatically fetching more input if necessary. These methods take
//     a position relative to the start of Decoder.buf as an argument and
//     return the end of the consumed JSON token as a position,
//     also relative to the start of Decoder.buf.
//
//   - In the event of an I/O errors or state machine violations,
//     the implementation avoids mutating the state of Decoder
//     (aside from the book-keeping needed to implement Decoder.fetch).
//     For this reason, only Decoder.ReadToken and Decoder.ReadValue are
//     responsible for updated Decoder.prevStart and Decoder.prevEnd.
//
//   - For performance, much of the implementation uses the pattern of calling
//     the inlinable consumeXXX functions first, and if more work is necessary,
//     then it calls the slower Decoder.consumeXXX methods.
//     TODO: Revisit this pattern if the Go compiler provides finer control
//     over exactly which calls are inlined or not.

// Decoder is a streaming decoder for raw JSON tokens and values.
// It is used to read a stream of top-level JSON values,
// each separated by optional whitespace characters.
//
// [Decoder.ReadToken] and [Decoder.ReadValue] calls may be interleaved.
// For example, the following JSON value:
//
//	{"name":"value","array":[null,false,true,3.14159],"object":{"k":"v"}}
//
// can be parsed with the following calls (ignoring errors for brevity):
//
//	d.ReadToken() // {
//	d.ReadToken() // "name"
//	d.ReadToken() // "value"
//	d.ReadValue() // "array"
//	d.ReadToken() // [
//	d.ReadToken() // null
//	d.ReadToken() // false
//	d.ReadValue() // true
//	d.ReadToken() // 3.14159
//	d.ReadToken() // ]
//	d.ReadValue() // "object"
//	d.ReadValue() // {"k":"v"}
//	d.ReadToken() // }
//
// The above is one of many possible sequence of calls and
// may not represent the most sensible method to call for any given token/value.
// For example, it is probably more common to call [Decoder.ReadToken] to obtain a
// string token for object names.
type Decoder struct {
	s decoderState
}

// decoderState is the low-level state of Decoder.
// It has exported fields and method for use by the "json" package.
type decoderState struct {
	state
	decodeBuffer
	jsonopts.Struct

	StringCache *[256]string // only used when unmarshaling; identical to json.stringCache
}

// decodeBuffer is a buffer split into 4 segments:
//
//   - buf[0:prevEnd]         // already read portion of the buffer
//   - buf[prevStart:prevEnd] // previously read value
//   - buf[prevEnd:len(buf)]  // unread portion of the buffer
//   - buf[len(buf):cap(buf)] // unused portion of the buffer
//
// Invariants:
//
//	0 ≤ prevStart ≤ prevEnd ≤ len(buf) ≤ cap(buf)
type decodeBuffer struct {
	peekPos int   // non-zero if valid offset into buf for start of next token
	peekErr error // implies peekPos is -1

	buf       []byte // may alias rd if it is a bytes.Buffer
	prevStart int
	prevEnd   int

	// baseOffset is added to prevStart and prevEnd to obtain
	// the absolute offset relative to the start of io.Reader stream.
	baseOffset int64

	rd io.Reader
}

// NewDecoder constructs a new streaming decoder reading from r.
//
// If r is a [bytes.Buffer], then the decoder parses directly from the buffer
// without first copying the contents to an intermediate buffer.
// Additional writes to the buffer must not occur while the decoder is in use.
func NewDecoder(r io.Reader, opts ...Options) *Decoder {
	d := new(Decoder)
	d.Reset(r, opts...)
	return d
}

// Reset resets a decoder such that it is reading afresh from r and
// configured with the provided options. Reset must not be called on an
// a Decoder passed to the [encoding/json/v2.UnmarshalerFrom.UnmarshalJSONFrom] method
// or the [encoding/json/v2.UnmarshalFromFunc] function.
func (d *Decoder) Reset(r io.Reader, opts ...Options) {
	switch {
	case d == nil:
		panic("jsontext: invalid nil Decoder")
	case r == nil:
		panic("jsontext: invalid nil io.Reader")
	case d.s.Flags.Get(jsonflags.WithinArshalCall):
		panic("jsontext: cannot reset Decoder passed to json.UnmarshalerFrom")
	}
	// Reuse the buffer if it does not alias a previous [bytes.Buffer].
	b := d.s.buf[:0]
	if _, ok := d.s.rd.(*bytes.Buffer); ok {
		b = nil
	}
	d.s.reset(b, r, opts...)
}

func (d *decoderState) reset(b []byte, r io.Reader, opts ...Options) {
	d.state.reset()
	d.decodeBuffer = decodeBuffer{buf: b, rd: r}
	opts2 := jsonopts.Struct{} // avoid mutating d.Struct in case it is part of opts
	opts2.Join(opts...)
	d.Struct = opts2
}

// Options returns the options used to construct the encoder and
// may additionally contain semantic options passed to a
// [encoding/json/v2.UnmarshalDecode] call.
//
// If operating within
// a [encoding/json/v2.UnmarshalerFrom.UnmarshalJSONFrom] method call or
// a [encoding/json/v2.UnmarshalFromFunc] function call,
// then the returned options are only valid within the call.
func (d *Decoder) Options() Options {
	return &d.s.Struct
}

var errBufferWriteAfterNext = errors.New("invalid bytes.Buffer.Write call after calling bytes.Buffer.Next")

// fetch reads at least 1 byte from the underlying io.Reader.
// It returns io.ErrUnexpectedEOF if zero bytes were read and io.EOF was seen.
func (d *decoderState) fetch() error {
	if d.rd == nil {
		return io.ErrUnexpectedEOF
	}

	// Inform objectNameStack that we are about to fetch new buffer content.
	d.Names.copyQuotedBuffer(d.buf)

	// Specialize bytes.Buffer for better performance.
	if bb, ok := d.rd.(*bytes.Buffer); ok {
		switch {
		case bb.Len() == 0:
			return io.ErrUnexpectedEOF
		case len(d.buf) == 0:
			d.buf = bb.Next(bb.Len()) // "read" all data in the buffer
			return nil
		default:
			// This only occurs if a partially filled bytes.Buffer was provided
			// and more data is written to it while Decoder is reading from it.
			// This practice will lead to data corruption since future writes
			// may overwrite the contents of the current buffer.
			//
			// The user is trying to use a bytes.Buffer as a pipe,
			// but a bytes.Buffer is poor implementation of a pipe,
			// the purpose-built io.Pipe should be used instead.
			return &ioError{action: "read", err: errBufferWriteAfterNext}
		}
	}

	// Allocate initial buffer if empty.
	if cap(d.buf) == 0 {
		d.buf = make([]byte, 0, 64)
	}

	// Check whether to grow the buffer.
	const maxBufferSize = 4 << 10
	const growthSizeFactor = 2 // higher value is faster
	const growthRateFactor = 2 // higher value is slower
	// By default, grow if below the maximum buffer size.
	grow := cap(d.buf) <= maxBufferSize/growthSizeFactor
	// Growing can be expensive, so only grow
	// if a sufficient number of bytes have been processed.
	grow = grow && int64(cap(d.buf)) < d.previousOffsetEnd()/growthRateFactor
	// If prevStart==0, then fetch was called in order to fetch more data
	// to finish consuming a large JSON value contiguously.
	// Grow if less than 25% of the remaining capacity is available.
	// Note that this may cause the input buffer to exceed maxBufferSize.
	grow = grow || (d.prevStart == 0 && len(d.buf) >= 3*cap(d.buf)/4)

	if grow {
		// Allocate a new buffer and copy the contents of the old buffer over.
		// TODO: Provide a hard limit on the maximum internal buffer size?
		buf := make([]byte, 0, cap(d.buf)*growthSizeFactor)
		d.buf = append(buf, d.buf[d.prevStart:]...)
	} else {
		// Move unread portion of the data to the front.
		n := copy(d.buf[:cap(d.buf)], d.buf[d.prevStart:])
		d.buf = d.buf[:n]
	}
	d.baseOffset += int64(d.prevStart)
	d.prevEnd -= d.prevStart
	d.prevStart = 0

	// Read more data into the internal buffer.
	for {
		n, err := d.rd.Read(d.buf[len(d.buf):cap(d.buf)])
		switch {
		case n > 0:
			d.buf = d.buf[:len(d.buf)+n]
			return nil // ignore errors if any bytes are read
		case err == io.EOF:
			return io.ErrUnexpectedEOF
		case err != nil:
			return &ioError{action: "read", err: err}
		default:
			continue // Read returned (0, nil)
		}
	}
}

const invalidateBufferByte = '#' // invalid starting character for JSON grammar

// invalidatePreviousRead invalidates buffers returned by Peek and Read calls
// so that the first byte is an invalid character.
// This Hyrum-proofs the API against faulty application code that assumes
// values returned by ReadValue remain valid past subsequent Read calls.
func (d *decodeBuffer) invalidatePreviousRead() {
	// Avoid mutating the buffer if d.rd is nil which implies that d.buf
	// is provided by the user code and may not expect mutations.
	isBytesBuffer := func(r io.Reader) bool {
		_, ok := r.(*bytes.Buffer)
		return ok
	}
	if d.rd != nil && !isBytesBuffer(d.rd) && d.prevStart < d.prevEnd && uint(d.prevStart) < uint(len(d.buf)) {
		d.buf[d.prevStart] = invalidateBufferByte
		d.prevStart = d.prevEnd
	}
}

// needMore reports whether there are no more unread bytes.
func (d *decodeBuffer) needMore(pos int) bool {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	return pos == len(d.buf)
}

func (d *decodeBuffer) offsetAt(pos int) int64     { return d.baseOffset + int64(pos) }
func (d *decodeBuffer) previousOffsetStart() int64 { return d.baseOffset + int64(d.prevStart) }
func (d *decodeBuffer) previousOffsetEnd() int64   { return d.baseOffset + int64(d.prevEnd) }
func (d *decodeBuffer) previousBuffer() []byte     { return d.buf[d.prevStart:d.prevEnd] }
func (d *decodeBuffer) unreadBuffer() []byte       { return d.buf[d.prevEnd:len(d.buf)] }

// PreviousTokenOrValue returns the previously read token or value
// unless it has been invalidated by a call to PeekKind.
// If a token is just a delimiter, then this returns a 1-byte buffer.
// This method is used for error reporting at the semantic layer.
func (d *decodeBuffer) PreviousTokenOrValue() []byte {
	b := d.previousBuffer()
	// If peek was called, then the previous token or buffer is invalidated.
	if d.peekPos > 0 || len(b) > 0 && b[0] == invalidateBufferByte {
		return nil
	}
	// ReadToken does not preserve the buffer for null, bools, or delimiters.
	// Manually re-construct that buffer.
	if len(b) == 0 {
		b = d.buf[:d.prevEnd] // entirety of the previous buffer
		for _, tok := range []string{"null", "false", "true", "{", "}", "[", "]"} {
			if len(b) >= len(tok) && string(b[len(b)-len(tok):]) == tok {
				return b[len(b)-len(tok):]
			}
		}
	}
	return b
}

// PeekKind retrieves the next token kind, but does not advance the read offset.
//
// It returns [KindInvalid] if an error occurs. Any such error is cached until
// the next read call and it is the caller's responsibility to eventually
// follow up a PeekKind call with a read call.
func (d *Decoder) PeekKind() Kind {
	return d.s.PeekKind()
}
func (d *decoderState) PeekKind() Kind {
	// Check whether we have a cached peek result.
	if d.peekPos > 0 {
		return Kind(d.buf[d.peekPos]).normalize()
	}

	var err error
	d.invalidatePreviousRead()
	pos := d.prevEnd

	// Consume leading whitespace.
	pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
	if d.needMore(pos) {
		if pos, err = d.consumeWhitespace(pos); err != nil {
			if err == io.ErrUnexpectedEOF && d.Tokens.Depth() == 1 {
				err = io.EOF // EOF possibly if no Tokens present after top-level value
			}
			d.peekPos, d.peekErr = -1, wrapSyntacticError(d, err, pos, 0)
			return invalidKind
		}
	}

	// Consume colon or comma.
	var delim byte
	if c := d.buf[pos]; c == ':' || c == ',' {
		delim = c
		pos += 1
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				err = wrapSyntacticError(d, err, pos, 0)
				d.peekPos, d.peekErr = -1, d.checkDelimBeforeIOError(delim, err)
				return invalidKind
			}
		}
	}
	next := Kind(d.buf[pos]).normalize()
	if d.Tokens.needDelim(next) != delim {
		d.peekPos, d.peekErr = -1, d.checkDelim(delim, next)
		return invalidKind
	}

	// This may set peekPos to zero, which is indistinguishable from
	// the uninitialized state. While a small hit to performance, it is correct
	// since ReadValue and ReadToken will disregard the cached result and
	// recompute the next kind.
	d.peekPos, d.peekErr = pos, nil
	return next
}

// checkDelimBeforeIOError checks whether the delim is even valid
// before returning an IO error, which occurs after the delim.
func (d *decoderState) checkDelimBeforeIOError(delim byte, err error) error {
	// Since an IO error occurred, we do not know what the next kind is.
	// However, knowing the next kind is necessary to validate
	// whether the current delim is at least potentially valid.
	// Since a JSON string is always valid as the next token,
	// conservatively assume that is the next kind for validation.
	const next = Kind('"')
	if d.Tokens.needDelim(next) != delim {
		err = d.checkDelim(delim, next)
	}
	return err
}

// CountNextDelimWhitespace counts the number of upcoming bytes of
// delimiter or whitespace characters.
// This method is used for error reporting at the semantic layer.
func (d *decoderState) CountNextDelimWhitespace() int {
	d.PeekKind() // populate unreadBuffer
	return len(d.unreadBuffer()) - len(bytes.TrimLeft(d.unreadBuffer(), ",: \n\r\t"))
}

// checkDelim checks whether delim is valid for the given next kind.
func (d *decoderState) checkDelim(delim byte, next Kind) error {
	where := "at start of value"
	switch d.Tokens.needDelim(next) {
	case delim:
		return nil
	case ':':
		where = "after object name (expecting ':')"
	case ',':
		if d.Tokens.Last.isObject() {
			where = "after object value (expecting ',' or '}')"
		} else {
			where = "after array element (expecting ',' or ']')"
		}
	}
	pos := d.prevEnd // restore position to right after leading whitespace
	pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
	err := jsonwire.NewInvalidCharacterError(d.buf[pos:], where)
	return wrapSyntacticError(d, err, pos, 0)
}

// SkipValue is semantically equivalent to calling [Decoder.ReadValue] and discarding
// the result except that memory is not wasted trying to hold the entire result.
func (d *Decoder) SkipValue() error {
	return d.s.SkipValue()
}
func (d *decoderState) SkipValue() error {
	switch d.PeekKind() {
	case '{', '[':
		// For JSON objects and arrays, keep skipping all tokens
		// until the depth matches the starting depth.
		depth := d.Tokens.Depth()
		for {
			if _, err := d.ReadToken(); err != nil {
				return err
			}
			if depth >= d.Tokens.Depth() {
				return nil
			}
		}
	default:
		// Trying to skip a value when the next token is a '}' or ']'
		// will result in an error being returned here.
		var flags jsonwire.ValueFlags
		if _, err := d.ReadValue(&flags); err != nil {
			return err
		}
		return nil
	}
}

// SkipValueRemainder skips the remainder of a value
// after reading a '{' or '[' token.
func (d *decoderState) SkipValueRemainder() error {
	if d.Tokens.Depth()-1 > 0 && d.Tokens.Last.Length() == 0 {
		for n := d.Tokens.Depth(); d.Tokens.Depth() >= n; {
			if _, err := d.ReadToken(); err != nil {
				return err
			}
		}
	}
	return nil
}

// SkipUntil skips all tokens until the state machine
// is at or past the specified depth and length.
func (d *decoderState) SkipUntil(depth int, length int64) error {
	for d.Tokens.Depth() > depth || (d.Tokens.Depth() == depth && d.Tokens.Last.Length() < length) {
		if _, err := d.ReadToken(); err != nil {
			return err
		}
	}
	return nil
}

// ReadToken reads the next [Token], advancing the read offset.
// The returned token is only valid until the next Peek, Read, or Skip call.
// It returns [io.EOF] if there are no more tokens.
func (d *Decoder) ReadToken() (Token, error) {
	return d.s.ReadToken()
}
func (d *decoderState) ReadToken() (Token, error) {
	// Determine the next kind.
	var err error
	var next Kind
	pos := d.peekPos
	if pos != 0 {
		// Use cached peek result.
		if d.peekErr != nil {
			err := d.peekErr
			d.peekPos, d.peekErr = 0, nil // possibly a transient I/O error
			return Token{}, err
		}
		next = Kind(d.buf[pos]).normalize()
		d.peekPos = 0 // reset cache
	} else {
		d.invalidatePreviousRead()
		pos = d.prevEnd

		// Consume leading whitespace.
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				if err == io.ErrUnexpectedEOF && d.Tokens.Depth() == 1 {
					err = io.EOF // EOF possibly if no Tokens present after top-level value
				}
				return Token{}, wrapSyntacticError(d, err, pos, 0)
			}
		}

		// Consume colon or comma.
		var delim byte
		if c := d.buf[pos]; c == ':' || c == ',' {
			delim = c
			pos += 1
			pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
			if d.needMore(pos) {
				if pos, err = d.consumeWhitespace(pos); err != nil {
					err = wrapSyntacticError(d, err, pos, 0)
					return Token{}, d.checkDelimBeforeIOError(delim, err)
				}
			}
		}
		next = Kind(d.buf[pos]).normalize()
		if d.Tokens.needDelim(next) != delim {
			return Token{}, d.checkDelim(delim, next)
		}
	}

	// Handle the next token.
	var n int
	switch next {
	case 'n':
		if jsonwire.ConsumeNull(d.buf[pos:]) == 0 {
			pos, err = d.consumeLiteral(pos, "null")
			if err != nil {
				return Token{}, wrapSyntacticError(d, err, pos, +1)
			}
		} else {
			pos += len("null")
		}
		if err = d.Tokens.appendLiteral(); err != nil {
			return Token{}, wrapSyntacticError(d, err, pos-len("null"), +1) // report position at start of literal
		}
		d.prevStart, d.prevEnd = pos, pos
		return Null, nil

	case 'f':
		if jsonwire.ConsumeFalse(d.buf[pos:]) == 0 {
			pos, err = d.consumeLiteral(pos, "false")
			if err != nil {
				return Token{}, wrapSyntacticError(d, err, pos, +1)
			}
		} else {
			pos += len("false")
		}
		if err = d.Tokens.appendLiteral(); err != nil {
			return Token{}, wrapSyntacticError(d, err, pos-len("false"), +1) // report position at start of literal
		}
		d.prevStart, d.prevEnd = pos, pos
		return False, nil

	case 't':
		if jsonwire.ConsumeTrue(d.buf[pos:]) == 0 {
			pos, err = d.consumeLiteral(pos, "true")
			if err != nil {
				return Token{}, wrapSyntacticError(d, err, pos, +1)
			}
		} else {
			pos += len("true")
		}
		if err = d.Tokens.appendLiteral(); err != nil {
			return Token{}, wrapSyntacticError(d, err, pos-len("true"), +1) // report position at start of literal
		}
		d.prevStart, d.prevEnd = pos, pos
		return True, nil

	case '"':
		var flags jsonwire.ValueFlags // TODO: Preserve this in Token?
		if n = jsonwire.ConsumeSimpleString(d.buf[pos:]); n == 0 {
			oldAbsPos := d.baseOffset + int64(pos)
			pos, err = d.consumeString(&flags, pos)
			newAbsPos := d.baseOffset + int64(pos)
			n = int(newAbsPos - oldAbsPos)
			if err != nil {
				return Token{}, wrapSyntacticError(d, err, pos, +1)
			}
		} else {
			pos += n
		}
		if d.Tokens.Last.NeedObjectName() {
			if !d.Flags.Get(jsonflags.AllowDuplicateNames) {
				if !d.Tokens.Last.isValidNamespace() {
					return Token{}, wrapSyntacticError(d, errInvalidNamespace, pos-n, +1)
				}
				if d.Tokens.Last.isActiveNamespace() && !d.Namespaces.Last().insertQuoted(d.buf[pos-n:pos], flags.IsVerbatim()) {
					err = wrapWithObjectName(ErrDuplicateName, d.buf[pos-n:pos])
					return Token{}, wrapSyntacticError(d, err, pos-n, +1) // report position at start of string
				}
			}
			d.Names.ReplaceLastQuotedOffset(pos - n) // only replace if insertQuoted succeeds
		}
		if err = d.Tokens.appendString(); err != nil {
			return Token{}, wrapSyntacticError(d, err, pos-n, +1) // report position at start of string
		}
		d.prevStart, d.prevEnd = pos-n, pos
		return Token{raw: &d.decodeBuffer, num: uint64(d.previousOffsetStart())}, nil

	case '0':
		// NOTE: Since JSON numbers are not self-terminating,
		// we need to make sure that the next byte is not part of a number.
		if n = jsonwire.ConsumeSimpleNumber(d.buf[pos:]); n == 0 || d.needMore(pos+n) {
			oldAbsPos := d.baseOffset + int64(pos)
			pos, err = d.consumeNumber(pos)
			newAbsPos := d.baseOffset + int64(pos)
			n = int(newAbsPos - oldAbsPos)
			if err != nil {
				return Token{}, wrapSyntacticError(d, err, pos, +1)
			}
		} else {
			pos += n
		}
		if err = d.Tokens.appendNumber(); err != nil {
			return Token{}, wrapSyntacticError(d, err, pos-n, +1) // report position at start of number
		}
		d.prevStart, d.prevEnd = pos-n, pos
		return Token{raw: &d.decodeBuffer, num: uint64(d.previousOffsetStart())}, nil

	case '{':
		if err = d.Tokens.pushObject(); err != nil {
			return Token{}, wrapSyntacticError(d, err, pos, +1)
		}
		d.Names.push()
		if !d.Flags.Get(jsonflags.AllowDuplicateNames) {
			d.Namespaces.push()
		}
		pos += 1
		d.prevStart, d.prevEnd = pos, pos
		return BeginObject, nil

	case '}':
		if err = d.Tokens.popObject(); err != nil {
			return Token{}, wrapSyntacticError(d, err, pos, +1)
		}
		d.Names.pop()
		if !d.Flags.Get(jsonflags.AllowDuplicateNames) {
			d.Namespaces.pop()
		}
		pos += 1
		d.prevStart, d.prevEnd = pos, pos
		return EndObject, nil

	case '[':
		if err = d.Tokens.pushArray(); err != nil {
			return Token{}, wrapSyntacticError(d, err, pos, +1)
		}
		pos += 1
		d.prevStart, d.prevEnd = pos, pos
		return BeginArray, nil

	case ']':
		if err = d.Tokens.popArray(); err != nil {
			return Token{}, wrapSyntacticError(d, err, pos, +1)
		}
		pos += 1
		d.prevStart, d.prevEnd = pos, pos
		return EndArray, nil

	default:
		err = jsonwire.NewInvalidCharacterError(d.buf[pos:], "at start of value")
		return Token{}, wrapSyntacticError(d, err, pos, +1)
	}
}

// ReadValue returns the next raw JSON value, advancing the read offset.
// The value is stripped of any leading or trailing whitespace and
// contains the exact bytes of the input, which may contain invalid UTF-8
// if [AllowInvalidUTF8] is specified.
//
// The returned value is only valid until the next Peek, Read, or Skip call and
// may not be mutated while the Decoder remains in use.
// If the decoder is currently at the end token for an object or array,
// then it reports a [SyntacticError] and the internal state remains unchanged.
// It returns [io.EOF] if there are no more values.
func (d *Decoder) ReadValue() (Value, error) {
	var flags jsonwire.ValueFlags
	return d.s.ReadValue(&flags)
}
func (d *decoderState) ReadValue(flags *jsonwire.ValueFlags) (Value, error) {
	// Determine the next kind.
	var err error
	var next Kind
	pos := d.peekPos
	if pos != 0 {
		// Use cached peek result.
		if d.peekErr != nil {
			err := d.peekErr
			d.peekPos, d.peekErr = 0, nil // possibly a transient I/O error
			return nil, err
		}
		next = Kind(d.buf[pos]).normalize()
		d.peekPos = 0 // reset cache
	} else {
		d.invalidatePreviousRead()
		pos = d.prevEnd

		// Consume leading whitespace.
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				if err == io.ErrUnexpectedEOF && d.Tokens.Depth() == 1 {
					err = io.EOF // EOF possibly if no Tokens present after top-level value
				}
				return nil, wrapSyntacticError(d, err, pos, 0)
			}
		}

		// Consume colon or comma.
		var delim byte
		if c := d.buf[pos]; c == ':' || c == ',' {
			delim = c
			pos += 1
			pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
			if d.needMore(pos) {
				if pos, err = d.consumeWhitespace(pos); err != nil {
					err = wrapSyntacticError(d, err, pos, 0)
					return nil, d.checkDelimBeforeIOError(delim, err)
				}
			}
		}
		next = Kind(d.buf[pos]).normalize()
		if d.Tokens.needDelim(next) != delim {
			return nil, d.checkDelim(delim, next)
		}
	}

	// Handle the next value.
	oldAbsPos := d.baseOffset + int64(pos)
	pos, err = d.consumeValue(flags, pos, d.Tokens.Depth())
	newAbsPos := d.baseOffset + int64(pos)
	n := int(newAbsPos - oldAbsPos)
	if err != nil {
		return nil, wrapSyntacticError(d, err, pos, +1)
	}
	switch next {
	case 'n', 't', 'f':
		err = d.Tokens.appendLiteral()
	case '"':
		if d.Tokens.Last.NeedObjectName() {
			if !d.Flags.Get(jsonflags.AllowDuplicateNames) {
				if !d.Tokens.Last.isValidNamespace() {
					err = errInvalidNamespace
					break
				}
				if d.Tokens.Last.isActiveNamespace() && !d.Namespaces.Last().insertQuoted(d.buf[pos-n:pos], flags.IsVerbatim()) {
					err = wrapWithObjectName(ErrDuplicateName, d.buf[pos-n:pos])
					break
				}
			}
			d.Names.ReplaceLastQuotedOffset(pos - n) // only replace if insertQuoted succeeds
		}
		err = d.Tokens.appendString()
	case '0':
		err = d.Tokens.appendNumber()
	case '{':
		if err = d.Tokens.pushObject(); err != nil {
			break
		}
		if err = d.Tokens.popObject(); err != nil {
			panic("BUG: popObject should never fail immediately after pushObject: " + err.Error())
		}
	case '[':
		if err = d.Tokens.pushArray(); err != nil {
			break
		}
		if err = d.Tokens.popArray(); err != nil {
			panic("BUG: popArray should never fail immediately after pushArray: " + err.Error())
		}
	}
	if err != nil {
		return nil, wrapSyntacticError(d, err, pos-n, +1) // report position at start of value
	}
	d.prevEnd = pos
	d.prevStart = pos - n
	return d.buf[pos-n : pos : pos], nil
}

// CheckNextValue checks whether the next value is syntactically valid,
// but does not advance the read offset.
// If last, it verifies that the stream cleanly terminates with [io.EOF].
func (d *decoderState) CheckNextValue(last bool) error {
	d.PeekKind() // populates d.peekPos and d.peekErr
	pos, err := d.peekPos, d.peekErr
	d.peekPos, d.peekErr = 0, nil
	if err != nil {
		return err
	}

	var flags jsonwire.ValueFlags
	if pos, err := d.consumeValue(&flags, pos, d.Tokens.Depth()); err != nil {
		return wrapSyntacticError(d, err, pos, +1)
	} else if last {
		return d.checkEOF(pos)
	}
	return nil
}

// AtEOF reports whether the decoder is at EOF.
func (d *decoderState) AtEOF() bool {
	_, err := d.consumeWhitespace(d.prevEnd)
	return err == io.ErrUnexpectedEOF
}

// CheckEOF verifies that the input has no more data.
func (d *decoderState) CheckEOF() error {
	return d.checkEOF(d.prevEnd)
}
func (d *decoderState) checkEOF(pos int) error {
	switch pos, err := d.consumeWhitespace(pos); err {
	case nil:
		err := jsonwire.NewInvalidCharacterError(d.buf[pos:], "after top-level value")
		return wrapSyntacticError(d, err, pos, 0)
	case io.ErrUnexpectedEOF:
		return nil
	default:
		return err
	}
}

// consumeWhitespace consumes all whitespace starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the last whitespace.
// If it returns nil, there is guaranteed to at least be one unread byte.
//
// The following pattern is common in this implementation:
//
//	pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
//	if d.needMore(pos) {
//		if pos, err = d.consumeWhitespace(pos); err != nil {
//			return ...
//		}
//	}
//
// It is difficult to simplify this without sacrificing performance since
// consumeWhitespace must be inlined. The body of the if statement is
// executed only in rare situations where we need to fetch more data.
// Since fetching may return an error, we also need to check the error.
func (d *decoderState) consumeWhitespace(pos int) (newPos int, err error) {
	for {
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				return pos, err
			}
			continue
		}
		return pos, nil
	}
}

// consumeValue consumes a single JSON value starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the value.
func (d *decoderState) consumeValue(flags *jsonwire.ValueFlags, pos, depth int) (newPos int, err error) {
	for {
		var n int
		var err error
		switch next := Kind(d.buf[pos]).normalize(); next {
		case 'n':
			if n = jsonwire.ConsumeNull(d.buf[pos:]); n == 0 {
				n, err = jsonwire.ConsumeLiteral(d.buf[pos:], "null")
			}
		case 'f':
			if n = jsonwire.ConsumeFalse(d.buf[pos:]); n == 0 {
				n, err = jsonwire.ConsumeLiteral(d.buf[pos:], "false")
			}
		case 't':
			if n = jsonwire.ConsumeTrue(d.buf[pos:]); n == 0 {
				n, err = jsonwire.ConsumeLiteral(d.buf[pos:], "true")
			}
		case '"':
			if n = jsonwire.ConsumeSimpleString(d.buf[pos:]); n == 0 {
				return d.consumeString(flags, pos)
			}
		case '0':
			// NOTE: Since JSON numbers are not self-terminating,
			// we need to make sure that the next byte is not part of a number.
			if n = jsonwire.ConsumeSimpleNumber(d.buf[pos:]); n == 0 || d.needMore(pos+n) {
				return d.consumeNumber(pos)
			}
		case '{':
			return d.consumeObject(flags, pos, depth)
		case '[':
			return d.consumeArray(flags, pos, depth)
		default:
			if (d.Tokens.Last.isObject() && next == ']') || (d.Tokens.Last.isArray() && next == '}') {
				return pos, errMismatchDelim
			}
			return pos, jsonwire.NewInvalidCharacterError(d.buf[pos:], "at start of value")
		}
		if err == io.ErrUnexpectedEOF {
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				return pos + n, err
			}
			continue
		}
		return pos + n, err
	}
}

// consumeLiteral consumes a single JSON literal starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the literal.
func (d *decoderState) consumeLiteral(pos int, lit string) (newPos int, err error) {
	for {
		n, err := jsonwire.ConsumeLiteral(d.buf[pos:], lit)
		if err == io.ErrUnexpectedEOF {
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				return pos + n, err
			}
			continue
		}
		return pos + n, err
	}
}

// consumeString consumes a single JSON string starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the string.
func (d *decoderState) consumeString(flags *jsonwire.ValueFlags, pos int) (newPos int, err error) {
	var n int
	for {
		n, err = jsonwire.ConsumeStringResumable(flags, d.buf[pos:], n, !d.Flags.Get(jsonflags.AllowInvalidUTF8))
		if err == io.ErrUnexpectedEOF {
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				return pos + n, err
			}
			continue
		}
		return pos + n, err
	}
}

// consumeNumber consumes a single JSON number starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the number.
func (d *decoderState) consumeNumber(pos int) (newPos int, err error) {
	var n int
	var state jsonwire.ConsumeNumberState
	for {
		n, state, err = jsonwire.ConsumeNumberResumable(d.buf[pos:], n, state)
		// NOTE: Since JSON numbers are not self-terminating,
		// we need to make sure that the next byte is not part of a number.
		if err == io.ErrUnexpectedEOF || d.needMore(pos+n) {
			mayTerminate := err == nil
			absPos := d.baseOffset + int64(pos)
			err = d.fetch() // will mutate d.buf and invalidate pos
			pos = int(absPos - d.baseOffset)
			if err != nil {
				if mayTerminate && err == io.ErrUnexpectedEOF {
					return pos + n, nil
				}
				return pos, err
			}
			continue
		}
		return pos + n, err
	}
}

// consumeObject consumes a single JSON object starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the object.
func (d *decoderState) consumeObject(flags *jsonwire.ValueFlags, pos, depth int) (newPos int, err error) {
	var n int
	var names *objectNamespace
	if !d.Flags.Get(jsonflags.AllowDuplicateNames) {
		d.Namespaces.push()
		defer d.Namespaces.pop()
		names = d.Namespaces.Last()
	}

	// Handle before start.
	if uint(pos) >= uint(len(d.buf)) || d.buf[pos] != '{' {
		panic("BUG: consumeObject must be called with a buffer that starts with '{'")
	} else if depth == maxNestingDepth+1 {
		return pos, errMaxDepth
	}
	pos++

	// Handle after start.
	pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
	if d.needMore(pos) {
		if pos, err = d.consumeWhitespace(pos); err != nil {
			return pos, err
		}
	}
	if d.buf[pos] == '}' {
		pos++
		return pos, nil
	}

	depth++
	for {
		// Handle before name.
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		var flags2 jsonwire.ValueFlags
		if n = jsonwire.ConsumeSimpleString(d.buf[pos:]); n == 0 {
			oldAbsPos := d.baseOffset + int64(pos)
			pos, err = d.consumeString(&flags2, pos)
			newAbsPos := d.baseOffset + int64(pos)
			n = int(newAbsPos - oldAbsPos)
			flags.Join(flags2)
			if err != nil {
				return pos, err
			}
		} else {
			pos += n
		}
		quotedName := d.buf[pos-n : pos]
		if !d.Flags.Get(jsonflags.AllowDuplicateNames) && !names.insertQuoted(quotedName, flags2.IsVerbatim()) {
			return pos - n, wrapWithObjectName(ErrDuplicateName, quotedName)
		}

		// Handle after name.
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, wrapWithObjectName(err, quotedName)
			}
		}
		if d.buf[pos] != ':' {
			err := jsonwire.NewInvalidCharacterError(d.buf[pos:], "after object name (expecting ':')")
			return pos, wrapWithObjectName(err, quotedName)
		}
		pos++

		// Handle before value.
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, wrapWithObjectName(err, quotedName)
			}
		}
		pos, err = d.consumeValue(flags, pos, depth)
		if err != nil {
			return pos, wrapWithObjectName(err, quotedName)
		}

		// Handle after value.
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		switch d.buf[pos] {
		case ',':
			pos++
			continue
		case '}':
			pos++
			return pos, nil
		default:
			return pos, jsonwire.NewInvalidCharacterError(d.buf[pos:], "after object value (expecting ',' or '}')")
		}
	}
}

// consumeArray consumes a single JSON array starting at d.buf[pos:].
// It returns the new position in d.buf immediately after the array.
func (d *decoderState) consumeArray(flags *jsonwire.ValueFlags, pos, depth int) (newPos int, err error) {
	// Handle before start.
	if uint(pos) >= uint(len(d.buf)) || d.buf[pos] != '[' {
		panic("BUG: consumeArray must be called with a buffer that starts with '['")
	} else if depth == maxNestingDepth+1 {
		return pos, errMaxDepth
	}
	pos++

	// Handle after start.
	pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
	if d.needMore(pos) {
		if pos, err = d.consumeWhitespace(pos); err != nil {
			return pos, err
		}
	}
	if d.buf[pos] == ']' {
		pos++
		return pos, nil
	}

	var idx int64
	depth++
	for {
		// Handle before value.
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		pos, err = d.consumeValue(flags, pos, depth)
		if err != nil {
			return pos, wrapWithArrayIndex(err, idx)
		}

		// Handle after value.
		pos += jsonwire.ConsumeWhitespace(d.buf[pos:])
		if d.needMore(pos) {
			if pos, err = d.consumeWhitespace(pos); err != nil {
				return pos, err
			}
		}
		switch d.buf[pos] {
		case ',':
			pos++
			idx++
			continue
		case ']':
			pos++
			return pos, nil
		default:
			return pos, jsonwire.NewInvalidCharacterError(d.buf[pos:], "after array element (expecting ',' or ']')")
		}
	}
}

// InputOffset returns the current input byte offset. It gives the location
// of the next byte immediately after the most recently returned token or value.
// The number of bytes actually read from the underlying [io.Reader] may be more
// than this offset due to internal buffering effects.
func (d *Decoder) InputOffset() int64 {
	return d.s.previousOffsetEnd()
}

// UnreadBuffer returns the data remaining in the unread buffer,
// which may contain zero or more bytes.
// The returned buffer must not be mutated while Decoder continues to be used.
// The buffer contents are valid until the next Peek, Read, or Skip call.
func (d *Decoder) UnreadBuffer() []byte {
	return d.s.unreadBuffer()
}

// StackDepth returns the depth of the state machine for read JSON data.
// Each level on the stack represents a nested JSON object or array.
// It is incremented whenever an [BeginObject] or [BeginArray] token is encountered
// and decremented whenever an [EndObject] or [EndArray] token is encountered.
// The depth is zero-indexed, where zero represents the top-level JSON value.
func (d *Decoder) StackDepth() int {
	// NOTE: Keep in sync with Encoder.StackDepth.
	return d.s.Tokens.Depth() - 1
}

// StackIndex returns information about the specified stack level.
// It must be a number between 0 and [Decoder.StackDepth], inclusive.
// For each level, it reports the kind:
//
//   - [KindInvalid] for a level of zero,
//   - [KindBeginObject] for a level representing a JSON object, and
//   - [KindBeginArray] for a level representing a JSON array.
//
// It also reports the length of that JSON object or array.
// Each name and value in a JSON object is counted separately,
// so the effective number of members would be half the length.
// A complete JSON object must have an even length.
func (d *Decoder) StackIndex(i int) (Kind, int64) {
	// NOTE: Keep in sync with Encoder.StackIndex.
	switch s := d.s.Tokens.index(i); {
	case i > 0 && s.isObject():
		return '{', s.Length()
	case i > 0 && s.isArray():
		return '[', s.Length()
	default:
		return 0, s.Length()
	}
}

// StackPointer returns a JSON Pointer (RFC 6901) to the most recently read value.
func (d *Decoder) StackPointer() Pointer {
	return Pointer(d.s.AppendStackPointer(nil, -1))
}

func (d *decoderState) AppendStackPointer(b []byte, where int) []byte {
	d.Names.copyQuotedBuffer(d.buf)
	return d.state.appendStackPointer(b, where)
}
