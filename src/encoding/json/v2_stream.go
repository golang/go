// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"bytes"
	"errors"
	"io"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
)

// A Decoder reads and decodes JSON values from an input stream.
type Decoder struct {
	dec  *jsontext.Decoder
	opts jsonv2.Options
	err  error

	// hadPeeked reports whether [Decoder.More] was called.
	// It is reset by [Decoder.Decode] and [Decoder.Token].
	hadPeeked bool
}

// NewDecoder returns a new decoder that reads from r.
//
// The decoder introduces its own buffering and may
// read data from r beyond the JSON values requested.
func NewDecoder(r io.Reader) *Decoder {
	// Hide bytes.Buffer from jsontext since it implements optimizations that
	// also limits certain ways it could be used. For example, one cannot write
	// to the bytes.Buffer while it is in use by jsontext.Decoder.
	if _, ok := r.(*bytes.Buffer); ok {
		r = struct{ io.Reader }{r}
	}

	dec := new(Decoder)
	dec.opts = DefaultOptionsV1()
	dec.dec = jsontext.NewDecoder(r, dec.opts)
	return dec
}

// UseNumber causes the Decoder to unmarshal a number into an
// interface value as a [Number] instead of as a float64.
func (dec *Decoder) UseNumber() {
	if useNumber, _ := jsonv2.GetOption(dec.opts, unmarshalAnyWithRawNumber); !useNumber {
		dec.opts = jsonv2.JoinOptions(dec.opts, unmarshalAnyWithRawNumber(true))
	}
}

// DisallowUnknownFields causes the Decoder to return an error when the destination
// is a struct and the input contains object keys which do not match any
// non-ignored, exported fields in the destination.
func (dec *Decoder) DisallowUnknownFields() {
	if reject, _ := jsonv2.GetOption(dec.opts, jsonv2.RejectUnknownMembers); !reject {
		dec.opts = jsonv2.JoinOptions(dec.opts, jsonv2.RejectUnknownMembers(true))
	}
}

// Decode reads the next JSON-encoded value from its
// input and stores it in the value pointed to by v.
//
// See the documentation for [Unmarshal] for details about
// the conversion of JSON into a Go value.
func (dec *Decoder) Decode(v any) error {
	if dec.err != nil {
		return dec.err
	}
	b, err := dec.dec.ReadValue()
	if err != nil {
		dec.err = transformSyntacticError(err)
		if dec.err.Error() == errUnexpectedEnd.Error() {
			// NOTE: Decode has always been inconsistent with Unmarshal
			// with regard to the exact error value for truncated input.
			dec.err = io.ErrUnexpectedEOF
		}
		return dec.err
	}
	dec.hadPeeked = false
	return jsonv2.Unmarshal(b, v, dec.opts)
}

// Buffered returns a reader of the data remaining in the Decoder's
// buffer. The reader is valid until the next call to [Decoder.Decode].
func (dec *Decoder) Buffered() io.Reader {
	return bytes.NewReader(dec.dec.UnreadBuffer())
}

// An Encoder writes JSON values to an output stream.
type Encoder struct {
	w    io.Writer
	opts jsonv2.Options
	err  error

	buf       bytes.Buffer
	indentBuf bytes.Buffer

	indentPrefix string
	indentValue  string
}

// NewEncoder returns a new encoder that writes to w.
func NewEncoder(w io.Writer) *Encoder {
	enc := new(Encoder)
	enc.w = w
	enc.opts = DefaultOptionsV1()
	return enc
}

// Encode writes the JSON encoding of v to the stream,
// followed by a newline character.
//
// See the documentation for [Marshal] for details about the
// conversion of Go values to JSON.
func (enc *Encoder) Encode(v any) error {
	if enc.err != nil {
		return enc.err
	}

	buf := &enc.buf
	buf.Reset()
	if err := jsonv2.MarshalWrite(buf, v, enc.opts); err != nil {
		return err
	}
	if len(enc.indentPrefix)+len(enc.indentValue) > 0 {
		enc.indentBuf.Reset()
		if err := Indent(&enc.indentBuf, buf.Bytes(), enc.indentPrefix, enc.indentValue); err != nil {
			return err
		}
		buf = &enc.indentBuf
	}
	buf.WriteByte('\n')

	if _, err := enc.w.Write(buf.Bytes()); err != nil {
		enc.err = err
		return err
	}
	return nil
}

// SetIndent instructs the encoder to format each subsequent encoded
// value as if indented by the package-level function Indent(dst, src, prefix, indent).
// Calling SetIndent("", "") disables indentation.
func (enc *Encoder) SetIndent(prefix, indent string) {
	enc.indentPrefix = prefix
	enc.indentValue = indent
}

// SetEscapeHTML specifies whether problematic HTML characters
// should be escaped inside JSON quoted strings.
// The default behavior is to escape &, <, and > to \u0026, \u003c, and \u003e
// to avoid certain safety problems that can arise when embedding JSON in HTML.
//
// In non-HTML settings where the escaping interferes with the readability
// of the output, SetEscapeHTML(false) disables this behavior.
func (enc *Encoder) SetEscapeHTML(on bool) {
	if escape, _ := jsonv2.GetOption(enc.opts, jsontext.EscapeForHTML); escape != on {
		enc.opts = jsonv2.JoinOptions(enc.opts, jsontext.EscapeForHTML(on))
	}
}

// RawMessage is a raw encoded JSON value.
// It implements [Marshaler] and [Unmarshaler] and can
// be used to delay JSON decoding or precompute a JSON encoding.
type RawMessage = jsontext.Value

// A Token holds a value of one of these types:
//
//   - [Delim], for the four JSON delimiters [ ] { }
//   - bool, for JSON booleans
//   - float64, for JSON numbers
//   - [Number], for JSON numbers
//   - string, for JSON string literals
//   - nil, for JSON null
type Token any

// A Delim is a JSON array or object delimiter, one of [ ] { or }.
type Delim rune

func (d Delim) String() string {
	return string(d)
}

// Token returns the next JSON token in the input stream.
// At the end of the input stream, Token returns nil, [io.EOF].
//
// Token guarantees that the delimiters [ ] { } it returns are
// properly nested and matched: if Token encounters an unexpected
// delimiter in the input, it will return an error.
//
// The input stream consists of basic JSON values—bool, string,
// number, and null—along with delimiters [ ] { } of type [Delim]
// to mark the start and end of arrays and objects.
// Commas and colons are elided.
func (dec *Decoder) Token() (Token, error) {
	if dec.err != nil {
		return nil, dec.err
	}
	tok, err := dec.dec.ReadToken()
	if err != nil {
		// Historically, v1 would report just [io.EOF] if
		// the stream is a prefix of a valid JSON value.
		// It reports an unwrapped [io.ErrUnexpectedEOF] if
		// truncated within a JSON token such as a literal, number, or string.
		if errors.Is(err, io.ErrUnexpectedEOF) {
			if len(bytes.Trim(dec.dec.UnreadBuffer(), " \r\n\t,:")) == 0 {
				return nil, io.EOF
			}
			return nil, io.ErrUnexpectedEOF
		}
		return nil, transformSyntacticError(err)
	}
	dec.hadPeeked = false
	switch k := tok.Kind(); k {
	case 'n':
		return nil, nil
	case 'f':
		return false, nil
	case 't':
		return true, nil
	case '"':
		return tok.String(), nil
	case '0':
		if useNumber, _ := jsonv2.GetOption(dec.opts, unmarshalAnyWithRawNumber); useNumber {
			return Number(tok.String()), nil
		}
		return tok.Float(), nil
	case '{', '}', '[', ']':
		return Delim(k), nil
	default:
		panic("unreachable")
	}
}

// More reports whether there is another element in the
// current array or object being parsed.
func (dec *Decoder) More() bool {
	dec.hadPeeked = true
	k := dec.dec.PeekKind()
	if k == 0 {
		if dec.err == nil {
			// PeekKind doesn't distinguish between EOF and error,
			// so read the next token to see which we get.
			_, err := dec.dec.ReadToken()
			if err == nil {
				// This is only possible if jsontext violates its documentation.
				err = errors.New("json: successful read after failed peek")
			}
			dec.err = transformSyntacticError(err)
		}
		return dec.err != io.EOF
	}
	return k != ']' && k != '}'
}

// InputOffset returns the input stream byte offset of the current decoder position.
// The offset gives the location of the end of the most recently returned token
// and the beginning of the next token.
func (dec *Decoder) InputOffset() int64 {
	offset := dec.dec.InputOffset()
	if dec.hadPeeked {
		// Historically, InputOffset reported the location of
		// the end of the most recently returned token
		// unless [Decoder.More] is called, in which case, it reported
		// the beginning of the next token.
		unreadBuffer := dec.dec.UnreadBuffer()
		trailingTokens := bytes.TrimLeft(unreadBuffer, " \n\r\t")
		if len(trailingTokens) > 0 {
			leadingWhitespace := len(unreadBuffer) - len(trailingTokens)
			offset += int64(leadingWhitespace)
		}
	}
	return offset
}
