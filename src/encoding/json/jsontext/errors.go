// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"bytes"
	"io"
	"strconv"

	"encoding/json/internal/jsonwire"
)

const errorPrefix = "jsontext: "

type ioError struct {
	action string // either "read" or "write"
	err    error
}

func (e *ioError) Error() string {
	return errorPrefix + e.action + " error: " + e.err.Error()
}
func (e *ioError) Unwrap() error {
	return e.err
}

// SyntacticError is a description of a syntactic error that occurred when
// encoding or decoding JSON according to the grammar.
//
// The contents of this error as produced by this package may change over time.
type SyntacticError struct {
	requireKeyedLiterals
	nonComparable

	// ByteOffset indicates that an error occurred after this byte offset.
	ByteOffset int64
	// JSONPointer indicates that an error occurred within this JSON value
	// as indicated using the JSON Pointer notation (see RFC 6901).
	JSONPointer Pointer

	// Err is the underlying error.
	Err error
}

// wrapSyntacticError wraps an error and annotates it with a precise location
// using the provided [encoderState] or [decoderState].
// If err is an [ioError] or [io.EOF], then it is not wrapped.
//
// It takes a relative offset pos that can be resolved into
// an absolute offset using state.offsetAt.
//
// It takes a where that specify how the JSON pointer is derived.
// If the underlying error is a [pointerSuffixError],
// then the suffix is appended to the derived pointer.
func wrapSyntacticError(state interface {
	offsetAt(pos int) int64
	AppendStackPointer(b []byte, where int) []byte
}, err error, pos, where int) error {
	if _, ok := err.(*ioError); err == io.EOF || ok {
		return err
	}
	offset := state.offsetAt(pos)
	ptr := state.AppendStackPointer(nil, where)
	if serr, ok := err.(*pointerSuffixError); ok {
		ptr = serr.appendPointer(ptr)
		err = serr.error
	}
	if d, ok := state.(*decoderState); ok && err == errMismatchDelim {
		where := "at start of value"
		if len(d.Tokens.Stack) > 0 && d.Tokens.Last.Length() > 0 {
			switch {
			case d.Tokens.Last.isArray():
				where = "after array element (expecting ',' or ']')"
				ptr = []byte(Pointer(ptr).Parent()) // problem is with parent array
			case d.Tokens.Last.isObject():
				where = "after object value (expecting ',' or '}')"
				ptr = []byte(Pointer(ptr).Parent()) // problem is with parent object
			}
		}
		err = jsonwire.NewInvalidCharacterError(d.buf[pos:], where)
	}
	return &SyntacticError{ByteOffset: offset, JSONPointer: Pointer(ptr), Err: err}
}

func (e *SyntacticError) Error() string {
	pointer := e.JSONPointer
	offset := e.ByteOffset
	b := []byte(errorPrefix)
	if e.Err != nil {
		b = append(b, e.Err.Error()...)
		if e.Err == ErrDuplicateName {
			b = strconv.AppendQuote(append(b, ' '), pointer.LastToken())
			pointer = pointer.Parent()
			offset = 0 // not useful to print offset for duplicate names
		}
	} else {
		b = append(b, "syntactic error"...)
	}
	if pointer != "" {
		b = strconv.AppendQuote(append(b, " within "...), jsonwire.TruncatePointer(string(pointer), 100))
	}
	if offset > 0 {
		b = strconv.AppendInt(append(b, " after offset "...), offset, 10)
	}
	return string(b)
}

func (e *SyntacticError) Unwrap() error {
	return e.Err
}

// pointerSuffixError represents a JSON pointer suffix to be appended
// to [SyntacticError.JSONPointer]. It is an internal error type
// used within this package and does not appear in the public API.
//
// This type is primarily used to annotate errors in Encoder.WriteValue
// and Decoder.ReadValue with precise positions.
// At the time WriteValue or ReadValue is called, a JSON pointer to the
// upcoming value can be constructed using the Encoder/Decoder state.
// However, tracking pointers within values during normal operation
// would incur a performance penalty in the error-free case.
//
// To provide precise error locations without this overhead,
// the error is wrapped with object names or array indices
// as the call stack is popped when an error occurs.
// Since this happens in reverse order, pointerSuffixError holds
// the pointer in reverse and is only later reversed when appending to
// the pointer prefix.
//
// For example, if the encoder is at "/alpha/bravo/charlie"
// and an error occurs in WriteValue at "/xray/yankee/zulu", then
// the final pointer should be "/alpha/bravo/charlie/xray/yankee/zulu".
//
// As pointerSuffixError is populated during the error return path,
// it first contains "/zulu", then "/zulu/yankee",
// and finally "/zulu/yankee/xray".
// These tokens are reversed and concatenated to "/alpha/bravo/charlie"
// to form the full pointer.
type pointerSuffixError struct {
	error

	// reversePointer is a JSON pointer, but with each token in reverse order.
	reversePointer []byte
}

// wrapWithObjectName wraps err with a JSON object name access,
// which must be a valid quoted JSON string.
func wrapWithObjectName(err error, quotedName []byte) error {
	serr, _ := err.(*pointerSuffixError)
	if serr == nil {
		serr = &pointerSuffixError{error: err}
	}
	name := jsonwire.UnquoteMayCopy(quotedName, false)
	serr.reversePointer = appendEscapePointerName(append(serr.reversePointer, '/'), name)
	return serr
}

// wrapWithArrayIndex wraps err with a JSON array index access.
func wrapWithArrayIndex(err error, index int64) error {
	serr, _ := err.(*pointerSuffixError)
	if serr == nil {
		serr = &pointerSuffixError{error: err}
	}
	serr.reversePointer = strconv.AppendUint(append(serr.reversePointer, '/'), uint64(index), 10)
	return serr
}

// appendPointer appends the path encoded in e to the end of pointer.
func (e *pointerSuffixError) appendPointer(pointer []byte) []byte {
	// Copy each token in reversePointer to the end of pointer in reverse order.
	// Double reversal means that the appended suffix is now in forward order.
	bi, bo := e.reversePointer, pointer
	for len(bi) > 0 {
		i := bytes.LastIndexByte(bi, '/')
		bi, bo = bi[:i], append(bo, bi[i:]...)
	}
	return bo
}
