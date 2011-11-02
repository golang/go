// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package error contains common error types for the OpenPGP packages.
package error

import (
	"strconv"
)

// A StructuralError is returned when OpenPGP data is found to be syntactically
// invalid.
type StructuralError string

func (s StructuralError) Error() string {
	return "OpenPGP data invalid: " + string(s)
}

// UnsupportedError indicates that, although the OpenPGP data is valid, it
// makes use of currently unimplemented features.
type UnsupportedError string

func (s UnsupportedError) Error() string {
	return "OpenPGP feature unsupported: " + string(s)
}

// InvalidArgumentError indicates that the caller is in error and passed an
// incorrect value.
type InvalidArgumentError string

func (i InvalidArgumentError) Error() string {
	return "OpenPGP argument invalid: " + string(i)
}

// SignatureError indicates that a syntactically valid signature failed to
// validate.
type SignatureError string

func (b SignatureError) Error() string {
	return "OpenPGP signature invalid: " + string(b)
}

type keyIncorrectError int

func (ki keyIncorrectError) Error() string {
	return "the given key was incorrect"
}

var KeyIncorrectError = keyIncorrectError(0)

type unknownIssuerError int

func (unknownIssuerError) Error() string {
	return "signature make by unknown entity"
}

var UnknownIssuerError = unknownIssuerError(0)

type UnknownPacketTypeError uint8

func (upte UnknownPacketTypeError) Error() string {
	return "unknown OpenPGP packet type: " + strconv.Itoa(int(upte))
}
