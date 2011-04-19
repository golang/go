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

func (s StructuralError) String() string {
	return "OpenPGP data invalid: " + string(s)
}

// UnsupportedError indicates that, although the OpenPGP data is valid, it
// makes use of currently unimplemented features.
type UnsupportedError string

func (s UnsupportedError) String() string {
	return "OpenPGP feature unsupported: " + string(s)
}

// InvalidArgumentError indicates that the caller is in error and passed an
// incorrect value.
type InvalidArgumentError string

func (i InvalidArgumentError) String() string {
	return "OpenPGP argument invalid: " + string(i)
}

// SignatureError indicates that a syntactically valid signature failed to
// validate.
type SignatureError string

func (b SignatureError) String() string {
	return "OpenPGP signature invalid: " + string(b)
}

type keyIncorrect int

func (ki keyIncorrect) String() string {
	return "the given key was incorrect"
}

var KeyIncorrectError = keyIncorrect(0)

type unknownIssuer int

func (unknownIssuer) String() string {
	return "signature make by unknown entity"
}

var UnknownIssuerError = unknownIssuer(0)

type UnknownPacketTypeError uint8

func (upte UnknownPacketTypeError) String() string {
	return "unknown OpenPGP packet type: " + strconv.Itoa(int(upte))
}
