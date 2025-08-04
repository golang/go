// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package internal

import "errors"

// NotForPublicUse is a marker type that an API is for internal use only.
// It does not perfectly prevent usage of that API, but helps to restrict usage.
// Anything with this marker is not covered by the Go compatibility agreement.
type NotForPublicUse struct{}

// AllowInternalUse is passed from "json" to "jsontext" to authenticate
// that the caller can have access to internal functionality.
var AllowInternalUse NotForPublicUse

// Sentinel error values internally shared between jsonv1 and jsonv2.
var (
	ErrCycle           = errors.New("encountered a cycle")
	ErrNonNilReference = errors.New("value must be passed as a non-nil pointer reference")
	ErrNilInterface    = errors.New("cannot derive concrete type for nil interface with finite type set")
)

var (
	// TransformMarshalError converts a v2 error into a v1 error.
	// It is called only at the top-level of a Marshal function.
	TransformMarshalError func(any, error) error
	// NewMarshalerError constructs a jsonv1.MarshalerError.
	// It is called after a user-defined Marshal method/function fails.
	NewMarshalerError func(any, error, string) error
	// TransformUnmarshalError converts a v2 error into a v1 error.
	// It is called only at the top-level of a Unmarshal function.
	TransformUnmarshalError func(any, error) error

	// NewRawNumber returns new(jsonv1.Number).
	NewRawNumber func() any
	// RawNumberOf returns jsonv1.Number(b).
	RawNumberOf func(b []byte) any
)
