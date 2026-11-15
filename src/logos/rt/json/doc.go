// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package json is the runtime support package of the loGo/s static JSON
// feature (M11). The compiler generates MarshalJSON and UnmarshalJSON
// methods for annotated structs and those methods call the helpers in
// this package instead of reflecting over the receiver.
//
// The marshal side (AppendEscaped, AppendQuoted, EscapedLen, QuotedLen,
// IntLen, UintLen, AppendInt, AppendUint, AppendFloat, FloatErr,
// AppendBase64, IsZero and MarshalField) produces bytes identical to
// encoding/json for the same value, under both settings of the jsonv2
// GOEXPERIMENT. The two settings differ in exactly one case, the encoding
// of an invalid UTF-8 byte inside a string, which is isolated in
// escape_v1.go and escape_v2.go. Walk carries one root MarshalJSON call's
// traversal state, so cycles spanning generated encoders and dynamic
// values terminate with the encoding/json cycle error and nesting stops at
// the encoding/json depth bound. Walk.AppendAny walks a value whose type
// is known only at run time with that state (walk.go, walkfields.go, with
// mode-specific visit keys and depth checks in walk_v1.go and walk_v2.go).
//
// The unmarshal side is the Scanner, a cursor over one complete JSON
// value held in memory, plus the typed readers (Bool, Int, Uint, Float,
// String, their quoted forms and Slice). The Scanner validates while it
// decodes, never pre-allocates from untrusted input and reports the same
// error values encoding/json reports: type errors are saved and decoding
// continues, syntax errors are fatal and are taken from encoding/json
// itself so that message and offset are identical.
//
// The package imports only encoding/json, math, reflect, strconv, unicode,
// unicode/utf8 and unsafe.
package json
