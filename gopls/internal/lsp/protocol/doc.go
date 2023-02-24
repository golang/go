// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run ./generate

// Package protocol contains the structs that map directly to the
// request and response messages of the Language Server Protocol.
//
// It is a literal transcription, with unmodified comments, and only the changes
// required to make it go code.
// Names are uppercased to export them.
// All fields have JSON tags added to correct the names.
// Fields marked with a ? are also marked as "omitempty"
// Fields that are "|| null" are made pointers
// Fields that are string or number are left as string
// Fields that are type "number" are made float64
package protocol
