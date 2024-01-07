// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lostcancel defines an Analyzer that checks for failure to
// call a context cancellation function.
//
// # Analyzer lostcancel
//
// lostcancel: check cancel func returned by context.WithCancel is called
//
// The cancellation function returned by context.WithCancel, WithTimeout,
// and WithDeadline must be called or the new context will remain live
// until its parent context is cancelled.
// (The background context is never cancelled.)
package lostcancel
