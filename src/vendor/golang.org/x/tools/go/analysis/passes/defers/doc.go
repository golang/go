// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package defers defines an Analyzer that checks for common mistakes in defer
// statements.
//
// # Analyzer defers
//
// defers: report common mistakes in defer statements
//
// The defers analyzer reports a diagnostic when a defer statement would
// result in a non-deferred call to time.Since, as experience has shown
// that this is nearly always a mistake.
//
// For example:
//
//	start := time.Now()
//	...
//	defer recordLatency(time.Since(start)) // error: call to time.Since is not deferred
//
// The correct code is:
//
//	defer func() { recordLatency(time.Since(start)) }()
package defers
