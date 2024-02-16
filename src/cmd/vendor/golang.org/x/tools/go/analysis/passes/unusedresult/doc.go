// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unusedresult defines an analyzer that checks for unused
// results of calls to certain pure functions.
//
// # Analyzer unusedresult
//
// unusedresult: check for unused results of calls to some functions
//
// Some functions like fmt.Errorf return a result and have no side
// effects, so it is always a mistake to discard the result. Other
// functions may return an error that must not be ignored, or a cleanup
// operation that must be called. This analyzer reports calls to
// functions like these when the result of the call is ignored.
//
// The set of functions may be controlled using flags.
package unusedresult
