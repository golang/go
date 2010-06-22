// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Sigrecv returns a bitmask of signals that have arrived since the last call to Sigrecv.
// It blocks until at least one signal arrives.
func Sigrecv() uint32

// Signame returns a string describing the signal, or "" if the signal is unknown.
func Signame(sig int32) string

// Siginit enables receipt of signals via Sigrecv.  It should typically
// be called during initialization.
func Siginit()
