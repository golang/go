// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build logosejected

package rt

// TraceErr is a no-op in ejected code: traces exist only in binaries built
// by logos with the runtime trace ring.
func TraceErr(err error, s *Site) {}

// ErrorTrace returns no sites in ejected code.
func ErrorTrace() []Site { return nil }
