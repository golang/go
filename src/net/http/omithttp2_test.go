// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build nethttpomithttp2

package http

// ResetPools reinitializes pools containing channels.
// Call this at the start and end of any test using synctest,
// to avoid leaking bubbled channels into/out of bubbles.
func ResetPools() {
}
