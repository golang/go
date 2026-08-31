// errorcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && 386

package p

var _ struct {
	a, b [1 << 30]byte // ERROR "type struct .* too large"
}
