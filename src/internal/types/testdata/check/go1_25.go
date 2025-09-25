// -lang=go1.25

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check Go language version-specific errors.

//go:build go1.25

package p

var _ = new /* ERROR "new(expr) requires go1.26 or later" */ (123)
