// errorcheck -lang=go1.4

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.4

package p

const c = 0o123 // ERROR "file declares //go:build go1.4"
