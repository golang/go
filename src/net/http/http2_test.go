// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !nethttpomithttp2

package http

func init() {
	// Disable HTTP/2 internal channel pooling which interferes with synctest.
	http2inTests = true
}
