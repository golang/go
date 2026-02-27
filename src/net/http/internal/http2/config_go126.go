// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package http2

import (
	"net/http"
)

func http2ConfigStrictMaxConcurrentRequests(h2 *http.HTTP2Config) bool {
	return h2.StrictMaxConcurrentRequests
}
