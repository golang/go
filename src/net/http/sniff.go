// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import "net/http/internal"

// DetectContentType implements the algorithm described
// at https://mimesniff.spec.whatwg.org/ to determine the
// Content-Type of the given data. It considers at most the
// first 512 bytes of data. DetectContentType always returns
// a valid MIME type: if it cannot determine a more specific one, it
// returns "application/octet-stream".
func DetectContentType(data []byte) string {
	return internal.DetectContentType(data)
}
