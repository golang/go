// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

// Content-type sniffing algorithm.
// http://tools.ietf.org/html/draft-ietf-websec-mime-sniff-03

// The algorithm prefers to use sniffLen bytes to make its decision.
const sniffLen = 1024

// detectContentType returns the sniffed Content-Type string
// for the given data.
func detectContentType(data []byte) string {
	// TODO(dsymonds,rsc): Implement algorithm from draft.
	return "text/html; charset=utf-8"
}
