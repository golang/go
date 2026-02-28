// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.8
// +build go1.8

package logopt

import "net/url"

func pathEscape(s string) string {
	return url.PathEscape(s)
}
