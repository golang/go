// +build !go1.13

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import "path/filepath"

// TODO: use proxy functionality in golang.org/x/tools/go/packages/packagestest
// instead of copying it here.

func proxyDirToURL(dir string) string {
	// Prior to go1.13, the Go command on Windows only accepted GOPROXY file URLs
	// of the form file://C:/path/to/proxy. This was incorrect: when parsed, "C:"
	// is interpreted as the host. See golang.org/issue/6027. This has been
	// fixed in go1.13, but we emit the old format for old releases.
	return "file://" + filepath.ToSlash(dir)
}
