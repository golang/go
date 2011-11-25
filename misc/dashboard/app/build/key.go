// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import "appengine"

// Delete this init function before deploying to production.
func init() {
	if !appengine.IsDevAppServer() {
		panic("please read key.go")
	}
}

const secretKey = "" // Important! Put a secret here before deploying!
