// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package godoc

import "appengine"

func init() {
	onAppengine = !appengine.IsDevAppServer()
}
