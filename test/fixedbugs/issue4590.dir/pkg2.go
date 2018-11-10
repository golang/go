// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkg2

import "./pkg1"

var T = struct{ pkg1.A }{nil}
var U = struct{ pkg1.B }{nil}
var V pkg1.A = struct{ *pkg1.C }{nil}
var W = interface {
	Write() error
	Hello()
}(nil)
