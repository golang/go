// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkg1

type A interface {
	Write() error
}

type B interface {
	Hello()
	world()
}

type C struct{}

func (c C) Write() error { return nil }

var T = struct{ A }{nil}
var U = struct{ B }{nil}
var V A = struct{ *C }{nil}
var W = interface {
	Write() error
	Hello()
}(nil)
