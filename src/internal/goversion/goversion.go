// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goversion

// Version is the current Go 1.x version. During development cycles on
// the master branch it changes to be the version of the next Go 1.x
// release.
//
// When incrementing this, also add to the list at src/go/build/doc.go
// (search for "onward").
const Version = 13
