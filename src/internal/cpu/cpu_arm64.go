// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

const CacheLineSize = 64

// TODO: delete this once https://go-review.googlesource.com/c/go/+/76490 lands.
// These will just be false for now.
var ARM64 struct {
	HasSHA1 bool
	HasSHA2 bool
}
