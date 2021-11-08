// errorcheck -G=3 -lang=go1.17

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type _ interface {
	int // ERROR "embedding non-interface type int requires go1\.18 or later \(-lang was set to go1\.17; check go.mod\)"
}
