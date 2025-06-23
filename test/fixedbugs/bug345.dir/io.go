// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io

type Writer interface {
	WrongWrite()
}

type SectionReader struct {
	X int
}

func SR(*SectionReader) {}
