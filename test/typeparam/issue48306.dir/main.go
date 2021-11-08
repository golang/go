// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "a"

type S struct{}

func (*S) F() *S { return nil }

func main() {
	var _ a.I[*S] = &S{}
}
