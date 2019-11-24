// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "time"

// for golang.org/issue/27250

func init() {
	register("After1", After1)
}

func After1() {
	<-time.After(1 * time.Second)
}
