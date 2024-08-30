// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"time"
)

// for golang.org/issue/27250

func init() {
	register("After1", After1)
}

func After1() {
	os.Stdout.WriteString("ready\n")
	os.Stdout.Close()
	<-time.After(1 * time.Second)
}
