// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./pkg1"
)

type message struct { // Presence of this creates a crash
	data pkg1.Data
}

func main() {
	pkg1.CrashCall()
}
