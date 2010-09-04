// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This used to crash because the scheduler
// tried to kick off a new scheduling thread for f
// when time.Nanoseconds went into the system call.
// It's not okay to schedule new goroutines
// until main has started.

package main

import "time"

func f() {
}

func init() {
	go f()
	time.Nanoseconds()
}

func main() {
}

