// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Starting in Go 1.20, the global rand is auto-seeded,
// with a better value than the current Unix nanoseconds.
// Only seed if we're using older versions of Go.

//go:build !go1.20

package main

import (
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}
