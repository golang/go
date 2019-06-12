// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.12

package debug

import (
	"fmt"
	"io"
)

func printBuildInfo(w io.Writer, verbose bool, mode PrintMode) {
	fmt.Fprintf(w, "version %s, built in $GOPATH mode\n", Version)
}
