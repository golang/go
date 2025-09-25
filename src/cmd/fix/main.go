// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"os"
)

var (
	_ = flag.Bool("diff", false, "obsolete, no effect")
	_ = flag.String("go", "", "obsolete, no effect")
	_ = flag.String("r", "", "obsolete, no effect")
	_ = flag.String("force", "", "obsolete, no effect")
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go tool fix [-diff] [-r ignored] [-force ignored] ...\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func main() {
	flag.Usage = usage
	flag.Parse()

	os.Exit(0)
}
