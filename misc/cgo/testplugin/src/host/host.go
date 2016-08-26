// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"plugin"

	"common"
)

func init() {
	common.X *= 5
}

func main() {
	if got, want := common.X, 3*5; got != want {
		log.Fatalf("before plugin load common.X=%d, want %d", got, want)
	}

	p, err := plugin.Open("plugin1.so")
	if err != nil {
		log.Fatalf("plugin.Open failed: %v", err)
	}

	const wantX = 3 * 5 * 7
	if got := common.X; got != wantX {
		log.Fatalf("after plugin load common.X=%d, want %d", got, wantX)
	}

	seven, err := p.Lookup("Seven")
	if err != nil {
		log.Fatalf(`Lookup("Seven") failed: %v`, err)
	}
	if got, want := *seven.(*int), 7; got != want {
		log.Fatalf("via lookup plugin1.Seven=%d, want %d", got, want)
	}

	readFunc, err := p.Lookup("ReadCommonX")
	if err != nil {
		log.Fatalf(`Lookup("ReadCommonX") failed: %v`, err)
	}
	if got := readFunc.(func() int)(); got != wantX {
		log.Fatalf("via lookup plugin1.ReadCommonX()=%d, want %d", got, wantX)
	}

	fmt.Println("PASS")
}
