// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>

extern int goCallback(int v);

static int target(int *p, int doCall) {
	int v = *p;
	if (doCall) {
		v = goCallback(v);
	}
	return v + 1;
}

#cgo noescape target
#cgo nocallback target
*/
import "C"

import (
	"fmt"
	"os"
	"runtime/debug"
	"strings"
)

//export goCallback
func goCallback(v C.int) C.int { return v + 1 }

//go:noinline
func forbidden(v C.int, want string) (panicked bool, bad string) {
	defer func() {
		if r := recover(); r != nil {
			if strings.Contains(fmt.Sprint(r), want) {
				panicked = true
			} else {
				bad = fmt.Sprint("wrong panic class: ", r)
			}
		}
	}()
	got := C.target(&v, 1)
	bad = fmt.Sprintf("returned %d instead of panicking", int(got))
	return false, bad
}

func repeatWant() string {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return "unexpected setting cgoNoCallback"
	}
	for _, kv := range bi.Settings {
		if kv.Key != "GOEXPERIMENT" {
			continue
		}
		for name := range strings.SplitSeq(kv.Value, ",") {
			if name == "logosfastcgo" {
				return "called back into Go"
			}
		}
	}
	return "unexpected setting cgoNoCallback"
}

func safemain() string {
	for i, want := range []string{
		"called back into Go",
		repeatWant(),
		repeatWant(),
	} {
		ok, bad := forbidden(C.int(41+i), want)
		if bad != "" {
			return fmt.Sprintf("call %d %s", i, bad)
		}
		if !ok {
			return fmt.Sprintf("call %d did not panic", i)
		}
	}
	return ""
}

func run() (err string) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Sprint("unexpected panic: ", r)
		}
	}()
	return safemain()
}

func main() {
	if err := run(); err != "" {
		fmt.Println("LAW cgo both forbid seq", err)
		os.Exit(1)
	}
	fmt.Println("ok cgo both forbid seq")
}
