// runoutput

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test run-time behavior of 3-index slice expressions.

package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
)

var bout *bufio.Writer

func main() {
	bout = bufio.NewWriter(os.Stdout)

	fmt.Fprintf(bout, "%s", programTop)
	fmt.Fprintf(bout, "func main() {\n")

	index := []string{
		"0",
		"1",
		"2",
		"3",
		"10",
		"20",
		"vminus1",
		"v0",
		"v1",
		"v2",
		"v3",
		"v10",
		"v20",
	}

	parse := func(s string) (n int, isconst bool) {
		if s == "vminus1" {
			return -1, false
		}
		isconst = true
		if s[0] == 'v' {
			isconst = false
			s = s[1:]
		}
		n, _ = strconv.Atoi(s)
		return n, isconst
	}

	const Cap = 10 // cap of slice, array

	for _, base := range []string{"array", "slice"} {
		for _, i := range index {
			iv, iconst := parse(i)
			for _, j := range index {
				jv, jconst := parse(j)
				for _, k := range index {
					kv, kconst := parse(k)
					// Avoid errors that would make the program not compile.
					// Those are tested by slice3err.go.
					switch {
					case iconst && jconst && iv > jv,
						jconst && kconst && jv > kv,
						iconst && kconst && iv > kv,
						iconst && base == "array" && iv > Cap,
						jconst && base == "array" && jv > Cap,
						kconst && base == "array" && kv > Cap:
						continue
					}

					expr := base + "[" + i + ":" + j + ":" + k + "]"
					var xbase, xlen, xcap int
					if iv > jv || jv > kv || kv > Cap || iv < 0 || jv < 0 || kv < 0 {
						xbase, xlen, xcap = -1, -1, -1
					} else {
						xbase = iv
						xlen = jv - iv
						xcap = kv - iv
					}
					fmt.Fprintf(bout, "\tcheckSlice(%q, func() []byte { return %s }, %d, %d, %d)\n", expr, expr, xbase, xlen, xcap)
				}
			}
		}
	}

	fmt.Fprintf(bout, "\tif !ok { os.Exit(1) }\n")
	fmt.Fprintf(bout, "}\n")
	bout.Flush()
}

var programTop = `
package main

import (
	"fmt"
	"os"
	"unsafe"
)

var ok = true

var (
	array = new([10]byte)
	slice = array[:]

	vminus1 = -1
	v0 = 0
	v1 = 1
	v2 = 2
	v3 = 3
	v4 = 4
	v5 = 5
	v10 = 10
	v20 = 20
)

func notOK() {
	if ok {
		println("BUG:")
		ok = false
	}
}

func checkSlice(desc string, f func() []byte, xbase, xlen, xcap int) {
	defer func() {
		if err := recover(); err != nil {
			if xbase >= 0 {
				notOK()
				println(desc, " unexpected panic: ", fmt.Sprint(err))
			}
		}
		// "no panic" is checked below
	}()
	
	x := f()

	arrayBase := uintptr(unsafe.Pointer(array))
	raw := *(*[3]uintptr)(unsafe.Pointer(&x))
	base, len, cap := raw[0] - arrayBase, raw[1], raw[2]
	if xbase < 0 {
		notOK()
		println(desc, "=", base, len, cap, "want panic")
		return
	}
	if cap != 0 && base != uintptr(xbase) || base >= 10 || len != uintptr(xlen) || cap != uintptr(xcap) {
		notOK()
		if cap == 0 {
			println(desc, "=", base, len, cap, "want", "0-9", xlen, xcap)
		} else {
			println(desc, "=", base, len, cap, "want", xbase, xlen, xcap)
		}
	}
}

`
