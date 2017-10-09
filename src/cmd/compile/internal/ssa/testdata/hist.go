// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func ensure(n int, sl []int) []int {
	for len(sl) <= n {
		sl = append(sl, 0)
	}
	return sl
}

var cannedInput string = `1
1
1
1
2
2
2
4
4
8
`

func main() {
	hist := make([]int, 10)
	var reader io.Reader = strings.NewReader(cannedInput) //gdb-dbg=(hist/A,cannedInput/A)
	if len(os.Args) > 1 {
		var err error
		reader, err = os.Open(os.Args[1])
		if err != nil {
			fmt.Fprintf(os.Stderr, "There was an error opening %s: %v\n", os.Args[1], err)
			return
		}
	}
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		s := scanner.Text()
		i, err := strconv.ParseInt(s, 10, 64)
		if err != nil { //gdb-dbg=(i)
			fmt.Fprintf(os.Stderr, "There was an error: %v\n", err)
			return
		}
		hist = ensure(int(i), hist)
		hist[int(i)]++
	}
	t := 0
	n := 0
	for i, a := range hist {
		if a == 0 {
			continue
		}
		t += i * a
		n += a
		fmt.Fprintf(os.Stderr, "%d\t%d\t%d\t%d\t%d\n", i, a, n, i*a, t) //gdb-dbg=(n,i,t)
	}

}
