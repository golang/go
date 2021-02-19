// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"
)

func fromStrings3[T any](s []string, set func(*T, string)) []T {
        results := make([]T, len(s))
        for i, v := range s {
                set(&results[i], v)
        }
        return results
}

type Settable int

func (p *Settable) Set(s string) {
        i, err := strconv.Atoi(s)
        if err != nil {
                panic(err)
        }
        *p = Settable(i)
}

func main() {
        s := fromStrings3([]string{"1"},
                func(p *Settable, s string) { p.Set(s) })
        if len(s) != 1 || s[0] != 1 {
                panic(fmt.Sprintf("got %v, want %v", s, []int{1}))
        }
}
