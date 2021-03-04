// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"
)

type Setter[B any] interface {
        Set(string)
	type *B
}

func fromStrings1[T any, PT Setter[T]](s []string) []T {
        result := make([]T, len(s))
        for i, v := range s {
                // The type of &result[i] is *T which is in the type list
                // of Setter, so we can convert it to PT.
                p := PT(&result[i])
                // PT has a Set method.
                p.Set(v)
        }
        return result
}

func fromStrings2[T any](s []string, set func(*T, string)) []T {
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
        s := fromStrings1[Settable, *Settable]([]string{"1"})
        if len(s) != 1 || s[0] != 1 {
                panic(fmt.Sprintf("got %v, want %v", s, []int{1}))
        }

        s = fromStrings2([]string{"1"}, func(p *Settable, s string) { p.Set(s) })
        if len(s) != 1 || s[0] != 1 {
                panic(fmt.Sprintf("got %v, want %v", s, []int{1}))
        }
}
