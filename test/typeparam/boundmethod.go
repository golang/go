// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test illustrates how a type bound method (String below) can be implemented
// either by a concrete type (myint below) or a instantiated generic type
// (StringInt[myint] below).

package main

import (
        "fmt"
        "reflect"
        "strconv"
)

type myint int

//go:noinline
func (m myint) String() string {
        return strconv.Itoa(int(m))
}

type Stringer interface {
        String() string
}

func stringify[T Stringer](s []T) (ret []string) {
        for _, v := range s {
                ret = append(ret, v.String())
        }
        return ret
}

type StringInt[T any] T

//go:noinline
func (m StringInt[T]) String() string {
        return "aa"
}

func main() {
        x := []myint{myint(1), myint(2), myint(3)}

        got := stringify(x)
        want := []string{"1", "2", "3"}
        if !reflect.DeepEqual(got, want) {
                panic(fmt.Sprintf("got %s, want %s", got, want))
        }

        x2 := []StringInt[myint]{StringInt[myint](1), StringInt[myint](2), StringInt[myint](3)}

        got2 := stringify(x2)
        want2 := []string{"aa", "aa", "aa"}
        if !reflect.DeepEqual(got2, want2) {
                panic(fmt.Sprintf("got %s, want %s", got2, want2))
        }
}
