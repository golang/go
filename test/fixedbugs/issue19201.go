// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/binary"
)

var (
	ch1 = make(chan int)
	ch2 = make(chan int)

	bin  = []byte("a\000\000\001")
	want = binary.BigEndian.Uint32(bin)

	c consumer = noopConsumer{}
)

type msg struct {
	code uint32
}

type consumer interface {
	consume(msg)
}

type noopConsumer struct{}

func (noopConsumer) consume(msg) {}

func init() {
	close(ch1)
}

func main() {
	var m msg
	m.code = binary.BigEndian.Uint32(bin)

	select {
	case <-ch1:
		c.consume(m)
		if m.code != want {
			// can not use m.code here, or it will work
			panic("BigEndian read failed")
		}
	case <-ch2:
	}
}
