// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package query

type Conn string

func (c Conn) DoQuery(query string) Result {
	return Result("result")
}

type Result string

func Query(conns []Conn, query string) Result {
	ch := make(chan Result, 1)
	for _, conn := range conns {
		go func(c Conn) {
			select {
			case ch <- c.DoQuery(query):
			default:
			}
		}(conn)
	}
	return <-ch
}

// STOP OMIT
