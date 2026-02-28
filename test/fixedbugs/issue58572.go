// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func New() resource {
	return &Client{}
}

type resource interface {
	table()
}

type Client struct {
	m map[Key1]int
}

func (c *Client) table() {}

type Key1 struct {
	K Key2
}

type Key2 struct {
	f [2]any
}
