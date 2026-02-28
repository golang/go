// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type None struct{}

type Response interface {
	send(ctx *struct{})
}

type HandlerFunc[Input any] func(Input) Response

func Operation[Input any](method, path string, h HandlerFunc[Input]) {
	var input Input
	h(input)
}

func Get[Body any](path string, h HandlerFunc[struct{ Body Body }]) {
	Operation("GET", path, h)
}

func main() {
	Get("/", func(req struct{ Body None }) Response {
		return nil
	})
}
