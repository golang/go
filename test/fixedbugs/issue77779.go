// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
)

type Renderer interface {
	Render() error
}

type ZeroSize struct{}

func (ZeroSize) Render() error { return nil }

type Data struct {
	X, Y, Z int
}

// Container is pointer-sized (8 bytes): zero-size embed + one pointer field.
// This triggers Go 1.26 interface inlining, which produces a nil data pointer
// for the zero-size field when extracted via reflect.Value.Interface().
type Container struct {
	ZeroSize
	Data *Data
}

func main() {
	render(Container{})
	render(&Container{})
}

func render(iface any) {
	if reflect.ValueOf(iface).Kind() == reflect.Ptr {
		_ = reflect.ValueOf(iface).Elem().Field(0).Interface().(Renderer).Render()
		return
	}

	_ = reflect.ValueOf(iface).Field(0).Interface().(Renderer).Render()
}
