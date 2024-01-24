// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type explainer struct {
	m map[string]string
}

func init() {
	RegisterExplainer(newExplainer())
}

type Explainer interface {
	Name() string
	Map() map[string]string
}

func (e explainer) Name() string {
	return "HelloWorldExplainer"
}

func (e explainer) Map() map[string]string {
	return e.m
}

//go:noinline
func newExplainer() explainer {
	m := make(map[string]string)
	m["Hello"] = "World!"
	return explainer{m}
}

var explainers = make(map[string]Explainer)

func RegisterExplainer(e Explainer) {
	explainers[e.Name()] = e
}

func main() {

}
