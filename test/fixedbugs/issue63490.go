// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type ResourceFunc struct {
	junk [8]int
	base assignmentBaseResource
}

type SubscriptionAssignmentResource struct {
	base assignmentBaseResource
}

type assignmentBaseResource struct{}

//go:noinline
func (a assignmentBaseResource) f(s string) ResourceFunc {
	println(s)
	return ResourceFunc{}
}

//go:noinline
func (r SubscriptionAssignmentResource) Hi() ResourceFunc {
	rf := r.base.f("Hello world")
	rf.base = r.base
	return rf
}

func main() {
	var r SubscriptionAssignmentResource
	r.Hi()
}
