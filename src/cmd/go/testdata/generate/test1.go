// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simple test for go generate.

// We include a build tag that go generate should ignore.

// +build ignore

//go:generate echo Success

package p
