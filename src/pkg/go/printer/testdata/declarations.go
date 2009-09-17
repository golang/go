// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import "io"

import (
	_ "io"
)

import _ "io"

import (
	"io";
	"io";
	"io";
)

import (
	"io";
	aLongRename "io";
	b "io";
	c "i" "o";
)

func _() {
	// the following decls need a semicolon at the end
	type _ int;
	type _ *int;
	type _ []int;
	type _ map[string]int;
	type _ chan int;
	type _ func() int;

	var _ int;
	var _ *int;
	var _ []int;
	var _ map[string]int;
	var _ chan int;
	var _ func() int;

	// the following decls don't need a semicolon at the end
	type _ struct{}
	type _ *struct{}
	type _ []struct{}
	type _ map[string]struct{}
	type _ chan struct{}
	type _ func() struct{}

	type _ interface{}
	type _ *interface{}
	type _ []interface{}
	type _ map[string]interface{}
	type _ chan interface{}
	type _ func() interface{}

	var _ struct{}
	var _ *struct{}
	var _ []struct{}
	var _ map[string]struct{}
	var _ chan struct{}
	var _ func() struct{}

	var _ interface{}
	var _ *interface{}
	var _ []interface{}
	var _ map[string]interface{}
	var _ chan interface{}
	var _ func() interface{}
}

// TODO(gri) add more test cases
