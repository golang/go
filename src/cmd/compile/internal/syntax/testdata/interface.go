// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains test cases for interfaces containing
// constraint elements.

package p

type _ interface {
	m()
	E
}

type _ interface {
	m()
	~int
	int | string
	int | ~string
	~int | ~string
}

type _ interface {
	m()
	~int
	T[int, string] | string
	int | ~T[string, struct{}]
	~int | ~string
}

type _ interface {
	int
	[]byte
	[10]int
	struct{}
	*int
	func()
	interface{}
	map[string]int
	chan T
	chan<- T
	<-chan T
	T[int]
}

type _ interface {
	int | string
	[]byte | string
	[10]int | string
	struct{} | string
	*int | string
	func() | string
	interface{} | string
	map[string]int | string
	chan T | string
	chan<- T | string
	<-chan T | string
	T[int] | string
}

type _ interface {
	~int | string
	~[]byte | string
	~[10]int | string
	~struct{} | string
	~*int | string
	~func() | string
	~interface{} | string
	~map[string]int | string
	~chan T | string
	~chan<- T | string
	~<-chan T | string
	~T[int] | string
}
