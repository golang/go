// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Item interface {
	Print() string
}

type ListItem struct {
	item Item
	next *ListItem
}

type List struct {
	head *ListItem
}

func (list *List) Init() {
	list.head = nil
}

func (list *List) Insert(i Item) {
	item := new(ListItem)
	item.item = i
	item.next = list.head
	list.head = item
}

func (list *List) Print() string {
	r := ""
	i := list.head
	for i != nil {
		r += i.item.Print()
		i = i.next
	}
	return r
}

// Something to put in a list
type Integer struct {
	val int
}

func (this *Integer) Init(i int) *Integer {
	this.val = i
	return this
}

func (this *Integer) Print() string {
	return string(this.val + '0')
}

func main() {
	list := new(List)
	list.Init()
	for i := 0; i < 10; i = i + 1 {
		integer := new(Integer)
		integer.Init(i)
		list.Insert(integer)
	}

	r := list.Print()
	if r != "9876543210" {
		panic(r)
	}
}
