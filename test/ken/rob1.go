// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Item interface
{
	Print_BUG	func();
}

type ListItem struct
{
	item    Item;
	next    *ListItem;
}

type List struct
{
	head    *ListItem;
}

func (list *List)
Init()
{
	list.head = nil;
}

func (list *List)
Insert(i Item)
{
	item := new(ListItem);
	item.item = i;
	item.next = list.head;
	list.head = item;
}

func (list *List)
Print()
{
	i := list.head;
	for i != nil {
		i.item.Print_BUG();
		i = i.next;
	}
}

// Something to put in a list
type Integer struct
{
	val		int;
}

func (this *Integer)
Init_BUG(i int) *Integer
{
	this.val = i;
	return this;
}

func (this *Integer)
Print_BUG()
{
	print this.val;
}

func
main() int32
{
	list := new(List);
	list.Init();
	for i := 0; i < 10; i = i + 1 {
		integer := new(Integer);
		integer.Init_BUG(i);
		list.Insert(integer);
	}

	list.Print();
	return 0;
}
