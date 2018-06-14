// errorcheck -0 -race

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 17449: race instrumentation copies over previous instrumented nodes from parents block into child's Ninit block.
// This code surfaces the duplication at compile time because of generated inline labels.

package master

type PriorityList struct {
    elems []interface{}
}

func (x *PriorityList) Len() int { return len(x.elems) }

func (l *PriorityList) remove(i int) interface{} {
    elem := l.elems[i]
    l.elems = append(l.elems[:i], l.elems[i+1:]...)
    return elem
}

func (l *PriorityList) Next() interface{} {
    return l.remove(l.Len() - 1)
}

var l *PriorityList

func Foo() {
    // It would fail here if instrumented code (including inline-label) was copied.
    for elem := l.Next(); elem != nil; elem = l.Next() {
    }
}
