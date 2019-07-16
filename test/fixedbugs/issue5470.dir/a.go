// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Foo interface {
	Hi() string
}

func Test1() Foo { return make(tst1) }

type tst1 map[string]bool

func (r tst1) Hi() string { return "Hi!" }

func Test2() Foo { return make(tst2, 0) }

type tst2 []string

func (r tst2) Hi() string { return "Hi!" }

func Test3() Foo { return make(tst3) }

type tst3 chan string

func (r tst3) Hi() string { return "Hi!" }
