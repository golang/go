// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f1

type Foo struct {
	doneChan chan bool
	Name     string
	fOO      int
	hook     func()
}

func (f *Foo) Exported() {
}

func (f *Foo) unexported() {
}
