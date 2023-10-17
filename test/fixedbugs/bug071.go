// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bug071

type rat struct  {
	den  int;
}

func (u *rat) pr() {
}

type dch struct {
	dat chan  *rat;
}

func dosplit(in *dch){
	dat := <-in.dat;
	_ = dat;
}
