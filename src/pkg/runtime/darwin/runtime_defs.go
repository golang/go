// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go definitions of internal structures. Master is runtime.h

package runtime

type lock struct {
	key  uint32
	sema uint32
}

type usema struct {
	u uint32
	k uint32
}


type note struct {
	wakeup int32
	sema   usema
}
