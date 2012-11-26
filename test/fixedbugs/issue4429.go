// errorcheck

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type a struct {
  a int
}

func main() {
  av := a{};
  *a(av); // ERROR "invalid indirect"
}
