// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test makes sure that we statically load a type from an itab, instead
// of doing a indirect load from thet itab.

package codegen

type M interface{ M() }
type A interface{ A() }

type Impl struct{}

func (*Impl) M() {}
func (*Impl) A() {}

func main() {
	var a M = &Impl{}
	// amd64:`LEAQ type:.*Impl`
	a.(A).A()
}
