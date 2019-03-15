// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

type EmbedImported struct {
	a.NoitfStruct
}

func Test() []string {
	bad := []string{}
	x := interface{}(new(a.NoitfStruct))
	if _, ok := x.(interface {
		NoInterfaceMethod()
	}); ok {
		bad = append(bad, "fail 1")
	}

	x = interface{}(new(EmbedImported))
	if _, ok := x.(interface {
		NoInterfaceMethod()
	}); ok {
		bad = append(bad, "fail 2")
	}
	return bad
}
