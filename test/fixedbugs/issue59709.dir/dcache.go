// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dcache

import (
	"./aconfig"
	"./bresource"
	"./cmem"
)

type Module struct {
	cfg  *aconfig.Config
	err  error
	last any
}

//go:noinline
func TD() {
}

func (m *Module) Configure(x string) error {
	if m.err != nil {
		return m.err
	}
	res := cmem.NewResource(m.cfg)
	m.last = res

	return nil
}

func (m *Module) Blurb(x string, e error) bool {
	res, ok := m.last.(*bresource.Resource[*int])
	if !ok {
		panic("bad")
	}
	return bresource.Should(res, e)
}
