// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godebug

import (
	"internal/godebug"
)

type Setting godebug.Setting

func New(name string) *Setting {
	return (*Setting)(godebug.New(name))
}

func (s *Setting) Value() string {
	return (*godebug.Setting)(s).Value()
}

func Value(name string) string {
	return godebug.New(name).Value()
}
