// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows
// +build windows

package registry

func (k Key) SetValue(name string, valtype uint32, data []byte) error {
	return k.setValue(name, valtype, data)
}
