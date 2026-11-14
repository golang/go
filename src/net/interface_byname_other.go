// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux

package net

// interfaceByName returns the interface with the given name, or
// errNoSuchInterface.
func interfaceByName(name string) (*Interface, error) {
	return interfaceByNameFromTable(name)
}
