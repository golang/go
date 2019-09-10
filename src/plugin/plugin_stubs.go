// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux,!freebsd,!darwin !cgo

package plugin

import "errors"

func lookup(p *Plugin, symName string) (Symbol, error) {
	return nil, errors.New("plugin: not implemented")
}

func open(name string) (*Plugin, error) {
	return nil, errors.New("plugin: not implemented")
}
