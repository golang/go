// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

// modFile holds all of the information we know about a mod file.
type modFile struct {
	fileBase
}

func (*modFile) setContent(content []byte) {}
func (*modFile) filename() string          { return "" }
func (*modFile) isActive() bool            { return false }
