// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !racelite

package runtime

func raceliteinit()             {}
func racelitecount()            {}
func racelitetick(delay uint32) {}
