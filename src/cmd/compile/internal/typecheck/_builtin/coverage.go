// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// NOTE: If you change this file you must run "go generate"
// to update builtin.go. This is not done automatically
// to avoid depending on having a working compiler binary.

//go:build ignore

package coverage

func initHook(istest bool)
