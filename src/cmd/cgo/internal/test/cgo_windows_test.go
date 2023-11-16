// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && windows

package cgotest

import "testing"

func TestCallbackCallersSEH(t *testing.T) { testCallbackCallersSEH(t) }
