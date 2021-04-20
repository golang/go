// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (js && wasm) || windows
// +build js,wasm windows

package net

const readMsgFlags = 0

func setReadMsgCloseOnExec(oob []byte) {}
