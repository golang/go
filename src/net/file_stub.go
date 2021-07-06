// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm
// +build js,wasm

package net

import (
	"os"
	"syscall"
)

func fileConn(f *os.File) (Conn, error)             { return nil, syscall.ENOPROTOOPT }
func fileListener(f *os.File) (Listener, error)     { return nil, syscall.ENOPROTOOPT }
func filePacketConn(f *os.File) (PacketConn, error) { return nil, syscall.ENOPROTOOPT }
