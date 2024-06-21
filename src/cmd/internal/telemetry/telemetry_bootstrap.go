// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cmd_go_bootstrap || compiler_bootstrap

package telemetry

func MaybeParent()              {}
func MaybeChild()               {}
func Mode() string              { return "" }
func SetMode(mode string) error { return nil }
func Dir() string               { return "" }
