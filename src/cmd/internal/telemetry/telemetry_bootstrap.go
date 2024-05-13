// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cmd_go_bootstrap || compiler_bootstrap

package telemetry

import "flag"

type dummyCounter struct{}

func (dc dummyCounter) Inc() {}

func Start()                                                              {}
func StartWithUpload()                                                    {}
func Inc(name string)                                                     {}
func NewCounter(name string) dummyCounter                                 { return dummyCounter{} }
func NewStackCounter(name string, depth int) dummyCounter                 { return dummyCounter{} }
func CountFlags(name string, flagSet flag.FlagSet)                        {}
func CountFlagValue(prefix string, flagSet flag.FlagSet, flagName string) {}
