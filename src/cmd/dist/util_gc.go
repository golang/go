// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !gccgo

package main

// useVFPv1 tries to execute one VFPv1 instruction on ARM.
// It will crash the current process if VFPv1 is missing.
func useVFPv1()

// useVFPv3 tries to execute one VFPv3 instruction on ARM.
// It will crash the current process if VFPv3 is missing.
func useVFPv3()

// useARMv6K tries to run ARMv6K instructions on ARM.
// It will crash the current process if it doesn't implement
// ARMv6K or above.
func useARMv6K()
