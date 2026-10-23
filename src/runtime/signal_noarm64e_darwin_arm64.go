// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.arm64e

package runtime

func (c *sigctxt) r29() uint64 { return c.regs().fp }
func (c *sigctxt) lr() uint64  { return c.regs().lr }
func (c *sigctxt) sp() uint64  { return c.regs().sp }

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return c.regs().pc }

func (c *sigctxt) set_pc(x uint64) { c.regs().pc = x }
func (c *sigctxt) set_sp(x uint64) { c.regs().sp = x }
func (c *sigctxt) set_lr(x uint64) { c.regs().lr = x }
