// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgrouptest

import (
	"fmt"
	"testing"
)

func TestInCgroupV2(t *testing.T) {
	InCgroupV2(t, func(c *CgroupV2) {
		fmt.Println("Created", c.Path())
		if err := c.SetCPUMax(500000, 100000); err != nil {
			t.Errorf("Erroring setting cpu.max: %v", err)
		}
	})
}
