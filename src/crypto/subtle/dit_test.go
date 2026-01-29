// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package subtle

import (
	"internal/cpu"
	"internal/runtime/sys"
	"testing"
)

func TestWithDataIndependentTiming(t *testing.T) {
	if !cpu.ARM64.HasDIT {
		t.Skip("CPU does not support DIT")
	}

	ditAlreadyEnabled := sys.DITEnabled()

	WithDataIndependentTiming(func() {
		if !sys.DITEnabled() {
			t.Fatal("dit not enabled within WithDataIndependentTiming closure")
		}

		WithDataIndependentTiming(func() {
			if !sys.DITEnabled() {
				t.Fatal("dit not enabled within nested WithDataIndependentTiming closure")
			}
		})

		if !sys.DITEnabled() {
			t.Fatal("dit not enabled after return from nested WithDataIndependentTiming closure")
		}
	})

	if !ditAlreadyEnabled && sys.DITEnabled() {
		t.Fatal("dit not unset after returning from WithDataIndependentTiming closure")
	}
}

func TestDITPanic(t *testing.T) {
	if !cpu.ARM64.HasDIT {
		t.Skip("CPU does not support DIT")
	}

	ditAlreadyEnabled := sys.DITEnabled()

	defer func() {
		e := recover()
		if e == nil {
			t.Fatal("didn't panic")
		}
		if !ditAlreadyEnabled && sys.DITEnabled() {
			t.Error("DIT still enabled after panic inside of WithDataIndependentTiming closure")
		}
	}()

	WithDataIndependentTiming(func() {
		if !sys.DITEnabled() {
			t.Fatal("dit not enabled within WithDataIndependentTiming closure")
		}

		panic("bad")
	})
}

func TestDITGoroutineInheritance(t *testing.T) {
	if !cpu.ARM64.HasDIT {
		t.Skip("CPU does not support DIT")
	}

	ditAlreadyEnabled := sys.DITEnabled()

	WithDataIndependentTiming(func() {
		done := make(chan struct{})
		go func() {
			if !sys.DITEnabled() {
				t.Error("DIT not enabled in new goroutine")
			}
			close(done)
		}()
		<-done
		if !ditAlreadyEnabled && !sys.DITEnabled() {
			t.Fatal("dit unset after returning from goroutine started in WithDataIndependentTiming closure")
		}
	})
}
