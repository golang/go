// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"context"
	"flag"
	"fmt"
	"testing"
	"time"
)

// Pilosa is a repository that has historically caused significant memory
// problems for Gopls. We use it for a simple stress test that types
// arbitrarily in a file with lots of dependents.

var pilosaPath = flag.String("pilosa_path", "", "Path to a directory containing "+
	"github.com/pilosa/pilosa, for stress testing. Do not set this unless you "+
	"know what you're doing!")

func stressTestOptions(dir string) []RunOption {
	return []RunOption{
		// Run in an existing directory, since we're trying to simulate known cases
		// that cause gopls memory problems.
		InExistingDir(dir),

		// Enable live debugging.
		WithDebugAddress(":8087"),

		// Skip logs as they buffer up memory unnaturally.
		SkipLogs(),
		// Similarly to logs: disable hooks so that they don't affect performance.
		SkipHooks(true),
		// The Debug server only makes sense if running in singleton mode.
		WithModes(Singleton),
		// Set a generous timeout. Individual tests should control their own
		// graceful termination.
		WithTimeout(20 * time.Minute),

		// Use the actual proxy, since we want our builds to succeed.
		WithGOPROXY("https://proxy.golang.org"),
	}
}

func TestPilosaStress(t *testing.T) {
	if *pilosaPath == "" {
		t.Skip("-pilosa_path not configured")
	}
	opts := stressTestOptions(*pilosaPath)

	withOptions(opts...).run(t, "", func(t *testing.T, env *Env) {
		files := []string{
			"cmd.go",
			"internal/private.pb.go",
			"roaring/roaring.go",
			"roaring/roaring_internal_test.go",
			"server/handler_test.go",
		}
		for _, file := range files {
			env.OpenFile(file)
		}
		ctx, cancel := context.WithTimeout(env.Ctx, 10*time.Minute)
		defer cancel()

		i := 1
		// MagicNumber is an identifier that occurs in roaring.go. Just change it
		// arbitrarily.
		env.RegexpReplace("roaring/roaring.go", "MagicNumber", fmt.Sprintf("MagicNumber%d", 1))
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}
			env.RegexpReplace("roaring/roaring.go", fmt.Sprintf("MagicNumber%d", i), fmt.Sprintf("MagicNumber%d", i+1))
			time.Sleep(20 * time.Millisecond)
			i++
		}
	})
}
