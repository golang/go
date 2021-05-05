// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"context"
	"flag"
	"fmt"
	"testing"
	"time"

	. "golang.org/x/tools/internal/lsp/regtest"
)

// Pilosa is a repository that has historically caused significant memory
// problems for Gopls. We use it for a simple stress test that types
// arbitrarily in a file with lots of dependents.

var pilosaPath = flag.String("pilosa_path", "", "Path to a directory containing "+
	"github.com/pilosa/pilosa, for stress testing. Do not set this unless you "+
	"know what you're doing!")

func stressTestOptions(dir string) []RunOption {
	opts := benchmarkOptions(dir)
	opts = append(opts, SkipHooks(true), DebugAddress(":8087"))
	return opts
}

func TestPilosaStress(t *testing.T) {
	if *pilosaPath == "" {
		t.Skip("-pilosa_path not configured")
	}
	opts := stressTestOptions(*pilosaPath)

	WithOptions(opts...).Run(t, "", func(_ *testing.T, env *Env) {
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
