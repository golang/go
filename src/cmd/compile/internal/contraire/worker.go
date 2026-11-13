// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || linux

package contraire

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"strings"
	"syscall"
)

// runWorker executes a shard of trials sequentially.
func runWorker(trials int, cfgPath, outPath, lockPath string, std bool) error {
	if trials < 1 {
		return fmt.Errorf("trials %d must be positive", trials)
	}

	// Load config if provided; otherwise use the default.
	c := cfg
	if cfgPath != "" {
		data, err := os.ReadFile(cfgPath)
		if err != nil {
			return fmt.Errorf("couldn't read config %q: %v", cfgPath, err)
		}
		c = new(config)
		if err := json.Unmarshal(data, c); err != nil {
			return fmt.Errorf("couldn't decode config %q: %v", cfgPath, err)
		}
	}

	// The driver must have created the lock file.
	f, err := os.Open(lockPath)
	if err != nil {
		return fmt.Errorf("couldn't open lock file %q: %v", lockPath, err)
	}
	defer f.Close()
	fd := int(f.Fd())

	write := func(out *output) error {
		data, err := json.Marshal(out)
		if err != nil {
			return fmt.Errorf("couldn't marshal output: %v", err)
		}
		// Hold the lock while writing so the driver never observes a partial write.
		if err := syscall.Flock(fd, syscall.LOCK_EX); err != nil {
			return fmt.Errorf("couldn't acquire lock file %q: %v", lockPath, err)
		}
		if err := os.WriteFile(outPath, data, 0o644); err != nil {
			return fmt.Errorf("couldn't write output to %q: %v", outPath, err)
		}
		if err := syscall.Flock(fd, syscall.LOCK_UN); err != nil {
			return fmt.Errorf("couldn't release lock file %q: %v", lockPath, err)
		}
		return nil
	}

	g := newGenerator(c)
	var (
		sc scorer
		pr printer
	)
	genNext := func() *scoredProg {
		p := g.genProg()
		return &scoredProg{Source: pr.print(p), Score: sc.score(p)}
	}
	next := genNext()

	// Populate the WAL to signal to the driver that startup is complete.
	// We are about to enter the main loop and the driver can begin to
	// monitor the worker's progress. The driver should have ensured
	// that outPath doesn't exist yet.
	if err := write(&output{Next: next}); err != nil {
		return err
	}

	var (
		trialsDone  int
		parseErrors int
		typeErrors  int
		successes   int
		minPanic    *scoredProg
	)
	for trialsDone < trials {
		// Type-check the in-hand program.
		switch check(next.Source, std) {
		case parseErr:
			parseErrors++
		case typeErr:
			typeErrors++
		case success:
			successes++
		case panik:
			minPanic = min(minPanic, next)
		}
		trialsDone++

		// Prepare the next program if needed.
		if trialsDone < trials {
			next = genNext()
		} else {
			next = nil
		}

		// Report progress back to the driver.
		if err := write(&output{
			TrialsDone:  trialsDone,
			Next:        next,
			ParseErrors: parseErrors,
			TypeErrors:  typeErrors,
			Successes:   successes,
			Panic:       minPanic,
		}); err != nil {
			return err
		}
	}
	return nil
}

// A tcOutcome is the result of type-checking a program. The outcomes form a
// partition.
type tcOutcome int

const (
	parseErr tcOutcome = iota
	typeErr
	success
	panik
)

// check parses and type-checks src.
func check(src string, std bool) (out tcOutcome) {
	defer func() {
		if r := recover(); r != nil {
			out = panik
		}
	}()

	if std {
		return checkStd(src)
	}
	return checkGC(src)
}

// checkStd parses and type-checks src using the standard library's facilities.
func checkStd(src string) tcOutcome {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, 0)
	if err != nil {
		return parseErr
	}
	conf := types.Config{Error: func(error) {}}
	if _, err := conf.Check("p", fset, []*ast.File{f}, nil); err != nil {
		return typeErr
	}
	return success
}

// checkGC parses and type-checks src using the compiler's facilities.
func checkGC(src string) tcOutcome {
	f, err := syntax.Parse(syntax.NewFileBase("p.go"), strings.NewReader(src), nil, nil, 0)
	if err != nil {
		return parseErr
	}
	conf := types2.Config{Error: func(error) {}}
	if _, err := conf.Check("p", []*syntax.File{f}, nil); err != nil {
		return typeErr
	}
	return success
}
