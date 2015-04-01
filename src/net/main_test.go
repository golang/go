// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"net/internal/socktest"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
)

var (
	sw socktest.Switch

	// uninstallTestHooks runs just before a run of benchmarks.
	testHookUninstaller sync.Once
)

func TestMain(m *testing.M) {
	installTestHooks()

	st := m.Run()

	testHookUninstaller.Do(func() { uninstallTestHooks() })
	if !testing.Short() {
		printLeakedGoroutines()
		printLeakedSockets()
		printSocketStats()
	}
	forceCloseSockets()
	os.Exit(st)
}

func printLeakedGoroutines() {
	gss := leakedGoroutines()
	if len(gss) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "Leaked goroutines:\n")
	for _, gs := range gss {
		fmt.Fprintf(os.Stderr, "%v\n", gs)
	}
	fmt.Fprintf(os.Stderr, "\n")
}

// leakedGoroutines returns a list of remaining goroutins used in test
// cases.
func leakedGoroutines() []string {
	var gss []string
	b := make([]byte, 2<<20)
	b = b[:runtime.Stack(b, true)]
	for _, s := range strings.Split(string(b), "\n\n") {
		ss := strings.SplitN(s, "\n", 2)
		if len(ss) != 2 {
			continue
		}
		stack := strings.TrimSpace(ss[1])
		if !strings.Contains(stack, "created by net") {
			continue
		}
		gss = append(gss, stack)
	}
	sort.Strings(gss)
	return gss
}

func printLeakedSockets() {
	sos := sw.Sockets()
	if len(sos) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "Leaked sockets:\n")
	for s, so := range sos {
		fmt.Fprintf(os.Stderr, "%v: %+v\n", s, so)
	}
	fmt.Fprintf(os.Stderr, "\n")
}

func printSocketStats() {
	sts := sw.Stats()
	if len(sts) == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "Socket statistical information:\n")
	for _, st := range sts {
		fmt.Fprintf(os.Stderr, "%+v\n", st)
	}
	fmt.Fprintf(os.Stderr, "\n")
}
