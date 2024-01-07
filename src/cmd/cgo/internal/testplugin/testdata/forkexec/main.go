// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"os/exec"
	_ "plugin"
	"sync"
)

func main() {
	if os.Args[1] != "1" {
		return
	}

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// does not matter what we exec, just exec itself
			cmd := exec.Command("./forkexec.exe", "0")
			cmd.Run()
		}()
	}
	wg.Wait()
}
