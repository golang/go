// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"os"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestRead(t *testing.T) {
	t.Run("SpecialFile", func { t ->
		var wg sync.WaitGroup
		for _, p := range specialFiles() {
			for i := 0; i < 4; i++ {
				wg.Add(1)
				go func(p string) {
					defer wg.Done()
					for i := 0; i < 100; i++ {
						if _, err := os.ReadFile(p); err != nil {
							t.Error(err)
							return
						}
						time.Sleep(time.Nanosecond)
					}
				}(p)
			}
		}
		wg.Wait()
	})
}

func specialFiles() []string {
	var ps []string
	switch runtime.GOOS {
	case "darwin", "ios", "dragonfly", "freebsd", "netbsd", "openbsd":
		ps = []string{
			"/dev/null",
		}
	case "linux":
		ps = []string{
			"/dev/null",
			"/proc/stat",
			"/sys/devices/system/cpu/online",
		}
	}
	nps := ps[:0]
	for _, p := range ps {
		f, err := os.Open(p)
		if err != nil {
			continue
		}
		f.Close()
		nps = append(nps, p)
	}
	return nps
}
