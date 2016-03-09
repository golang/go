// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"encoding/binary"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"time"
)

func cstring(x []byte) string {
	i := bytes.IndexByte(x, '\x00')
	if i >= 0 {
		x = x[:i]
	}
	return string(x)
}

func tokenize(s string) []string {
	var f []string
	for {
		s = strings.TrimLeft(s, " \t\r\n")
		if s == "" {
			break
		}
		quote := false
		i := 0
		for ; i < len(s); i++ {
			if s[i] == '\'' {
				if quote && i+1 < len(s) && s[i+1] == '\'' {
					i++
					continue
				}
				quote = !quote
			}
			if !quote && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n') {
				break
			}
		}
		next := s[:i]
		s = s[i:]
		if strings.Contains(next, "'") {
			var buf []byte
			quote := false
			for i := 0; i < len(next); i++ {
				if next[i] == '\'' {
					if quote && i+1 < len(next) && next[i+1] == '\'' {
						i++
						buf = append(buf, '\'')
					}
					quote = !quote
					continue
				}
				buf = append(buf, next[i])
			}
			next = string(buf)
		}
		f = append(f, next)
	}
	return f
}

var atExitFuncs []func()

func AtExit(f func()) {
	atExitFuncs = append(atExitFuncs, f)
}

func Exit(code int) {
	for i := len(atExitFuncs) - 1; i >= 0; i-- {
		f := atExitFuncs[i]
		atExitFuncs = atExitFuncs[:i]
		f()
	}
	os.Exit(code)
}

var (
	cpuprofile     string
	memprofile     string
	memprofilerate int64
)

func startProfile() {
	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			log.Fatalf("%v", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("%v", err)
		}
		AtExit(pprof.StopCPUProfile)
	}
	if memprofile != "" {
		if memprofilerate != 0 {
			runtime.MemProfileRate = int(memprofilerate)
		}
		f, err := os.Create(memprofile)
		if err != nil {
			log.Fatalf("%v", err)
		}
		AtExit(func() {
			runtime.GC() // profile all outstanding allocations
			if err := pprof.WriteHeapProfile(f); err != nil {
				log.Fatalf("%v", err)
			}
		})
	}
}

func artrim(x []byte) string {
	i := 0
	j := len(x)
	for i < len(x) && x[i] == ' ' {
		i++
	}
	for j > i && x[j-1] == ' ' {
		j--
	}
	return string(x[i:j])
}

func stringtouint32(x []uint32, s string) {
	for i := 0; len(s) > 0; i++ {
		var buf [4]byte
		s = s[copy(buf[:], s):]
		x[i] = binary.LittleEndian.Uint32(buf[:])
	}
}

var start = time.Now()

func elapsed() float64 {
	return time.Since(start).Seconds()
}
