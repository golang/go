// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
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

// Exit exits with code after executing all atExitFuncs.
func Exit(code int) {
	for i := len(atExitFuncs) - 1; i >= 0; i-- {
		atExitFuncs[i]()
	}
	os.Exit(code)
}

// Exitf logs an error message then calls Exit(2).
func Exitf(format string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, os.Args[0]+": "+format+"\n", a...)
	if coutbuf.f != nil {
		coutbuf.f.Close()
		mayberemoveoutfile()
	}
	Exit(2)
}

// Errorf logs an error message.
//
// If more than 20 errors have been printed, exit with an error.
//
// Logging an error means that on exit cmd/link will delete any
// output file and return a non-zero error code.
func Errorf(s *Symbol, format string, args ...interface{}) {
	if s != nil {
		format = s.Name + ": " + format
	}
	format += "\n"
	fmt.Fprintf(os.Stderr, format, args...)
	nerrors++
	if *flagH {
		panic("error")
	}
	if nerrors > 20 {
		Exitf("too many errors")
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
