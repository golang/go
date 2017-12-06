// run
// +build !nacl

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runs a build -S to capture the assembly language
// output, checks that the line numbers associated with
// the stream of instructions do not change "too much".
// The changes that fixes this (that reduces the amount
// of change) does so by treating register spill, reload,
// copy, and rematerializations as being "unimportant" and
// just assigns them the line numbers of whatever "real"
// instructions preceded them.

// nacl is excluded because this runs a compiler.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// updateEnv modifies env to ensure that key=val
func updateEnv(env *[]string, key, val string) {
	if val != "" {
		var found bool
		key = key + "="
		for i, kv := range *env {
			if strings.HasPrefix(kv, key) {
				(*env)[i] = key + val
				found = true
				break
			}
		}
		if !found {
			*env = append(*env, key+val)
		}
	}
}

func main() {
	testarch := os.Getenv("TESTARCH")     // Targets other platform in test compilation.
	debug := os.Getenv("TESTDEBUG") != "" // Output the relevant assembly language.

	cmd := exec.Command("go", "tool", "compile", "-S", "fixedbugs/issue18902b.go")
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cmd.Env = os.Environ()

	if testarch != "" {
		updateEnv(&cmd.Env, "GOARCH", testarch)
		updateEnv(&cmd.Env, "GOOS", "linux") // Simplify multi-arch testing
	}

	err := cmd.Run()
	if err != nil {
		fmt.Printf("%s\n%s", err, buf.Bytes())
		return
	}
	begin := "\"\".(*gcSortBuf).flush" // Text at beginning of relevant dissassembly.
	s := buf.String()
	i := strings.Index(s, begin)
	if i < 0 {
		fmt.Printf("Failed to find expected symbol %s in output\n%s\n", begin, s)
		return
	}
	s = s[i:]
	r := strings.NewReader(s)
	scanner := bufio.NewScanner(r)
	first := true                         // The first line after the begin text will be skipped
	beforeLineNumber := "issue18902b.go:" // Text preceding line number in each line.
	lbln := len(beforeLineNumber)

	var scannedCount, changes, sumdiffs float64

	prevVal := 0
	for scanner.Scan() {
		line := scanner.Text()
		if first {
			first = false
			continue
		}
		i = strings.Index(line, beforeLineNumber)
		if i < 0 {
			// Done reading lines
			const minLines = 150
			if scannedCount <= minLines { // When test was written, 251 lines observed on amd64; arm64 now obtains 184
				fmt.Printf("Scanned only %d lines, was expecting more than %d\n", int(scannedCount), minLines)
				return
			}
			// Note: when test was written, before changes=92, after=50 (was 62 w/o rematerialization NoXPos in *Value.copyInto())
			// and before sumdiffs=784, after=180 (was 446 w/o rematerialization NoXPos in *Value.copyInto())
			// Set the dividing line between pass and fail at the midpoint.
			// Normalize against instruction count in case we unroll loops, etc.
			if changes/scannedCount >= (50+92)/(2*scannedCount) || sumdiffs/scannedCount >= (180+784)/(2*scannedCount) {
				fmt.Printf("Line numbers change too much, # of changes=%.f, sumdiffs=%.f, # of instructions=%.f\n", changes, sumdiffs, scannedCount)
			}
			return
		}
		scannedCount++
		i += lbln
		lineVal, err := strconv.Atoi(line[i : i+3])
		if err != nil {
			fmt.Printf("Expected 3-digit line number after %s in %s\n", beforeLineNumber, line)
		}
		if prevVal == 0 {
			prevVal = lineVal
		}
		diff := lineVal - prevVal
		if diff < 0 {
			diff = -diff
		}
		if diff != 0 {
			changes++
			sumdiffs += float64(diff)
		}
		// If things change too much, set environment variable TESTDEBUG to help figure out what's up.
		// The "before" behavior can be recreated in DebugFriendlySetPosFrom (currently in gc/ssa.go)
		// by inserting unconditional
		//   	s.SetPos(v.Pos)
		// at the top of the function.

		if debug {
			fmt.Printf("%d %.f %.f %s\n", lineVal, changes, sumdiffs, line)
		}
		prevVal = lineVal
	}
	if err := scanner.Err(); err != nil {
		fmt.Println("Reading standard input:", err)
		return
	}
}
