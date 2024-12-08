// run

//go:build !nacl && !js && !wasip1 && !android && gc

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"log"
	"os/exec"
	"runtime"
	"strings"
)

func main() {
	// The cannot open file error indicates that the parsing of -B flag
	// succeeded and it failed at a later step.
	checkLinkOutput("0", "-B argument must start with 0x")
	checkLinkOutput("0x", "cannot open file nonexistent.o")
	checkLinkOutput("0x0", "-B argument must have even number of digits")
	checkLinkOutput("0x00", "cannot open file nonexistent.o")
	checkLinkOutput("0xYZ", "-B argument contains invalid hex digit")

	maxLen := 32
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		maxLen = 16
	}
	checkLinkOutput("0x"+strings.Repeat("00", maxLen), "cannot open file nonexistent.o")
	checkLinkOutput("0x"+strings.Repeat("00", maxLen+1), "-B option too long")
}

func checkLinkOutput(buildid string, message string) {
	cmd := exec.Command("go", "tool", "link", "-B", buildid, "nonexistent.o")
	out, err := cmd.CombinedOutput()
	if err == nil {
		log.Fatalf("expected cmd/link to fail")
	}

	firstLine := string(bytes.SplitN(out, []byte("\n"), 2)[0])
	if strings.HasPrefix(firstLine, "panic") {
		log.Fatalf("cmd/link panicked:\n%s", out)
	}

	if !strings.Contains(firstLine, message) {
		log.Fatalf("%s: cmd/link output did not include expected message %q: %s", buildid, message, firstLine)
	}
}
