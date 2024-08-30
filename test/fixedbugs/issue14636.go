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
	"strings"
)

func main() {
	checkLinkOutput("", "-B argument must start with 0x")
	checkLinkOutput("0", "-B argument must start with 0x")
	checkLinkOutput("0x", "usage")
	checkLinkOutput("0x0", "-B argument must have even number of digits")
	checkLinkOutput("0x00", "usage")
	checkLinkOutput("0xYZ", "-B argument contains invalid hex digit")
	checkLinkOutput("0x"+strings.Repeat("00", 32), "usage")
	checkLinkOutput("0x"+strings.Repeat("00", 33), "-B option too long (max 32 digits)")
}

func checkLinkOutput(buildid string, message string) {
	cmd := exec.Command("go", "tool", "link", "-B", buildid)
	out, err := cmd.CombinedOutput()
	if err == nil {
		log.Fatalf("expected cmd/link to fail")
	}

	firstLine := string(bytes.SplitN(out, []byte("\n"), 2)[0])
	if strings.HasPrefix(firstLine, "panic") {
		log.Fatalf("cmd/link panicked:\n%s", out)
	}

	if !strings.Contains(firstLine, message) {
		log.Fatalf("cmd/link output did not include expected message %q: %s", message, firstLine)
	}
}
