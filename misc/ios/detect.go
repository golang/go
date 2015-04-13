// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// detect attempts to autodetect the correct
// values of the environment variables
// used by go_darwin_arm_exec.
package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
)

func main() {
	devID := detectDevID()
	fmt.Printf("export GOIOS_DEV_ID=%s\n", devID)

	udid := detectUDID()
	mp := detectMobileProvisionFile(udid)

	f, err := ioutil.TempFile("", "go_ios_detect_")
	check(err)
	fname := f.Name()
	defer os.Remove(fname)

	out := combinedOutput(parseMobileProvision(mp))
	_, err = f.Write(out)
	check(err)
	check(f.Close())

	appID, err := plistExtract(fname, "ApplicationIdentifierPrefix:0")
	check(err)
	fmt.Printf("export GOIOS_APP_ID=%s\n", appID)

	teamID, err := plistExtract(fname, "Entitlements:com.apple.developer.team-identifier")
	check(err)
	fmt.Printf("export GOIOS_TEAM_ID=%s\n", teamID)
}

func detectDevID() string {
	cmd := exec.Command("security", "find-identity", "-p", "codesigning", "-v")
	lines := getLines(cmd)

	for _, line := range lines {
		if !bytes.Contains(line, []byte("iPhone Developer")) {
			continue
		}
		fields := bytes.Fields(line)
		return string(fields[1])
	}
	fail("no code signing identity found")
	panic("unreachable")
}

var udidPrefix = []byte("UniqueDeviceID: ")

func detectUDID() []byte {
	cmd := exec.Command("ideviceinfo")
	lines := getLines(cmd)
	for _, line := range lines {
		if bytes.HasPrefix(line, udidPrefix) {
			return bytes.TrimPrefix(line, udidPrefix)
		}
	}
	fail("udid not found; is the device connected?")
	panic("unreachable")
}

func detectMobileProvisionFile(udid []byte) string {
	cmd := exec.Command("mdfind", "-name", ".mobileprovision")
	lines := getLines(cmd)

	for _, line := range lines {
		if len(line) == 0 {
			continue
		}
		xmlLines := getLines(parseMobileProvision(string(line)))
		for _, xmlLine := range xmlLines {
			if bytes.Contains(xmlLine, udid) {
				return string(line)
			}
		}
	}
	fail("did not find mobile provision matching device udid %s", udid)
	panic("ureachable")
}

func parseMobileProvision(fname string) *exec.Cmd {
	return exec.Command("security", "cms", "-D", "-i", string(fname))
}

func plistExtract(fname string, path string) ([]byte, error) {
	out, err := exec.Command("/usr/libexec/PlistBuddy", "-c", "Print "+path, fname).CombinedOutput()
	if err != nil {
		return nil, err
	}
	return bytes.TrimSpace(out), nil
}

func getLines(cmd *exec.Cmd) [][]byte {
	out := combinedOutput(cmd)
	return bytes.Split(out, []byte("\n"))
}

func combinedOutput(cmd *exec.Cmd) []byte {
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(strings.Join(cmd.Args, "\n"))
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	return out
}

func check(err error) {
	if err != nil {
		fail(err.Error())
	}
}

func fail(msg string, v ...interface{}) {
	fmt.Fprintf(os.Stderr, msg, v...)
	fmt.Fprintln(os.Stderr)
	os.Exit(1)
}
