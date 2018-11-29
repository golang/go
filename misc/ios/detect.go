// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// detect attempts to autodetect the correct
// values of the environment variables
// used by go_darwin_arm_exec.
// detect shells out to ideviceinfo, a third party program that can
// be obtained by following the instructions at
// https://github.com/libimobiledevice/libimobiledevice.
package main

import (
	"bytes"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
)

func main() {
	udids := getLines(exec.Command("idevice_id", "-l"))
	if len(udids) == 0 {
		fail("no udid found; is a device connected?")
	}

	mps := detectMobileProvisionFiles(udids)
	if len(mps) == 0 {
		fail("did not find mobile provision matching device udids %q", udids)
	}

	fmt.Println("# Available provisioning profiles below.")
	fmt.Println("# NOTE: Any existing app on the device with the app id specified by GOIOS_APP_ID")
	fmt.Println("# will be overwritten when running Go programs.")
	for _, mp := range mps {
		fmt.Println()
		f, err := ioutil.TempFile("", "go_ios_detect_")
		check(err)
		fname := f.Name()
		defer os.Remove(fname)

		out := output(parseMobileProvision(mp))
		_, err = f.Write(out)
		check(err)
		check(f.Close())

		cert, err := plistExtract(fname, "DeveloperCertificates:0")
		check(err)
		pcert, err := x509.ParseCertificate(cert)
		check(err)
		fmt.Printf("export GOIOS_DEV_ID=\"%s\"\n", pcert.Subject.CommonName)

		appID, err := plistExtract(fname, "Entitlements:application-identifier")
		check(err)
		fmt.Printf("export GOIOS_APP_ID=%s\n", appID)

		teamID, err := plistExtract(fname, "Entitlements:com.apple.developer.team-identifier")
		check(err)
		fmt.Printf("export GOIOS_TEAM_ID=%s\n", teamID)
	}
}

func detectMobileProvisionFiles(udids [][]byte) []string {
	cmd := exec.Command("mdfind", "-name", ".mobileprovision")
	lines := getLines(cmd)

	var files []string
	for _, line := range lines {
		if len(line) == 0 {
			continue
		}
		xmlLines := getLines(parseMobileProvision(string(line)))
		matches := 0
		for _, udid := range udids {
			for _, xmlLine := range xmlLines {
				if bytes.Contains(xmlLine, udid) {
					matches++
				}
			}
		}
		if matches == len(udids) {
			files = append(files, string(line))
		}
	}
	return files
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
	out := output(cmd)
	lines := bytes.Split(out, []byte("\n"))
	// Skip the empty line at the end.
	if len(lines[len(lines)-1]) == 0 {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func output(cmd *exec.Cmd) []byte {
	out, err := cmd.Output()
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
