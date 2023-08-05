// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

var (
	cpuSetRE = regexp.MustCompile(`(\d,?)+`)
)

func init() {
	register("FreeBSDNumCPU", FreeBSDNumCPU)
	register("FreeBSDNumCPUHelper", FreeBSDNumCPUHelper)
}

func FreeBSDNumCPUHelper() {
	fmt.Printf("%d\n", runtime.NumCPU())
}

func FreeBSDNumCPU() {
	_, err := exec.LookPath("cpuset")
	if err != nil {
		// Can not test without cpuset command.
		fmt.Println("OK")
		return
	}
	_, err = exec.LookPath("sysctl")
	if err != nil {
		// Can not test without sysctl command.
		fmt.Println("OK")
		return
	}
	cmd := exec.Command("sysctl", "-n", "kern.smp.active")
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("fail to launch '%s', error: %s, output: %s\n", strings.Join(cmd.Args, " "), err, output)
		return
	}
	if !bytes.Equal(output, []byte("1\n")) {
		// SMP mode deactivated in kernel.
		fmt.Println("OK")
		return
	}

	list, err := getList()
	if err != nil {
		fmt.Printf("%s\n", err)
		return
	}
	err = checkNCPU(list)
	if err != nil {
		fmt.Printf("%s\n", err)
		return
	}
	if len(list) >= 2 {
		err = checkNCPU(list[:len(list)-1])
		if err != nil {
			fmt.Printf("%s\n", err)
			return
		}
	}
	fmt.Println("OK")
	return
}

func getList() ([]string, error) {
	pid := syscall.Getpid()

	// Launch cpuset to print a list of available CPUs: pid <PID> mask: 0, 1, 2, 3.
	cmd := exec.Command("cpuset", "-g", "-p", strconv.Itoa(pid))
	cmdline := strings.Join(cmd.Args, " ")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("fail to execute '%s': %s", cmdline, err)
	}
	output, _, ok := bytes.Cut(output, []byte("\n"))
	if !ok {
		return nil, fmt.Errorf("invalid output from '%s', '\\n' not found: %s", cmdline, output)
	}

	_, cpus, ok := bytes.Cut(output, []byte(":"))
	if !ok {
		return nil, fmt.Errorf("invalid output from '%s', ':' not found: %s", cmdline, output)
	}

	var list []string
	for _, val := range bytes.Split(cpus, []byte(",")) {
		index := string(bytes.TrimSpace(val))
		if len(index) == 0 {
			continue
		}
		list = append(list, index)
	}
	if len(list) == 0 {
		return nil, fmt.Errorf("empty CPU list from '%s': %s", cmdline, output)
	}
	return list, nil
}

func checkNCPU(list []string) error {
	listString := strings.Join(list, ",")
	if len(listString) == 0 {
		return fmt.Errorf("could not check against an empty CPU list")
	}

	cListString := cpuSetRE.FindString(listString)
	if len(cListString) == 0 {
		return fmt.Errorf("invalid cpuset output '%s'", listString)
	}
	// Launch FreeBSDNumCPUHelper() with specified CPUs list.
	cmd := exec.Command("cpuset", "-l", cListString, os.Args[0], "FreeBSDNumCPUHelper")
	cmdline := strings.Join(cmd.Args, " ")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("fail to launch child '%s', error: %s, output: %s", cmdline, err, output)
	}

	// NumCPU from FreeBSDNumCPUHelper come with '\n'.
	output = bytes.TrimSpace(output)
	n, err := strconv.Atoi(string(output))
	if err != nil {
		return fmt.Errorf("fail to parse output from child '%s', error: %s, output: %s", cmdline, err, output)
	}
	if n != len(list) {
		return fmt.Errorf("runtime.NumCPU() expected to %d, got %d when run with CPU list %s", len(list), n, cListString)
	}
	return nil
}
