// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sysinfo

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"strings"
)

func readLinuxProcCPUInfo(buf []byte) error {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.ReadFull(f, buf)
	if err != nil && err != io.ErrUnexpectedEOF {
		return err
	}

	return nil
}

func osCPUInfoName() string {
	modelName := ""
	cpuMHz := ""

	// The 512-byte buffer is enough to hold the contents of CPU0
	buf := make([]byte, 512)
	err := readLinuxProcCPUInfo(buf)
	if err != nil {
		return ""
	}

	scanner := bufio.NewScanner(bytes.NewReader(buf))
	for scanner.Scan() {
		key, value, found := strings.Cut(scanner.Text(), ": ")
		if !found {
			continue
		}
		switch strings.TrimSpace(key) {
		case "Model Name", "model name":
			modelName = value
		case "CPU MHz", "cpu MHz":
			cpuMHz = value
		}
	}

	if modelName == "" {
		return ""
	}

	if cpuMHz == "" {
		return modelName
	}

	// The modelName field already contains the frequency information,
	// so the cpuMHz field information is not needed.
	// modelName filed example:
	//	Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
	f := [...]string{"GHz", "MHz"}
	for _, v := range f {
		if strings.Contains(modelName, v) {
			return modelName
		}
	}

	return modelName + " @ " + cpuMHz + "MHz"
}
