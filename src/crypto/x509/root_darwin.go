// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run root_darwin_arm_gen.go -output root_darwin_armx.go

package x509

import (
	"bytes"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
)

func (c *Certificate) systemVerify(opts *VerifyOptions) (chains [][]*Certificate, err error) {
	return nil, nil
}

// This code is only used when compiling without cgo.
// It is here, instead of root_nocgo_darwin.go, so that tests can check it
// even if the tests are run with cgo enabled.
// The linker will not include these unused functions in binaries built with cgo enabled.

func execSecurityRoots() (*CertPool, error) {
	cmd := exec.Command("/usr/bin/security", "find-certificate", "-a", "-p", "/System/Library/Keychains/SystemRootCertificates.keychain")
	data, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	var (
		mu    sync.Mutex
		roots = NewCertPool()
	)
	add := func(cert *Certificate) {
		mu.Lock()
		defer mu.Unlock()
		roots.AddCert(cert)
	}
	blockCh := make(chan *pem.Block)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for block := range blockCh {
				verifyCertWithSystem(block, add)
			}
		}()
	}
	for len(data) > 0 {
		var block *pem.Block
		block, data = pem.Decode(data)
		if block == nil {
			break
		}
		if block.Type != "CERTIFICATE" || len(block.Headers) != 0 {
			continue
		}
		blockCh <- block
	}
	close(blockCh)
	wg.Wait()
	return roots, nil
}

func verifyCertWithSystem(block *pem.Block, add func(*Certificate)) {
	data := pem.EncodeToMemory(block)
	var cmd *exec.Cmd
	if needsTmpFiles() {
		f, err := ioutil.TempFile("", "cert")
		if err != nil {
			fmt.Fprintf(os.Stderr, "can't create temporary file for cert: %v", err)
			return
		}
		defer os.Remove(f.Name())
		if _, err := f.Write(data); err != nil {
			fmt.Fprintf(os.Stderr, "can't write temporary file for cert: %v", err)
			return
		}
		if err := f.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "can't write temporary file for cert: %v", err)
			return
		}
		cmd = exec.Command("/usr/bin/security", "verify-cert", "-c", f.Name(), "-l")
	} else {
		cmd = exec.Command("/usr/bin/security", "verify-cert", "-c", "/dev/stdin", "-l")
		cmd.Stdin = bytes.NewReader(data)
	}
	if cmd.Run() == nil {
		// Non-zero exit means untrusted
		cert, err := ParseCertificate(block.Bytes)
		if err != nil {
			return
		}

		add(cert)
	}
}

var versionCache struct {
	sync.Once
	major int
}

// needsTmpFiles reports whether the OS is <= 10.11 (which requires real
// files as arguments to the security command).
func needsTmpFiles() bool {
	versionCache.Do(func() {
		release, err := syscall.Sysctl("kern.osrelease")
		if err != nil {
			return
		}
		for i, c := range release {
			if c == '.' {
				release = release[:i]
				break
			}
		}
		major, err := strconv.Atoi(release)
		if err != nil {
			return
		}
		versionCache.major = major
	})
	return versionCache.major <= 15
}
