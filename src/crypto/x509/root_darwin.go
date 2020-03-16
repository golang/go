// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run root_darwin_arm_gen.go -output root_darwin_armx.go

package x509

import (
	"bufio"
	"bytes"
	"crypto/sha1"
	"encoding/pem"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

var debugDarwinRoots = strings.Contains(os.Getenv("GODEBUG"), "x509roots=1")

func (c *Certificate) systemVerify(opts *VerifyOptions) (chains [][]*Certificate, err error) {
	return nil, nil
}

// This code is only used when compiling without cgo.
// It is here, instead of root_nocgo_darwin.go, so that tests can check it
// even if the tests are run with cgo enabled.
// The linker will not include these unused functions in binaries built with cgo enabled.

// execSecurityRoots finds the macOS list of trusted root certificates
// using only command-line tools. This is our fallback path when cgo isn't available.
//
// The strategy is as follows:
//
// 1. Run "security trust-settings-export" and "security
//    trust-settings-export -d" to discover the set of certs with some
//    user-tweaked trust policy. We're too lazy to parse the XML
//    (Issue 26830) to understand what the trust
//    policy actually is. We just learn that there is _some_ policy.
//
// 2. Run "security find-certificate" to dump the list of system root
//    CAs in PEM format.
//
// 3. For each dumped cert, conditionally verify it with "security
//    verify-cert" if that cert was in the set discovered in Step 1.
//    Without the Step 1 optimization, running "security verify-cert"
//    150-200 times takes 3.5 seconds. With the optimization, the
//    whole process takes about 180 milliseconds with 1 untrusted root
//    CA. (Compared to 110ms in the cgo path)
func execSecurityRoots() (*CertPool, error) {
	hasPolicy, err := getCertsWithTrustPolicy()
	if err != nil {
		return nil, err
	}
	if debugDarwinRoots {
		fmt.Fprintf(os.Stderr, "crypto/x509: %d certs have a trust policy\n", len(hasPolicy))
	}

	keychains := []string{"/Library/Keychains/System.keychain"}

	// Note that this results in trusting roots from $HOME/... (the environment
	// variable), which might not be expected.
	home, err := os.UserHomeDir()
	if err != nil {
		if debugDarwinRoots {
			fmt.Fprintf(os.Stderr, "crypto/x509: can't get user home directory: %v\n", err)
		}
	} else {
		keychains = append(keychains,
			filepath.Join(home, "/Library/Keychains/login.keychain"),

			// Fresh installs of Sierra use a slightly different path for the login keychain
			filepath.Join(home, "/Library/Keychains/login.keychain-db"),
		)
	}

	type rootCandidate struct {
		c      *Certificate
		system bool
	}

	var (
		mu          sync.Mutex
		roots       = NewCertPool()
		numVerified int // number of execs of 'security verify-cert', for debug stats
		wg          sync.WaitGroup
		verifyCh    = make(chan rootCandidate)
	)

	// Using 4 goroutines to pipe into verify-cert seems to be
	// about the best we can do. The verify-cert binary seems to
	// just RPC to another server with coarse locking anyway, so
	// running 16 at a time for instance doesn't help at all. Due
	// to the "if hasPolicy" check below, though, we will rarely
	// (or never) call verify-cert on stock macOS systems, though.
	// The hope is that we only call verify-cert when the user has
	// tweaked their trust policy. These 4 goroutines are only
	// defensive in the pathological case of many trust edits.
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for cert := range verifyCh {
				sha1CapHex := fmt.Sprintf("%X", sha1.Sum(cert.c.Raw))

				var valid bool
				verifyChecks := 0
				if hasPolicy[sha1CapHex] {
					verifyChecks++
					valid = verifyCertWithSystem(cert.c)
				} else {
					// Certificates not in SystemRootCertificates without user
					// or admin trust settings are not trusted.
					valid = cert.system
				}

				mu.Lock()
				numVerified += verifyChecks
				if valid {
					roots.AddCert(cert.c)
				}
				mu.Unlock()
			}
		}()
	}
	err = forEachCertInKeychains(keychains, func(cert *Certificate) {
		verifyCh <- rootCandidate{c: cert, system: false}
	})
	if err != nil {
		close(verifyCh)
		return nil, err
	}
	err = forEachCertInKeychains([]string{
		"/System/Library/Keychains/SystemRootCertificates.keychain",
	}, func(cert *Certificate) {
		verifyCh <- rootCandidate{c: cert, system: true}
	})
	if err != nil {
		close(verifyCh)
		return nil, err
	}
	close(verifyCh)
	wg.Wait()

	if debugDarwinRoots {
		fmt.Fprintf(os.Stderr, "crypto/x509: ran security verify-cert %d times\n", numVerified)
	}

	return roots, nil
}

func forEachCertInKeychains(paths []string, f func(*Certificate)) error {
	args := append([]string{"find-certificate", "-a", "-p"}, paths...)
	cmd := exec.Command("/usr/bin/security", args...)
	data, err := cmd.Output()
	if err != nil {
		return err
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
		cert, err := ParseCertificate(block.Bytes)
		if err != nil {
			continue
		}
		f(cert)
	}
	return nil
}

func verifyCertWithSystem(cert *Certificate) bool {
	data := pem.EncodeToMemory(&pem.Block{
		Type: "CERTIFICATE", Bytes: cert.Raw,
	})

	f, err := ioutil.TempFile("", "cert")
	if err != nil {
		fmt.Fprintf(os.Stderr, "can't create temporary file for cert: %v", err)
		return false
	}
	defer os.Remove(f.Name())
	if _, err := f.Write(data); err != nil {
		fmt.Fprintf(os.Stderr, "can't write temporary file for cert: %v", err)
		return false
	}
	if err := f.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "can't write temporary file for cert: %v", err)
		return false
	}
	cmd := exec.Command("/usr/bin/security", "verify-cert", "-p", "ssl", "-c", f.Name(), "-l", "-L")
	var stderr bytes.Buffer
	if debugDarwinRoots {
		cmd.Stderr = &stderr
	}
	if err := cmd.Run(); err != nil {
		if debugDarwinRoots {
			fmt.Fprintf(os.Stderr, "crypto/x509: verify-cert rejected %s: %q\n", cert.Subject, bytes.TrimSpace(stderr.Bytes()))
		}
		return false
	}
	if debugDarwinRoots {
		fmt.Fprintf(os.Stderr, "crypto/x509: verify-cert approved %s\n", cert.Subject)
	}
	return true
}

// getCertsWithTrustPolicy returns the set of certs that have a
// possibly-altered trust policy. The keys of the map are capitalized
// sha1 hex of the raw cert.
// They are the certs that should be checked against `security
// verify-cert` to see whether the user altered the default trust
// settings. This code is only used for cgo-disabled builds.
func getCertsWithTrustPolicy() (map[string]bool, error) {
	set := map[string]bool{}
	td, err := ioutil.TempDir("", "x509trustpolicy")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(td)
	run := func(file string, args ...string) error {
		file = filepath.Join(td, file)
		args = append(args, file)
		cmd := exec.Command("/usr/bin/security", args...)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			// If there are no trust settings, the
			// `security trust-settings-export` command
			// fails with:
			//    exit status 1, SecTrustSettingsCreateExternalRepresentation: No Trust Settings were found.
			// Rather than match on English substrings that are probably
			// localized on macOS, just interpret any failure to mean that
			// there are no trust settings.
			if debugDarwinRoots {
				fmt.Fprintf(os.Stderr, "crypto/x509: exec %q: %v, %s\n", cmd.Args, err, stderr.Bytes())
			}
			return nil
		}

		f, err := os.Open(file)
		if err != nil {
			return err
		}
		defer f.Close()

		// Gather all the runs of 40 capitalized hex characters.
		br := bufio.NewReader(f)
		var hexBuf bytes.Buffer
		for {
			b, err := br.ReadByte()
			isHex := ('A' <= b && b <= 'F') || ('0' <= b && b <= '9')
			if isHex {
				hexBuf.WriteByte(b)
			} else {
				if hexBuf.Len() == 40 {
					set[hexBuf.String()] = true
				}
				hexBuf.Reset()
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				return err
			}
		}

		return nil
	}
	if err := run("user", "trust-settings-export"); err != nil {
		return nil, fmt.Errorf("dump-trust-settings (user): %v", err)
	}
	if err := run("admin", "trust-settings-export", "-d"); err != nil {
		return nil, fmt.Errorf("dump-trust-settings (admin): %v", err)
	}
	return set, nil
}
