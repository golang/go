// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Generates root_darwin_armx.go.
//
// As of iOS 8, there is no API for querying the system trusted X.509 root
// certificates. We could use SecTrustEvaluate to verify that a trust chain
// exists for a certificate, but the x509 API requires returning the entire
// chain.
//
// Apple publishes the list of trusted root certificates for iOS on
// support.apple.com. So we parse the list and extract the certificates from
// an OS X machine and embed them into the x509 package.
package main

import (
	"bytes"
	"crypto/x509"
	"encoding/pem"
	"flag"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
	"math/big"
	"net/http"
	"os/exec"
	"strings"
)

var output = flag.String("output", "root_darwin_armx.go", "file name to write")

func main() {
	certs, err := selectCerts()
	if err != nil {
		log.Fatal(err)
	}

	buf := new(bytes.Buffer)

	fmt.Fprintf(buf, "// Created by root_darwin_arm_gen --output %s; DO NOT EDIT\n", *output)
	fmt.Fprintf(buf, "%s", header)

	fmt.Fprintf(buf, "const systemRootsPEM = `\n")
	for _, cert := range certs {
		b := &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: cert.Raw,
		}
		if err := pem.Encode(buf, b); err != nil {
			log.Fatal(err)
		}
	}
	fmt.Fprintf(buf, "`")

	source, err := format.Source(buf.Bytes())
	if err != nil {
		log.Fatal("source format error:", err)
	}
	if err := ioutil.WriteFile(*output, source, 0644); err != nil {
		log.Fatal(err)
	}
}

func selectCerts() ([]*x509.Certificate, error) {
	ids, err := fetchCertIDs()
	if err != nil {
		return nil, err
	}

	scerts, err := sysCerts()
	if err != nil {
		return nil, err
	}

	var certs []*x509.Certificate
	for _, id := range ids {
		sn, ok := big.NewInt(0).SetString(id.serialNumber, 0) // 0x prefix selects hex
		if !ok {
			return nil, fmt.Errorf("invalid serial number: %q", id.serialNumber)
		}
		ski, ok := big.NewInt(0).SetString(id.subjectKeyID, 0)
		if !ok {
			return nil, fmt.Errorf("invalid Subject Key ID: %q", id.subjectKeyID)
		}

		for _, cert := range scerts {
			if sn.Cmp(cert.SerialNumber) != 0 {
				continue
			}
			cski := big.NewInt(0).SetBytes(cert.SubjectKeyId)
			if ski.Cmp(cski) != 0 {
				continue
			}
			certs = append(certs, cert)
			break
		}
	}
	return certs, nil
}

func sysCerts() (certs []*x509.Certificate, err error) {
	cmd := exec.Command("/usr/bin/security", "find-certificate", "-a", "-p", "/System/Library/Keychains/SystemRootCertificates.keychain")
	data, err := cmd.Output()
	if err != nil {
		return nil, err
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

		cert, err := x509.ParseCertificate(block.Bytes)
		if err != nil {
			continue
		}
		certs = append(certs, cert)
	}
	return certs, nil
}

type certID struct {
	serialNumber string
	subjectKeyID string
}

// fetchCertIDs fetches IDs of iOS X509 certificates from apple.com.
func fetchCertIDs() ([]certID, error) {
	resp, err := http.Get("https://support.apple.com/en-us/HT204132")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	text := string(body)
	text = text[strings.Index(text, "<section id=trusted"):]
	text = text[:strings.Index(text, "</section>")]

	lines := strings.Split(text, "\n")
	var ids []certID
	var id certID
	for i, ln := range lines {
		if i == len(lines)-1 {
			break
		}
		const sn = "Serial Number:"
		if ln == sn {
			id.serialNumber = "0x" + strings.Replace(strings.TrimSpace(lines[i+1]), ":", "", -1)
			continue
		}
		if strings.HasPrefix(ln, sn) {
			// extract hex value from parentheses.
			id.serialNumber = ln[strings.Index(ln, "(")+1 : len(ln)-1]
			continue
		}
		if strings.TrimSpace(ln) == "X509v3 Subject Key Identifier:" {
			id.subjectKeyID = "0x" + strings.Replace(strings.TrimSpace(lines[i+1]), ":", "", -1)
			ids = append(ids, id)
			id = certID{}
		}
	}
	return ids, nil
}

const header = `
// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo
// +build darwin
// +build arm arm64

package x509

func initSystemRoots() {
	systemRoots = NewCertPool()
	systemRoots.AppendCertsFromPEM([]byte(systemRootsPEM))
}
`
