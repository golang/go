// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generate a self-signed X.509 certificate for a TLS server. Outputs to
// 'cert.pem' and 'key.pem' and will overwrite existing files.

package main

import (
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"log"
	"os"
	"time"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Printf("Usage: %s <hostname of server>\n", os.Args[0])
		return
	}

	hostName := os.Args[1]

	urandom, err := os.Open("/dev/urandom", os.O_RDONLY, 0)
	if err != nil {
		log.Crashf("failed to open /dev/urandom: %s\n", err)
		return
	}

	log.Stdoutf("Generating RSA key\n")
	priv, err := rsa.GenerateKey(urandom, 1024)
	if err != nil {
		log.Crashf("failed to generate private key: %s\n", err)
		return
	}

	now := time.Seconds()

	template := x509.Certificate{
		SerialNumber: []byte{0},
		Subject: x509.Name{
			CommonName:   hostName,
			Organization: "Acme Co",
		},
		NotBefore: time.SecondsToUTC(now - 300),
		NotAfter:  time.SecondsToUTC(now + 86400*365), // valid for 1 year.

		SubjectKeyId: []byte{1, 2, 3, 4},
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
	}

	derBytes, err := x509.CreateCertificate(urandom, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		log.Crashf("Failed to create certificate: %s", err)
		return
	}

	certOut, err := os.Open("cert.pem", os.O_WRONLY|os.O_CREAT, 0644)
	if err != nil {
		log.Crashf("failed to open cert.pem for writing: %s\n", err)
		return
	}
	pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	certOut.Close()
	log.Stdoutf("written cert.pem\n")

	keyOut, err := os.Open("key.pem", os.O_WRONLY|os.O_CREAT, 0600)
	if err != nil {
		log.Crashf("failed to open key.pem for writing: %s\n", err)
		return
	}
	pem.Encode(keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})
	keyOut.Close()
	log.Stdoutf("written key.pem\n")
}
