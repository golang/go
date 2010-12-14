// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generate a self-signed X.509 certificate for a TLS server. Outputs to
// 'cert.pem' and 'key.pem' and will overwrite existing files.

package main

import (
	"crypto/rsa"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"flag"
	"log"
	"os"
	"time"
)

var hostName *string = flag.String("host", "127.0.0.1", "Hostname to generate a certificate for")

func main() {
	flag.Parse()

	priv, err := rsa.GenerateKey(rand.Reader, 1024)
	if err != nil {
		log.Exitf("failed to generate private key: %s", err)
		return
	}

	now := time.Seconds()

	template := x509.Certificate{
		SerialNumber: []byte{0},
		Subject: x509.Name{
			CommonName:   *hostName,
			Organization: []string{"Acme Co"},
		},
		NotBefore: time.SecondsToUTC(now - 300),
		NotAfter:  time.SecondsToUTC(now + 60*60*24*365), // valid for 1 year.

		SubjectKeyId: []byte{1, 2, 3, 4},
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		log.Exitf("Failed to create certificate: %s", err)
		return
	}

	certOut, err := os.Open("cert.pem", os.O_WRONLY|os.O_CREAT, 0644)
	if err != nil {
		log.Exitf("failed to open cert.pem for writing: %s", err)
		return
	}
	pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	certOut.Close()
	log.Print("written cert.pem\n")

	keyOut, err := os.Open("key.pem", os.O_WRONLY|os.O_CREAT, 0600)
	if err != nil {
		log.Print("failed to open key.pem for writing:", err)
		return
	}
	pem.Encode(keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})
	keyOut.Close()
	log.Print("written key.pem\n")
}
