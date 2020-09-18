// Copyright 2018 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package transport provides a mechanism to send requests with https cert,
// key, and CA.
package transport

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"
	"sync"

	"github.com/google/pprof/internal/plugin"
)

type transport struct {
	cert       *string
	key        *string
	ca         *string
	caCertPool *x509.CertPool
	certs      []tls.Certificate
	initOnce   sync.Once
	initErr    error
}

const extraUsage = `    -tls_cert             TLS client certificate file for fetching profile and symbols
    -tls_key              TLS private key file for fetching profile and symbols
    -tls_ca               TLS CA certs file for fetching profile and symbols`

// New returns a round tripper for making requests with the
// specified cert, key, and ca. The flags tls_cert, tls_key, and tls_ca are
// added to the flagset to allow a user to specify the cert, key, and ca. If
// the flagset is nil, no flags will be added, and users will not be able to
// use these flags.
func New(flagset plugin.FlagSet) http.RoundTripper {
	if flagset == nil {
		return &transport{}
	}
	flagset.AddExtraUsage(extraUsage)
	return &transport{
		cert: flagset.String("tls_cert", "", "TLS client certificate file for fetching profile and symbols"),
		key:  flagset.String("tls_key", "", "TLS private key file for fetching profile and symbols"),
		ca:   flagset.String("tls_ca", "", "TLS CA certs file for fetching profile and symbols"),
	}
}

// initialize uses the cert, key, and ca to initialize the certs
// to use these when making requests.
func (tr *transport) initialize() error {
	var cert, key, ca string
	if tr.cert != nil {
		cert = *tr.cert
	}
	if tr.key != nil {
		key = *tr.key
	}
	if tr.ca != nil {
		ca = *tr.ca
	}

	if cert != "" && key != "" {
		tlsCert, err := tls.LoadX509KeyPair(cert, key)
		if err != nil {
			return fmt.Errorf("could not load certificate/key pair specified by -tls_cert and -tls_key: %v", err)
		}
		tr.certs = []tls.Certificate{tlsCert}
	} else if cert == "" && key != "" {
		return fmt.Errorf("-tls_key is specified, so -tls_cert must also be specified")
	} else if cert != "" && key == "" {
		return fmt.Errorf("-tls_cert is specified, so -tls_key must also be specified")
	}

	if ca != "" {
		caCertPool := x509.NewCertPool()
		caCert, err := ioutil.ReadFile(ca)
		if err != nil {
			return fmt.Errorf("could not load CA specified by -tls_ca: %v", err)
		}
		caCertPool.AppendCertsFromPEM(caCert)
		tr.caCertPool = caCertPool
	}

	return nil
}

// RoundTrip executes a single HTTP transaction, returning
// a Response for the provided Request.
func (tr *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	tr.initOnce.Do(func() {
		tr.initErr = tr.initialize()
	})
	if tr.initErr != nil {
		return nil, tr.initErr
	}

	tlsConfig := &tls.Config{
		RootCAs:      tr.caCertPool,
		Certificates: tr.certs,
	}

	if req.URL.Scheme == "https+insecure" {
		// Make shallow copy of request, and req.URL, so the request's URL can be
		// modified.
		r := *req
		*r.URL = *req.URL
		req = &r
		tlsConfig.InsecureSkipVerify = true
		req.URL.Scheme = "https"
	}

	transport := http.Transport{
		Proxy:           http.ProxyFromEnvironment,
		TLSClientConfig: tlsConfig,
	}

	return transport.RoundTrip(req)
}
