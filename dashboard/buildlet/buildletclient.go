// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

// Package buildlet contains client tools for working with a buildlet
// server.
package buildlet // import "golang.org/x/tools/dashboard/buildlet"

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
)

// KeyPair is the TLS public certificate PEM file and its associated
// private key PEM file that a builder will use for its HTTPS
// server. The zero value means no HTTPs, which is used by the
// coordinator for machines running within a firewall.
type KeyPair struct {
	CertPEM string
	KeyPEM  string
}

// NoKeyPair is used by the coordinator to speak http directly to buildlets,
// inside their firewall, without TLS.
var NoKeyPair = KeyPair{}

// NewClient returns a *Client that will manipulate ipPort,
// authenticated using the provided keypair.
//
// This constructor returns immediately without testing the host or auth.
func NewClient(ipPort string, tls KeyPair) *Client {
	return &Client{
		ipPort: ipPort,
		tls:    tls,
	}
}

// A Client interacts with a single buildlet.
type Client struct {
	ipPort string
	tls    KeyPair
}

// URL returns the buildlet's URL prefix, without a trailing slash.
func (c *Client) URL() string {
	if c.tls != NoKeyPair {
		return "http://" + strings.TrimSuffix(c.ipPort, ":80")
	}
	return "https://" + strings.TrimSuffix(c.ipPort, ":443")
}

func (c *Client) PutTarball(r io.Reader) error {
	req, err := http.NewRequest("PUT", c.URL()+"/writetgz", r)
	if err != nil {
		return err
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode/100 != 2 {
		slurp, _ := ioutil.ReadAll(io.LimitReader(res.Body, 4<<10))
		return fmt.Errorf("%v; body: %s", res.Status, slurp)
	}
	return nil
}
