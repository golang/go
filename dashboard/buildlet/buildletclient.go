// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

// Package buildlet contains client tools for working with a buildlet
// server.
package buildlet // import "golang.org/x/tools/dashboard/buildlet"

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
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

// PutTarball writes files to the remote buildlet.
// The Reader must be of a tar.gz file.
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

// ExecOpts are options for a remote command invocation.
type ExecOpts struct {
	// Output is the output of stdout and stderr.
	// If nil, the output is discarded.
	Output io.Writer

	// OnStartExec is an optional hook that runs after the 200 OK
	// response from the buildlet, but before the output begins
	// writing to Output.
	OnStartExec func()
}

// Exec runs cmd on the buildlet.
//
// Two errors are returned: one is whether the command succeeded
// remotely (remoteErr), and the second (execErr) is whether there
// were system errors preventing the command from being started or
// seen to completition. If execErr is non-nil, the remoteErr is
// meaningless.
func (c *Client) Exec(cmd string, opts ExecOpts) (remoteErr, execErr error) {
	res, err := http.PostForm(c.URL()+"/exec", url.Values{"cmd": {cmd}})
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		slurp, _ := ioutil.ReadAll(io.LimitReader(res.Body, 4<<10))
		return nil, fmt.Errorf("buildlet: HTTP status %v: %s", res.Status, slurp)
	}
	condRun(opts.OnStartExec)

	// Stream the output:
	out := opts.Output
	if out == nil {
		out = ioutil.Discard
	}
	if _, err := io.Copy(out, res.Body); err != nil {
		return nil, fmt.Errorf("error copying response: %v", err)
	}

	// Don't record to the dashboard unless we heard the trailer from
	// the buildlet, otherwise it was probably some unrelated error
	// (like the VM being killed, or the buildlet crashing due to
	// e.g. https://golang.org/issue/9309, since we require a tip
	// build of the buildlet to get Trailers support)
	state := res.Trailer.Get("Process-State")
	if state == "" {
		return nil, errors.New("missing Process-State trailer from HTTP response; buildlet built with old (<= 1.4) Go?")
	}
	if state != "ok" {
		return errors.New(state), nil
	}
	return nil, nil
}

func condRun(fn func()) {
	if fn != nil {
		fn()
	}
}
