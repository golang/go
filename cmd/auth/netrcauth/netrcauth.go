// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// netrcauth uses a .netrc file (or _netrc file on Windows) to implement the
// GOAUTH protocol described in https://golang.org/issue/26232.
// It expects the location of the file as the first command-line argument.
//
// Example GOAUTH usage:
//
//	export GOAUTH="netrcauth $HOME/.netrc"
//
// See https://www.gnu.org/software/inetutils/manual/html_node/The-_002enetrc-file.html
// or run 'man 5 netrc' for a description of the .netrc file format.
package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s NETRCFILE [URL]", os.Args[0])
		os.Exit(2)
	}

	log.SetPrefix("netrcauth: ")

	if len(os.Args) != 2 {
		// An explicit URL was passed on the command line, but netrcauth does not
		// have any URL-specific output: it dumps the entire .netrc file at the
		// first call.
		return
	}

	path := os.Args[1]

	data, err := ioutil.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return
		}
		log.Fatalf("failed to read %s: %v\n", path, err)
	}

	u := &url.URL{Scheme: "https"}
	lines := parseNetrc(string(data))
	for _, l := range lines {
		u.Host = l.machine
		fmt.Printf("%s\n\n", u)

		req := &http.Request{Header: make(http.Header)}
		req.SetBasicAuth(l.login, l.password)
		req.Header.Write(os.Stdout)
		fmt.Println()
	}
}

// The following functions were extracted from src/cmd/go/internal/web2/web.go
// as of https://golang.org/cl/161698.

type netrcLine struct {
	machine  string
	login    string
	password string
}

func parseNetrc(data string) []netrcLine {
	// See https://www.gnu.org/software/inetutils/manual/html_node/The-_002enetrc-file.html
	// for documentation on the .netrc format.
	var nrc []netrcLine
	var l netrcLine
	inMacro := false
	for _, line := range strings.Split(data, "\n") {
		if inMacro {
			if line == "" {
				inMacro = false
			}
			continue
		}

		f := strings.Fields(line)
		i := 0
		for ; i < len(f)-1; i += 2 {
			// Reset at each "machine" token.
			// “The auto-login process searches the .netrc file for a machine token
			// that matches […]. Once a match is made, the subsequent .netrc tokens
			// are processed, stopping when the end of file is reached or another
			// machine or a default token is encountered.”
			switch f[i] {
			case "machine":
				l = netrcLine{machine: f[i+1]}
			case "default":
				break
			case "login":
				l.login = f[i+1]
			case "password":
				l.password = f[i+1]
			case "macdef":
				// “A macro is defined with the specified name; its contents begin with
				// the next .netrc line and continue until a null line (consecutive
				// new-line characters) is encountered.”
				inMacro = true
			}
			if l.machine != "" && l.login != "" && l.password != "" {
				nrc = append(nrc, l)
				l = netrcLine{}
			}
		}

		if i < len(f) && f[i] == "default" {
			// “There can be only one default token, and it must be after all machine tokens.”
			break
		}
	}

	return nrc
}
