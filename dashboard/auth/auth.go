// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

// Package auth contains shared code related to OAuth2 and obtaining
// tokens for a project.
package auth

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

func homedir() string {
	if runtime.GOOS == "windows" {
		return os.Getenv("HOMEDRIVE") + os.Getenv("HOMEPATH")
	}
	return os.Getenv("HOME")
}

// ProjectTokenSource returns an OAuth2 TokenSource for the given Google Project ID.
func ProjectTokenSource(proj string, scopes ...string) (oauth2.TokenSource, error) {
	// TODO(bradfitz): try different strategies too, like
	// three-legged flow if the service account doesn't exist, and
	// then cache the token file on disk somewhere. Or maybe that should be an
	// option, for environments without stdin/stdout available to the user.
	// We'll figure it out as needed.
	fileName := filepath.Join(homedir(), "keys", proj+".key.json")
	jsonConf, err := ioutil.ReadFile(fileName)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("Missing JSON key configuration. Download the Service Account JSON key from https://console.developers.google.com/project/%s/apiui/credential and place it at %s", proj, fileName)
		}
		return nil, err
	}
	conf, err := google.JWTConfigFromJSON(jsonConf, scopes...)
	if err != nil {
		return nil, fmt.Errorf("reading JSON config from %s: %v", fileName, err)
	}
	return conf.TokenSource(oauth2.NoContext), nil
}
