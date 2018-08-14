// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine

package playground

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
)

func post(ctx context.Context, url, contentType string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return nil, fmt.Errorf("http.NewRequest: %v", err)
	}
	return http.DefaultClient.Do(req.WithContext(ctx))
}

func contextFunc(_ *http.Request) context.Context {
	return context.Background()
}

func report(r *http.Request, err error) {
	log.Println(err)
}
