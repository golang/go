// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build autocert

// This file contains autocert and cloud.google.com/go/storage
// dependencies we want to hide by default from the Go build system,
// which currently doesn't know how to fetch non-golang.org/x/*
// dependencies. The Dockerfile builds the production binary
// with this code using --tags=autocert.

package main

import (
	"context"
	"crypto/tls"
	"log"
	"net/http"

	"cloud.google.com/go/storage"
	"golang.org/x/build/autocertcache"
	"golang.org/x/crypto/acme/autocert"
)

func init() {
	runHTTPS = runHTTPSAutocert
}

func runHTTPSAutocert(h http.Handler) error {
	var cache autocert.Cache
	if b := *autoCertCacheBucket; b != "" {
		sc, err := storage.NewClient(context.Background())
		if err != nil {
			log.Fatalf("storage.NewClient: %v", err)
		}
		cache = autocertcache.NewGoogleCloudStorageCache(sc, b)
	}
	m := autocert.Manager{
		Prompt:     autocert.AcceptTOS,
		HostPolicy: autocert.HostWhitelist(*autoCertDomain),
		Cache:      cache,
	}
	s := &http.Server{
		Addr:      ":https",
		Handler:   h,
		TLSConfig: &tls.Config{GetCertificate: m.GetCertificate},
	}
	return s.ListenAndServeTLS("", "")
}
