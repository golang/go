// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package http provides HTTP client and server implementations.

Get, Head, Post, and PostForm make HTTP requests:

	resp, err := http.Get("http://example.com/")
	...
	resp, err := http.Post("http://example.com/upload", "image/jpeg", &buf)
	...
	resp, err := http.PostForm("http://example.com/form",
 		url.Values{"key": {"Value"}, "id": {"123"}})

The client must close the response body when finished with it:

	resp, err := http.Get("http://example.com/")
	if err != nil {
		// handle error
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	// ...

For control over HTTP client headers, redirect policy, and other
settings, create a Client:

	client := &http.Client{
		CheckRedirect: redirectPolicyFunc,
	}

	resp, err := client.Get("http://example.com")
	// ...

	req := http.NewRequest("GET", "http://example.com", nil)
	req.Header.Add("If-None-Match", `W/"wyzzy"`)
	resp, err := client.Do(req)
	// ...

For control over proxies, TLS configuration, keep-alives,
compression, and other settings, create a Transport:

	tr := &http.Transport{
		TLSClientConfig:    &tls.Config{RootCAs: pool},
		DisableCompression: true,
	}
	client := &http.Client{Transport: tr}
	resp, err := client.Get("https://example.com")

Clients and Transports are safe for concurrent use by multiple
goroutines and for efficiency should only be created once and re-used.

ListenAndServe starts an HTTP server with a given address and handler.
The handler is usually nil, which means to use DefaultServeMux.
Handle and HandleFunc add handlers to DefaultServeMux:

	http.Handle("/foo", fooHandler)

	http.HandleFunc("/bar", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello, %q", html.EscapeString(r.URL.RawPath))
	})

	log.Fatal(http.ListenAndServe(":8080", nil))

More control over the server's behavior is available by creating a
custom Server:

	s := &http.Server{
		Addr:           ":8080",
		Handler:        myHandler,
		ReadTimeout:    10 * time.Second,
		WriteTimeout:   10 * time.Second,
		MaxHeaderBytes: 1 << 20,
	}
	log.Fatal(s.ListenAndServe())
*/
package http
