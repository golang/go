// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package url_test

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"strings"
)

func ExamplePathEscape() {
	path := url.PathEscape("my/cool+blog&about,stuff")
	fmt.Println(path)

	// Output:
	// my%2Fcool+blog&about%2Cstuff
}

func ExamplePathUnescape() {
	escapedPath := "my%2Fcool+blog&about%2Cstuff"
	path, err := url.PathUnescape(escapedPath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(path)

	// Output:
	// my/cool+blog&about,stuff
}

func ExampleQueryEscape() {
	query := url.QueryEscape("my/cool+blog&about,stuff")
	fmt.Println(query)

	// Output:
	// my%2Fcool%2Bblog%26about%2Cstuff
}

func ExampleQueryUnescape() {
	escapedQuery := "my%2Fcool%2Bblog%26about%2Cstuff"
	query, err := url.QueryUnescape(escapedQuery)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(query)

	// Output:
	// my/cool+blog&about,stuff
}

func ExampleValues() {
	v := url.Values{}
	v.Set("name", "Ava")
	v.Add("friend", "Jess")
	v.Add("friend", "Sarah")
	v.Add("friend", "Zoe")
	// v.Encode() == "name=Ava&friend=Jess&friend=Sarah&friend=Zoe"
	fmt.Println(v.Get("name"))
	fmt.Println(v.Get("friend"))
	fmt.Println(v["friend"])
	// Output:
	// Ava
	// Jess
	// [Jess Sarah Zoe]
}

func ExampleValues_Add() {
	v := url.Values{}
	v.Add("cat sounds", "meow")
	v.Add("cat sounds", "mew")
	v.Add("cat sounds", "mau")
	fmt.Println(v["cat sounds"])

	// Output:
	// [meow mew mau]
}

func ExampleValues_Del() {
	v := url.Values{}
	v.Add("cat sounds", "meow")
	v.Add("cat sounds", "mew")
	v.Add("cat sounds", "mau")
	fmt.Println(v["cat sounds"])

	v.Del("cat sounds")
	fmt.Println(v["cat sounds"])

	// Output:
	// [meow mew mau]
	// []
}

func ExampleValues_Encode() {
	v := url.Values{}
	v.Add("cat sounds", "meow")
	v.Add("cat sounds", "mew/")
	v.Add("cat sounds", "mau$")
	fmt.Println(v.Encode())

	// Output:
	// cat+sounds=meow&cat+sounds=mew%2F&cat+sounds=mau%24
}

func ExampleValues_Get() {
	v := url.Values{}
	v.Add("cat sounds", "meow")
	v.Add("cat sounds", "mew")
	v.Add("cat sounds", "mau")
	fmt.Printf("%q\n", v.Get("cat sounds"))
	fmt.Printf("%q\n", v.Get("dog sounds"))

	// Output:
	// "meow"
	// ""
}

func ExampleValues_Has() {
	v := url.Values{}
	v.Add("cat sounds", "meow")
	v.Add("cat sounds", "mew")
	v.Add("cat sounds", "mau")
	fmt.Println(v.Has("cat sounds"))
	fmt.Println(v.Has("dog sounds"))

	// Output:
	// true
	// false
}

func ExampleValues_Set() {
	v := url.Values{}
	v.Add("cat sounds", "meow")
	v.Add("cat sounds", "mew")
	v.Add("cat sounds", "mau")
	fmt.Println(v["cat sounds"])

	v.Set("cat sounds", "meow")
	fmt.Println(v["cat sounds"])

	// Output:
	// [meow mew mau]
	// [meow]
}

func ExampleURL() {
	u, err := url.Parse("http://bing.com/search?q=dotnet")
	if err != nil {
		log.Fatal(err)
	}
	u.Scheme = "https"
	u.Host = "google.com"
	q := u.Query()
	q.Set("q", "golang")
	u.RawQuery = q.Encode()
	fmt.Println(u)
	// Output: https://google.com/search?q=golang
}

func ExampleURL_roundtrip() {
	// Parse + String preserve the original encoding.
	u, err := url.Parse("https://example.com/foo%2fbar")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(u.Path)
	fmt.Println(u.RawPath)
	fmt.Println(u.String())
	// Output:
	// /foo/bar
	// /foo%2fbar
	// https://example.com/foo%2fbar
}

func ExampleURL_ResolveReference() {
	u, err := url.Parse("../../..//search?q=dotnet")
	if err != nil {
		log.Fatal(err)
	}
	base, err := url.Parse("http://example.com/directory/")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(base.ResolveReference(u))
	// Output:
	// http://example.com/search?q=dotnet
}

func ExampleParseQuery() {
	m, err := url.ParseQuery(`x=1&y=2&y=3`)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(toJSON(m))
	// Output:
	// {"x":["1"], "y":["2", "3"]}
}

func ExampleURL_EscapedPath() {
	u, err := url.Parse("http://example.com/x/y%2Fz")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Path:", u.Path)
	fmt.Println("RawPath:", u.RawPath)
	fmt.Println("EscapedPath:", u.EscapedPath())
	// Output:
	// Path: /x/y/z
	// RawPath: /x/y%2Fz
	// EscapedPath: /x/y%2Fz
}

func ExampleURL_EscapedFragment() {
	u, err := url.Parse("http://example.com/#x/y%2Fz")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Fragment:", u.Fragment)
	fmt.Println("RawFragment:", u.RawFragment)
	fmt.Println("EscapedFragment:", u.EscapedFragment())
	// Output:
	// Fragment: x/y/z
	// RawFragment: x/y%2Fz
	// EscapedFragment: x/y%2Fz
}

func ExampleURL_Hostname() {
	u, err := url.Parse("https://example.org:8000/path")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(u.Hostname())
	u, err = url.Parse("https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:17000")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(u.Hostname())
	// Output:
	// example.org
	// 2001:0db8:85a3:0000:0000:8a2e:0370:7334
}

func ExampleURL_IsAbs() {
	u := url.URL{Host: "example.com", Path: "foo"}
	fmt.Println(u.IsAbs())
	u.Scheme = "http"
	fmt.Println(u.IsAbs())
	// Output:
	// false
	// true
}

func ExampleURL_MarshalBinary() {
	u, _ := url.Parse("https://example.org")
	b, err := u.MarshalBinary()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", b)
	// Output:
	// https://example.org
}

func ExampleURL_Parse() {
	u, err := url.Parse("https://example.org")
	if err != nil {
		log.Fatal(err)
	}
	rel, err := u.Parse("/foo")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(rel)
	_, err = u.Parse(":foo")
	if _, ok := err.(*url.Error); !ok {
		log.Fatal(err)
	}
	// Output:
	// https://example.org/foo
}

func ExampleURL_Port() {
	u, err := url.Parse("https://example.org")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(u.Port())
	u, err = url.Parse("https://example.org:8080")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(u.Port())
	// Output:
	//
	// 8080
}

func ExampleURL_Query() {
	u, err := url.Parse("https://example.org/?a=1&a=2&b=&=3&&&&")
	if err != nil {
		log.Fatal(err)
	}
	q := u.Query()
	fmt.Println(q["a"])
	fmt.Println(q.Get("b"))
	fmt.Println(q.Get(""))
	// Output:
	// [1 2]
	//
	// 3
}

func ExampleURL_String() {
	u := &url.URL{
		Scheme:   "https",
		User:     url.UserPassword("me", "pass"),
		Host:     "example.com",
		Path:     "foo/bar",
		RawQuery: "x=1&y=2",
		Fragment: "anchor",
	}
	fmt.Println(u.String())
	u.Opaque = "opaque"
	fmt.Println(u.String())
	// Output:
	// https://me:pass@example.com/foo/bar?x=1&y=2#anchor
	// https:opaque?x=1&y=2#anchor
}

func ExampleURL_UnmarshalBinary() {
	u := &url.URL{}
	err := u.UnmarshalBinary([]byte("https://example.org/foo"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", u)
	// Output:
	// https://example.org/foo
}

func ExampleURL_Redacted() {
	u := &url.URL{
		Scheme: "https",
		User:   url.UserPassword("user", "password"),
		Host:   "example.com",
		Path:   "foo/bar",
	}
	fmt.Println(u.Redacted())
	u.User = url.UserPassword("me", "newerPassword")
	fmt.Println(u.Redacted())
	// Output:
	// https://user:xxxxx@example.com/foo/bar
	// https://me:xxxxx@example.com/foo/bar
}

func ExampleURL_RequestURI() {
	u, err := url.Parse("https://example.org/path?foo=bar")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(u.RequestURI())
	// Output: /path?foo=bar
}

func toJSON(m interface{}) string {
	js, err := json.Marshal(m)
	if err != nil {
		log.Fatal(err)
	}
	return strings.ReplaceAll(string(js), ",", ", ")
}
