// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"mime/multipart"
	"net/url"
	"testing"
)

func TestRequestCloneURLValues(t *testing.T) {

	dst := url.Values{"a": []string{"1"}}

	if cloneURLValues(dst).Get("a") != "1" {
		t.Fatalf("cloneURLValues not up to expectations")
	}
}

func TestRequestCloneURL(t *testing.T) {

	if cloneURL(nil) != nil {
		t.Fatalf("cloneURL not up to expectations")
	}

	u, _ := url.Parse("http://www.host.test/")
	u.User = url.UserPassword("test", "pwd123")

	res := cloneURL(u)

	if res.User.Username() != "test" {
		t.Fatalf("cloneURL not up to expectations")
	}
}

func TestRequestCloneMultipartForm(t *testing.T) {

	var (
		f        multipart.Form
		valueMap = map[string][]string{}
		fileMap  = make(map[string][]*multipart.FileHeader)
		fileSli  = make([]*multipart.FileHeader, 0)
	)

	valueMap["field1"] = []string{"value1", "value2"}

	tmp := &multipart.FileHeader{
		Filename: "123.txt",
		Header:   nil,
		Size:     128,
	}

	fileSli = append(fileSli, tmp)
	fileMap["file1"] = fileSli

	f.Value = valueMap
	f.File = fileMap

	res := cloneMultipartForm(&f)

	if res.Value["field1"][0] != "value1" || res.File["file1"][0].Size != 128 {
		t.Fatalf("cloneMultipartForm not up to expectations")
	}
}

func TestRequestCloneMultipartFileHeader(t *testing.T) {
	if cloneMultipartFileHeader(nil) != nil {
		t.Fatalf("cloneMultipartFileHeader not up to expectations")
	}
}
