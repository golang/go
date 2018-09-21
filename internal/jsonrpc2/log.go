// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jsonrpc2

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
)

// Log is an implementation of Logger that outputs using log.Print
// It is not used by default, but is provided for easy logging in users code.
func Log(mode string, id *ID, method string, payload *json.RawMessage, err *Error) {
	buf := &bytes.Buffer{}
	fmt.Fprint(buf, mode)
	if id == nil {
		fmt.Fprintf(buf, " []")
	} else {
		fmt.Fprintf(buf, " [%v]", id)
	}
	if method != "" {
		fmt.Fprintf(buf, " %s", method)
	}
	if payload != nil {
		fmt.Fprintf(buf, " %s", *payload)
	}
	if err != nil {
		fmt.Fprintf(buf, " failed: %s", err)
	}
	log.Print(buf)
}
