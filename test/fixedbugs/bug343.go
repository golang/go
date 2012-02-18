// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1900

package main

func getArgs(data map[string]interface{}, keys ...string) map[string]string {
       ret := map[string]string{}
       var ok bool
       for _, k := range keys {
               ret[k], ok = data[k].(string)
               if !ok {}
       }
       return ret
}

func main() {
	x := getArgs(map[string]interface{}{"x":"y"}, "x")
	if x["x"] != "y" {
		println("BUG bug343", x)
	}
}
	

/*
typecheck [1008592b0]
.   INDREG a(1) l(15) x(24) tc(2) runtime.ret G0 string
bug343.go:15: internal compiler error: typecheck INDREG
*/
