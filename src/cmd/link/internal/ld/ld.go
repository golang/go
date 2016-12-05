// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/obj.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/span.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package ld

import (
	"cmd/internal/obj"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

func addlib(ctxt *Link, src string, obj string, pathname string) *Library {
	name := path.Clean(pathname)

	// runtime.a -> runtime, runtime.6 -> runtime
	pkg := name
	if len(pkg) >= 2 && pkg[len(pkg)-2] == '.' {
		pkg = pkg[:len(pkg)-2]
	}

	// already loaded?
	for i := 0; i < len(ctxt.Library); i++ {
		if ctxt.Library[i].Pkg == pkg {
			return ctxt.Library[i]
		}
	}

	var pname string
	isshlib := false
	if filepath.IsAbs(name) {
		pname = name
	} else {
		// try dot, -L "libdir", and then goroot.
		for _, dir := range ctxt.Libdir {
			if *FlagLinkshared {
				pname = dir + "/" + pkg + ".shlibname"
				if _, err := os.Stat(pname); err == nil {
					isshlib = true
					break
				}
			}
			pname = dir + "/" + name
			if _, err := os.Stat(pname); err == nil {
				break
			}
		}
	}

	pname = path.Clean(pname)

	if ctxt.Debugvlog > 1 {
		ctxt.Logf("%5.2f addlib: %s %s pulls in %s isshlib %v\n", elapsed(), obj, src, pname, isshlib)
	}

	if isshlib {
		return addlibpath(ctxt, src, obj, "", pkg, pname)
	}
	return addlibpath(ctxt, src, obj, pname, pkg, "")
}

/*
 * add library to library list, return added library.
 *	srcref: src file referring to package
 *	objref: object file referring to package
 *	file: object file, e.g., /home/rsc/go/pkg/container/vector.a
 *	pkg: package import path, e.g. container/vector
 */
func addlibpath(ctxt *Link, srcref string, objref string, file string, pkg string, shlibnamefile string) *Library {
	for i := 0; i < len(ctxt.Library); i++ {
		if pkg == ctxt.Library[i].Pkg {
			return ctxt.Library[i]
		}
	}

	if ctxt.Debugvlog > 1 {
		ctxt.Logf("%5.2f addlibpath: srcref: %s objref: %s file: %s pkg: %s shlibnamefile: %s\n", obj.Cputime(), srcref, objref, file, pkg, shlibnamefile)
	}

	ctxt.Library = append(ctxt.Library, &Library{})
	l := ctxt.Library[len(ctxt.Library)-1]
	l.Objref = objref
	l.Srcref = srcref
	l.File = file
	l.Pkg = pkg
	if shlibnamefile != "" {
		shlibbytes, err := ioutil.ReadFile(shlibnamefile)
		if err != nil {
			Errorf(nil, "cannot read %s: %v", shlibnamefile, err)
		}
		l.Shlib = strings.TrimSpace(string(shlibbytes))
	}
	return l
}

func atolwhex(s string) int64 {
	n, _ := strconv.ParseInt(s, 0, 64)
	return n
}
