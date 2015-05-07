// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"strings"
)

func addlib(ctxt *Link, src string, obj string, pathname string) {
	name := path.Clean(pathname)

	// runtime.a -> runtime, runtime.6 -> runtime
	pkg := name
	if len(pkg) >= 2 && pkg[len(pkg)-2] == '.' {
		pkg = pkg[:len(pkg)-2]
	}

	// already loaded?
	for i := 0; i < len(ctxt.Library); i++ {
		if ctxt.Library[i].Pkg == pkg {
			return
		}
	}

	var pname string
	isshlib := false
	if (ctxt.Windows == 0 && strings.HasPrefix(name, "/")) || (ctxt.Windows != 0 && len(name) >= 2 && name[1] == ':') {
		pname = name
	} else {
		// try dot, -L "libdir", and then goroot.
		for _, dir := range ctxt.Libdir {
			if Linkshared {
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

	if ctxt.Debugvlog > 1 && ctxt.Bso != nil {
		fmt.Fprintf(ctxt.Bso, "%5.2f addlib: %s %s pulls in %s isshlib %v\n", elapsed(), obj, src, pname, isshlib)
	}

	if isshlib {
		addlibpath(ctxt, src, obj, "", pkg, pname)
	} else {
		addlibpath(ctxt, src, obj, pname, pkg, "")
	}
}

/*
 * add library to library list.
 *	srcref: src file referring to package
 *	objref: object file referring to package
 *	file: object file, e.g., /home/rsc/go/pkg/container/vector.a
 *	pkg: package import path, e.g. container/vector
 */
func addlibpath(ctxt *Link, srcref string, objref string, file string, pkg string, shlibnamefile string) {
	for i := 0; i < len(ctxt.Library); i++ {
		if pkg == ctxt.Library[i].Pkg {
			return
		}
	}

	if ctxt.Debugvlog > 1 && ctxt.Bso != nil {
		fmt.Fprintf(ctxt.Bso, "%5.2f addlibpath: srcref: %s objref: %s file: %s pkg: %s shlibnamefile: %s\n", obj.Cputime(), srcref, objref, file, pkg, shlibnamefile)
	}

	ctxt.Library = append(ctxt.Library, Library{})
	l := &ctxt.Library[len(ctxt.Library)-1]
	l.Objref = objref
	l.Srcref = srcref
	l.File = file
	l.Pkg = pkg
	if shlibnamefile != "" {
		shlibbytes, err := ioutil.ReadFile(shlibnamefile)
		if err != nil {
			Diag("cannot read %s: %v", shlibnamefile, err)
		}
		l.Shlib = strings.TrimSpace(string(shlibbytes))
	}
}

func atolwhex(s string) int64 {
	n, _ := strconv.ParseInt(s, 0, 64)
	return n
}
