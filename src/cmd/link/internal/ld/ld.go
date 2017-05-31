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
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

func (ctxt *Link) readImportCfg(file string) {
	ctxt.PackageFile = make(map[string]string)
	ctxt.PackageShlib = make(map[string]string)
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatalf("-importcfg: %v", err)
	}

	for lineNum, line := range strings.Split(string(data), "\n") {
		lineNum++ // 1-based
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		var verb, args string
		if i := strings.Index(line, " "); i < 0 {
			verb = line
		} else {
			verb, args = line[:i], strings.TrimSpace(line[i+1:])
		}
		var before, after string
		if i := strings.Index(args, "="); i >= 0 {
			before, after = args[:i], args[i+1:]
		}
		switch verb {
		default:
			log.Fatalf("%s:%d: unknown directive %q", file, lineNum, verb)
		case "packagefile":
			if before == "" || after == "" {
				log.Fatalf(`%s:%d: invalid packagefile: syntax is "packagefile path=filename"`, file, lineNum)
			}
			ctxt.PackageFile[before] = after
		case "packageshlib":
			if before == "" || after == "" {
				log.Fatalf(`%s:%d: invalid packageshlib: syntax is "packageshlib path=filename"`, file, lineNum)
			}
			ctxt.PackageShlib[before] = after
		}
	}
}

func addlib(ctxt *Link, src string, obj string, pathname string) *Library {
	name := path.Clean(pathname)

	// runtime.a -> runtime, runtime.6 -> runtime
	pkg := name
	if len(pkg) >= 2 && pkg[len(pkg)-2] == '.' {
		pkg = pkg[:len(pkg)-2]
	}

	// already loaded?
	if l := ctxt.LibraryByPkg[pkg]; l != nil {
		return l
	}

	var pname string
	isshlib := false

	if *FlagLinkshared && ctxt.PackageShlib[name] != "" {
		pname = ctxt.PackageShlib[name]
		isshlib = true
	} else if ctxt.PackageFile != nil {
		pname = ctxt.PackageFile[name]
		if pname == "" {
			ctxt.Logf("cannot find package %s (using -importcfg)\n", name)
			return nil
		}
	} else {
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
	}

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
	if l := ctxt.LibraryByPkg[pkg]; l != nil {
		return l
	}

	if ctxt.Debugvlog > 1 {
		ctxt.Logf("%5.2f addlibpath: srcref: %s objref: %s file: %s pkg: %s shlibnamefile: %s\n", Cputime(), srcref, objref, file, pkg, shlibnamefile)
	}

	l := &Library{}
	ctxt.LibraryByPkg[pkg] = l
	ctxt.Library = append(ctxt.Library, l)
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
