// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
This program reads a file containing function prototypes
(like syscall_aix.go) and generates system call bodies.
The prototypes are marked by lines beginning with "//sys"
and read like func declarations if //sys is replaced by func, but:
	* The parameter lists must give a name for each argument.
	  This includes return parameters.
	* The parameter lists must give a type for each argument:
	  the (x, y, z int) shorthand is not allowed.
	* If the return parameter is an error number, it must be named err.
	* If go func name needs to be different than its libc name,
	* or the function is not in libc, name could be specified
	* at the end, after "=" sign, like
	  //sys getsockopt(s int, level int, name int, val uintptr, vallen *_Socklen) (err error) = libsocket.getsockopt


This program will generate three files and handle both gc and gccgo implementation:
  - zsyscall_aix_ppc64.go: the common part of each implementation (error handler, pointer creation)
  - zsyscall_aix_ppc64_gc.go: gc part with //go_cgo_import_dynamic and a call to syscall6
  - zsyscall_aix_ppc64_gccgo.go: gccgo part with C function and conversion to C type.

 The generated code looks like this

zsyscall_aix_ppc64.go
func asyscall(...) (n int, err error) {
	 // Pointer Creation
	 r1, e1 := callasyscall(...)
	 // Type Conversion
	 // Error Handler
	 return
}

zsyscall_aix_ppc64_gc.go
//go:cgo_import_dynamic libc_asyscall asyscall "libc.a/shr_64.o"
//go:linkname libc_asyscall libc_asyscall
var asyscall syscallFunc

func callasyscall(...) (r1 uintptr, e1 Errno) {
	 r1, _, e1 = syscall6(uintptr(unsafe.Pointer(&libc_asyscall)), "nb_args", ... )
	 return
}

zsyscall_aix_ppc64_ggcgo.go

// int asyscall(...)

import "C"

func callasyscall(...) (r1 uintptr, e1 Errno) {
	 r1 = uintptr(C.asyscall(...))
	 e1 = syscall.GetErrno()
	 return
}
*/

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strings"
)

var (
	b32  = flag.Bool("b32", false, "32bit big-endian")
	l32  = flag.Bool("l32", false, "32bit little-endian")
	aix  = flag.Bool("aix", false, "aix")
	tags = flag.String("tags", "", "build tags")
)

// cmdLine returns this programs's commandline arguments
func cmdLine() string {
	return "go run mksyscall_aix_ppc64.go " + strings.Join(os.Args[1:], " ")
}

// buildTags returns build tags
func buildTags() string {
	return *tags
}

// Param is function parameter
type Param struct {
	Name string
	Type string
}

// usage prints the program usage
func usage() {
	fmt.Fprintf(os.Stderr, "usage: go run mksyscall_aix_ppc64.go [-b32 | -l32] [-tags x,y] [file ...]\n")
	os.Exit(1)
}

// parseParamList parses parameter list and returns a slice of parameters
func parseParamList(list string) []string {
	list = strings.TrimSpace(list)
	if list == "" {
		return []string{}
	}
	return regexp.MustCompile(`\s*,\s*`).Split(list, -1)
}

// parseParam splits a parameter into name and type
func parseParam(p string) Param {
	ps := regexp.MustCompile(`^(\S*) (\S*)$`).FindStringSubmatch(p)
	if ps == nil {
		fmt.Fprintf(os.Stderr, "malformed parameter: %s\n", p)
		os.Exit(1)
	}
	return Param{ps[1], ps[2]}
}

func main() {
	flag.Usage = usage
	flag.Parse()
	if len(flag.Args()) <= 0 {
		fmt.Fprintf(os.Stderr, "no files to parse provided\n")
		usage()
	}

	endianness := ""
	if *b32 {
		endianness = "big-endian"
	} else if *l32 {
		endianness = "little-endian"
	}

	pack := ""
	// GCCGO
	textgccgo := ""
	cExtern := "/*\n#include <stdint.h>\n"
	// GC
	textgc := ""
	dynimports := ""
	linknames := ""
	var vars []string
	// COMMON
	textcommon := ""
	for _, path := range flag.Args() {
		file, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, err.Error())
			os.Exit(1)
		}
		s := bufio.NewScanner(file)
		for s.Scan() {
			t := s.Text()
			t = strings.TrimSpace(t)
			t = regexp.MustCompile(`\s+`).ReplaceAllString(t, ` `)
			if p := regexp.MustCompile(`^package (\S+)$`).FindStringSubmatch(t); p != nil && pack == "" {
				pack = p[1]
			}
			nonblock := regexp.MustCompile(`^\/\/sysnb `).FindStringSubmatch(t)
			if regexp.MustCompile(`^\/\/sys `).FindStringSubmatch(t) == nil && nonblock == nil {
				continue
			}

			// Line must be of the form
			//	func Open(path string, mode int, perm int) (fd int, err error)
			// Split into name, in params, out params.
			f := regexp.MustCompile(`^\/\/sys(nb)? (\w+)\(([^()]*)\)\s*(?:\(([^()]+)\))?\s*(?:=\s*(?:(\w*)\.)?(\w*))?$`).FindStringSubmatch(t)
			if f == nil {
				fmt.Fprintf(os.Stderr, "%s:%s\nmalformed //sys declaration\n", path, t)
				os.Exit(1)
			}
			funct, inps, outps, modname, sysname := f[2], f[3], f[4], f[5], f[6]

			// Split argument lists on comma.
			in := parseParamList(inps)
			out := parseParamList(outps)

			inps = strings.Join(in, ", ")
			outps = strings.Join(out, ", ")

			if sysname == "" {
				sysname = funct
			}

			onlyCommon := false
			if funct == "readlen" || funct == "writelen" || funct == "FcntlInt" || funct == "FcntlFlock" {
				// This function call another syscall which is already implemented.
				// Therefore, the gc and gccgo part must not be generated.
				onlyCommon = true
			}

			// Try in vain to keep people from editing this file.
			// The theory is that they jump into the middle of the file
			// without reading the header.

			textcommon += "// THIS FILE IS GENERATED BY THE COMMAND AT THE TOP; DO NOT EDIT\n\n"
			if !onlyCommon {
				textgccgo += "// THIS FILE IS GENERATED BY THE COMMAND AT THE TOP; DO NOT EDIT\n\n"
				textgc += "// THIS FILE IS GENERATED BY THE COMMAND AT THE TOP; DO NOT EDIT\n\n"
			}

			// Check if value return, err return available
			errvar := ""
			rettype := ""
			for _, param := range out {
				p := parseParam(param)
				if p.Type == "error" {
					errvar = p.Name
				} else {
					rettype = p.Type
				}
			}

			sysname = regexp.MustCompile(`([a-z])([A-Z])`).ReplaceAllString(sysname, `${1}_$2`)
			sysname = strings.ToLower(sysname) // All libc functions are lowercase.

			// GCCGO Prototype return type
			cRettype := ""
			if rettype == "unsafe.Pointer" {
				cRettype = "uintptr_t"
			} else if rettype == "uintptr" {
				cRettype = "uintptr_t"
			} else if regexp.MustCompile(`^_`).FindStringSubmatch(rettype) != nil {
				cRettype = "uintptr_t"
			} else if rettype == "int" {
				cRettype = "int"
			} else if rettype == "int32" {
				cRettype = "int"
			} else if rettype == "int64" {
				cRettype = "long long"
			} else if rettype == "uint32" {
				cRettype = "unsigned int"
			} else if rettype == "uint64" {
				cRettype = "unsigned long long"
			} else {
				cRettype = "int"
			}
			if sysname == "exit" {
				cRettype = "void"
			}

			// GCCGO Prototype arguments type
			var cIn []string
			for i, param := range in {
				p := parseParam(param)
				if regexp.MustCompile(`^\*`).FindStringSubmatch(p.Type) != nil {
					cIn = append(cIn, "uintptr_t")
				} else if p.Type == "string" {
					cIn = append(cIn, "uintptr_t")
				} else if regexp.MustCompile(`^\[\](.*)`).FindStringSubmatch(p.Type) != nil {
					cIn = append(cIn, "uintptr_t", "size_t")
				} else if p.Type == "unsafe.Pointer" {
					cIn = append(cIn, "uintptr_t")
				} else if p.Type == "uintptr" {
					cIn = append(cIn, "uintptr_t")
				} else if regexp.MustCompile(`^_`).FindStringSubmatch(p.Type) != nil {
					cIn = append(cIn, "uintptr_t")
				} else if p.Type == "int" {
					if (i == 0 || i == 2) && funct == "fcntl" {
						// These fcntl arguments needs to be uintptr to be able to call FcntlInt and FcntlFlock
						cIn = append(cIn, "uintptr_t")
					} else {
						cIn = append(cIn, "int")
					}

				} else if p.Type == "int32" {
					cIn = append(cIn, "int")
				} else if p.Type == "int64" {
					cIn = append(cIn, "long long")
				} else if p.Type == "uint32" {
					cIn = append(cIn, "unsigned int")
				} else if p.Type == "uint64" {
					cIn = append(cIn, "unsigned long long")
				} else {
					cIn = append(cIn, "int")
				}
			}

			if !onlyCommon {
				// GCCGO Prototype Generation
				// Imports of system calls from libc
				cExtern += fmt.Sprintf("%s %s", cRettype, sysname)
				cIn := strings.Join(cIn, ", ")
				cExtern += fmt.Sprintf("(%s);\n", cIn)
			}
			// GC Library name
			if modname == "" {
				modname = "libc.a/shr_64.o"
			} else {
				fmt.Fprintf(os.Stderr, "%s: only syscall using libc are available\n", funct)
				os.Exit(1)
			}
			sysvarname := fmt.Sprintf("libc_%s", sysname)

			if !onlyCommon {
				// GC Runtime import of function to allow cross-platform builds.
				dynimports += fmt.Sprintf("//go:cgo_import_dynamic %s %s \"%s\"\n", sysvarname, sysname, modname)
				// GC Link symbol to proc address variable.
				linknames += fmt.Sprintf("//go:linkname %s %s\n", sysvarname, sysvarname)
				// GC Library proc address variable.
				vars = append(vars, sysvarname)
			}

			strconvfunc := "BytePtrFromString"
			strconvtype := "*byte"

			// Go function header.
			if outps != "" {
				outps = fmt.Sprintf(" (%s)", outps)
			}
			if textcommon != "" {
				textcommon += "\n"
			}

			textcommon += fmt.Sprintf("func %s(%s)%s {\n", funct, strings.Join(in, ", "), outps)

			// Prepare arguments tocall.
			var argscommon []string // Arguments in the common part
			var argscall []string   // Arguments for call prototype
			var argsgc []string     // Arguments for gc call (with syscall6)
			var argsgccgo []string  // Arguments for gccgo call (with C.name_of_syscall)
			n := 0
			argN := 0
			for _, param := range in {
				p := parseParam(param)
				if regexp.MustCompile(`^\*`).FindStringSubmatch(p.Type) != nil {
					argscommon = append(argscommon, fmt.Sprintf("uintptr(unsafe.Pointer(%s))", p.Name))
					argscall = append(argscall, fmt.Sprintf("%s uintptr", p.Name))
					argsgc = append(argsgc, p.Name)
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.uintptr_t(%s)", p.Name))
				} else if p.Type == "string" && errvar != "" {
					textcommon += fmt.Sprintf("\tvar _p%d %s\n", n, strconvtype)
					textcommon += fmt.Sprintf("\t_p%d, %s = %s(%s)\n", n, errvar, strconvfunc, p.Name)
					textcommon += fmt.Sprintf("\tif %s != nil {\n\t\treturn\n\t}\n", errvar)

					argscommon = append(argscommon, fmt.Sprintf("uintptr(unsafe.Pointer(_p%d))", n))
					argscall = append(argscall, fmt.Sprintf("_p%d uintptr ", n))
					argsgc = append(argsgc, fmt.Sprintf("_p%d", n))
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.uintptr_t(_p%d)", n))
					n++
				} else if p.Type == "string" {
					fmt.Fprintf(os.Stderr, path+":"+funct+" uses string arguments, but has no error return\n")
					textcommon += fmt.Sprintf("\tvar _p%d %s\n", n, strconvtype)
					textcommon += fmt.Sprintf("\t_p%d, %s = %s(%s)\n", n, errvar, strconvfunc, p.Name)
					textcommon += fmt.Sprintf("\tif %s != nil {\n\t\treturn\n\t}\n", errvar)

					argscommon = append(argscommon, fmt.Sprintf("uintptr(unsafe.Pointer(_p%d))", n))
					argscall = append(argscall, fmt.Sprintf("_p%d uintptr", n))
					argsgc = append(argsgc, fmt.Sprintf("_p%d", n))
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.uintptr_t(_p%d)", n))
					n++
				} else if m := regexp.MustCompile(`^\[\](.*)`).FindStringSubmatch(p.Type); m != nil {
					// Convert slice into pointer, length.
					// Have to be careful not to take address of &a[0] if len == 0:
					// pass nil in that case.
					textcommon += fmt.Sprintf("\tvar _p%d *%s\n", n, m[1])
					textcommon += fmt.Sprintf("\tif len(%s) > 0 {\n\t\t_p%d = &%s[0]\n\t}\n", p.Name, n, p.Name)
					argscommon = append(argscommon, fmt.Sprintf("uintptr(unsafe.Pointer(_p%d))", n), fmt.Sprintf("len(%s)", p.Name))
					argscall = append(argscall, fmt.Sprintf("_p%d uintptr", n), fmt.Sprintf("_lenp%d int", n))
					argsgc = append(argsgc, fmt.Sprintf("_p%d", n), fmt.Sprintf("uintptr(_lenp%d)", n))
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.uintptr_t(_p%d)", n), fmt.Sprintf("C.size_t(_lenp%d)", n))
					n++
				} else if p.Type == "int64" && endianness != "" {
					fmt.Fprintf(os.Stderr, path+":"+funct+" uses int64 with 32 bits mode. Case not yet implemented\n")
				} else if p.Type == "bool" {
					fmt.Fprintf(os.Stderr, path+":"+funct+" uses bool. Case not yet implemented\n")
				} else if regexp.MustCompile(`^_`).FindStringSubmatch(p.Type) != nil || p.Type == "unsafe.Pointer" {
					argscommon = append(argscommon, fmt.Sprintf("uintptr(%s)", p.Name))
					argscall = append(argscall, fmt.Sprintf("%s uintptr", p.Name))
					argsgc = append(argsgc, p.Name)
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.uintptr_t(%s)", p.Name))
				} else if p.Type == "int" {
					if (argN == 0 || argN == 2) && ((funct == "fcntl") || (funct == "FcntlInt") || (funct == "FcntlFlock")) {
						// These fcntl arguments need to be uintptr to be able to call FcntlInt and FcntlFlock
						argscommon = append(argscommon, fmt.Sprintf("uintptr(%s)", p.Name))
						argscall = append(argscall, fmt.Sprintf("%s uintptr", p.Name))
						argsgc = append(argsgc, p.Name)
						argsgccgo = append(argsgccgo, fmt.Sprintf("C.uintptr_t(%s)", p.Name))

					} else {
						argscommon = append(argscommon, p.Name)
						argscall = append(argscall, fmt.Sprintf("%s int", p.Name))
						argsgc = append(argsgc, fmt.Sprintf("uintptr(%s)", p.Name))
						argsgccgo = append(argsgccgo, fmt.Sprintf("C.int(%s)", p.Name))
					}
				} else if p.Type == "int32" {
					argscommon = append(argscommon, p.Name)
					argscall = append(argscall, fmt.Sprintf("%s int32", p.Name))
					argsgc = append(argsgc, fmt.Sprintf("uintptr(%s)", p.Name))
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.int(%s)", p.Name))
				} else if p.Type == "int64" {
					argscommon = append(argscommon, p.Name)
					argscall = append(argscall, fmt.Sprintf("%s int64", p.Name))
					argsgc = append(argsgc, fmt.Sprintf("uintptr(%s)", p.Name))
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.longlong(%s)", p.Name))
				} else if p.Type == "uint32" {
					argscommon = append(argscommon, p.Name)
					argscall = append(argscall, fmt.Sprintf("%s uint32", p.Name))
					argsgc = append(argsgc, fmt.Sprintf("uintptr(%s)", p.Name))
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.uint(%s)", p.Name))
				} else if p.Type == "uint64" {
					argscommon = append(argscommon, p.Name)
					argscall = append(argscall, fmt.Sprintf("%s uint64", p.Name))
					argsgc = append(argsgc, fmt.Sprintf("uintptr(%s)", p.Name))
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.ulonglong(%s)", p.Name))
				} else if p.Type == "uintptr" {
					argscommon = append(argscommon, p.Name)
					argscall = append(argscall, fmt.Sprintf("%s uintptr", p.Name))
					argsgc = append(argsgc, p.Name)
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.uintptr_t(%s)", p.Name))
				} else {
					argscommon = append(argscommon, fmt.Sprintf("int(%s)", p.Name))
					argscall = append(argscall, fmt.Sprintf("%s int", p.Name))
					argsgc = append(argsgc, fmt.Sprintf("uintptr(%s)", p.Name))
					argsgccgo = append(argsgccgo, fmt.Sprintf("C.int(%s)", p.Name))
				}
				argN++
			}
			nargs := len(argsgc)

			// COMMON function generation
			argscommonlist := strings.Join(argscommon, ", ")
			callcommon := fmt.Sprintf("call%s(%s)", sysname, argscommonlist)
			ret := []string{"_", "_"}
			body := ""
			doErrno := false
			for i := 0; i < len(out); i++ {
				p := parseParam(out[i])
				reg := ""
				if p.Name == "err" {
					reg = "e1"
					ret[1] = reg
					doErrno = true
				} else {
					reg = "r0"
					ret[0] = reg
				}
				if p.Type == "bool" {
					reg = fmt.Sprintf("%s != 0", reg)
				}
				if reg != "e1" {
					body += fmt.Sprintf("\t%s = %s(%s)\n", p.Name, p.Type, reg)
				}
			}
			if ret[0] == "_" && ret[1] == "_" {
				textcommon += fmt.Sprintf("\t%s\n", callcommon)
			} else {
				textcommon += fmt.Sprintf("\t%s, %s := %s\n", ret[0], ret[1], callcommon)
			}
			textcommon += body

			if doErrno {
				textcommon += "\tif e1 != 0 {\n"
				textcommon += "\t\terr = errnoErr(e1)\n"
				textcommon += "\t}\n"
			}
			textcommon += "\treturn\n"
			textcommon += "}\n"

			if onlyCommon {
				continue
			}

			// CALL Prototype
			callProto := fmt.Sprintf("func call%s(%s) (r1 uintptr, e1 Errno) {\n", sysname, strings.Join(argscall, ", "))

			// GC function generation
			asm := "syscall6"
			if nonblock != nil {
				asm = "rawSyscall6"
			}

			if len(argsgc) <= 6 {
				for len(argsgc) < 6 {
					argsgc = append(argsgc, "0")
				}
			} else {
				fmt.Fprintf(os.Stderr, "%s: too many arguments to system call", funct)
				os.Exit(1)
			}
			argsgclist := strings.Join(argsgc, ", ")
			callgc := fmt.Sprintf("%s(uintptr(unsafe.Pointer(&%s)), %d, %s)", asm, sysvarname, nargs, argsgclist)

			textgc += callProto
			textgc += fmt.Sprintf("\tr1, _, e1 = %s\n", callgc)
			textgc += "\treturn\n}\n"

			// GCCGO function generation
			argsgccgolist := strings.Join(argsgccgo, ", ")
			callgccgo := fmt.Sprintf("C.%s(%s)", sysname, argsgccgolist)
			textgccgo += callProto
			textgccgo += fmt.Sprintf("\tr1 = uintptr(%s)\n", callgccgo)
			textgccgo += "\te1 = syscall.GetErrno()\n"
			textgccgo += "\treturn\n}\n"
		}
		if err := s.Err(); err != nil {
			fmt.Fprintf(os.Stderr, err.Error())
			os.Exit(1)
		}
		file.Close()
	}
	imp := ""
	if pack != "unix" {
		imp = "import \"golang.org/x/sys/unix\"\n"

	}

	// Print zsyscall_aix_ppc64.go
	err := ioutil.WriteFile("zsyscall_aix_ppc64.go",
		[]byte(fmt.Sprintf(srcTemplate1, cmdLine(), buildTags(), pack, imp, textcommon)),
		0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, err.Error())
		os.Exit(1)
	}

	// Print zsyscall_aix_ppc64_gc.go
	vardecls := "\t" + strings.Join(vars, ",\n\t")
	vardecls += " syscallFunc"
	err = ioutil.WriteFile("zsyscall_aix_ppc64_gc.go",
		[]byte(fmt.Sprintf(srcTemplate2, cmdLine(), buildTags(), pack, imp, dynimports, linknames, vardecls, textgc)),
		0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, err.Error())
		os.Exit(1)
	}

	// Print zsyscall_aix_ppc64_gccgo.go
	err = ioutil.WriteFile("zsyscall_aix_ppc64_gccgo.go",
		[]byte(fmt.Sprintf(srcTemplate3, cmdLine(), buildTags(), pack, cExtern, imp, textgccgo)),
		0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, err.Error())
		os.Exit(1)
	}
}

const srcTemplate1 = `// %s
// Code generated by the command above; see README.md. DO NOT EDIT.

// +build %s

package %s

import (
	"unsafe"
)


%s

%s
`
const srcTemplate2 = `// %s
// Code generated by the command above; see README.md. DO NOT EDIT.

// +build %s
// +build !gccgo

package %s

import (
	"unsafe"
)
%s
%s
%s
type syscallFunc uintptr

var (
%s
)

// Implemented in runtime/syscall_aix.go.
func rawSyscall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)
func syscall6(trap, nargs, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err Errno)

%s
`
const srcTemplate3 = `// %s
// Code generated by the command above; see README.md. DO NOT EDIT.

// +build %s
// +build gccgo

package %s

%s
*/
import "C"
import (
	"syscall"
)


%s

%s
`
