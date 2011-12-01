// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/token"
	"strings"
)

func init() {
	register(timefileinfoFix)
}

var timefileinfoFix = fix{
	"time+fileinfo",
	"2011-11-29",
	timefileinfo,
	`Rewrite for new time and os.FileInfo APIs.

This fix applies some of the more mechanical changes,
but most code will still need manual cleanup.

http://codereview.appspot.com/5392041
http://codereview.appspot.com/5416060
`,
}

var timefileinfoTypeConfig = &TypeConfig{
	Type: map[string]*Type{
		"os.File": &Type{
			Method: map[string]string{
				"Readdir": "func() []*os.FileInfo",
				"Stat":    "func() (*os.FileInfo, error)",
			},
		},
		"time.Time": &Type{
			Method: map[string]string{
				"Seconds":     "time.raw",
				"Nanoseconds": "time.raw",
			},
		},
	},
	Func: map[string]string{
		"ioutil.ReadDir":              "([]*os.FileInfo, error)",
		"os.Stat":                     "(*os.FileInfo, error)",
		"os.Lstat":                    "(*os.FileInfo, error)",
		"time.LocalTime":              "*time.Time",
		"time.UTC":                    "*time.Time",
		"time.SecondsToLocalTime":     "*time.Time",
		"time.SecondsToUTC":           "*time.Time",
		"time.NanosecondsToLocalTime": "*time.Time",
		"time.NanosecondsToUTC":       "*time.Time",
		"time.Parse":                  "(*time.Time, error)",
		"time.Nanoseconds":            "time.raw",
		"time.Seconds":                "time.raw",
	},
}

// timefileinfoIsOld reports whether f has evidence of being
// "old code", from before the API changes.  Evidence means:
//
//	a mention of *os.FileInfo (the pointer)
//	a mention of *time.Time (the pointer)
//	a mention of old functions from package time
//	an attempt to call time.UTC
//
func timefileinfoIsOld(f *ast.File, typeof map[interface{}]string) bool {
	old := false

	// called records the expressions that appear as
	// the function part of a function call, so that
	// we can distinguish a ref to the possibly new time.UTC
	// from the definitely old time.UTC() function call.
	called := make(map[interface{}]bool)

	before := func(n interface{}) {
		if old {
			return
		}
		if star, ok := n.(*ast.StarExpr); ok {
			if isPkgDot(star.X, "os", "FileInfo") || isPkgDot(star.X, "time", "Time") {
				old = true
				return
			}
		}
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if isTopName(sel.X, "time") {
				if timefileinfoOldTimeFunc[sel.Sel.Name] {
					old = true
					return
				}
			}
			if typeof[sel.X] == "os.FileInfo" || typeof[sel.X] == "*os.FileInfo" {
				switch sel.Sel.Name {
				case "Mtime_ns", "IsDirectory", "IsRegular":
					old = true
					return
				case "Name", "Mode", "Size":
					if !called[sel] {
						old = true
						return
					}
				}
			}
		}
		call, ok := n.(*ast.CallExpr)
		if ok && isPkgDot(call.Fun, "time", "UTC") {
			old = true
			return
		}
		if ok {
			called[call.Fun] = true
		}
	}
	walkBeforeAfter(f, before, nop)
	return old
}

var timefileinfoOldTimeFunc = map[string]bool{
	"LocalTime":              true,
	"SecondsToLocalTime":     true,
	"SecondsToUTC":           true,
	"NanosecondsToLocalTime": true,
	"NanosecondsToUTC":       true,
	"Seconds":                true,
	"Nanoseconds":            true,
}

var isTimeNow = map[string]bool{
	"LocalTime":   true,
	"UTC":         true,
	"Seconds":     true,
	"Nanoseconds": true,
}

func timefileinfo(f *ast.File) bool {
	if !imports(f, "os") && !imports(f, "time") && !imports(f, "io/ioutil") {
		return false
	}

	typeof, _ := typecheck(timefileinfoTypeConfig, f)

	if !timefileinfoIsOld(f, typeof) {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		p, ok := n.(*ast.Expr)
		if !ok {
			return
		}
		nn := *p

		// Rewrite *os.FileInfo and *time.Time to drop the pointer.
		if star, ok := nn.(*ast.StarExpr); ok {
			if isPkgDot(star.X, "os", "FileInfo") || isPkgDot(star.X, "time", "Time") {
				fixed = true
				*p = star.X
				return
			}
		}

		// Rewrite old time API calls to new calls.
		// The code will still not compile after this edit,
		// but the compiler will catch that, and the replacement
		// code will be the correct functions to use in the new API.
		if sel, ok := nn.(*ast.SelectorExpr); ok && isTopName(sel.X, "time") {
			fn := sel.Sel.Name
			if fn == "LocalTime" || fn == "Seconds" || fn == "Nanoseconds" {
				fixed = true
				sel.Sel.Name = "Now"
				return
			}
		}

		if call, ok := nn.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				// Rewrite time.UTC but only when called (there's a new time.UTC var now).
				if isPkgDot(sel, "time", "UTC") {
					fixed = true
					sel.Sel.Name = "Now"
					// rewrite time.Now() into time.Now().UTC()
					*p = &ast.CallExpr{
						Fun: &ast.SelectorExpr{
							X:   call,
							Sel: ast.NewIdent("UTC"),
						},
					}
					return
				}

				// Rewrite conversions.
				if ok && isTopName(sel.X, "time") && len(call.Args) == 1 {
					fn := sel.Sel.Name
					switch fn {
					case "SecondsToLocalTime", "SecondsToUTC",
						"NanosecondsToLocalTime", "NanosecondsToUTC":
						fixed = true
						sel.Sel.Name = "Unix"
						call.Args = append(call.Args, nil)
						if strings.HasPrefix(fn, "Seconds") {
							// Unix(sec, 0)
							call.Args[1] = ast.NewIdent("0")
						} else {
							// Unix(0, nsec)
							call.Args[1] = call.Args[0]
							call.Args[0] = ast.NewIdent("0")
						}
						if strings.HasSuffix(fn, "ToUTC") {
							// rewrite call into call.UTC()
							*p = &ast.CallExpr{
								Fun: &ast.SelectorExpr{
									X:   call,
									Sel: ast.NewIdent("UTC"),
								},
							}
						}
						return
					}
				}

				// Rewrite method calls.
				switch typeof[sel.X] {
				case "*time.Time", "time.Time":
					switch sel.Sel.Name {
					case "Seconds":
						fixed = true
						sel.Sel.Name = "Unix"
						return
					case "Nanoseconds":
						fixed = true
						sel.Sel.Name = "UnixNano"
						return
					}

				case "*os.FileInfo", "os.FileInfo":
					switch sel.Sel.Name {
					case "IsDirectory":
						fixed = true
						sel.Sel.Name = "IsDir"
						return
					case "IsRegular":
						fixed = true
						sel.Sel.Name = "IsDir"
						*p = &ast.UnaryExpr{
							Op: token.NOT,
							X:  call,
						}
						return
					}
				}
			}
		}

		// Rewrite subtraction of two times.
		// Cannot handle +=/-=.
		if bin, ok := nn.(*ast.BinaryExpr); ok &&
			bin.Op == token.SUB &&
			(typeof[bin.X] == "time.raw" || typeof[bin.Y] == "time.raw") {
			fixed = true
			*p = &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   bin.X,
					Sel: ast.NewIdent("Sub"),
				},
				Args: []ast.Expr{bin.Y},
			}
		}

		// Rewrite field references for os.FileInfo.
		if sel, ok := nn.(*ast.SelectorExpr); ok {
			if typ := typeof[sel.X]; typ == "*os.FileInfo" || typ == "os.FileInfo" {
				addCall := false
				switch sel.Sel.Name {
				case "Name", "Size", "Mode":
					fixed = true
					addCall = true
				case "Mtime_ns":
					fixed = true
					sel.Sel.Name = "ModTime"
					addCall = true
				}
				if addCall {
					*p = &ast.CallExpr{
						Fun: sel,
					}
					return
				}
			}
		}
	})

	return true
}
