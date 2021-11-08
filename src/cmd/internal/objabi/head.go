// Derived from Inferno utils/6l/l.h and related files.
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/l.h
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

package objabi

import "fmt"

// HeadType is the executable header type.
type HeadType uint8

const (
	Hunknown HeadType = iota
	Hdarwin
	Hdragonfly
	Hfreebsd
	Hjs
	Hlinux
	Hnetbsd
	Hopenbsd
	Hplan9
	Hsolaris
	Hwindows
	Haix
)

func (h *HeadType) Set(s string) error {
	switch s {
	case "aix":
		*h = Haix
	case "darwin", "ios":
		*h = Hdarwin
	case "dragonfly":
		*h = Hdragonfly
	case "freebsd":
		*h = Hfreebsd
	case "js":
		*h = Hjs
	case "linux", "android":
		*h = Hlinux
	case "netbsd":
		*h = Hnetbsd
	case "openbsd":
		*h = Hopenbsd
	case "plan9":
		*h = Hplan9
	case "illumos", "solaris":
		*h = Hsolaris
	case "windows":
		*h = Hwindows
	default:
		return fmt.Errorf("invalid headtype: %q", s)
	}
	return nil
}

func (h *HeadType) String() string {
	switch *h {
	case Haix:
		return "aix"
	case Hdarwin:
		return "darwin"
	case Hdragonfly:
		return "dragonfly"
	case Hfreebsd:
		return "freebsd"
	case Hjs:
		return "js"
	case Hlinux:
		return "linux"
	case Hnetbsd:
		return "netbsd"
	case Hopenbsd:
		return "openbsd"
	case Hplan9:
		return "plan9"
	case Hsolaris:
		return "solaris"
	case Hwindows:
		return "windows"
	}
	return fmt.Sprintf("HeadType(%d)", *h)
}
