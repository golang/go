// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
)

var (
	hexDumpRE = regexp.MustCompile(`^\t(0x[0-9a-f]{4,})(( ([0-9a-f]{2}|  )){16})  [ -\x7F]{1,16}\n`)
	listingRE = regexp.MustCompile(`^\t(0x[0-9a-f]{4,}) ([0-9]{4,}) \(.*:[0-9]+\)\t`)
)

// okdiffs lists regular expressions for lines to consider minor mismatches.
// If one of these regexps matches both of a pair of unequal lines, the mismatch
// is reported but not treated as the one worth looking for.
// For example, differences in the TEXT line are typically frame size
// changes due to optimization decisions made in the body of the function.
// Better to keep looking for the actual difference.
// Similarly, forward jumps might have different offsets due to a
// change in instruction encoding later on.
// Better to find that change.
var okdiffs = []*regexp.Regexp{
	regexp.MustCompile(`\)	TEXT[ 	].*,\$`),
	regexp.MustCompile(`\)	WORD[ 	].*,\$`),
	regexp.MustCompile(`\)	(B|BR|JMP)	`),
	regexp.MustCompile(`\)	FUNCDATA	`),
	regexp.MustCompile(`\\(z|x00)`),
	regexp.MustCompile(`\$\([0-9]\.[0-9]+e[+\-][0-9]+\)`),
	regexp.MustCompile(`size=.*value=.*args=.*locals=`),
}

func compareLogs(outfile string) string {
	f1, err := os.Open(outfile + ".log")
	if err != nil {
		log.Fatal(err)
	}
	defer f1.Close()

	f2, err := os.Open(outfile + ".stash.log")
	if err != nil {
		log.Fatal(err)
	}
	defer f2.Close()

	b1 := bufio.NewReader(f1)
	b2 := bufio.NewReader(f2)

	offset := int64(0)
	textOffset := offset
	textLineno := 0
	lineno := 0
	var line1, line2 string
	var prefix bytes.Buffer
Reading:
	for {
		var err1, err2 error
		line1, err1 = b1.ReadString('\n')
		line2, err2 = b2.ReadString('\n')
		if strings.Contains(line1, ")\tTEXT\t") {
			textOffset = offset
			textLineno = lineno
		}
		offset += int64(len(line1))
		lineno++
		if err1 == io.EOF && err2 == io.EOF {
			return "no differences in debugging output"
		}

		if lineno == 1 || line1 == line2 && err1 == nil && err2 == nil {
			continue
		}
		// Lines are inconsistent. Worth stopping?
		for _, re := range okdiffs {
			if re.MatchString(line1) && re.MatchString(line2) {
				fmt.Fprintf(&prefix, "inconsistent log line:\n%s:%d:\n\t%s\n%s:%d:\n\t%s\n\n",
					f1.Name(), lineno, strings.TrimSuffix(line1, "\n"),
					f2.Name(), lineno, strings.TrimSuffix(line2, "\n"))
				continue Reading
			}
		}

		if err1 != nil {
			line1 = err1.Error()
		}
		if err2 != nil {
			line2 = err2.Error()
		}
		break
	}

	msg := fmt.Sprintf("inconsistent log line:\n%s:%d:\n\t%s\n%s:%d:\n\t%s",
		f1.Name(), lineno, strings.TrimSuffix(line1, "\n"),
		f2.Name(), lineno, strings.TrimSuffix(line2, "\n"))

	if m := hexDumpRE.FindStringSubmatch(line1); m != nil {
		target, err := strconv.ParseUint(m[1], 0, 64)
		if err != nil {
			goto Skip
		}

		m2 := hexDumpRE.FindStringSubmatch(line2)
		if m2 == nil {
			goto Skip
		}

		fields1 := strings.Fields(m[2])
		fields2 := strings.Fields(m2[2])
		i := 0
		for i < len(fields1) && i < len(fields2) && fields1[i] == fields2[i] {
			i++
		}
		target += uint64(i)

		f1.Seek(textOffset, 0)
		b1 = bufio.NewReader(f1)
		last := ""
		lineno := textLineno
		limitAddr := uint64(0)
		lastAddr := uint64(0)
		for {
			line1, err1 := b1.ReadString('\n')
			if err1 != nil {
				break
			}
			lineno++
			if m := listingRE.FindStringSubmatch(line1); m != nil {
				addr, _ := strconv.ParseUint(m[1], 0, 64)
				if addr > target {
					limitAddr = addr
					break
				}
				last = line1
				lastAddr = addr
			} else if hexDumpRE.FindStringSubmatch(line1) != nil {
				break
			}
		}
		if last != "" {
			msg = fmt.Sprintf("assembly instruction at %#04x-%#04x:\n%s:%d\n\t%s\n\n%s",
				lastAddr, limitAddr, f1.Name(), lineno-1, strings.TrimSuffix(last, "\n"), msg)
		}
	}
Skip:

	return prefix.String() + msg
}
