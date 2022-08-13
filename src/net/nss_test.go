// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package net

import (
	"reflect"
	"strings"
	"testing"
)

const ubuntuTrustyAvahi = `# /etc/nsswitch.conf
#
# Example configuration of GNU Name Service Switch functionality.
# If you have the libc-doc-reference' and nfo' packages installed, try:
# nfo libc "Name Service Switch"' for information about this file.

passwd:         compat
group:          compat
shadow:         compat

hosts:          files mdns4_minimal [NOTFOUND=return] dns mdns4
networks:       files

protocols:      db files
services:       db files
ethers:         db files
rpc:            db files

netgroup:       nis
`

func TestParseNSSConf(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want *nssConf
	}{
		{
			name: "no_newline",
			in:   "foo: a b",
			want: &nssConf{
				sources: map[string][]nssSource{
					"foo": {{source: "a"}, {source: "b"}},
				},
			},
		},
		{
			name: "newline",
			in:   "foo: a b\n",
			want: &nssConf{
				sources: map[string][]nssSource{
					"foo": {{source: "a"}, {source: "b"}},
				},
			},
		},
		{
			name: "whitespace",
			in:   "   foo:a    b    \n",
			want: &nssConf{
				sources: map[string][]nssSource{
					"foo": {{source: "a"}, {source: "b"}},
				},
			},
		},
		{
			name: "comment1",
			in:   "   foo:a    b#c\n",
			want: &nssConf{
				sources: map[string][]nssSource{
					"foo": {{source: "a"}, {source: "b"}},
				},
			},
		},
		{
			name: "comment2",
			in:   "   foo:a    b #c \n",
			want: &nssConf{
				sources: map[string][]nssSource{
					"foo": {{source: "a"}, {source: "b"}},
				},
			},
		},
		{
			name: "crit",
			in:   "   foo:a    b [!a=b    X=Y ] c#d \n",
			want: &nssConf{
				sources: map[string][]nssSource{
					"foo": {
						{source: "a"},
						{
							source: "b",
							criteria: []nssCriterion{
								{
									negate: true,
									status: "a",
									action: "b",
								},
								{
									status: "x",
									action: "y",
								},
							},
						},
						{source: "c"},
					},
				},
			},
		},

		// Ubuntu Trusty w/ avahi-daemon, libavahi-* etc installed.
		{
			name: "ubuntu_trusty_avahi",
			in:   ubuntuTrustyAvahi,
			want: &nssConf{
				sources: map[string][]nssSource{
					"passwd": {{source: "compat"}},
					"group":  {{source: "compat"}},
					"shadow": {{source: "compat"}},
					"hosts": {
						{source: "files"},
						{
							source: "mdns4_minimal",
							criteria: []nssCriterion{
								{
									negate: false,
									status: "notfound",
									action: "return",
								},
							},
						},
						{source: "dns"},
						{source: "mdns4"},
					},
					"networks": {{source: "files"}},
					"protocols": {
						{source: "db"},
						{source: "files"},
					},
					"services": {
						{source: "db"},
						{source: "files"},
					},
					"ethers": {
						{source: "db"},
						{source: "files"},
					},
					"rpc": {
						{source: "db"},
						{source: "files"},
					},
					"netgroup": {
						{source: "nis"},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		gotConf := parseNSSConf(strings.NewReader(tt.in))
		if !reflect.DeepEqual(gotConf, tt.want) {
			t.Errorf("%s: mismatch\n got %#v\nwant %#v", tt.name, gotConf, tt.want)
		}
	}
}
