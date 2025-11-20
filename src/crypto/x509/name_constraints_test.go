// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/hex"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"net/url"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	// testNameConstraintsAgainstOpenSSL can be set to true to run tests
	// against the system OpenSSL. This is disabled by default because Go
	// cannot depend on having OpenSSL installed at testing time.
	testNameConstraintsAgainstOpenSSL = false

	// debugOpenSSLFailure can be set to true, when
	// testNameConstraintsAgainstOpenSSL is also true, to cause
	// intermediate files to be preserved for debugging.
	debugOpenSSLFailure = false
)

type nameConstraintsTest struct {
	name          string
	roots         []constraintsSpec
	intermediates [][]constraintsSpec
	leaf          leafSpec
	requestedEKUs []ExtKeyUsage
	expectedError string
	noOpenSSL     bool
	ignoreCN      bool
}

type constraintsSpec struct {
	ok   []string
	bad  []string
	ekus []string
}

type leafSpec struct {
	sans []string
	ekus []string
	cn   string
}

var nameConstraintsTests = []nameConstraintsTest{
	{
		name:  "certificate generation process",
		roots: make([]constraintsSpec, 1),
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name:  "single level of intermediate",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name:  "two levels of intermediates",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{},
			},
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name: "matching DNS constraint in root",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name:  "matching DNS constraint in intermediate",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					ok: []string{"dns:example.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name: "leading period only matches subdomains",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
		expectedError: "\"example.com\" is not permitted",
	},
	{
		name:  "leading period matches subdomains",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					ok: []string{"dns:.example.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.example.com"},
		},
	},
	{
		name: "leading period matches multiple levels of subdomains",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.bar.example.com"},
		},
	},
	{
		name: "specifying a permitted list of names does not exclude other name types",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:10.1.1.1"},
		},
	},
	{
		name: "specifying a permitted list of names does not exclude other name types",
		roots: []constraintsSpec{
			{
				ok: []string{"ip:10.0.0.0/8"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name: "intermediates can try to permit other names, which isn't forbidden if the leaf doesn't mention them",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					ok: []string{"dns:example.com", "dns:foo.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name: "intermediates cannot add permitted names that the root doesn't grant them",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					ok: []string{"dns:foo.example.com", "dns:foo.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.com"},
		},
		expectedError: "\"foo.com\" is not permitted",
	},
	{
		name: "intermediates can further limit their scope if they wish",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					ok: []string{"dns:.bar.example.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.bar.example.com"},
		},
	},
	{
		name: "intermediates can further limit their scope and that limitation is effective",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					ok: []string{"dns:.bar.example.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.notbar.example.com"},
		},
		expectedError: "\"foo.notbar.example.com\" is not permitted",
	},
	{
		name: "roots can exclude subtrees and that doesn't affect other names",
		roots: []constraintsSpec{
			{
				bad: []string{"dns:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.com"},
		},
	},
	{
		name: "roots exclusions are effective",
		roots: []constraintsSpec{
			{
				bad: []string{"dns:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.example.com"},
		},
		expectedError: "\"foo.example.com\" is excluded",
	},
	{
		name:  "intermediates can also exclude names and that doesn't affect other names",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					bad: []string{"dns:.example.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.com"},
		},
	},
	{
		name:  "intermediate exclusions are effective",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					bad: []string{"dns:.example.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.example.com"},
		},
		expectedError: "\"foo.example.com\" is excluded",
	},
	{
		name: "having an exclusion doesn't prohibit other types of names",
		roots: []constraintsSpec{
			{
				bad: []string{"dns:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.com", "ip:10.1.1.1"},
		},
	},
	{
		name: "IP-based exclusions are permitted and don't affect unrelated IP addresses",
		roots: []constraintsSpec{
			{
				bad: []string{"ip:10.0.0.0/8"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:192.168.1.1"},
		},
	},
	{
		name: "IP-based exclusions are effective",
		roots: []constraintsSpec{
			{
				bad: []string{"ip:10.0.0.0/8"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:10.0.0.1"},
		},
		expectedError: "\"10.0.0.1\" is excluded",
	},
	{
		name: "intermediates can further constrain IP ranges",
		roots: []constraintsSpec{
			{
				bad: []string{"ip:0.0.0.0/1"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					bad: []string{"ip:11.0.0.0/8"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:11.0.0.1"},
		},
		expectedError: "\"11.0.0.1\" is excluded",
	},
	{
		name:  "multiple intermediates with incompatible constraints",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					ok: []string{"dns:.foo.com"},
				},
				{
					ok: []string{"dns:.example.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.example.com"},
		},
		noOpenSSL: true, // OpenSSL's chain building is not informed by constraints.
	},
	{
		name:  "multiple intermediates with incompatible constraints swapped",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					ok: []string{"dns:.example.com"},
				},
				{
					ok: []string{"dns:.foo.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.example.com"},
		},
		noOpenSSL: true, // OpenSSL's chain building is not informed by constraints.
	},
	{
		name: "multiple roots with incompatible constraints",
		roots: []constraintsSpec{
			{},
			{
				ok: []string{"dns:foo.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
		noOpenSSL: true, // OpenSSL's chain building is not informed by constraints.
	},
	{
		name: "multiple roots with incompatible constraints swapped",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:foo.com"},
			},
			{},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
		noOpenSSL: true, // OpenSSL's chain building is not informed by constraints.
	},
	{
		name: "chain building with multiple intermediates and roots",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:foo.com"},
			},
			{
				ok: []string{"dns:example.com"},
			},
			{},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
				{
					ok: []string{"dns:foo.com"},
				},
			},
			{
				{},
				{
					ok: []string{"dns:foo.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:bar.com"},
		},
		noOpenSSL: true, // OpenSSL's chain building is not informed by constraints.
	},
	{
		name: "chain building fails with no valid path",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:foo.com"},
			},
			{
				ok: []string{"dns:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
				{
					ok: []string{"dns:foo.com"},
				},
			},
			{
				{
					ok: []string{"dns:bar.com"},
				},
				{
					ok: []string{"dns:foo.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:bar.com"},
		},
		expectedError: "\"bar.com\" is not permitted",
	},
	{
		name:  "unknown name types are unconstrained",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"unknown:"},
		},
	},
	{
		name: "unknown name types allowed in constrained chain",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:foo.com", "dns:.foo.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"unknown:"},
		},
	},
	{
		name: "CN is ignored in constrained chain",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:foo.com", "dns:.foo.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{},
			cn:   "foo.com",
		},
	},
	{
		name: "IPv6 permitted constraint",
		roots: []constraintsSpec{
			{
				ok: []string{"ip:2000:abcd::/32"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:2000:abcd:1234::"},
		},
	},
	{
		name: "IPv6 permitted constraint is effective",
		roots: []constraintsSpec{
			{
				ok: []string{"ip:2000:abcd::/32"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:2000:1234:abcd::"},
		},
		expectedError: "\"2000:1234:abcd::\" is not permitted",
	},
	{
		name: "IPv6 permitted constraint does not affect DNS",
		roots: []constraintsSpec{
			{
				ok: []string{"ip:2000:abcd::/32"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:2000:abcd::", "dns:foo.com"},
		},
	},
	{
		name: "IPv6 excluded constraint",
		roots: []constraintsSpec{
			{
				bad: []string{"ip:2000:abcd::/32"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:2000:1234::"},
		},
	},
	{
		name: "IPv6 excluded constraint is effective",
		roots: []constraintsSpec{
			{
				bad: []string{"ip:2000:abcd::/32"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:2000:abcd::"},
		},
		expectedError: "\"2000:abcd::\" is excluded",
	},
	{
		name: "IPv6 constraint does not permit IPv4",
		roots: []constraintsSpec{
			{
				ok: []string{"ip:2000:abcd::/32"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:10.0.0.1"},
		},
		expectedError: "\"10.0.0.1\" is not permitted",
	},
	{
		name: "IPv4 constraint does not permit IPv6",
		roots: []constraintsSpec{
			{
				ok: []string{"ip:10.0.0.0/8"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:2000:abcd::"},
		},
		expectedError: "\"2000:abcd::\" is not permitted",
	},
	{
		name: "unknown excluded constraint does not affect other names",
		roots: []constraintsSpec{
			{
				bad: []string{"unknown:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name: "unknown permitted constraint does not affect other names",
		roots: []constraintsSpec{
			{
				ok: []string{"unknown:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name: "exact email constraint",
		roots: []constraintsSpec{
			{
				ok: []string{"email:foo@example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:foo@example.com"},
		},
	},
	{
		name: "exact email constraint is effective",
		roots: []constraintsSpec{
			{
				ok: []string{"email:foo@example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:bar@example.com"},
		},
		expectedError: "\"bar@example.com\" is not permitted",
	},
	{
		name: "email canonicalization",
		roots: []constraintsSpec{
			{
				ok: []string{"email:foo@example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:\"\\f\\o\\o\"@example.com"},
		},
		noOpenSSL: true, // OpenSSL doesn't canonicalise email addresses before matching
	},
	{
		name: "email host constraint",
		roots: []constraintsSpec{
			{
				ok: []string{"email:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:foo@example.com"},
		},
	},
	{
		name: "email subdomain constraint",
		roots: []constraintsSpec{
			{
				ok: []string{"email:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:foo@sub.example.com"},
		},
	},
	{
		name: "email subdomain constraint does not match parent",
		roots: []constraintsSpec{
			{
				ok: []string{"email:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:foo@example.com"},
		},
		expectedError: "\"foo@example.com\" is not permitted",
	},
	{
		name: "email subdomain constraint matches deeper subdomains",
		roots: []constraintsSpec{
			{
				ok: []string{"email:.example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:foo@sub.sub.example.com"},
		},
	},
	{
		name: "email local part is case-sensitive",
		roots: []constraintsSpec{
			{
				ok: []string{"email:foo@example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:Foo@example.com"},
		},
		expectedError: "\"Foo@example.com\" is not permitted",
	},
	{
		name: "email domain part is case-insensitive",
		roots: []constraintsSpec{
			{
				ok: []string{"email:foo@EXAMPLE.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:foo@example.com"},
		},
	},
	{
		name: "DNS domain is case-insensitive",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:EXAMPLE.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name: "URI constraint covers host",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{
				"uri:http://example.com/bar",
				"uri:http://example.com:8080/",
				"uri:https://example.com/wibble#bar",
			},
		},
	},
	{
		name: "URI with IP is rejected",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:http://1.2.3.4/"},
		},
		expectedError: "URI with IP",
	},
	{
		name: "URI with IP and port is rejected",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:http://1.2.3.4:43/"},
		},
		expectedError: "URI with IP",
	},
	{
		name: "URI with IPv6 is rejected",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:http://[2006:abcd::1]/"},
		},
		expectedError: "URI with IP",
	},
	{
		name: "URI with IPv6 and port is rejected",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:http://[2006:abcd::1]:16/"},
		},
		expectedError: "URI with IP",
	},
	{
		name: "URI permitted constraint is effective",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:http://bar.com/"},
		},
		expectedError: "\"http://bar.com/\" is not permitted",
	},
	{
		name: "URI excluded constraint is effective",
		roots: []constraintsSpec{
			{
				bad: []string{"uri:foo.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:http://foo.com/"},
		},
		expectedError: "\"http://foo.com/\" is excluded",
	},
	{
		name: "URI subdomain constraint",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:.foo.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:http://www.foo.com/"},
		},
	},
	{
		name: "IPv4-mapped-IPv6 exclusion does not affect IPv4",
		roots: []constraintsSpec{
			{
				bad: []string{"ip:::ffff:1.2.3.4/128"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:1.2.3.4"},
		},
	},
	{
		name: "URI constraint not matched by URN",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:urn:example"},
		},
		expectedError: "URI with empty host",
	},
	{
		name: "IPv6 exclusion does not exclude all IPv4",
		roots: []constraintsSpec{
			{
				ok:  []string{"ip:1.2.3.0/24"},
				bad: []string{"ip:::0/0"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"ip:1.2.3.4"},
		},
	},
	{
		name:  "empty EKU in CA means any is ok",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"serverAuth", "other"},
		},
	},
	{
		name:  "any EKU means any is ok",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					ekus: []string{"any"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"serverAuth", "other"},
		},
	},
	// default.)
	{
		name:  "intermediate with enumerated EKUs",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					ekus: []string{"email"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"serverAuth"},
		},
		expectedError: "incompatible key usage",
	},
	{
		name:  "unknown EKU in leaf",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					ekus: []string{"email"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"other"},
		},
		requestedEKUs: []ExtKeyUsage{ExtKeyUsageAny},
	},
	// certificate doesn't use them.
	{
		name: "intermediate cannot add EKUs not in root if leaf uses them",
		roots: []constraintsSpec{
			{
				ekus: []string{"serverAuth"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					ekus: []string{"serverAuth", "email"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"serverAuth"},
		},
	},
	{
		name: "EKUs in root are effective",
		roots: []constraintsSpec{
			{
				ekus: []string{"email"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					ekus: []string{"serverAuth"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"serverAuth"},
		},
		expectedError: "incompatible key usage",
	},
	{
		name: "netscapeSGC EKU does not permit server/client auth",
		roots: []constraintsSpec{
			{},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					ekus: []string{"netscapeSGC"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"serverAuth", "clientAuth"},
		},
		expectedError: "incompatible key usage",
	},
	{
		name:  "msSGC EKU does not permit server/client auth",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					ekus: []string{"msSGC"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"serverAuth", "clientAuth"},
		},
		expectedError: "incompatible key usage",
	},
	{
		name: "empty DNS permitted constraint allows anything",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
	},
	{
		name: "empty DNS excluded constraint rejects everything",
		roots: []constraintsSpec{
			{
				bad: []string{"dns:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
		},
		expectedError: "\"example.com\" is excluded",
	},
	{
		name: "empty email permitted constraint allows anything",
		roots: []constraintsSpec{
			{
				ok: []string{"email:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:foo@example.com"},
		},
	},
	{
		name: "empty email excluded constraint rejects everything",
		roots: []constraintsSpec{
			{
				bad: []string{"email:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:foo@example.com"},
		},
		expectedError: "\"foo@example.com\" is excluded",
	},
	{
		name: "empty URI permitted constraint allows anything",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:https://example.com/test"},
		},
	},
	{
		name: "empty URI excluded constraint rejects everything",
		roots: []constraintsSpec{
			{
				bad: []string{"uri:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:https://example.com/test"},
		},
		expectedError: "\"https://example.com/test\" is excluded",
	},
	{
		name:  "serverAuth EKU does not permit clientAuth",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"serverAuth"},
		},
		requestedEKUs: []ExtKeyUsage{ExtKeyUsageClientAuth},
		expectedError: "incompatible key usage",
	},
	{
		name:  "msSGC EKU does not permit serverAuth",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"msSGC"},
		},
		requestedEKUs: []ExtKeyUsage{ExtKeyUsageServerAuth},
		expectedError: "incompatible key usage",
	},
	{
		// An invalid DNS SAN should be detected only at validation time so
		// that we can process CA certificates in the wild that have invalid SANs.
		// See https://github.com/golang/go/issues/23995
		name:  "invalid SANs are ignored with no constraints",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:this is invalid", "email:this @ is invalid"},
		},
	},
	{
		name: "invalid DNS SAN detected with constraints",
		roots: []constraintsSpec{
			{
				bad: []string{"uri:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:this is invalid"},
		},
		expectedError: "cannot parse dnsName",
	},
	{
		name: "invalid email SAN detected with constraints",
		roots: []constraintsSpec{
			{
				bad: []string{"uri:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"email:this @ is invalid"},
		},
		expectedError: "cannot parse rfc822Name",
	},
	{
		name:  "any requested EKU is sufficient",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			ekus: []string{"email"},
		},
		requestedEKUs: []ExtKeyUsage{ExtKeyUsageClientAuth, ExtKeyUsageEmailProtection},
	},
	{
		name:  "unrequested EKUs not required to be nested",
		roots: make([]constraintsSpec, 1),
		intermediates: [][]constraintsSpec{
			{
				{
					ekus: []string{"serverAuth"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:example.com"},
			// There's no email EKU in the intermediate. This would be rejected if
			// full nesting was required.
			ekus: []string{"email", "serverAuth"},
		},
	},
	{
		name: "empty leaf is accepted in constrained chain",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:foo.com", "dns:.foo.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{},
		},
	},
	{
		name: "no SANs and non-hostname CN is accepted in constrained chain",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:foo.com", "dns:.foo.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{},
			cn:   "foo.bar",
		},
	},
	{
		name: "constraints don't apply to CN",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:foo.com", "dns:.foo.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:foo.com"},
			cn:   "foo.bar",
		},
	},
	{
		name:          "DNS SAN cannot use leading period form",
		roots:         []constraintsSpec{{ok: []string{"dns:example.com"}}},
		leaf:          leafSpec{sans: []string{"dns:.example.com"}},
		expectedError: "cannot parse dnsName \".example.com\"",
	},
	{
		name: "URI with IPv6 and zone is rejected",
		roots: []constraintsSpec{
			{
				ok: []string{"uri:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"uri:http://[2006:abcd::1%25.example.com]:16/"},
		},
		expectedError: "URI with IP",
	},
	{
		name: "intermediate can narrow permitted dns scope",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					ok: []string{"dns:example.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:test.com"},
		},
		expectedError: "\"test.com\" is not permitted",
	},
	{
		name: "intermediate cannot narrow excluded dns scope",
		roots: []constraintsSpec{
			{
				bad: []string{"dns:"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					bad: []string{"dns:example.com"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:test.com"},
		},
		expectedError: "\"test.com\" is excluded by constraint \"\"",
	},
	{
		name: "intermediate can narrow excluded dns scope",
		roots: []constraintsSpec{
			{
				bad: []string{"dns:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{
					bad: []string{"dns:"},
				},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:test.com"},
		},
		expectedError: "\"test.com\" is excluded by constraint \"\"",
	},
	{
		name: "permitted dns constraint is not a prefix match",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:testexample.com"},
		},
		expectedError: "\"testexample.com\" is not permitted",
	},
	{
		name: "subdomain constraint does not allow wildcard",
		roots: []constraintsSpec{
			{
				ok: []string{"dns:a.com", "dns:foo.example.com", "dns:z.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:*.example.com"},
		},
		expectedError: "\"*.example.com\" is not permitted",
	},
	{
		name: "excluded dns constraint is not a prefix match",
		roots: []constraintsSpec{
			{
				bad: []string{"dns:example.com"},
			},
		},
		intermediates: [][]constraintsSpec{
			{
				{},
			},
		},
		leaf: leafSpec{
			sans: []string{"dns:testexample.com"},
		},
	},
}

func makeConstraintsCACert(constraints constraintsSpec, name string, key *ecdsa.PrivateKey, parent *Certificate, parentKey *ecdsa.PrivateKey) (*Certificate, error) {
	var serialBytes [16]byte
	rand.Read(serialBytes[:])

	template := &Certificate{
		SerialNumber: new(big.Int).SetBytes(serialBytes[:]),
		Subject: pkix.Name{
			CommonName: name,
		},
		NotBefore:             time.Unix(1000, 0),
		NotAfter:              time.Unix(2000, 0),
		KeyUsage:              KeyUsageCertSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	if err := addConstraintsToTemplate(constraints, template); err != nil {
		return nil, err
	}

	if parent == nil {
		parent = template
	}
	derBytes, err := CreateCertificate(rand.Reader, template, parent, &key.PublicKey, parentKey)
	if err != nil {
		return nil, err
	}

	caCert, err := ParseCertificate(derBytes)
	if err != nil {
		return nil, err
	}

	return caCert, nil
}

func makeConstraintsLeafCert(leaf leafSpec, key *ecdsa.PrivateKey, parent *Certificate, parentKey *ecdsa.PrivateKey) (*Certificate, error) {
	var serialBytes [16]byte
	rand.Read(serialBytes[:])

	template := &Certificate{
		SerialNumber: new(big.Int).SetBytes(serialBytes[:]),
		Subject: pkix.Name{
			OrganizationalUnit: []string{"Leaf"},
			CommonName:         leaf.cn,
		},
		NotBefore:             time.Unix(1000, 0),
		NotAfter:              time.Unix(2000, 0),
		KeyUsage:              KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
		IsCA:                  false,
	}

	for _, name := range leaf.sans {
		switch {
		case strings.HasPrefix(name, "dns:"):
			template.DNSNames = append(template.DNSNames, name[4:])

		case strings.HasPrefix(name, "ip:"):
			ip := net.ParseIP(name[3:])
			if ip == nil {
				return nil, fmt.Errorf("cannot parse IP %q", name[3:])
			}
			template.IPAddresses = append(template.IPAddresses, ip)

		case strings.HasPrefix(name, "invalidip:"):
			ipBytes, err := hex.DecodeString(name[10:])
			if err != nil {
				return nil, fmt.Errorf("cannot parse invalid IP: %s", err)
			}
			template.IPAddresses = append(template.IPAddresses, net.IP(ipBytes))

		case strings.HasPrefix(name, "email:"):
			template.EmailAddresses = append(template.EmailAddresses, name[6:])

		case strings.HasPrefix(name, "uri:"):
			uri, err := url.Parse(name[4:])
			if err != nil {
				return nil, fmt.Errorf("cannot parse URI %q: %s", name[4:], err)
			}
			template.URIs = append(template.URIs, uri)

		case strings.HasPrefix(name, "unknown:"):
			// This is a special case for testing unknown
			// name types. A custom SAN extension is
			// injected into the certificate.
			if len(leaf.sans) != 1 {
				panic("when using unknown name types, it must be the sole name")
			}

			template.ExtraExtensions = append(template.ExtraExtensions, pkix.Extension{
				Id: []int{2, 5, 29, 17},
				Value: []byte{
					0x30, // SEQUENCE
					3,    // three bytes
					9,    // undefined GeneralName type 9
					1,
					1,
				},
			})

		default:
			return nil, fmt.Errorf("unknown name type %q", name)
		}
	}

	var err error
	if template.ExtKeyUsage, template.UnknownExtKeyUsage, err = parseEKUs(leaf.ekus); err != nil {
		return nil, err
	}

	if parent == nil {
		parent = template
	}

	derBytes, err := CreateCertificate(rand.Reader, template, parent, &key.PublicKey, parentKey)
	if err != nil {
		return nil, err
	}

	return ParseCertificate(derBytes)
}

func customConstraintsExtension(typeNum int, constraint []byte, isExcluded bool) pkix.Extension {
	appendConstraint := func(contents []byte, tag uint8) []byte {
		contents = append(contents, tag|32 /* constructed */ |0x80 /* context-specific */)
		contents = append(contents, byte(4+len(constraint)) /* length */)
		contents = append(contents, 0x30 /* SEQUENCE */)
		contents = append(contents, byte(2+len(constraint)) /* length */)
		contents = append(contents, byte(typeNum) /* GeneralName type */)
		contents = append(contents, byte(len(constraint)))
		return append(contents, constraint...)
	}

	var contents []byte
	if !isExcluded {
		contents = appendConstraint(contents, 0 /* tag 0 for permitted */)
	} else {
		contents = appendConstraint(contents, 1 /* tag 1 for excluded */)
	}

	var value []byte
	value = append(value, 0x30 /* SEQUENCE */)
	value = append(value, byte(len(contents)))
	value = append(value, contents...)

	return pkix.Extension{
		Id:    []int{2, 5, 29, 30},
		Value: value,
	}
}

func addConstraintsToTemplate(constraints constraintsSpec, template *Certificate) error {
	parse := func(constraints []string) (dnsNames []string, ips []*net.IPNet, emailAddrs []string, uriDomains []string, err error) {
		for _, constraint := range constraints {
			switch {
			case strings.HasPrefix(constraint, "dns:"):
				dnsNames = append(dnsNames, constraint[4:])

			case strings.HasPrefix(constraint, "ip:"):
				_, ipNet, err := net.ParseCIDR(constraint[3:])
				if err != nil {
					return nil, nil, nil, nil, err
				}
				ips = append(ips, ipNet)

			case strings.HasPrefix(constraint, "email:"):
				emailAddrs = append(emailAddrs, constraint[6:])

			case strings.HasPrefix(constraint, "uri:"):
				uriDomains = append(uriDomains, constraint[4:])

			default:
				return nil, nil, nil, nil, fmt.Errorf("unknown constraint %q", constraint)
			}
		}

		return dnsNames, ips, emailAddrs, uriDomains, err
	}

	handleSpecialConstraint := func(constraint string, isExcluded bool) bool {
		switch {
		case constraint == "unknown:":
			template.ExtraExtensions = append(template.ExtraExtensions, customConstraintsExtension(9 /* undefined GeneralName type */, []byte{1}, isExcluded))

		default:
			return false
		}

		return true
	}

	if len(constraints.ok) == 1 && len(constraints.bad) == 0 {
		if handleSpecialConstraint(constraints.ok[0], false) {
			return nil
		}
	}

	if len(constraints.bad) == 1 && len(constraints.ok) == 0 {
		if handleSpecialConstraint(constraints.bad[0], true) {
			return nil
		}
	}

	var err error
	template.PermittedDNSDomains, template.PermittedIPRanges, template.PermittedEmailAddresses, template.PermittedURIDomains, err = parse(constraints.ok)
	if err != nil {
		return err
	}

	template.ExcludedDNSDomains, template.ExcludedIPRanges, template.ExcludedEmailAddresses, template.ExcludedURIDomains, err = parse(constraints.bad)
	if err != nil {
		return err
	}

	if template.ExtKeyUsage, template.UnknownExtKeyUsage, err = parseEKUs(constraints.ekus); err != nil {
		return err
	}

	return nil
}

func parseEKUs(ekuStrs []string) (ekus []ExtKeyUsage, unknowns []asn1.ObjectIdentifier, err error) {
	for _, s := range ekuStrs {
		switch s {
		case "serverAuth":
			ekus = append(ekus, ExtKeyUsageServerAuth)
		case "clientAuth":
			ekus = append(ekus, ExtKeyUsageClientAuth)
		case "email":
			ekus = append(ekus, ExtKeyUsageEmailProtection)
		case "netscapeSGC":
			ekus = append(ekus, ExtKeyUsageNetscapeServerGatedCrypto)
		case "msSGC":
			ekus = append(ekus, ExtKeyUsageMicrosoftServerGatedCrypto)
		case "any":
			ekus = append(ekus, ExtKeyUsageAny)
		case "other":
			unknowns = append(unknowns, asn1.ObjectIdentifier{2, 4, 1, 2, 3})
		default:
			return nil, nil, fmt.Errorf("unknown EKU %q", s)
		}
	}

	return
}

func TestConstraintCases(t *testing.T) {
	privateKeys := sync.Pool{
		New: func() any {
			priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
			if err != nil {
				panic(err)
			}
			return priv
		},
	}

	for i, test := range nameConstraintsTests {
		t.Run(test.name, func(t *testing.T) {
			rootPool := NewCertPool()
			rootKey := privateKeys.Get().(*ecdsa.PrivateKey)
			rootName := "Root " + strconv.Itoa(i)

			// keys keeps track of all the private keys used in a given
			// test and puts them back in the privateKeys pool at the end.
			keys := []*ecdsa.PrivateKey{rootKey}

			// At each level (root, intermediate(s), leaf), parent points to
			// an example parent certificate and parentKey the key for the
			// parent level. Since all certificates at a given level have
			// the same name and public key, any parent certificate is
			// sufficient to get the correct issuer name and authority
			// key ID.
			var parent *Certificate
			parentKey := rootKey

			for _, root := range test.roots {
				rootCert, err := makeConstraintsCACert(root, rootName, rootKey, nil, rootKey)
				if err != nil {
					t.Fatalf("failed to create root: %s", err)
				}

				parent = rootCert
				rootPool.AddCert(rootCert)
			}

			intermediatePool := NewCertPool()

			for level, intermediates := range test.intermediates {
				levelKey := privateKeys.Get().(*ecdsa.PrivateKey)
				keys = append(keys, levelKey)
				levelName := "Intermediate level " + strconv.Itoa(level)
				var last *Certificate

				for _, intermediate := range intermediates {
					caCert, err := makeConstraintsCACert(intermediate, levelName, levelKey, parent, parentKey)
					if err != nil {
						t.Fatalf("failed to create %q: %s", levelName, err)
					}

					last = caCert
					intermediatePool.AddCert(caCert)
				}

				parent = last
				parentKey = levelKey
			}

			leafKey := privateKeys.Get().(*ecdsa.PrivateKey)
			keys = append(keys, leafKey)

			leafCert, err := makeConstraintsLeafCert(test.leaf, leafKey, parent, parentKey)
			if err != nil {
				t.Fatalf("cannot create leaf: %s", err)
			}

			// Skip tests with CommonName set because OpenSSL will try to match it
			// against name constraints, while we ignore it when it's not hostname-looking.
			if !test.noOpenSSL && testNameConstraintsAgainstOpenSSL && test.leaf.cn == "" {
				output, err := testChainAgainstOpenSSL(t, leafCert, intermediatePool, rootPool)
				if err == nil && len(test.expectedError) > 0 {
					t.Error("unexpectedly succeeded against OpenSSL")
					if debugOpenSSLFailure {
						return
					}
				}

				if err != nil {
					if _, ok := err.(*exec.ExitError); !ok {
						t.Errorf("OpenSSL failed to run: %s", err)
					} else if len(test.expectedError) == 0 {
						t.Errorf("OpenSSL unexpectedly failed: %v", output)
						if debugOpenSSLFailure {
							return
						}
					}
				}
			}

			verifyOpts := VerifyOptions{
				Roots:         rootPool,
				Intermediates: intermediatePool,
				CurrentTime:   time.Unix(1500, 0),
				KeyUsages:     test.requestedEKUs,
			}
			_, err = leafCert.Verify(verifyOpts)

			logInfo := false
			if len(test.expectedError) == 0 {
				if err != nil {
					t.Errorf("unexpected failure: %s", err)
				} else {
					logInfo = false
				}
			} else {
				if err == nil {
					t.Error("unexpected success")
				} else if !strings.Contains(err.Error(), test.expectedError) {
					t.Errorf("expected error containing %q, but got: %s", test.expectedError, err)
				} else {
					logInfo = false
				}
			}

			if logInfo {
				certAsPEM := func(cert *Certificate) string {
					var buf bytes.Buffer
					pem.Encode(&buf, &pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw})
					return buf.String()
				}
				t.Errorf("root:\n%s", certAsPEM(rootPool.mustCert(t, 0)))
				if intermediates := allCerts(t, intermediatePool); len(intermediates) > 0 {
					for ii, intermediate := range intermediates {
						t.Errorf("intermediate %d:\n%s", ii, certAsPEM(intermediate))
					}
				}
				t.Errorf("leaf:\n%s", certAsPEM(leafCert))
			}

			for _, key := range keys {
				privateKeys.Put(key)
			}
		})
	}
}

func writePEMsToTempFile(certs []*Certificate) *os.File {
	file, err := os.CreateTemp("", "name_constraints_test")
	if err != nil {
		panic("cannot create tempfile")
	}

	pemBlock := &pem.Block{Type: "CERTIFICATE"}
	for _, cert := range certs {
		pemBlock.Bytes = cert.Raw
		pem.Encode(file, pemBlock)
	}

	return file
}

func testChainAgainstOpenSSL(t *testing.T, leaf *Certificate, intermediates, roots *CertPool) (string, error) {
	args := []string{"verify", "-no_check_time"}

	rootsFile := writePEMsToTempFile(allCerts(t, roots))
	if debugOpenSSLFailure {
		println("roots file:", rootsFile.Name())
	} else {
		defer os.Remove(rootsFile.Name())
	}
	args = append(args, "-CAfile", rootsFile.Name())

	if intermediates.len() > 0 {
		intermediatesFile := writePEMsToTempFile(allCerts(t, intermediates))
		if debugOpenSSLFailure {
			println("intermediates file:", intermediatesFile.Name())
		} else {
			defer os.Remove(intermediatesFile.Name())
		}
		args = append(args, "-untrusted", intermediatesFile.Name())
	}

	leafFile := writePEMsToTempFile([]*Certificate{leaf})
	if debugOpenSSLFailure {
		println("leaf file:", leafFile.Name())
	} else {
		defer os.Remove(leafFile.Name())
	}
	args = append(args, leafFile.Name())

	var output bytes.Buffer
	cmd := exec.Command("openssl", args...)
	cmd.Stdout = &output
	cmd.Stderr = &output

	err := cmd.Run()
	return output.String(), err
}

var rfc2821Tests = []struct {
	in                string
	localPart, domain string
}{
	{"foo@example.com", "foo", "example.com"},
	{"@example.com", "", ""},
	{"\"@example.com", "", ""},
	{"\"\"@example.com", "", "example.com"},
	{"\"a\"@example.com", "a", "example.com"},
	{"\"\\a\"@example.com", "a", "example.com"},
	{"a\"@example.com", "", ""},
	{"foo..bar@example.com", "", ""},
	{".foo.bar@example.com", "", ""},
	{"foo.bar.@example.com", "", ""},
	{"|{}?'@example.com", "|{}?'", "example.com"},

	// Examples from RFC 3696
	{"Abc\\@def@example.com", "Abc@def", "example.com"},
	{"Fred\\ Bloggs@example.com", "Fred Bloggs", "example.com"},
	{"Joe.\\\\Blow@example.com", "Joe.\\Blow", "example.com"},
	{"\"Abc@def\"@example.com", "Abc@def", "example.com"},
	{"\"Fred Bloggs\"@example.com", "Fred Bloggs", "example.com"},
	{"customer/department=shipping@example.com", "customer/department=shipping", "example.com"},
	{"$A12345@example.com", "$A12345", "example.com"},
	{"!def!xyz%abc@example.com", "!def!xyz%abc", "example.com"},
	{"_somename@example.com", "_somename", "example.com"},
}

func TestRFC2821Parsing(t *testing.T) {
	for i, test := range rfc2821Tests {
		mailbox, ok := parseRFC2821Mailbox(test.in)
		expectedFailure := len(test.localPart) == 0 && len(test.domain) == 0

		if ok && expectedFailure {
			t.Errorf("#%d: %q unexpectedly parsed as (%q, %q)", i, test.in, mailbox.local, mailbox.domain)
			continue
		}

		if !ok && !expectedFailure {
			t.Errorf("#%d: unexpected failure for %q", i, test.in)
			continue
		}

		if !ok {
			continue
		}

		if mailbox.local != test.localPart || mailbox.domain != test.domain {
			t.Errorf("#%d: %q parsed as (%q, %q), but wanted (%q, %q)", i, test.in, mailbox.local, mailbox.domain, test.localPart, test.domain)
		}
	}
}

func TestBadNamesInConstraints(t *testing.T) {
	constraintParseError := func(err error) bool {
		str := err.Error()
		return strings.Contains(str, "failed to parse ") && strings.Contains(str, "constraint")
	}

	encodingError := func(err error) bool {
		return strings.Contains(err.Error(), "cannot be encoded as an IA5String")
	}

	// Bad names in constraints should not parse.
	badNames := []struct {
		name    string
		matcher func(error) bool
	}{
		{"dns:foo.com.", constraintParseError},
		{"email:abc@foo.com.", constraintParseError},
		{"email:foo.com.", constraintParseError},
		{"uri:example.com.", constraintParseError},
		{"uri:1.2.3.4", constraintParseError},
		{"uri:ffff::1", constraintParseError},
		{"dns:not–hyphen.com", encodingError},
		{"email:foo@not–hyphen.com", encodingError},
		{"uri:not–hyphen.com", encodingError},
	}

	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		panic(err)
	}

	for _, test := range badNames {
		_, err := makeConstraintsCACert(constraintsSpec{
			ok: []string{test.name},
		}, "TestAbsoluteNamesInConstraints", priv, nil, priv)

		if err == nil {
			t.Errorf("bad name %q unexpectedly accepted in name constraint", test.name)
			continue
		} else {
			if !test.matcher(err) {
				t.Errorf("bad name %q triggered unrecognised error: %s", test.name, err)
			}
		}
	}
}

func TestBadNamesInSANs(t *testing.T) {
	// Bad names in URI and IP SANs should not parse. Bad DNS and email SANs
	// will parse and are tested in name constraint tests at the top of this
	// file.
	badNames := []string{
		"uri:https://example.com./dsf",
		"invalidip:0102",
		"invalidip:0102030405",
	}

	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		panic(err)
	}

	for _, badName := range badNames {
		_, err := makeConstraintsLeafCert(leafSpec{sans: []string{badName}}, priv, nil, priv)

		if err == nil {
			t.Errorf("bad name %q unexpectedly accepted in SAN", badName)
			continue
		}

		if str := err.Error(); !strings.Contains(str, "cannot parse ") {
			t.Errorf("bad name %q triggered unrecognised error: %s", badName, str)
		}
	}
}
