// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"fmt"
	"net"
	"net/netip"
	"net/url"
	"slices"
	"strings"
)

// This file contains the data structures and functions necessary for
// efficiently checking X.509 name constraints. The method for constraint
// checking implemented in this file is based on a technique originally
// described by davidben@google.com.
//
// The basic concept is based on the fact that constraints describe possibly
// overlapping subtrees that we need to match against. If sorted in lexicographic
// order, and then pruned, removing any subtrees that overlap with preceding
// subtrees, a simple binary search can be used to find the nearest matching
// prefix. This reduces the complexity of name constraint checking from
// quadratic to log linear complexity.
//
// A close reading of RFC 5280 may suggest that constraints could also be
// implemented as a trie (or radix tree), which would present the possibility of
// doing construction and matching in linear time, but the memory cost of
// implementing them is actually quite high, and in the worst case (where each
// node has a high number of children) can be abused to require a program to use
// significant amounts of memory. The log linear approach taken here is
// extremely cheap in terms of memory because we directly alias the already
// parsed constraints, thus avoiding the need to do significant additional
// allocations.
//
// The basic data structure is nameConstraintsSet, which implements the sorting,
// pruning, and querying of the prefix sets.
//
// In order to check IP, DNS, URI, and email constraints, we need to use two
// different techniques, one for IP addresses, which is quite simple, and one
// for DNS names, which additionally compose the portions of URIs and emails we
// care about (technically we also need some special logic for email addresses
// as well for when constraints comprise of full email addresses) which is
// slightly more complex.
//
// IP addresses use two nameConstraintsSets, one for IPv4 addresses and one for
// IPv6 addresses, with no additional logic.
//
// DNS names require some extra logic in order to handle the distinctions
// between permitted and excluded subtrees, as well as for wildcards, and the
// semantics of leading period constraints (i.e. '.example.com'). This logic is
// implemented in the dnsConstraints type.
//
// Email addresses also require some additional logic, which does not make use
// of nameConstraintsSet, to handle constraints which define full email
// addresses (i.e. 'test@example.com'). For bare domain constraints, we use the
// dnsConstraints type described above, querying the domain portion of the email
// address. For full email addresses, we also hold a map of email addresses that
// map the local portion of the email to the domain. When querying full email
// addresses we then check if the local portion of the email is present in the
// map, and if so case insensitively compare the domain portion of the
// email.

type nameConstraintsSet[T *net.IPNet | string, V net.IP | string] struct {
	set []T
}

// sortAndPrune sorts the constraints using the provided comparison function, and then
// prunes any constraints that are subsets of preceding constraints using the
// provided subset function.
func (nc *nameConstraintsSet[T, V]) sortAndPrune(cmp func(T, T) int, subset func(T, T) bool) {
	if len(nc.set) < 2 {
		return
	}

	slices.SortFunc(nc.set, cmp)

	if len(nc.set) < 2 {
		return
	}
	writeIndex := 1
	for readIndex := 1; readIndex < len(nc.set); readIndex++ {
		if !subset(nc.set[writeIndex-1], nc.set[readIndex]) {
			nc.set[writeIndex] = nc.set[readIndex]
			writeIndex++
		}
	}
	nc.set = nc.set[:writeIndex]
}

// search does a binary search over the constraints set for the provided value
// s, using the provided comparison function cmp to find the lower bound, and
// the match function to determine if the found constraint is a prefix of s. If
// a matching constraint is found, it is returned along with true. If no
// matching constraint is found, the zero value of T and false are returned.
func (nc *nameConstraintsSet[T, V]) search(s V, cmp func(T, V) int, match func(T, V) bool) (lowerBound T, exactMatch bool) {
	if len(nc.set) == 0 {
		return lowerBound, false
	}
	// Look for the lower bound of s in the set.
	i, found := slices.BinarySearchFunc(nc.set, s, cmp)
	// If we found an exact match, return it
	if found {
		return nc.set[i], true
	}

	if i < 0 {
		return lowerBound, false
	}

	var constraint T
	if i == 0 {
		constraint = nc.set[0]
	} else {
		constraint = nc.set[i-1]
	}
	if match(constraint, s) {
		return constraint, true
	}
	return lowerBound, false
}

func ipNetworkSubset(a, b *net.IPNet) bool {
	if !a.Contains(b.IP) {
		return false
	}
	broadcast := make(net.IP, len(b.IP))
	for i := range b.IP {
		broadcast[i] = b.IP[i] | (^b.Mask[i])
	}
	return a.Contains(broadcast)
}

func ipNetworkCompare(a, b *net.IPNet) int {
	i := bytes.Compare(a.IP, b.IP)
	if i != 0 {
		return i
	}
	return bytes.Compare(a.Mask, b.Mask)
}

func ipBinarySearch(constraint *net.IPNet, target net.IP) int {
	return bytes.Compare(constraint.IP, target)
}

func ipMatch(constraint *net.IPNet, target net.IP) bool {
	return constraint.Contains(target)
}

type ipConstraints struct {
	// NOTE: we could store IP network prefixes as a pre-processed byte slice
	// (i.e. by masking the IP) and doing the byte prefix checking using faster
	// techniques, but this would require allocating new byte slices, which is
	// likely significantly more expensive than just operating on the
	// pre-allocated *net.IPNet and net.IP objects directly.

	ipv4 *nameConstraintsSet[*net.IPNet, net.IP]
	ipv6 *nameConstraintsSet[*net.IPNet, net.IP]
}

func newIPNetConstraints(l []*net.IPNet) interface {
	query(net.IP) (*net.IPNet, bool)
} {
	if len(l) == 0 {
		return nil
	}
	var ipv4, ipv6 []*net.IPNet
	for _, n := range l {
		if len(n.IP) == net.IPv4len {
			ipv4 = append(ipv4, n)
		} else {
			ipv6 = append(ipv6, n)
		}
	}
	var v4c, v6c *nameConstraintsSet[*net.IPNet, net.IP]
	if len(ipv4) > 0 {
		v4c = &nameConstraintsSet[*net.IPNet, net.IP]{
			set: ipv4,
		}
		v4c.sortAndPrune(ipNetworkCompare, ipNetworkSubset)
	}
	if len(ipv6) > 0 {
		v6c = &nameConstraintsSet[*net.IPNet, net.IP]{
			set: ipv6,
		}
		v6c.sortAndPrune(ipNetworkCompare, ipNetworkSubset)
	}
	return &ipConstraints{ipv4: v4c, ipv6: v6c}
}

func (ipc *ipConstraints) query(ip net.IP) (*net.IPNet, bool) {
	var c *nameConstraintsSet[*net.IPNet, net.IP]
	if len(ip) == net.IPv4len {
		c = ipc.ipv4
	} else {
		c = ipc.ipv6
	}
	if c == nil {
		return nil, false
	}
	return c.search(ip, ipBinarySearch, ipMatch)
}

// dnsHasSuffix case-insensitively checks if DNS name b is a label suffix of DNS
// name a, meaning that example.com is not considered a suffix of
// testexample.com, but is a suffix of test.example.com.
//
// dnsHasSuffix supports the URI "leading period" constraint semantics, which
// while not explicitly defined for dNSNames in RFC 5280, are widely supported
// (see errata 5997). In particular, a constraint of ".example.com" is
// considered to only match subdomains of example.com, but not example.com
// itself.
//
// a and b must both be non-empty strings representing (mostly) valid DNS names.
func dnsHasSuffix(a, b string) bool {
	lenA := len(a)
	lenB := len(b)
	if lenA > lenB {
		return false
	}
	i := lenA - 1
	offset := lenA - lenB
	for ; i >= 0; i-- {
		ar, br := a[i], b[i-(offset)]
		if ar == br {
			continue
		}
		if br < ar {
			ar, br = br, ar
		}
		if 'A' <= ar && ar <= 'Z' && br == ar+'a'-'A' {
			continue
		}
		return false
	}

	if a[0] != '.' && lenB > lenA && b[lenB-lenA-1] != '.' {
		return false
	}

	return true
}

// dnsCompareTable contains the ASCII alphabet mapped from a characters index in
// the table to its lowercased form.
var dnsCompareTable [256]byte

func init() {
	// NOTE: we don't actually need the
	// full alphabet, but calculating offsets would be more expensive than just
	// having redundant characters.
	for i := 0; i < 256; i++ {
		c := byte(i)
		if 'A' <= c && c <= 'Z' {
			// Lowercase uppercase characters A-Z.
			c += 'a' - 'A'
		}
		dnsCompareTable[i] = c
	}
	// Set the period character to 0 so that we get the right sorting behavior.
	//
	// In particular, we need the period character to sort before the only
	// other valid DNS name character which isn't a-z or 0-9, the hyphen,
	// otherwise a name with a dash would be incorrectly sorted into the middle
	// of another tree.
	//
	// For example, imagine a certificate with the constraints "a.com", "a.a.com", and
	// "a-a.com". These would sort as "a.com", "a-a.com", "a.a.com", which would break
	// the pruning step since we wouldn't see that "a.a.com" is a subset of "a.com".
	// Sorting the period before the hyphen ensures that "a.a.com" sorts before "a-a.com".
	dnsCompareTable['.'] = 0
}

// dnsCompare is a case-insensitive reversed implementation of strings.Compare
// that operates from the end to the start of the strings. This is more
// efficient that allocating reversed version of a and b and using
// strings.Compare directly (even though it is highly optimized).
//
// NOTE: this function treats the period character ('.') as sorting above every
// other character, which is necessary for us to properly sort names into their
// correct order. This is further discussed in the init function above.
func dnsCompare(a, b string) int {
	idxA := len(a) - 1
	idxB := len(b) - 1

	for idxA >= 0 && idxB >= 0 {
		byteA := dnsCompareTable[a[idxA]]
		byteB := dnsCompareTable[b[idxB]]
		if byteA == byteB {
			idxA--
			idxB--
			continue
		}
		ret := 1
		if byteA < byteB {
			ret = -1
		}
		return ret
	}

	ret := 0
	if idxA < idxB {
		ret = -1
	} else if idxB < idxA {
		ret = 1
	}
	return ret
}

type dnsConstraints struct {
	// all lets us short circuit the query logic if we see a zero length
	// constraint which permits or excludes everything.
	all bool

	// permitted indicates if these constraints are for permitted or excluded
	// names.
	permitted bool

	constraints *nameConstraintsSet[string, string]

	// parentConstraints contains a subset of constraints which are used for
	// wildcard SAN queries, which are constructed by removing the first label
	// from the constraints in constraints. parentConstraints is only populated
	// if permitted is false.
	parentConstraints map[string]string
}

func newDNSConstraints(l []string, permitted bool) interface{ query(string) (string, bool) } {
	if len(l) == 0 {
		return nil
	}
	for _, n := range l {
		if len(n) == 0 {
			return &dnsConstraints{all: true}
		}
	}
	constraints := slices.Clone(l)

	nc := &dnsConstraints{
		constraints: &nameConstraintsSet[string, string]{
			set: constraints,
		},
		permitted: permitted,
	}

	nc.constraints.sortAndPrune(dnsCompare, dnsHasSuffix)

	if !permitted {
		parentConstraints := map[string]string{}
		for _, name := range nc.constraints.set {
			trimmedName := trimFirstLabel(name)
			if trimmedName == "" {
				continue
			}
			parentConstraints[trimmedName] = name
		}
		if len(parentConstraints) > 0 {
			nc.parentConstraints = parentConstraints
		}
	}

	return nc
}

func (dnc *dnsConstraints) query(s string) (string, bool) {
	if dnc.all {
		return "", true
	}

	constraint, match := dnc.constraints.search(s, dnsCompare, dnsHasSuffix)
	if match {
		return constraint, true
	}

	if !dnc.permitted && s[0] == '*' {
		trimmed := trimFirstLabel(s)
		if constraint, found := dnc.parentConstraints[trimmed]; found {
			return constraint, true
		}
	}
	return "", false
}

type emailConstraints struct {
	dnsConstraints interface{ query(string) (string, bool) }

	fullEmails map[string]string
}

func newEmailConstraints(l []string, permitted bool) interface {
	query(parsedEmail) (string, bool)
} {
	if len(l) == 0 {
		return nil
	}
	exactMap := map[string]string{}
	var domains []string
	for _, c := range l {
		if !strings.ContainsRune(c, '@') {
			domains = append(domains, c)
			continue
		}
		parsed, ok := parseRFC2821Mailbox(c)
		if !ok {
			// We've already parsed these addresses in parseCertificate, and
			// treat failures as a hard failure for parsing. The only way we can
			// get a parse failure here is if the caller has mutated the
			// certificate since parsing.
			continue
		}
		exactMap[parsed.local] = parsed.domain
	}
	ec := &emailConstraints{
		fullEmails: exactMap,
	}
	if len(domains) > 0 {
		ec.dnsConstraints = newDNSConstraints(domains, permitted)
	}
	return ec
}

func (ec *emailConstraints) query(s parsedEmail) (string, bool) {
	if len(ec.fullEmails) > 0 && strings.ContainsRune(s.email, '@') {
		if domain, ok := ec.fullEmails[s.mailbox.local]; ok && strings.EqualFold(domain, s.mailbox.domain) {
			return ec.fullEmails[s.email] + "@" + s.mailbox.domain, true
		}
	}
	if ec.dnsConstraints == nil {
		return "", false
	}
	constraint, found := ec.dnsConstraints.query(s.mailbox.domain)
	return constraint, found
}

type constraints[T any, V any] struct {
	constraintType string
	permitted      interface{ query(V) (T, bool) }
	excluded       interface{ query(V) (T, bool) }
}

func checkConstraints[T string | *net.IPNet, V any, P string | net.IP | parsedURI | parsedEmail](c constraints[T, V], s V, p P) error {
	if c.permitted != nil {
		if _, found := c.permitted.query(s); !found {
			return fmt.Errorf("%s %q is not permitted by any constraint", c.constraintType, p)
		}
	}
	if c.excluded != nil {
		if constraint, found := c.excluded.query(s); found {
			return fmt.Errorf("%s %q is excluded by constraint %q", c.constraintType, p, constraint)
		}
	}
	return nil
}

type chainConstraints struct {
	ip    constraints[*net.IPNet, net.IP]
	dns   constraints[string, string]
	uri   constraints[string, string]
	email constraints[string, parsedEmail]

	index int
	next  *chainConstraints
}

func (cc *chainConstraints) check(dns []string, uris []parsedURI, emails []parsedEmail, ips []net.IP) error {
	for _, ip := range ips {
		if err := checkConstraints(cc.ip, ip, ip); err != nil {
			return err
		}
	}
	for _, d := range dns {
		if !domainNameValid(d, false) {
			return fmt.Errorf("x509: cannot parse dnsName %q", d)
		}
		if err := checkConstraints(cc.dns, d, d); err != nil {
			return err
		}
	}
	for _, u := range uris {
		if !domainNameValid(u.domain, false) {
			return fmt.Errorf("x509: internal error: URI SAN %q failed to parse", u)
		}
		if err := checkConstraints(cc.uri, u.domain, u); err != nil {
			return err
		}
	}
	for _, e := range emails {
		if !domainNameValid(e.mailbox.domain, false) {
			return fmt.Errorf("x509: cannot parse rfc822Name %q", e.mailbox)
		}
		if err := checkConstraints(cc.email, e, e); err != nil {
			return err
		}
	}
	return nil
}

func checkChainConstraints(chain []*Certificate) error {
	var currentConstraints *chainConstraints
	var last *chainConstraints
	for i, c := range chain {
		if !c.hasNameConstraints() {
			continue
		}
		cc := &chainConstraints{
			ip:    constraints[*net.IPNet, net.IP]{"IP address", newIPNetConstraints(c.PermittedIPRanges), newIPNetConstraints(c.ExcludedIPRanges)},
			dns:   constraints[string, string]{"DNS name", newDNSConstraints(c.PermittedDNSDomains, true), newDNSConstraints(c.ExcludedDNSDomains, false)},
			uri:   constraints[string, string]{"URI", newDNSConstraints(c.PermittedURIDomains, true), newDNSConstraints(c.ExcludedURIDomains, false)},
			email: constraints[string, parsedEmail]{"email address", newEmailConstraints(c.PermittedEmailAddresses, true), newEmailConstraints(c.ExcludedEmailAddresses, false)},
			index: i,
		}
		if currentConstraints == nil {
			currentConstraints = cc
			last = cc
		} else if last != nil {
			last.next = cc
			last = cc
		}
	}
	if currentConstraints == nil {
		return nil
	}

	for i, c := range chain {
		if !c.hasSANExtension() {
			continue
		}
		if i >= currentConstraints.index {
			for currentConstraints.index <= i {
				if currentConstraints.next == nil {
					return nil
				}
				currentConstraints = currentConstraints.next
			}
		}

		uris, err := parseURIs(c.URIs)
		if err != nil {
			return err
		}
		emails, err := parseMailboxes(c.EmailAddresses)
		if err != nil {
			return err
		}

		for n := currentConstraints; n != nil; n = n.next {
			if err := n.check(c.DNSNames, uris, emails, c.IPAddresses); err != nil {
				return err
			}
		}
	}

	return nil
}

type parsedURI struct {
	uri    *url.URL
	domain string
}

func (u parsedURI) String() string {
	return u.uri.String()
}

func parseURIs(uris []*url.URL) ([]parsedURI, error) {
	parsed := make([]parsedURI, 0, len(uris))
	for _, uri := range uris {
		host := strings.ToLower(uri.Host)
		if len(host) == 0 {
			return nil, fmt.Errorf("URI with empty host (%q) cannot be matched against constraints", uri.String())
		}
		if strings.Contains(host, ":") && !strings.HasSuffix(host, "]") {
			var err error
			host, _, err = net.SplitHostPort(uri.Host)
			if err != nil {
				return nil, fmt.Errorf("cannot parse URI host %q: %v", uri.Host, err)
			}
		}

		// netip.ParseAddr will reject the URI IPv6 literal form "[...]", so we
		// check if _either_ the string parses as an IP, or if it is enclosed in
		// square brackets.
		if _, err := netip.ParseAddr(host); err == nil || (strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]")) {
			return nil, fmt.Errorf("URI with IP (%q) cannot be matched against constraints", uri.String())
		}

		parsed = append(parsed, parsedURI{uri, host})
	}
	return parsed, nil
}

type parsedEmail struct {
	email   string
	mailbox *rfc2821Mailbox
}

func (e parsedEmail) String() string {
	return e.mailbox.local + "@" + e.mailbox.domain
}

func parseMailboxes(emails []string) ([]parsedEmail, error) {
	parsed := make([]parsedEmail, 0, len(emails))
	for _, email := range emails {
		mailbox, ok := parseRFC2821Mailbox(email)
		if !ok {
			return nil, fmt.Errorf("cannot parse rfc822Name %q", email)
		}
		mailbox.domain = strings.ToLower(mailbox.domain)
		parsed = append(parsed, parsedEmail{strings.ToLower(email), &mailbox})
	}
	return parsed, nil
}

func trimFirstLabel(dnsName string) string {
	firstDotInd := strings.IndexByte(dnsName, '.')
	if firstDotInd < 0 {
		// Constraint is a single label, we cannot trim it.
		return ""
	}
	return dnsName[firstDotInd:]
}
