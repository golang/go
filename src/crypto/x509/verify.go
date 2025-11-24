// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import (
	"bytes"
	"crypto"
	"crypto/x509/pkix"
	"errors"
	"fmt"
	"iter"
	"maps"
	"net"
	"net/netip"
	"net/url"
	"reflect"
	"runtime"
	"strings"
	"time"
	"unicode/utf8"
)

type InvalidReason int

const (
	// NotAuthorizedToSign results when a certificate is signed by another
	// which isn't marked as a CA certificate.
	NotAuthorizedToSign InvalidReason = iota
	// Expired results when a certificate has expired, based on the time
	// given in the VerifyOptions.
	Expired
	// CANotAuthorizedForThisName results when an intermediate or root
	// certificate has a name constraint which doesn't permit a DNS or
	// other name (including IP address) in the leaf certificate.
	CANotAuthorizedForThisName
	// TooManyIntermediates results when a path length constraint is
	// violated.
	TooManyIntermediates
	// IncompatibleUsage results when the certificate's key usage indicates
	// that it may only be used for a different purpose.
	IncompatibleUsage
	// NameMismatch results when the subject name of a parent certificate
	// does not match the issuer name in the child.
	NameMismatch
	// NameConstraintsWithoutSANs is a legacy error and is no longer returned.
	NameConstraintsWithoutSANs
	// UnconstrainedName results when a CA certificate contains permitted
	// name constraints, but leaf certificate contains a name of an
	// unsupported or unconstrained type.
	UnconstrainedName
	// TooManyConstraints results when the number of comparison operations
	// needed to check a certificate exceeds the limit set by
	// VerifyOptions.MaxConstraintComparisions. This limit exists to
	// prevent pathological certificates can consuming excessive amounts of
	// CPU time to verify.
	TooManyConstraints
	// CANotAuthorizedForExtKeyUsage results when an intermediate or root
	// certificate does not permit a requested extended key usage.
	CANotAuthorizedForExtKeyUsage
	// NoValidChains results when there are no valid chains to return.
	NoValidChains
)

// CertificateInvalidError results when an odd error occurs. Users of this
// library probably want to handle all these errors uniformly.
type CertificateInvalidError struct {
	Cert   *Certificate
	Reason InvalidReason
	Detail string
}

func (e CertificateInvalidError) Error() string {
	switch e.Reason {
	case NotAuthorizedToSign:
		return "x509: certificate is not authorized to sign other certificates"
	case Expired:
		return "x509: certificate has expired or is not yet valid: " + e.Detail
	case CANotAuthorizedForThisName:
		return "x509: a root or intermediate certificate is not authorized to sign for this name: " + e.Detail
	case CANotAuthorizedForExtKeyUsage:
		return "x509: a root or intermediate certificate is not authorized for an extended key usage: " + e.Detail
	case TooManyIntermediates:
		return "x509: too many intermediates for path length constraint"
	case IncompatibleUsage:
		return "x509: certificate specifies an incompatible key usage"
	case NameMismatch:
		return "x509: issuer name does not match subject from issuing certificate"
	case NameConstraintsWithoutSANs:
		return "x509: issuer has name constraints but leaf doesn't have a SAN extension"
	case UnconstrainedName:
		return "x509: issuer has name constraints but leaf contains unknown or unconstrained name: " + e.Detail
	case NoValidChains:
		s := "x509: no valid chains built"
		if e.Detail != "" {
			s = fmt.Sprintf("%s: %s", s, e.Detail)
		}
		return s
	}
	return "x509: unknown error"
}

// HostnameError results when the set of authorized names doesn't match the
// requested name.
type HostnameError struct {
	Certificate *Certificate
	Host        string
}

func (h HostnameError) Error() string {
	c := h.Certificate

	if !c.hasSANExtension() && matchHostnames(c.Subject.CommonName, h.Host) {
		return "x509: certificate relies on legacy Common Name field, use SANs instead"
	}

	var valid string
	if ip := net.ParseIP(h.Host); ip != nil {
		// Trying to validate an IP
		if len(c.IPAddresses) == 0 {
			return "x509: cannot validate certificate for " + h.Host + " because it doesn't contain any IP SANs"
		}
		for _, san := range c.IPAddresses {
			if len(valid) > 0 {
				valid += ", "
			}
			valid += san.String()
		}
	} else {
		valid = strings.Join(c.DNSNames, ", ")
	}

	if len(valid) == 0 {
		return "x509: certificate is not valid for any names, but wanted to match " + h.Host
	}
	return "x509: certificate is valid for " + valid + ", not " + h.Host
}

// UnknownAuthorityError results when the certificate issuer is unknown
type UnknownAuthorityError struct {
	Cert *Certificate
	// hintErr contains an error that may be helpful in determining why an
	// authority wasn't found.
	hintErr error
	// hintCert contains a possible authority certificate that was rejected
	// because of the error in hintErr.
	hintCert *Certificate
}

func (e UnknownAuthorityError) Error() string {
	s := "x509: certificate signed by unknown authority"
	if e.hintErr != nil {
		certName := e.hintCert.Subject.CommonName
		if len(certName) == 0 {
			if len(e.hintCert.Subject.Organization) > 0 {
				certName = e.hintCert.Subject.Organization[0]
			} else {
				certName = "serial:" + e.hintCert.SerialNumber.String()
			}
		}
		s += fmt.Sprintf(" (possibly because of %q while trying to verify candidate authority certificate %q)", e.hintErr, certName)
	}
	return s
}

// SystemRootsError results when we fail to load the system root certificates.
type SystemRootsError struct {
	Err error
}

func (se SystemRootsError) Error() string {
	msg := "x509: failed to load system roots and no roots provided"
	if se.Err != nil {
		return msg + "; " + se.Err.Error()
	}
	return msg
}

func (se SystemRootsError) Unwrap() error { return se.Err }

// errNotParsed is returned when a certificate without ASN.1 contents is
// verified. Platform-specific verification needs the ASN.1 contents.
var errNotParsed = errors.New("x509: missing ASN.1 contents; use ParseCertificate")

// VerifyOptions contains parameters for Certificate.Verify.
type VerifyOptions struct {
	// DNSName, if set, is checked against the leaf certificate with
	// Certificate.VerifyHostname or the platform verifier.
	DNSName string

	// Intermediates is an optional pool of certificates that are not trust
	// anchors, but can be used to form a chain from the leaf certificate to a
	// root certificate.
	Intermediates *CertPool
	// Roots is the set of trusted root certificates the leaf certificate needs
	// to chain up to. If nil, the system roots or the platform verifier are used.
	Roots *CertPool

	// CurrentTime is used to check the validity of all certificates in the
	// chain. If zero, the current time is used.
	CurrentTime time.Time

	// KeyUsages specifies which Extended Key Usage values are acceptable. A
	// chain is accepted if it allows any of the listed values. An empty list
	// means ExtKeyUsageServerAuth. To accept any key usage, include ExtKeyUsageAny.
	KeyUsages []ExtKeyUsage

	// MaxConstraintComparisions is the maximum number of comparisons to
	// perform when checking a given certificate's name constraints. If
	// zero, a sensible default is used. This limit prevents pathological
	// certificates from consuming excessive amounts of CPU time when
	// validating. It does not apply to the platform verifier.
	MaxConstraintComparisions int

	// CertificatePolicies specifies which certificate policy OIDs are
	// acceptable during policy validation. An empty CertificatePolices
	// field implies any valid policy is acceptable.
	CertificatePolicies []OID

	// The following policy fields are unexported, because we do not expect
	// users to actually need to use them, but are useful for testing the
	// policy validation code.

	// inhibitPolicyMapping indicates if policy mapping should be allowed
	// during path validation.
	inhibitPolicyMapping bool

	// requireExplicitPolicy indidicates if explicit policies must be present
	// for each certificate being validated.
	requireExplicitPolicy bool

	// inhibitAnyPolicy indicates if the anyPolicy policy should be
	// processed if present in a certificate being validated.
	inhibitAnyPolicy bool
}

const (
	leafCertificate = iota
	intermediateCertificate
	rootCertificate
)

// rfc2821Mailbox represents a “mailbox” (which is an email address to most
// people) by breaking it into the “local” (i.e. before the '@') and “domain”
// parts.
type rfc2821Mailbox struct {
	local, domain string
}

// parseRFC2821Mailbox parses an email address into local and domain parts,
// based on the ABNF for a “Mailbox” from RFC 2821. According to RFC 5280,
// Section 4.2.1.6 that's correct for an rfc822Name from a certificate: “The
// format of an rfc822Name is a "Mailbox" as defined in RFC 2821, Section 4.1.2”.
func parseRFC2821Mailbox(in string) (mailbox rfc2821Mailbox, ok bool) {
	if len(in) == 0 {
		return mailbox, false
	}

	localPartBytes := make([]byte, 0, len(in)/2)

	if in[0] == '"' {
		// Quoted-string = DQUOTE *qcontent DQUOTE
		// non-whitespace-control = %d1-8 / %d11 / %d12 / %d14-31 / %d127
		// qcontent = qtext / quoted-pair
		// qtext = non-whitespace-control /
		//         %d33 / %d35-91 / %d93-126
		// quoted-pair = ("\" text) / obs-qp
		// text = %d1-9 / %d11 / %d12 / %d14-127 / obs-text
		//
		// (Names beginning with “obs-” are the obsolete syntax from RFC 2822,
		// Section 4. Since it has been 16 years, we no longer accept that.)
		in = in[1:]
	QuotedString:
		for {
			if len(in) == 0 {
				return mailbox, false
			}
			c := in[0]
			in = in[1:]

			switch {
			case c == '"':
				break QuotedString

			case c == '\\':
				// quoted-pair
				if len(in) == 0 {
					return mailbox, false
				}
				if in[0] == 11 ||
					in[0] == 12 ||
					(1 <= in[0] && in[0] <= 9) ||
					(14 <= in[0] && in[0] <= 127) {
					localPartBytes = append(localPartBytes, in[0])
					in = in[1:]
				} else {
					return mailbox, false
				}

			case c == 11 ||
				c == 12 ||
				// Space (char 32) is not allowed based on the
				// BNF, but RFC 3696 gives an example that
				// assumes that it is. Several “verified”
				// errata continue to argue about this point.
				// We choose to accept it.
				c == 32 ||
				c == 33 ||
				c == 127 ||
				(1 <= c && c <= 8) ||
				(14 <= c && c <= 31) ||
				(35 <= c && c <= 91) ||
				(93 <= c && c <= 126):
				// qtext
				localPartBytes = append(localPartBytes, c)

			default:
				return mailbox, false
			}
		}
	} else {
		// Atom ("." Atom)*
	NextChar:
		for len(in) > 0 {
			// atext from RFC 2822, Section 3.2.4
			c := in[0]

			switch {
			case c == '\\':
				// Examples given in RFC 3696 suggest that
				// escaped characters can appear outside of a
				// quoted string. Several “verified” errata
				// continue to argue the point. We choose to
				// accept it.
				in = in[1:]
				if len(in) == 0 {
					return mailbox, false
				}
				fallthrough

			case ('0' <= c && c <= '9') ||
				('a' <= c && c <= 'z') ||
				('A' <= c && c <= 'Z') ||
				c == '!' || c == '#' || c == '$' || c == '%' ||
				c == '&' || c == '\'' || c == '*' || c == '+' ||
				c == '-' || c == '/' || c == '=' || c == '?' ||
				c == '^' || c == '_' || c == '`' || c == '{' ||
				c == '|' || c == '}' || c == '~' || c == '.':
				localPartBytes = append(localPartBytes, in[0])
				in = in[1:]

			default:
				break NextChar
			}
		}

		if len(localPartBytes) == 0 {
			return mailbox, false
		}

		// From RFC 3696, Section 3:
		// “period (".") may also appear, but may not be used to start
		// or end the local part, nor may two or more consecutive
		// periods appear.”
		twoDots := []byte{'.', '.'}
		if localPartBytes[0] == '.' ||
			localPartBytes[len(localPartBytes)-1] == '.' ||
			bytes.Contains(localPartBytes, twoDots) {
			return mailbox, false
		}
	}

	if len(in) == 0 || in[0] != '@' {
		return mailbox, false
	}
	in = in[1:]

	// The RFC species a format for domains, but that's known to be
	// violated in practice so we accept that anything after an '@' is the
	// domain part.
	if _, ok := domainToReverseLabels(in); !ok {
		return mailbox, false
	}

	mailbox.local = string(localPartBytes)
	mailbox.domain = in
	return mailbox, true
}

// domainToReverseLabels converts a textual domain name like foo.example.com to
// the list of labels in reverse order, e.g. ["com", "example", "foo"].
func domainToReverseLabels(domain string) (reverseLabels []string, ok bool) {
	reverseLabels = make([]string, 0, strings.Count(domain, ".")+1)
	for len(domain) > 0 {
		if i := strings.LastIndexByte(domain, '.'); i == -1 {
			reverseLabels = append(reverseLabels, domain)
			domain = ""
		} else {
			reverseLabels = append(reverseLabels, domain[i+1:])
			domain = domain[:i]
			if i == 0 { // domain == ""
				// domain is prefixed with an empty label, append an empty
				// string to reverseLabels to indicate this.
				reverseLabels = append(reverseLabels, "")
			}
		}
	}

	if len(reverseLabels) > 0 && len(reverseLabels[0]) == 0 {
		// An empty label at the end indicates an absolute value.
		return nil, false
	}

	for _, label := range reverseLabels {
		if len(label) == 0 {
			// Empty labels are otherwise invalid.
			return nil, false
		}

		for _, c := range label {
			if c < 33 || c > 126 {
				// Invalid character.
				return nil, false
			}
		}
	}

	return reverseLabels, true
}

func matchEmailConstraint(mailbox rfc2821Mailbox, constraint string, excluded bool, reversedDomainsCache map[string][]string, reversedConstraintsCache map[string][]string) (bool, error) {
	// If the constraint contains an @, then it specifies an exact mailbox
	// name.
	if strings.Contains(constraint, "@") {
		constraintMailbox, ok := parseRFC2821Mailbox(constraint)
		if !ok {
			return false, fmt.Errorf("x509: internal error: cannot parse constraint %q", constraint)
		}
		return mailbox.local == constraintMailbox.local && strings.EqualFold(mailbox.domain, constraintMailbox.domain), nil
	}

	// Otherwise the constraint is like a DNS constraint of the domain part
	// of the mailbox.
	return matchDomainConstraint(mailbox.domain, constraint, excluded, reversedDomainsCache, reversedConstraintsCache)
}

func matchURIConstraint(uri *url.URL, constraint string, excluded bool, reversedDomainsCache map[string][]string, reversedConstraintsCache map[string][]string) (bool, error) {
	// From RFC 5280, Section 4.2.1.10:
	// “a uniformResourceIdentifier that does not include an authority
	// component with a host name specified as a fully qualified domain
	// name (e.g., if the URI either does not include an authority
	// component or includes an authority component in which the host name
	// is specified as an IP address), then the application MUST reject the
	// certificate.”

	host := uri.Host
	if len(host) == 0 {
		return false, fmt.Errorf("URI with empty host (%q) cannot be matched against constraints", uri.String())
	}

	if strings.Contains(host, ":") && !strings.HasSuffix(host, "]") {
		var err error
		host, _, err = net.SplitHostPort(uri.Host)
		if err != nil {
			return false, err
		}
	}

	// netip.ParseAddr will reject the URI IPv6 literal form "[...]", so we
	// check if _either_ the string parses as an IP, or if it is enclosed in
	// square brackets.
	if _, err := netip.ParseAddr(host); err == nil || (strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]")) {
		return false, fmt.Errorf("URI with IP (%q) cannot be matched against constraints", uri.String())
	}

	return matchDomainConstraint(host, constraint, excluded, reversedDomainsCache, reversedConstraintsCache)
}

func matchIPConstraint(ip net.IP, constraint *net.IPNet) (bool, error) {
	if len(ip) != len(constraint.IP) {
		return false, nil
	}

	for i := range ip {
		if mask := constraint.Mask[i]; ip[i]&mask != constraint.IP[i]&mask {
			return false, nil
		}
	}

	return true, nil
}

func matchDomainConstraint(domain, constraint string, excluded bool, reversedDomainsCache map[string][]string, reversedConstraintsCache map[string][]string) (bool, error) {
	// The meaning of zero length constraints is not specified, but this
	// code follows NSS and accepts them as matching everything.
	if len(constraint) == 0 {
		return true, nil
	}

	domainLabels, found := reversedDomainsCache[domain]
	if !found {
		var ok bool
		domainLabels, ok = domainToReverseLabels(domain)
		if !ok {
			return false, fmt.Errorf("x509: internal error: cannot parse domain %q", domain)
		}
		reversedDomainsCache[domain] = domainLabels
	}

	wildcardDomain := false
	if len(domain) > 0 && domain[0] == '*' {
		wildcardDomain = true
	}

	// RFC 5280 says that a leading period in a domain name means that at
	// least one label must be prepended, but only for URI and email
	// constraints, not DNS constraints. The code also supports that
	// behaviour for DNS constraints.

	mustHaveSubdomains := false
	if constraint[0] == '.' {
		mustHaveSubdomains = true
		constraint = constraint[1:]
	}

	constraintLabels, found := reversedConstraintsCache[constraint]
	if !found {
		var ok bool
		constraintLabels, ok = domainToReverseLabels(constraint)
		if !ok {
			return false, fmt.Errorf("x509: internal error: cannot parse domain %q", constraint)
		}
		reversedConstraintsCache[constraint] = constraintLabels
	}

	if len(domainLabels) < len(constraintLabels) ||
		(mustHaveSubdomains && len(domainLabels) == len(constraintLabels)) {
		return false, nil
	}

	if excluded && wildcardDomain && len(domainLabels) > 1 && len(constraintLabels) > 0 {
		domainLabels = domainLabels[:len(domainLabels)-1]
		constraintLabels = constraintLabels[:len(constraintLabels)-1]
	}

	for i, constraintLabel := range constraintLabels {
		if !strings.EqualFold(constraintLabel, domainLabels[i]) {
			return false, nil
		}
	}

	return true, nil
}

// checkNameConstraints checks that c permits a child certificate to claim the
// given name, of type nameType. The argument parsedName contains the parsed
// form of name, suitable for passing to the match function. The total number
// of comparisons is tracked in the given count and should not exceed the given
// limit.
func (c *Certificate) checkNameConstraints(count *int,
	maxConstraintComparisons int,
	nameType string,
	name string,
	parsedName any,
	match func(parsedName, constraint any, excluded bool) (match bool, err error),
	permitted, excluded any) error {

	excludedValue := reflect.ValueOf(excluded)

	*count += excludedValue.Len()
	if *count > maxConstraintComparisons {
		return CertificateInvalidError{c, TooManyConstraints, ""}
	}

	for i := 0; i < excludedValue.Len(); i++ {
		constraint := excludedValue.Index(i).Interface()
		match, err := match(parsedName, constraint, true)
		if err != nil {
			return CertificateInvalidError{c, CANotAuthorizedForThisName, err.Error()}
		}

		if match {
			return CertificateInvalidError{c, CANotAuthorizedForThisName, fmt.Sprintf("%s %q is excluded by constraint %q", nameType, name, constraint)}
		}
	}

	permittedValue := reflect.ValueOf(permitted)

	*count += permittedValue.Len()
	if *count > maxConstraintComparisons {
		return CertificateInvalidError{c, TooManyConstraints, ""}
	}

	ok := true
	for i := 0; i < permittedValue.Len(); i++ {
		constraint := permittedValue.Index(i).Interface()

		var err error
		if ok, err = match(parsedName, constraint, false); err != nil {
			return CertificateInvalidError{c, CANotAuthorizedForThisName, err.Error()}
		}

		if ok {
			break
		}
	}

	if !ok {
		return CertificateInvalidError{c, CANotAuthorizedForThisName, fmt.Sprintf("%s %q is not permitted by any constraint", nameType, name)}
	}

	return nil
}

// isValid performs validity checks on c given that it is a candidate to append
// to the chain in currentChain.
func (c *Certificate) isValid(certType int, currentChain []*Certificate, opts *VerifyOptions) error {
	if len(c.UnhandledCriticalExtensions) > 0 {
		return UnhandledCriticalExtension{}
	}

	if len(currentChain) > 0 {
		child := currentChain[len(currentChain)-1]
		if !bytes.Equal(child.RawIssuer, c.RawSubject) {
			return CertificateInvalidError{c, NameMismatch, ""}
		}
	}

	now := opts.CurrentTime
	if now.IsZero() {
		now = time.Now()
	}
	if now.Before(c.NotBefore) {
		return CertificateInvalidError{
			Cert:   c,
			Reason: Expired,
			Detail: fmt.Sprintf("current time %s is before %s", now.Format(time.RFC3339), c.NotBefore.Format(time.RFC3339)),
		}
	} else if now.After(c.NotAfter) {
		return CertificateInvalidError{
			Cert:   c,
			Reason: Expired,
			Detail: fmt.Sprintf("current time %s is after %s", now.Format(time.RFC3339), c.NotAfter.Format(time.RFC3339)),
		}
	}

	maxConstraintComparisons := opts.MaxConstraintComparisions
	if maxConstraintComparisons == 0 {
		maxConstraintComparisons = 250000
	}
	comparisonCount := 0

	if certType == intermediateCertificate || certType == rootCertificate {
		if len(currentChain) == 0 {
			return errors.New("x509: internal error: empty chain when appending CA cert")
		}
	}

	// Each time we do constraint checking, we need to check the constraints in
	// the current certificate against all of the names that preceded it. We
	// reverse these names using domainToReverseLabels, which is a relatively
	// expensive operation. Since we check each name against each constraint,
	// this requires us to do N*C calls to domainToReverseLabels (where N is the
	// total number of names that preceed the certificate, and C is the total
	// number of constraints in the certificate). By caching the results of
	// calling domainToReverseLabels, we can reduce that to N+C calls at the
	// cost of keeping all of the parsed names and constraints in memory until
	// we return from isValid.
	reversedDomainsCache := map[string][]string{}
	reversedConstraintsCache := map[string][]string{}

	if (certType == intermediateCertificate || certType == rootCertificate) &&
		c.hasNameConstraints() {
		toCheck := []*Certificate{}
		for _, c := range currentChain {
			if c.hasSANExtension() {
				toCheck = append(toCheck, c)
			}
		}
		for _, sanCert := range toCheck {
			err := forEachSAN(sanCert.getSANExtension(), func(tag int, data []byte) error {
				switch tag {
				case nameTypeEmail:
					name := string(data)
					mailbox, ok := parseRFC2821Mailbox(name)
					if !ok {
						return fmt.Errorf("x509: cannot parse rfc822Name %q", mailbox)
					}

					if err := c.checkNameConstraints(&comparisonCount, maxConstraintComparisons, "email address", name, mailbox,
						func(parsedName, constraint any, excluded bool) (bool, error) {
							return matchEmailConstraint(parsedName.(rfc2821Mailbox), constraint.(string), excluded, reversedDomainsCache, reversedConstraintsCache)
						}, c.PermittedEmailAddresses, c.ExcludedEmailAddresses); err != nil {
						return err
					}

				case nameTypeDNS:
					name := string(data)
					if !domainNameValid(name, false) {
						return fmt.Errorf("x509: cannot parse dnsName %q", name)
					}

					if err := c.checkNameConstraints(&comparisonCount, maxConstraintComparisons, "DNS name", name, name,
						func(parsedName, constraint any, excluded bool) (bool, error) {
							return matchDomainConstraint(parsedName.(string), constraint.(string), excluded, reversedDomainsCache, reversedConstraintsCache)
						}, c.PermittedDNSDomains, c.ExcludedDNSDomains); err != nil {
						return err
					}

				case nameTypeURI:
					name := string(data)
					uri, err := url.Parse(name)
					if err != nil {
						return fmt.Errorf("x509: internal error: URI SAN %q failed to parse", name)
					}

					if err := c.checkNameConstraints(&comparisonCount, maxConstraintComparisons, "URI", name, uri,
						func(parsedName, constraint any, excluded bool) (bool, error) {
							return matchURIConstraint(parsedName.(*url.URL), constraint.(string), excluded, reversedDomainsCache, reversedConstraintsCache)
						}, c.PermittedURIDomains, c.ExcludedURIDomains); err != nil {
						return err
					}

				case nameTypeIP:
					ip := net.IP(data)
					if l := len(ip); l != net.IPv4len && l != net.IPv6len {
						return fmt.Errorf("x509: internal error: IP SAN %x failed to parse", data)
					}

					if err := c.checkNameConstraints(&comparisonCount, maxConstraintComparisons, "IP address", ip.String(), ip,
						func(parsedName, constraint any, _ bool) (bool, error) {
							return matchIPConstraint(parsedName.(net.IP), constraint.(*net.IPNet))
						}, c.PermittedIPRanges, c.ExcludedIPRanges); err != nil {
						return err
					}

				default:
					// Unknown SAN types are ignored.
				}

				return nil
			})

			if err != nil {
				return err
			}
		}
	}

	// KeyUsage status flags are ignored. From Engineering Security, Peter
	// Gutmann: A European government CA marked its signing certificates as
	// being valid for encryption only, but no-one noticed. Another
	// European CA marked its signature keys as not being valid for
	// signatures. A different CA marked its own trusted root certificate
	// as being invalid for certificate signing. Another national CA
	// distributed a certificate to be used to encrypt data for the
	// country’s tax authority that was marked as only being usable for
	// digital signatures but not for encryption. Yet another CA reversed
	// the order of the bit flags in the keyUsage due to confusion over
	// encoding endianness, essentially setting a random keyUsage in
	// certificates that it issued. Another CA created a self-invalidating
	// certificate by adding a certificate policy statement stipulating
	// that the certificate had to be used strictly as specified in the
	// keyUsage, and a keyUsage containing a flag indicating that the RSA
	// encryption key could only be used for Diffie-Hellman key agreement.

	if certType == intermediateCertificate && (!c.BasicConstraintsValid || !c.IsCA) {
		return CertificateInvalidError{c, NotAuthorizedToSign, ""}
	}

	if c.BasicConstraintsValid && c.MaxPathLen >= 0 {
		numIntermediates := len(currentChain) - 1
		if numIntermediates > c.MaxPathLen {
			return CertificateInvalidError{c, TooManyIntermediates, ""}
		}
	}

	return nil
}

// Verify attempts to verify c by building one or more chains from c to a
// certificate in opts.Roots, using certificates in opts.Intermediates if
// needed. If successful, it returns one or more chains where the first
// element of the chain is c and the last element is from opts.Roots.
//
// If opts.Roots is nil, the platform verifier might be used, and
// verification details might differ from what is described below. If system
// roots are unavailable the returned error will be of type SystemRootsError.
//
// Name constraints in the intermediates will be applied to all names claimed
// in the chain, not just opts.DNSName. Thus it is invalid for a leaf to claim
// example.com if an intermediate doesn't permit it, even if example.com is not
// the name being validated. Note that DirectoryName constraints are not
// supported.
//
// Name constraint validation follows the rules from RFC 5280, with the
// addition that DNS name constraints may use the leading period format
// defined for emails and URIs. When a constraint has a leading period
// it indicates that at least one additional label must be prepended to
// the constrained name to be considered valid.
//
// Extended Key Usage values are enforced nested down a chain, so an intermediate
// or root that enumerates EKUs prevents a leaf from asserting an EKU not in that
// list. (While this is not specified, it is common practice in order to limit
// the types of certificates a CA can issue.)
//
// Certificates that use SHA1WithRSA and ECDSAWithSHA1 signatures are not supported,
// and will not be used to build chains.
//
// Certificates other than c in the returned chains should not be modified.
//
// WARNING: this function doesn't do any revocation checking.
func (c *Certificate) Verify(opts VerifyOptions) (chains [][]*Certificate, err error) {
	// Platform-specific verification needs the ASN.1 contents so
	// this makes the behavior consistent across platforms.
	if len(c.Raw) == 0 {
		return nil, errNotParsed
	}
	for i := 0; i < opts.Intermediates.len(); i++ {
		c, _, err := opts.Intermediates.cert(i)
		if err != nil {
			return nil, fmt.Errorf("crypto/x509: error fetching intermediate: %w", err)
		}
		if len(c.Raw) == 0 {
			return nil, errNotParsed
		}
	}

	// Use platform verifiers, where available, if Roots is from SystemCertPool.
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		// Don't use the system verifier if the system pool was replaced with a non-system pool,
		// i.e. if SetFallbackRoots was called with x509usefallbackroots=1.
		systemPool := systemRootsPool()
		if opts.Roots == nil && (systemPool == nil || systemPool.systemPool) {
			return c.systemVerify(&opts)
		}
		if opts.Roots != nil && opts.Roots.systemPool {
			platformChains, err := c.systemVerify(&opts)
			// If the platform verifier succeeded, or there are no additional
			// roots, return the platform verifier result. Otherwise, continue
			// with the Go verifier.
			if err == nil || opts.Roots.len() == 0 {
				return platformChains, err
			}
		}
	}

	if opts.Roots == nil {
		opts.Roots = systemRootsPool()
		if opts.Roots == nil {
			return nil, SystemRootsError{systemRootsErr}
		}
	}

	err = c.isValid(leafCertificate, nil, &opts)
	if err != nil {
		return
	}

	if len(opts.DNSName) > 0 {
		err = c.VerifyHostname(opts.DNSName)
		if err != nil {
			return
		}
	}

	var candidateChains [][]*Certificate
	if opts.Roots.contains(c) {
		candidateChains = [][]*Certificate{{c}}
	} else {
		candidateChains, err = c.buildChains([]*Certificate{c}, nil, &opts)
		if err != nil {
			return nil, err
		}
	}

	chains = make([][]*Certificate, 0, len(candidateChains))

	var invalidPoliciesChains int
	for _, candidate := range candidateChains {
		if !policiesValid(candidate, opts) {
			invalidPoliciesChains++
			continue
		}
		chains = append(chains, candidate)
	}

	if len(chains) == 0 {
		return nil, CertificateInvalidError{c, NoValidChains, "all candidate chains have invalid policies"}
	}

	for _, eku := range opts.KeyUsages {
		if eku == ExtKeyUsageAny {
			// If any key usage is acceptable, no need to check the chain for
			// key usages.
			return chains, nil
		}
	}

	if len(opts.KeyUsages) == 0 {
		opts.KeyUsages = []ExtKeyUsage{ExtKeyUsageServerAuth}
	}

	candidateChains = chains
	chains = chains[:0]

	var incompatibleKeyUsageChains int
	for _, candidate := range candidateChains {
		if !checkChainForKeyUsage(candidate, opts.KeyUsages) {
			incompatibleKeyUsageChains++
			continue
		}
		chains = append(chains, candidate)
	}

	if len(chains) == 0 {
		var details []string
		if incompatibleKeyUsageChains > 0 {
			if invalidPoliciesChains == 0 {
				return nil, CertificateInvalidError{c, IncompatibleUsage, ""}
			}
			details = append(details, fmt.Sprintf("%d chains with incompatible key usage", incompatibleKeyUsageChains))
		}
		if invalidPoliciesChains > 0 {
			details = append(details, fmt.Sprintf("%d chains with invalid policies", invalidPoliciesChains))
		}
		err = CertificateInvalidError{c, NoValidChains, strings.Join(details, ", ")}
		return nil, err
	}

	return chains, nil
}

func appendToFreshChain(chain []*Certificate, cert *Certificate) []*Certificate {
	n := make([]*Certificate, len(chain)+1)
	copy(n, chain)
	n[len(chain)] = cert
	return n
}

// alreadyInChain checks whether a candidate certificate is present in a chain.
// Rather than doing a direct byte for byte equivalency check, we check if the
// subject, public key, and SAN, if present, are equal. This prevents loops that
// are created by mutual cross-signatures, or other cross-signature bridge
// oddities.
func alreadyInChain(candidate *Certificate, chain []*Certificate) bool {
	type pubKeyEqual interface {
		Equal(crypto.PublicKey) bool
	}

	var candidateSAN *pkix.Extension
	for _, ext := range candidate.Extensions {
		if ext.Id.Equal(oidExtensionSubjectAltName) {
			candidateSAN = &ext
			break
		}
	}

	for _, cert := range chain {
		if !bytes.Equal(candidate.RawSubject, cert.RawSubject) {
			continue
		}
		// We enforce the canonical encoding of SPKI (by only allowing the
		// correct AI paremeter encodings in parseCertificate), so it's safe to
		// directly compare the raw bytes.
		if !bytes.Equal(candidate.RawSubjectPublicKeyInfo, cert.RawSubjectPublicKeyInfo) {
			continue
		}
		var certSAN *pkix.Extension
		for _, ext := range cert.Extensions {
			if ext.Id.Equal(oidExtensionSubjectAltName) {
				certSAN = &ext
				break
			}
		}
		if candidateSAN == nil && certSAN == nil {
			return true
		} else if candidateSAN == nil || certSAN == nil {
			return false
		}
		if bytes.Equal(candidateSAN.Value, certSAN.Value) {
			return true
		}
	}
	return false
}

// maxChainSignatureChecks is the maximum number of CheckSignatureFrom calls
// that an invocation of buildChains will (transitively) make. Most chains are
// less than 15 certificates long, so this leaves space for multiple chains and
// for failed checks due to different intermediates having the same Subject.
const maxChainSignatureChecks = 100

func (c *Certificate) buildChains(currentChain []*Certificate, sigChecks *int, opts *VerifyOptions) (chains [][]*Certificate, err error) {
	var (
		hintErr  error
		hintCert *Certificate
	)

	considerCandidate := func(certType int, candidate potentialParent) {
		if candidate.cert.PublicKey == nil || alreadyInChain(candidate.cert, currentChain) {
			return
		}

		if sigChecks == nil {
			sigChecks = new(int)
		}
		*sigChecks++
		if *sigChecks > maxChainSignatureChecks {
			err = errors.New("x509: signature check attempts limit reached while verifying certificate chain")
			return
		}

		if err := c.CheckSignatureFrom(candidate.cert); err != nil {
			if hintErr == nil {
				hintErr = err
				hintCert = candidate.cert
			}
			return
		}

		err = candidate.cert.isValid(certType, currentChain, opts)
		if err != nil {
			if hintErr == nil {
				hintErr = err
				hintCert = candidate.cert
			}
			return
		}

		if candidate.constraint != nil {
			if err := candidate.constraint(currentChain); err != nil {
				if hintErr == nil {
					hintErr = err
					hintCert = candidate.cert
				}
				return
			}
		}

		switch certType {
		case rootCertificate:
			chains = append(chains, appendToFreshChain(currentChain, candidate.cert))
		case intermediateCertificate:
			var childChains [][]*Certificate
			childChains, err = candidate.cert.buildChains(appendToFreshChain(currentChain, candidate.cert), sigChecks, opts)
			chains = append(chains, childChains...)
		}
	}

	for _, root := range opts.Roots.findPotentialParents(c) {
		considerCandidate(rootCertificate, root)
	}
	for _, intermediate := range opts.Intermediates.findPotentialParents(c) {
		considerCandidate(intermediateCertificate, intermediate)
	}

	if len(chains) > 0 {
		err = nil
	}
	if len(chains) == 0 && err == nil {
		err = UnknownAuthorityError{c, hintErr, hintCert}
	}

	return
}

func validHostnamePattern(host string) bool { return validHostname(host, true) }
func validHostnameInput(host string) bool   { return validHostname(host, false) }

// validHostname reports whether host is a valid hostname that can be matched or
// matched against according to RFC 6125 2.2, with some leniency to accommodate
// legacy values.
func validHostname(host string, isPattern bool) bool {
	if !isPattern {
		host = strings.TrimSuffix(host, ".")
	}
	if len(host) == 0 {
		return false
	}
	if host == "*" {
		// Bare wildcards are not allowed, they are not valid DNS names,
		// nor are they allowed per RFC 6125.
		return false
	}

	for i, part := range strings.Split(host, ".") {
		if part == "" {
			// Empty label.
			return false
		}
		if isPattern && i == 0 && part == "*" {
			// Only allow full left-most wildcards, as those are the only ones
			// we match, and matching literal '*' characters is probably never
			// the expected behavior.
			continue
		}
		for j, c := range part {
			if 'a' <= c && c <= 'z' {
				continue
			}
			if '0' <= c && c <= '9' {
				continue
			}
			if 'A' <= c && c <= 'Z' {
				continue
			}
			if c == '-' && j != 0 {
				continue
			}
			if c == '_' {
				// Not a valid character in hostnames, but commonly
				// found in deployments outside the WebPKI.
				continue
			}
			return false
		}
	}

	return true
}

func matchExactly(hostA, hostB string) bool {
	if hostA == "" || hostA == "." || hostB == "" || hostB == "." {
		return false
	}
	return toLowerCaseASCII(hostA) == toLowerCaseASCII(hostB)
}

func matchHostnames(pattern, host string) bool {
	pattern = toLowerCaseASCII(pattern)
	host = toLowerCaseASCII(strings.TrimSuffix(host, "."))

	if len(pattern) == 0 || len(host) == 0 {
		return false
	}

	patternParts := strings.Split(pattern, ".")
	hostParts := strings.Split(host, ".")

	if len(patternParts) != len(hostParts) {
		return false
	}

	for i, patternPart := range patternParts {
		if i == 0 && patternPart == "*" {
			continue
		}
		if patternPart != hostParts[i] {
			return false
		}
	}

	return true
}

// toLowerCaseASCII returns a lower-case version of in. See RFC 6125 6.4.1. We use
// an explicitly ASCII function to avoid any sharp corners resulting from
// performing Unicode operations on DNS labels.
func toLowerCaseASCII(in string) string {
	// If the string is already lower-case then there's nothing to do.
	isAlreadyLowerCase := true
	for _, c := range in {
		if c == utf8.RuneError {
			// If we get a UTF-8 error then there might be
			// upper-case ASCII bytes in the invalid sequence.
			isAlreadyLowerCase = false
			break
		}
		if 'A' <= c && c <= 'Z' {
			isAlreadyLowerCase = false
			break
		}
	}

	if isAlreadyLowerCase {
		return in
	}

	out := []byte(in)
	for i, c := range out {
		if 'A' <= c && c <= 'Z' {
			out[i] += 'a' - 'A'
		}
	}
	return string(out)
}

// VerifyHostname returns nil if c is a valid certificate for the named host.
// Otherwise it returns an error describing the mismatch.
//
// IP addresses can be optionally enclosed in square brackets and are checked
// against the IPAddresses field. Other names are checked case insensitively
// against the DNSNames field. If the names are valid hostnames, the certificate
// fields can have a wildcard as the complete left-most label (e.g. *.example.com).
//
// Note that the legacy Common Name field is ignored.
func (c *Certificate) VerifyHostname(h string) error {
	// IP addresses may be written in [ ].
	candidateIP := h
	if len(h) >= 3 && h[0] == '[' && h[len(h)-1] == ']' {
		candidateIP = h[1 : len(h)-1]
	}
	if ip := net.ParseIP(candidateIP); ip != nil {
		// We only match IP addresses against IP SANs.
		// See RFC 6125, Appendix B.2.
		for _, candidate := range c.IPAddresses {
			if ip.Equal(candidate) {
				return nil
			}
		}
		return HostnameError{c, candidateIP}
	}

	candidateName := toLowerCaseASCII(h) // Save allocations inside the loop.
	validCandidateName := validHostnameInput(candidateName)

	for _, match := range c.DNSNames {
		// Ideally, we'd only match valid hostnames according to RFC 6125 like
		// browsers (more or less) do, but in practice Go is used in a wider
		// array of contexts and can't even assume DNS resolution. Instead,
		// always allow perfect matches, and only apply wildcard and trailing
		// dot processing to valid hostnames.
		if validCandidateName && validHostnamePattern(match) {
			if matchHostnames(match, candidateName) {
				return nil
			}
		} else {
			if matchExactly(match, candidateName) {
				return nil
			}
		}
	}

	return HostnameError{c, h}
}

func checkChainForKeyUsage(chain []*Certificate, keyUsages []ExtKeyUsage) bool {
	usages := make([]ExtKeyUsage, len(keyUsages))
	copy(usages, keyUsages)

	if len(chain) == 0 {
		return false
	}

	usagesRemaining := len(usages)

	// We walk down the list and cross out any usages that aren't supported
	// by each certificate. If we cross out all the usages, then the chain
	// is unacceptable.

NextCert:
	for i := len(chain) - 1; i >= 0; i-- {
		cert := chain[i]
		if len(cert.ExtKeyUsage) == 0 && len(cert.UnknownExtKeyUsage) == 0 {
			// The certificate doesn't have any extended key usage specified.
			continue
		}

		for _, usage := range cert.ExtKeyUsage {
			if usage == ExtKeyUsageAny {
				// The certificate is explicitly good for any usage.
				continue NextCert
			}
		}

		const invalidUsage ExtKeyUsage = -1

	NextRequestedUsage:
		for i, requestedUsage := range usages {
			if requestedUsage == invalidUsage {
				continue
			}

			for _, usage := range cert.ExtKeyUsage {
				if requestedUsage == usage {
					continue NextRequestedUsage
				}
			}

			usages[i] = invalidUsage
			usagesRemaining--
			if usagesRemaining == 0 {
				return false
			}
		}
	}

	return true
}

func mustNewOIDFromInts(ints []uint64) OID {
	oid, err := OIDFromInts(ints)
	if err != nil {
		panic(fmt.Sprintf("OIDFromInts(%v) unexpected error: %v", ints, err))
	}
	return oid
}

type policyGraphNode struct {
	validPolicy       OID
	expectedPolicySet []OID
	// we do not implement qualifiers, so we don't track qualifier_set

	parents  map[*policyGraphNode]bool
	children map[*policyGraphNode]bool
}

func newPolicyGraphNode(valid OID, parents []*policyGraphNode) *policyGraphNode {
	n := &policyGraphNode{
		validPolicy:       valid,
		expectedPolicySet: []OID{valid},
		children:          map[*policyGraphNode]bool{},
		parents:           map[*policyGraphNode]bool{},
	}
	for _, p := range parents {
		p.children[n] = true
		n.parents[p] = true
	}
	return n
}

type policyGraph struct {
	strata []map[string]*policyGraphNode
	// map of OID -> nodes at strata[depth-1] with OID in their expectedPolicySet
	parentIndex map[string][]*policyGraphNode
	depth       int
}

var anyPolicyOID = mustNewOIDFromInts([]uint64{2, 5, 29, 32, 0})

func newPolicyGraph() *policyGraph {
	root := policyGraphNode{
		validPolicy:       anyPolicyOID,
		expectedPolicySet: []OID{anyPolicyOID},
		children:          map[*policyGraphNode]bool{},
		parents:           map[*policyGraphNode]bool{},
	}
	return &policyGraph{
		depth:  0,
		strata: []map[string]*policyGraphNode{{string(anyPolicyOID.der): &root}},
	}
}

func (pg *policyGraph) insert(n *policyGraphNode) {
	pg.strata[pg.depth][string(n.validPolicy.der)] = n
}

func (pg *policyGraph) parentsWithExpected(expected OID) []*policyGraphNode {
	if pg.depth == 0 {
		return nil
	}
	return pg.parentIndex[string(expected.der)]
}

func (pg *policyGraph) parentWithAnyPolicy() *policyGraphNode {
	if pg.depth == 0 {
		return nil
	}
	return pg.strata[pg.depth-1][string(anyPolicyOID.der)]
}

func (pg *policyGraph) parents() iter.Seq[*policyGraphNode] {
	if pg.depth == 0 {
		return nil
	}
	return maps.Values(pg.strata[pg.depth-1])
}

func (pg *policyGraph) leaves() map[string]*policyGraphNode {
	return pg.strata[pg.depth]
}

func (pg *policyGraph) leafWithPolicy(policy OID) *policyGraphNode {
	return pg.strata[pg.depth][string(policy.der)]
}

func (pg *policyGraph) deleteLeaf(policy OID) {
	n := pg.strata[pg.depth][string(policy.der)]
	if n == nil {
		return
	}
	for p := range n.parents {
		delete(p.children, n)
	}
	for c := range n.children {
		delete(c.parents, n)
	}
	delete(pg.strata[pg.depth], string(policy.der))
}

func (pg *policyGraph) validPolicyNodes() []*policyGraphNode {
	var validNodes []*policyGraphNode
	for i := pg.depth; i >= 0; i-- {
		for _, n := range pg.strata[i] {
			if n.validPolicy.Equal(anyPolicyOID) {
				continue
			}

			if len(n.parents) == 1 {
				for p := range n.parents {
					if p.validPolicy.Equal(anyPolicyOID) {
						validNodes = append(validNodes, n)
					}
				}
			}
		}
	}
	return validNodes
}

func (pg *policyGraph) prune() {
	for i := pg.depth - 1; i > 0; i-- {
		for _, n := range pg.strata[i] {
			if len(n.children) == 0 {
				for p := range n.parents {
					delete(p.children, n)
				}
				delete(pg.strata[i], string(n.validPolicy.der))
			}
		}
	}
}

func (pg *policyGraph) incrDepth() {
	pg.parentIndex = map[string][]*policyGraphNode{}
	for _, n := range pg.strata[pg.depth] {
		for _, e := range n.expectedPolicySet {
			pg.parentIndex[string(e.der)] = append(pg.parentIndex[string(e.der)], n)
		}
	}

	pg.depth++
	pg.strata = append(pg.strata, map[string]*policyGraphNode{})
}

func policiesValid(chain []*Certificate, opts VerifyOptions) bool {
	// The following code implements the policy verification algorithm as
	// specified in RFC 5280 and updated by RFC 9618. In particular the
	// following sections are replaced by RFC 9618:
	//	* 6.1.2 (a)
	//	* 6.1.3 (d)
	//	* 6.1.3 (e)
	//	* 6.1.3 (f)
	//	* 6.1.4 (b)
	//	* 6.1.5 (g)

	if len(chain) == 1 {
		return true
	}

	// n is the length of the chain minus the trust anchor
	n := len(chain) - 1

	pg := newPolicyGraph()
	var inhibitAnyPolicy, explicitPolicy, policyMapping int
	if !opts.inhibitAnyPolicy {
		inhibitAnyPolicy = n + 1
	}
	if !opts.requireExplicitPolicy {
		explicitPolicy = n + 1
	}
	if !opts.inhibitPolicyMapping {
		policyMapping = n + 1
	}

	initialUserPolicySet := map[string]bool{}
	for _, p := range opts.CertificatePolicies {
		initialUserPolicySet[string(p.der)] = true
	}
	// If the user does not pass any policies, we consider
	// that equivalent to passing anyPolicyOID.
	if len(initialUserPolicySet) == 0 {
		initialUserPolicySet[string(anyPolicyOID.der)] = true
	}

	for i := n - 1; i >= 0; i-- {
		cert := chain[i]

		isSelfSigned := bytes.Equal(cert.RawIssuer, cert.RawSubject)

		// 6.1.3 (e) -- as updated by RFC 9618
		if len(cert.Policies) == 0 {
			pg = nil
		}

		// 6.1.3 (f) -- as updated by RFC 9618
		if explicitPolicy == 0 && pg == nil {
			return false
		}

		if pg != nil {
			pg.incrDepth()

			policies := map[string]bool{}

			// 6.1.3 (d) (1) -- as updated by RFC 9618
			for _, policy := range cert.Policies {
				policies[string(policy.der)] = true

				if policy.Equal(anyPolicyOID) {
					continue
				}

				// 6.1.3 (d) (1) (i) -- as updated by RFC 9618
				parents := pg.parentsWithExpected(policy)
				if len(parents) == 0 {
					// 6.1.3 (d) (1) (ii) -- as updated by RFC 9618
					if anyParent := pg.parentWithAnyPolicy(); anyParent != nil {
						parents = []*policyGraphNode{anyParent}
					}
				}
				if len(parents) > 0 {
					pg.insert(newPolicyGraphNode(policy, parents))
				}
			}

			// 6.1.3 (d) (2) -- as updated by RFC 9618
			// NOTE: in the check "n-i < n" our i is different from the i in the specification.
			// In the specification chains go from the trust anchor to the leaf, whereas our
			// chains go from the leaf to the trust anchor, so our i's our inverted. Our
			// check here matches the check "i < n" in the specification.
			if policies[string(anyPolicyOID.der)] && (inhibitAnyPolicy > 0 || (n-i < n && isSelfSigned)) {
				missing := map[string][]*policyGraphNode{}
				leaves := pg.leaves()
				for p := range pg.parents() {
					for _, expected := range p.expectedPolicySet {
						if leaves[string(expected.der)] == nil {
							missing[string(expected.der)] = append(missing[string(expected.der)], p)
						}
					}
				}

				for oidStr, parents := range missing {
					pg.insert(newPolicyGraphNode(OID{der: []byte(oidStr)}, parents))
				}
			}

			// 6.1.3 (d) (3) -- as updated by RFC 9618
			pg.prune()

			if i != 0 {
				// 6.1.4 (b) -- as updated by RFC 9618
				if len(cert.PolicyMappings) > 0 {
					// collect map of issuer -> []subject
					mappings := map[string][]OID{}

					for _, mapping := range cert.PolicyMappings {
						if policyMapping > 0 {
							if mapping.IssuerDomainPolicy.Equal(anyPolicyOID) || mapping.SubjectDomainPolicy.Equal(anyPolicyOID) {
								// Invalid mapping
								return false
							}
							mappings[string(mapping.IssuerDomainPolicy.der)] = append(mappings[string(mapping.IssuerDomainPolicy.der)], mapping.SubjectDomainPolicy)
						} else {
							// 6.1.4 (b) (3) (i) -- as updated by RFC 9618
							pg.deleteLeaf(mapping.IssuerDomainPolicy)

							// 6.1.4 (b) (3) (ii) -- as updated by RFC 9618
							pg.prune()
						}
					}

					for issuerStr, subjectPolicies := range mappings {
						// 6.1.4 (b) (1) -- as updated by RFC 9618
						if matching := pg.leafWithPolicy(OID{der: []byte(issuerStr)}); matching != nil {
							matching.expectedPolicySet = subjectPolicies
						} else if matching := pg.leafWithPolicy(anyPolicyOID); matching != nil {
							// 6.1.4 (b) (2) -- as updated by RFC 9618
							n := newPolicyGraphNode(OID{der: []byte(issuerStr)}, []*policyGraphNode{matching})
							n.expectedPolicySet = subjectPolicies
							pg.insert(n)
						}
					}
				}
			}
		}

		if i != 0 {
			// 6.1.4 (h)
			if !isSelfSigned {
				if explicitPolicy > 0 {
					explicitPolicy--
				}
				if policyMapping > 0 {
					policyMapping--
				}
				if inhibitAnyPolicy > 0 {
					inhibitAnyPolicy--
				}
			}

			// 6.1.4 (i)
			if (cert.RequireExplicitPolicy > 0 || cert.RequireExplicitPolicyZero) && cert.RequireExplicitPolicy < explicitPolicy {
				explicitPolicy = cert.RequireExplicitPolicy
			}
			if (cert.InhibitPolicyMapping > 0 || cert.InhibitPolicyMappingZero) && cert.InhibitPolicyMapping < policyMapping {
				policyMapping = cert.InhibitPolicyMapping
			}
			// 6.1.4 (j)
			if (cert.InhibitAnyPolicy > 0 || cert.InhibitAnyPolicyZero) && cert.InhibitAnyPolicy < inhibitAnyPolicy {
				inhibitAnyPolicy = cert.InhibitAnyPolicy
			}
		}
	}

	// 6.1.5 (a)
	if explicitPolicy > 0 {
		explicitPolicy--
	}

	// 6.1.5 (b)
	if chain[0].RequireExplicitPolicyZero {
		explicitPolicy = 0
	}

	// 6.1.5 (g) (1) -- as updated by RFC 9618
	var validPolicyNodeSet []*policyGraphNode
	// 6.1.5 (g) (2) -- as updated by RFC 9618
	if pg != nil {
		validPolicyNodeSet = pg.validPolicyNodes()
		// 6.1.5 (g) (3) -- as updated by RFC 9618
		if currentAny := pg.leafWithPolicy(anyPolicyOID); currentAny != nil {
			validPolicyNodeSet = append(validPolicyNodeSet, currentAny)
		}
	}

	// 6.1.5 (g) (4) -- as updated by RFC 9618
	authorityConstrainedPolicySet := map[string]bool{}
	for _, n := range validPolicyNodeSet {
		authorityConstrainedPolicySet[string(n.validPolicy.der)] = true
	}
	// 6.1.5 (g) (5) -- as updated by RFC 9618
	userConstrainedPolicySet := maps.Clone(authorityConstrainedPolicySet)
	// 6.1.5 (g) (6) -- as updated by RFC 9618
	if len(initialUserPolicySet) != 1 || !initialUserPolicySet[string(anyPolicyOID.der)] {
		// 6.1.5 (g) (6) (i) -- as updated by RFC 9618
		for p := range userConstrainedPolicySet {
			if !initialUserPolicySet[p] {
				delete(userConstrainedPolicySet, p)
			}
		}
		// 6.1.5 (g) (6) (ii) -- as updated by RFC 9618
		if authorityConstrainedPolicySet[string(anyPolicyOID.der)] {
			for policy := range initialUserPolicySet {
				userConstrainedPolicySet[policy] = true
			}
		}
	}

	if explicitPolicy == 0 && len(userConstrainedPolicySet) == 0 {
		return false
	}

	return true
}
