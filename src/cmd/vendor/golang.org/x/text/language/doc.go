// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package language implements BCP 47 language tags and related functionality.
//
// The most important function of package language is to match a list of
// user-preferred languages to a list of supported languages.
// It alleviates the developer of dealing with the complexity of this process
// and provides the user with the best experience
// (see https://blog.golang.org/matchlang).
//
// # Matching preferred against supported languages
//
// A Matcher for an application that supports English, Australian English,
// Danish, and standard Mandarin can be created as follows:
//
//	var matcher = language.NewMatcher([]language.Tag{
//	    language.English,   // The first language is used as fallback.
//	    language.MustParse("en-AU"),
//	    language.Danish,
//	    language.Chinese,
//	})
//
// This list of supported languages is typically implied by the languages for
// which there exists translations of the user interface.
//
// User-preferred languages usually come as a comma-separated list of BCP 47
// language tags.
// The MatchString finds best matches for such strings:
//
//	handler(w http.ResponseWriter, r *http.Request) {
//	    lang, _ := r.Cookie("lang")
//	    accept := r.Header.Get("Accept-Language")
//	    tag, _ := language.MatchStrings(matcher, lang.String(), accept)
//
//	    // tag should now be used for the initialization of any
//	    // locale-specific service.
//	}
//
// The Matcher's Match method can be used to match Tags directly.
//
// Matchers are aware of the intricacies of equivalence between languages, such
// as deprecated subtags, legacy tags, macro languages, mutual
// intelligibility between scripts and languages, and transparently passing
// BCP 47 user configuration.
// For instance, it will know that a reader of Bokmål Danish can read Norwegian
// and will know that Cantonese ("yue") is a good match for "zh-HK".
//
// # Using match results
//
// To guarantee a consistent user experience to the user it is important to
// use the same language tag for the selection of any locale-specific services.
// For example, it is utterly confusing to substitute spelled-out numbers
// or dates in one language in text of another language.
// More subtly confusing is using the wrong sorting order or casing
// algorithm for a certain language.
//
// All the packages in x/text that provide locale-specific services
// (e.g. collate, cases) should be initialized with the tag that was
// obtained at the start of an interaction with the user.
//
// Note that Tag that is returned by Match and MatchString may differ from any
// of the supported languages, as it may contain carried over settings from
// the user tags.
// This may be inconvenient when your application has some additional
// locale-specific data for your supported languages.
// Match and MatchString both return the index of the matched supported tag
// to simplify associating such data with the matched tag.
//
// # Canonicalization
//
// If one uses the Matcher to compare languages one does not need to
// worry about canonicalization.
//
// The meaning of a Tag varies per application. The language package
// therefore delays canonicalization and preserves information as much
// as possible. The Matcher, however, will always take into account that
// two different tags may represent the same language.
//
// By default, only legacy and deprecated tags are converted into their
// canonical equivalent. All other information is preserved. This approach makes
// the confidence scores more accurate and allows matchers to distinguish
// between variants that are otherwise lost.
//
// As a consequence, two tags that should be treated as identical according to
// BCP 47 or CLDR, like "en-Latn" and "en", will be represented differently. The
// Matcher handles such distinctions, though, and is aware of the
// equivalence relations. The CanonType type can be used to alter the
// canonicalization form.
//
// # References
//
// BCP 47 - Tags for Identifying Languages http://tools.ietf.org/html/bcp47
package language // import "golang.org/x/text/language"

// TODO: explanation on how to match languages for your own locale-specific
// service.
