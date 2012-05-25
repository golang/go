// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"strings"
)

func adjustAttributeNames(aa []Attribute, nameMap map[string]string) {
	for i := range aa {
		if newName, ok := nameMap[aa[i].Key]; ok {
			aa[i].Key = newName
		}
	}
}

func adjustForeignAttributes(aa []Attribute) {
	for i, a := range aa {
		if a.Key == "" || a.Key[0] != 'x' {
			continue
		}
		switch a.Key {
		case "xlink:actuate", "xlink:arcrole", "xlink:href", "xlink:role", "xlink:show",
			"xlink:title", "xlink:type", "xml:base", "xml:lang", "xml:space", "xmlns:xlink":
			j := strings.Index(a.Key, ":")
			aa[i].Namespace = a.Key[:j]
			aa[i].Key = a.Key[j+1:]
		}
	}
}

func htmlIntegrationPoint(n *Node) bool {
	if n.Type != ElementNode {
		return false
	}
	switch n.Namespace {
	case "math":
		if n.Data == "annotation-xml" {
			for _, a := range n.Attr {
				if a.Key == "encoding" {
					val := strings.ToLower(a.Val)
					if val == "text/html" || val == "application/xhtml+xml" {
						return true
					}
				}
			}
		}
	case "svg":
		switch n.Data {
		case "desc", "foreignObject", "title":
			return true
		}
	}
	return false
}

func mathMLTextIntegrationPoint(n *Node) bool {
	if n.Namespace != "math" {
		return false
	}
	switch n.Data {
	case "mi", "mo", "mn", "ms", "mtext":
		return true
	}
	return false
}

// Section 12.2.5.5.
var breakout = map[string]bool{
	"b":          true,
	"big":        true,
	"blockquote": true,
	"body":       true,
	"br":         true,
	"center":     true,
	"code":       true,
	"dd":         true,
	"div":        true,
	"dl":         true,
	"dt":         true,
	"em":         true,
	"embed":      true,
	"h1":         true,
	"h2":         true,
	"h3":         true,
	"h4":         true,
	"h5":         true,
	"h6":         true,
	"head":       true,
	"hr":         true,
	"i":          true,
	"img":        true,
	"li":         true,
	"listing":    true,
	"menu":       true,
	"meta":       true,
	"nobr":       true,
	"ol":         true,
	"p":          true,
	"pre":        true,
	"ruby":       true,
	"s":          true,
	"small":      true,
	"span":       true,
	"strong":     true,
	"strike":     true,
	"sub":        true,
	"sup":        true,
	"table":      true,
	"tt":         true,
	"u":          true,
	"ul":         true,
	"var":        true,
}

// Section 12.2.5.5.
var svgTagNameAdjustments = map[string]string{
	"altglyph":            "altGlyph",
	"altglyphdef":         "altGlyphDef",
	"altglyphitem":        "altGlyphItem",
	"animatecolor":        "animateColor",
	"animatemotion":       "animateMotion",
	"animatetransform":    "animateTransform",
	"clippath":            "clipPath",
	"feblend":             "feBlend",
	"fecolormatrix":       "feColorMatrix",
	"fecomponenttransfer": "feComponentTransfer",
	"fecomposite":         "feComposite",
	"feconvolvematrix":    "feConvolveMatrix",
	"fediffuselighting":   "feDiffuseLighting",
	"fedisplacementmap":   "feDisplacementMap",
	"fedistantlight":      "feDistantLight",
	"feflood":             "feFlood",
	"fefunca":             "feFuncA",
	"fefuncb":             "feFuncB",
	"fefuncg":             "feFuncG",
	"fefuncr":             "feFuncR",
	"fegaussianblur":      "feGaussianBlur",
	"feimage":             "feImage",
	"femerge":             "feMerge",
	"femergenode":         "feMergeNode",
	"femorphology":        "feMorphology",
	"feoffset":            "feOffset",
	"fepointlight":        "fePointLight",
	"fespecularlighting":  "feSpecularLighting",
	"fespotlight":         "feSpotLight",
	"fetile":              "feTile",
	"feturbulence":        "feTurbulence",
	"foreignobject":       "foreignObject",
	"glyphref":            "glyphRef",
	"lineargradient":      "linearGradient",
	"radialgradient":      "radialGradient",
	"textpath":            "textPath",
}

// Section 12.2.5.1
var mathMLAttributeAdjustments = map[string]string{
	"definitionurl": "definitionURL",
}

var svgAttributeAdjustments = map[string]string{
	"attributename":             "attributeName",
	"attributetype":             "attributeType",
	"basefrequency":             "baseFrequency",
	"baseprofile":               "baseProfile",
	"calcmode":                  "calcMode",
	"clippathunits":             "clipPathUnits",
	"contentscripttype":         "contentScriptType",
	"contentstyletype":          "contentStyleType",
	"diffuseconstant":           "diffuseConstant",
	"edgemode":                  "edgeMode",
	"externalresourcesrequired": "externalResourcesRequired",
	"filterres":                 "filterRes",
	"filterunits":               "filterUnits",
	"glyphref":                  "glyphRef",
	"gradienttransform":         "gradientTransform",
	"gradientunits":             "gradientUnits",
	"kernelmatrix":              "kernelMatrix",
	"kernelunitlength":          "kernelUnitLength",
	"keypoints":                 "keyPoints",
	"keysplines":                "keySplines",
	"keytimes":                  "keyTimes",
	"lengthadjust":              "lengthAdjust",
	"limitingconeangle":         "limitingConeAngle",
	"markerheight":              "markerHeight",
	"markerunits":               "markerUnits",
	"markerwidth":               "markerWidth",
	"maskcontentunits":          "maskContentUnits",
	"maskunits":                 "maskUnits",
	"numoctaves":                "numOctaves",
	"pathlength":                "pathLength",
	"patterncontentunits":       "patternContentUnits",
	"patterntransform":          "patternTransform",
	"patternunits":              "patternUnits",
	"pointsatx":                 "pointsAtX",
	"pointsaty":                 "pointsAtY",
	"pointsatz":                 "pointsAtZ",
	"preservealpha":             "preserveAlpha",
	"preserveaspectratio":       "preserveAspectRatio",
	"primitiveunits":            "primitiveUnits",
	"refx":                      "refX",
	"refy":                      "refY",
	"repeatcount":               "repeatCount",
	"repeatdur":                 "repeatDur",
	"requiredextensions":        "requiredExtensions",
	"requiredfeatures":          "requiredFeatures",
	"specularconstant":          "specularConstant",
	"specularexponent":          "specularExponent",
	"spreadmethod":              "spreadMethod",
	"startoffset":               "startOffset",
	"stddeviation":              "stdDeviation",
	"stitchtiles":               "stitchTiles",
	"surfacescale":              "surfaceScale",
	"systemlanguage":            "systemLanguage",
	"tablevalues":               "tableValues",
	"targetx":                   "targetX",
	"targety":                   "targetY",
	"textlength":                "textLength",
	"viewbox":                   "viewBox",
	"viewtarget":                "viewTarget",
	"xchannelselector":          "xChannelSelector",
	"ychannelselector":          "yChannelSelector",
	"zoomandpan":                "zoomAndPan",
}
